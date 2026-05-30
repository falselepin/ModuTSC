"""用户插件动态加载器。

支持运行时加载用户上传的 .py 文件作为自定义组件，
包括安全校验、接口契约检查、动态注册与卸载。
"""

import ast
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from modutsc.scheduling.registry import (
    _canonical_kind, _KIND_CONTRACT, _registry, register_dynamic,
)

# ── 用户插件目录 ──
USER_PLUGINS_DIR = Path(os.environ.get("USER_PLUGINS_DIR", "user_plugins"))
INDEX_FILE = USER_PLUGINS_DIR / "index.json"


# ── 安全白名单 ──
_ALLOWED_IMPORTS = {
    "modutsc", "modutsc.scheduling.registry",
    "modutsc.env",
    "modutsc.plugins", "modutsc.plugins.algorithms", "modutsc.plugins.observers",
    "modutsc.plugins.actors", "modutsc.plugins.collectors", "modutsc.plugins.rewards",
    "modutsc.plugins.trackers",
    "modutsc.orchestration",
    "torch", "torch.nn", "torch.nn.functional", "torch.optim",
    "numpy", "traci",
    "typing", "abc", "dataclasses",
    "collections", "math", "random", "itertools", "functools",
    "json", "logging", "warnings",
}

_FORBIDDEN_FUNCTIONS = {"eval", "exec", "compile", "__import__"}
_FORBIDDEN_MODULES = {"os", "sys", "subprocess", "shutil", "socket", "ctypes"}
# pickle 仅在 save/load 方法内允许（函数级 import）
_CONTEXT_ALLOWED_MODULES = {"pickle"}


def _validate_source_security(source: str) -> List[str]:
    """AST 级别安全检查，返回违规列表。"""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"语法错误: 第{e.lineno}行 - {e.msg}"]

    violations = []

    # 先收集每个类中 save/load 方法的行号范围，用于上下文豁免
    safe_lines = set()  # 允许 pickle / open('wb') 的行号
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in ("save", "load"):
                    for child in ast.walk(item):
                        if hasattr(child, "lineno"):
                            safe_lines.add(child.lineno)

    for node in ast.walk(tree):
        # 检查禁止的 import — ast.Import 与 ast.ImportFrom 分开处理
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _CONTEXT_ALLOWED_MODULES and node.lineno in safe_lines:
                    continue
                _check_import(alias.name, violations)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name in _CONTEXT_ALLOWED_MODULES and node.lineno in safe_lines:
                continue
            _check_import(module_name, violations)
            for alias in node.names:
                full = f"{module_name}.{alias.name}"
                _check_import(full, violations)

        # 检查禁止的函数调用
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_FUNCTIONS:
                    violations.append(
                        f"第{node.lineno}行: 禁止使用 {node.func.id}()"
                    )

        # 检查 open() 写入模式 — save/load 方法内允许
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "open" and len(node.args) >= 2:
                mode = node.args[1]
                if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
                    if any(c in mode.value for c in "wa+"):
                        if node.lineno not in safe_lines:
                            violations.append(
                                f"第{node.lineno}行: 禁止文件写入操作 open(..., '{mode.value}')"
                            )

        # 检查属性访问型危险调用
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if (node.func.value.id, node.func.attr) in {
                    ("os", "remove"), ("os", "rmdir"), ("os", "system"),
                    ("shutil", "rmtree"), ("shutil", "copy"),
                    ("subprocess", "run"), ("subprocess", "Popen"),
                }:
                    violations.append(
                        f"第{node.lineno}行: 禁止调用 {node.func.value.id}.{node.func.attr}()"
                    )

    return violations


def _check_import(name: str, violations: List[str]):
    """检查导入是否在白名单内。"""
    top = name.split(".")[0]
    if top in _FORBIDDEN_MODULES:
        violations.append(f"禁止导入 {name}")
        return
    # 逐级检查白名单
    parts = name.split(".")
    for i in range(len(parts)):
        prefix = ".".join(parts[: i + 1])
        if prefix in _ALLOWED_IMPORTS:
            return
    if top not in _ALLOWED_IMPORTS:
        violations.append(f"未授权的导入: {name}（仅允许: {', '.join(sorted(_ALLOWED_IMPORTS))}）")


def _find_register_class(source: str) -> Optional[str]:
    """查找源码中 @register 装饰的类名。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    if dec.func.id == "register":
                        return node.name
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                    if dec.func.attr == "register":
                        return node.name
    return None


def _ensure_dirs():
    """确保用户插件目录结构存在。"""
    USER_PLUGINS_DIR.mkdir(exist_ok=True)
    for kind in _KIND_CONTRACT:
        (USER_PLUGINS_DIR / kind).mkdir(exist_ok=True)


def _load_index() -> dict:
    """读取插件索引。"""
    if not INDEX_FILE.exists():
        return {"plugins": {}}
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"plugins": {}}


def _save_index(data: dict):
    """保存插件索引。"""
    _ensure_dirs()
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── 公共 API ──

def validate_plugin_source(source: str, kind: str) -> Dict:
    """上传前对源码进行安全检查 + 契约预检。

    Returns:
        {"valid": bool, "violations": [...], "class_name": str|None}
    """
    kind = _canonical_kind(kind)
    if kind not in _KIND_CONTRACT:
        return {"valid": False, "violations": [f"未知的组件类型: {kind}"], "class_name": None}

    violations = _validate_source_security(source)
    class_name = _find_register_class(source)

    if not class_name:
        violations.append("未找到 @register(kind, name) 装饰的类")

    # 检查必需方法
    required_methods = _KIND_CONTRACT.get(kind, [])
    if required_methods and class_name:
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    class_methods = {
                        n.name for n in node.body
                        if isinstance(n, ast.FunctionDef)
                    }
                    missing = [m for m in required_methods if m not in class_methods]
                    if missing:
                        violations.append(
                            f"类 '{class_name}' 缺少必需方法: {', '.join(missing)}"
                        )
        except SyntaxError:
            pass

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "class_name": class_name,
    }


def import_user_plugin(kind: str, file_path: str) -> Optional[type]:
    """加载单个用户 .py 文件并触发 @register 装饰器注册。

    Returns:
        注册后的类，失败返回 None。
    """
    kind = _canonical_kind(kind)
    file_path = str(file_path)

    # 读取源码进行安全校验
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError:
        return None

    violations = _validate_source_security(source)
    if violations:
        raise ValueError(f"安全检查失败: {'; '.join(violations)}")

    # 动态导入
    module_name = f"_user_plugin_{kind}_{Path(file_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"模块执行失败: {e}")

    # 查找注册的类
    registered_cls = None
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, type) and hasattr(obj, "setup"):
            registered_cls = obj
            break

    return registered_cls


def discover_user_plugins() -> Dict[str, Dict]:
    """扫描 user_plugins/ 目录，注册所有用户插件。

    Returns:
        加载结果统计 {"loaded": int, "failed": int, "details": [...]}
    """
    _ensure_dirs()
    result = {"loaded": 0, "failed": 0, "details": []}
    index = _load_index()
    
    # 先加载用户注册的组件类型
    kinds_dir = USER_PLUGINS_DIR / "kinds"
    if kinds_dir.exists() and kinds_dir.is_dir():
        for kind_dir in sorted(kinds_dir.iterdir()):
            if not kind_dir.is_dir():
                continue
            kind = _canonical_kind(kind_dir.name)
            # 查找基类文件
            base_file = kind_dir / f"{kind}_base.py"
            if not base_file.exists():
                # 尝试找任何 .py 文件
                py_files = list(kind_dir.glob("*.py"))
                if py_files:
                    base_file = py_files[0]
            if base_file.exists():
                try:
                    # 导入基类
                    import_user_plugin(kind, str(base_file))
                    # 注册类型
                    from modutsc.scheduling.registry import register_component_kind
                    register_component_kind(kind)
                    # 更新索引
                    if "kinds" not in index:
                        index["kinds"] = []
                    if kind not in index["kinds"]:
                        index["kinds"].append(kind)
                    result["details"].append({
                        "name": kind, "kind": "kind", "status": "loaded",
                    })
                except Exception as e:
                    result["details"].append({
                        "name": kind, "kind": "kind", "status": "failed",
                        "error": str(e)[:200],
                    })

    for kind_dir in sorted(USER_PLUGINS_DIR.iterdir()):
        if not kind_dir.is_dir() or kind_dir.name == "kinds":
            continue
        kind = _canonical_kind(kind_dir.name)
        if kind not in _KIND_CONTRACT:
            continue

        for py_file in sorted(kind_dir.glob("*.py")):
            plugin_name = py_file.stem
            try:
                import_user_plugin(kind, str(py_file))
                result["loaded"] += 1
                result["details"].append({
                    "name": plugin_name, "kind": kind, "status": "loaded",
                })
                # 确保已加载的插件出现在索引中（register_user_plugin 的冲突检查依赖索引）
                if plugin_name not in index.get("plugins", {}):
                    index.setdefault("plugins", {})[plugin_name] = {
                        "kind": kind,
                        "file": str(py_file.relative_to(USER_PLUGINS_DIR)),
                        "name": plugin_name,
                        "author": "",
                        "description": "",
                        "status": "valid",
                        "methods": [],
                        "config_keys": [],
                    }
                else:
                    index["plugins"][plugin_name]["status"] = "valid"
            except Exception as e:
                result["failed"] += 1
                result["details"].append({
                    "name": plugin_name, "kind": kind, "status": "failed",
                    "error": str(e)[:200],
                })
                # 更新索引状态
                if plugin_name in index.get("plugins", {}):
                    index["plugins"][plugin_name]["status"] = "error"

    _save_index(index)
    return result


def register_user_plugin(
    kind: str,
    name: str,
    source: str,
    author: str = "",
    description: str = "",
) -> Dict:
    """完整的用户插件注册流程：保存文件 → 加载 → 更新索引。

    Returns:
        {"id": str, "kind": str, "status": str, "methods": [...], "config_keys": [...]}
    """
    kind = _canonical_kind(kind)
    if kind not in _KIND_CONTRACT:
        raise ValueError(f"未知的组件类型: {kind}")

    from modutsc.scheduling.config_solver import scan_setup_cfg_keys

    # 安全检查
    result = validate_plugin_source(source, kind)
    if not result["valid"]:
        return {
            "id": name, "kind": kind, "status": "invalid",
            "violations": result["violations"],
            "methods": [], "config_keys": [],
        }

    # 冲突检查：不能与内置插件同名（用户插件允许覆盖）
    from modutsc.scheduling.registry import find as _find
    if _find(kind, name) is not None:
        index = _load_index()
        if name not in index.get("plugins", {}):
            # 不在用户索引中，检查是否是 discover_user_plugins 预加载的
            if (USER_PLUGINS_DIR / kind / f"{name}.py").exists():
                # 预加载的插件，在索引中补一条记录后允许覆盖
                index.setdefault("plugins", {})[name] = {
                    "kind": kind, "file": f"{kind}/{name}.py",
                    "name": name, "author": "", "description": "",
                    "status": "valid", "methods": [], "config_keys": [],
                }
                _save_index(index)
            else:
                raise ValueError(f"插件名 '{name}' 与内置插件冲突，请更换名称")

    # 创建目录
    _ensure_dirs()
    kind_dir = USER_PLUGINS_DIR / kind
    file_path = kind_dir / f"{name}.py"

    # 如果是覆盖已有文件，先卸载旧的
    if file_path.exists():
        unregister_user_plugin(kind, name)

    # 写入文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(source)

    # 动态加载
    cls = import_user_plugin(kind, str(file_path))
    if cls is None:
        raise ImportError(f"无法从文件中加载类: {name}")

    # 收集方法列表和配置参数
    required_methods = _KIND_CONTRACT.get(kind, [])
    methods = [m for m in required_methods if hasattr(cls, m)]
    config_keys = sorted(scan_setup_cfg_keys(cls)) if cls else []

    # 更新索引
    index = _load_index()
    index["plugins"][name] = {
        "kind": kind,
        "file": str(file_path.relative_to(USER_PLUGINS_DIR)),
        "name": name,
        "author": author,
        "description": description,
        "status": "valid",
        "methods": methods,
        "config_keys": config_keys,
    }
    _save_index(index)

    # 重建兼容性缓存
    try:
        from modutsc.api import invalidate_compatibility_cache
        invalidate_compatibility_cache()
    except ImportError:
        pass

    return {
        "id": name, "kind": kind, "status": "valid",
        "violations": [],
        "methods": methods,
        "config_keys": config_keys,
    }


def unregister_user_plugin(kind: str, name: str) -> bool:
    """删除用户插件：从 registry 移除、删除文件、更新索引。

    Returns:
        True 表示成功，False 表示插件不存在。
    """
    kind = _canonical_kind(kind)

    # 从 registry 移除
    key = (kind, str(name))  # registry key 是字符串 name
    if key in _registry:
        del _registry[key]
    else:
        return False

    # 删除文件
    file_path = USER_PLUGINS_DIR / kind / f"{name}.py"
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            pass

    # 更新索引
    index = _load_index()
    if name in index.get("plugins", {}):
        del index["plugins"][name]
        _save_index(index)

    # 重建兼容性缓存
    try:
        from modutsc.api import invalidate_compatibility_cache
        invalidate_compatibility_cache()
    except ImportError:
        pass

    return True


def list_user_plugins() -> Dict:
    """列出所有用户插件，按 kind 分组。"""
    index = _load_index()
    result = {}
    for plugin_name, info in index.get("plugins", {}).items():
        kind = info.get("kind", "unknown")
        if kind not in result:
            result[kind] = []
        result[kind].append({
            "id": plugin_name,
            "name": plugin_name,
            "kind": kind,
            "status": info.get("status", "unknown"),
            "author": info.get("author", ""),
            "description": info.get("description", ""),
            "methods": info.get("methods", []),
            "config_keys": info.get("config_keys", []),
        })
    # 确保所有 kind 都有键
    for kind in _KIND_CONTRACT:
        if kind not in result:
            result[kind] = []
    return result


def get_user_plugin_detail(name: str) -> Optional[Dict]:
    """获取单个插件的详细信息，包含源码。"""
    index = _load_index()
    info = index.get("plugins", {}).get(name)
    if not info:
        return None

    file_path = USER_PLUGINS_DIR / info["file"]
    source = ""
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError:
            pass

    return {
        **info,
        "source": source,
    }


def scaffold_plugin_template(kind: str, name: str) -> str:
    """生成插件代码模板（继承抽象基类，自动实现必需方法）。

    复用 modutsc.scheduling.scaffold 中已有的 _load_abc / _scan_abc / _generate_code，
    通过 inspect 自动扫描基类的 @abstractmethod，无需硬编码方法签名。
    """
    kind = _canonical_kind(kind)

    from modutsc.scheduling.scaffold import _load_abc, _scan_abc, _generate_code

    abc_cls = _load_abc(kind, abc_module=None, abc_class=None)
    methods = _scan_abc(abc_cls)

    if not methods:
        raise ValueError(f"基类 {abc_cls.__name__} 没有抽象方法，无法生成模板")

    return _generate_code(kind, name, abc_cls, methods)


def _to_class_name(name: str) -> str:
    """插件名转类名：my_plugin → MyPlugin"""
    return "".join(w.capitalize() for w in name.replace("-", "_").split("_"))


def _indent(text: str, level: int) -> str:
    """文本缩进"""
    prefix = "    " * level
    return prefix + text.replace("\n", "\n" + prefix)