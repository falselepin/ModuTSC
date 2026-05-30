"""自定义插件管理 API 路由。

支持上传 .py 文件、列出/查看/删除用户插件、生成代码模板。
"""
import base64
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from modutsc.scheduling.dynamic_loader import (
    register_user_plugin, unregister_user_plugin,
    list_user_plugins, get_user_plugin_detail,
    validate_plugin_source, scaffold_plugin_template,
)
from modutsc.scheduling.registry import (
    _KIND_CONTRACT, register_component_kind, unregister_component_kind,
)
from modutsc.api import invalidate_compatibility_cache

router = APIRouter(prefix="/api/plugins", tags=["plugins"])

USER_PLUGINS_DIR = Path("user_plugins")


class UploadRequest(BaseModel):
    kind: str
    name: str
    author: str = ""
    description: str = ""
    file: str  # base64 编码的 .py 文件内容


class ScaffoldRequest(BaseModel):
    kind: str
    name: str


class KindBaseRequest(BaseModel):
    kind_name: str
    file_b64: str


class OrchestratorUploadRequest(BaseModel):
    name: str
    file_b64: str


def _load_user_kinds() -> List[str]:
    """从索引文件加载用户注册的组件类型"""
    index_file = USER_PLUGINS_DIR / "index.json"
    if not index_file.exists():
        return []
    import json
    try:
        data = json.loads(index_file.read_text(encoding="utf-8"))
        return data.get("kinds", [])
    except Exception:
        return []


def _save_user_kinds(kinds: List[str]) -> None:
    """保存用户注册的组件类型到索引文件"""
    USER_PLUGINS_DIR.mkdir(exist_ok=True)
    index_file = USER_PLUGINS_DIR / "index.json"
    import json
    data = {}
    if index_file.exists():
        try:
            data = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    data["kinds"] = kinds
    index_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/kinds")
def get_all_kinds():
    """获取所有已注册的组件类型"""
    builtin_kinds = list(_KIND_CONTRACT.keys())
    user_kinds = _load_user_kinds()
    return {"kinds": builtin_kinds + user_kinds}


@router.delete("/kind/{kind_name}")
def delete_kind(kind_name: str):
    """删除一个用户注册的组件类型"""
    user_kinds = _load_user_kinds()
    if kind_name not in user_kinds:
        raise HTTPException(status_code=404, detail=f"未找到类型: {kind_name}")
    
    kind_dir = USER_PLUGINS_DIR / f"kinds/{kind_name}"
    if kind_dir.exists():
        import shutil
        shutil.rmtree(kind_dir)
    
    if kind_name in _KIND_CONTRACT:
        unregister_component_kind(kind_name)
    
    user_kinds.remove(kind_name)
    _save_user_kinds(user_kinds)
    
    invalidate_compatibility_cache()
    return {"status": "ok"}


@router.post("/scaffold/kind-base/save")
def save_kind_base(req: KindBaseRequest):
    """保存并注册组件类型基类"""
    if not req.kind_name or not req.kind_name.isidentifier():
        raise HTTPException(status_code=400, detail="类型名称必须是合法的 Python 标识符")
    
    try:
        source = base64.b64decode(req.file_b64).decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="文件内容不是有效的 base64 编码")
    
    kind_dir = USER_PLUGINS_DIR / f"kinds/{req.kind_name}"
    kind_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = f"{req.kind_name}_base.py"
    file_path = kind_dir / file_name
    file_path.write_text(source, encoding="utf-8")
    
    register_component_kind(req.kind_name)
    
    user_kinds = _load_user_kinds()
    if req.kind_name not in user_kinds:
        user_kinds.append(req.kind_name)
        _save_user_kinds(user_kinds)
    
    invalidate_compatibility_cache()
    return {"kind": req.kind_name, "status": "registered"}


@router.post("/orchestrator/upload")
def upload_orchestrator(req: OrchestratorUploadRequest):
    """上传编排器"""
    if not req.name or not req.name.isidentifier():
        raise HTTPException(status_code=400, detail="名称必须是合法的 Python 标识符")
    
    try:
        source = base64.b64decode(req.file_b64).decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="文件内容不是有效的 base64 编码")
    
    orch_dir = USER_PLUGINS_DIR / "orchestrator"
    orch_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = orch_dir / f"{req.name}.py"
    file_path.write_text(source, encoding="utf-8")
    
    try:
        result = register_user_plugin(
            kind="orchestrator",
            name=req.name,
            source=source,
            author="",
            description="",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=400, detail=f"加载失败: {str(e)}")
    
    invalidate_compatibility_cache()
    return result


@router.delete("/orchestrator/{orch_name}")
def delete_orchestrator(orch_name: str):
    """删除编排器"""
    success = unregister_user_plugin("orchestrator", orch_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"编排器不在注册表中: {orch_name}")
    
    orch_file = USER_PLUGINS_DIR / "orchestrator" / f"{orch_name}.py"
    if orch_file.exists():
        orch_file.unlink()
    
    invalidate_compatibility_cache()
    return {"id": orch_name, "status": "deleted"}


@router.post("/upload")
def upload_plugin(req: UploadRequest):
    """上传一个 .py 文件并注册为用户插件。"""
    # 解码 base64
    try:
        source = base64.b64decode(req.file).decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="文件内容不是有效的 base64 编码")

    # 校验文件名后缀
    if not req.name or not req.name.isidentifier():
        raise HTTPException(status_code=400, detail="插件名称必须是合法的 Python 标识符")

    # 先做安全校验（不上传）
    validation = validate_plugin_source(source, req.kind)
    if not validation["valid"]:
        return {
            "id": req.name,
            "kind": req.kind,
            "status": "invalid",
            "violations": validation["violations"],
            "methods": [],
            "config_keys": [],
        }

    # 注册插件
    try:
        result = register_user_plugin(
            kind=req.kind,
            name=req.name,
            source=source,
            author=req.author,
            description=req.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=400, detail=f"加载失败: {str(e)}")

    return result


@router.get("/user")
def get_user_plugins():
    """列出所有用户插件，按 kind 分组。"""
    return list_user_plugins()


@router.get("/user/{plugin_name}")
def get_user_plugin(plugin_name: str):
    """获取单个插件的详细信息（包含源码）。"""
    detail = get_user_plugin_detail(plugin_name)
    if not detail:
        raise HTTPException(status_code=404, detail=f"未找到插件: {plugin_name}")
    return detail


@router.delete("/user/{plugin_name}")
def delete_user_plugin(plugin_name: str):
    """删除一个用户插件。"""
    # 从索引中找到插件类型
    detail = get_user_plugin_detail(plugin_name)
    if not detail:
        raise HTTPException(status_code=404, detail=f"未找到插件: {plugin_name}")

    kind = detail.get("kind", "")
    if not kind:
        raise HTTPException(status_code=400, detail="插件类型未知")

    success = unregister_user_plugin(kind, plugin_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"插件不在注册表中: {plugin_name}")

    return {"id": plugin_name, "status": "deleted"}


@router.post("/validate")
def validate_plugin(req: UploadRequest):
    """仅验证上传的源码（不注册）。"""
    try:
        source = base64.b64decode(req.file).decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="文件内容不是有效的 base64 编码")

    return validate_plugin_source(source, req.kind)


@router.post("/scaffold")
def scaffold_plugin(req: ScaffoldRequest):
    """生成插件代码模板。"""
    if not req.name or not req.name.isidentifier():
        raise HTTPException(status_code=400, detail="插件名称必须是合法的 Python 标识符")

    try:
        template = scaffold_plugin_template(req.kind, req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "kind": req.kind,
        "name": req.name,
        "template": template,
    }