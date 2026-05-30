# 功能四：自定义插件上传 总体架构设计

## 一、功能概述

允许用户将本地编写的 `.py` 文件上传到平台，作为自定义组件（编排器 / 算法 / 观测器 / 收集器 / 奖励器 / 动作器等），与系统内置插件混合搭配使用。上传后自动完成语法校验、接口契约检查、兼容性分析，无缝融入现有的"自由装配 → 参数配置 → 启动实验"流水线。

---

## 二、整体交互流程

```
┌──────────┐    上传 .py     ┌──────────────┐    校验+注册    ┌──────────────┐
│  用户本地 │ ──────────────> │  后端 API      │ ──────────────> │  Registry    │
│  编写插件  │                │  /api/plugins  │                │  动态注册    │
└──────────┘                 └──────────────┘                └──────┬───────┘
                                                                   │
                                                          重建兼容性缓存
                                                                   │
      ┌────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────┐   选择组件     ┌──────────┐   constraint    ┌──────────┐   启动    ┌──────────┐
│ 自由装配  │ ────────────> │ 参数配置  │ ──────────────> │ 生成Config │ ────────> │  训练运行 │
│ 模块货架  │   (混合选择)   │ 约束校验  │    session     │ scaffold  │  launch  │  Launcher │
└──────────┘               └──────────┘                └──────────┘          └──────────┘
```

---

## 三、后端架构

### 3.1 目录结构

```
ModuTSC/
├── user_plugins/                  # 用户上传的插件（新增）
│   ├── index.json                 # 插件索引元数据
│   ├── orchestrator/              # 用户上传的编排器
│   │   └── my_orch.py
│   ├── algorithm/
│   │   └── my_dqn.py
│   ├── observer/
│   │   └── my_obs.py
│   ├── collector/
│   ├── reward/
│   └── actor/
├── app/
│   └── routers/
│       └── plugins.py             # 新增：插件上传统一路由
├── modutsc/
│   └── scheduling/
│       ├── registry.py            # 修改：新增 register_dynamic() 函数
│       └── dynamic_loader.py      # 新增：动态加载器模块
```

### 3.2 核心模块：动态加载器 (`modutsc/scheduling/dynamic_loader.py`)

```
功能职责：
  1. import_user_plugin(kind, file_path)   — 加载单个 .py 并注册
  2. discover_user_plugins()               — 扫描 user_plugins/ 目录，批量注册
  3. validate_plugin_source(source_code, kind) — 上传前的静态检查
  4. unregister_user_plugin(kind, name)    — 移除已注册的用户插件
```

#### 3.2.1 加载流程

```
用户上传 foo.py
       │
       ▼
┌─────────────────────────┐
│ 1. 静态安全检查          │
│    - AST 语法校验        │
│    - 禁止 import os/sys  │  ← 防止恶意代码
│    - 检查 @register 装饰器│
└──────────┬──────────────┘
           │ 通过
           ▼
┌─────────────────────────┐
│ 2. 写入 user_plugins/<kind>/ │
│    文件名自动去重         │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ 3. 动态 import            │
│    importlib.__import__() │
│    → @register 自动触发   │
│    → 类加入 _registry    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ 4. 接口契约验证           │
│    - 检查 kind 必需方法   │
│    - 检查 setup 签名      │
│    - 支持 Optional[Env]  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ 5. 重建兼容性缓存         │
│    invalidate_cache()    │
└─────────────────────────┘
```

#### 3.2.2 安全白名单

上传的 `.py` 文件在导入前进行 AST 级别的安全检查，禁止以下模式：
- `import os` / `import sys` / `import subprocess`
- `eval(` / `exec(` / `compile(`
- `__import__(`
- `open(` 非读取模式
- `shutil.rmtree` / `os.remove`
- 文件系统写入操作

白名单允许导入的模块仅限：`modutsc`, `torch`, `numpy`, `typing`, `abc`, `dataclasses`, `collections`, `math`, `random`, `itertools`, `functools`。

### 3.3 Registry 扩展现有能力

现有的 `discover()` 函数通过 `pkgutil.walk_packages("modutsc")` 递归发现插件包。需要新增：

```python
# registry.py 新增
def discover_user_plugins():
    """扫描 user_plugins/ 目录并注册所有用户插件"""
    from pathlib import Path
    import importlib.util
    user_dir = Path("user_plugins")
    if not user_dir.exists():
        return
    for kind_dir in user_dir.iterdir():
        if not kind_dir.is_dir():
            continue
        kind = _canonical_kind(kind_dir.name)
        for py_file in kind_dir.glob("*.py"):
            try:
                _import_user_module(kind, py_file)
            except Exception as e:
                print(f"[user_plugins] 跳过 {py_file}: {e}")

def register_dynamic(kind: str, name: str, cls):
    """运行时动态注册一个组件类（无需装饰器）"""
    kind = _canonical_kind(kind)
    required = _get_kind_contract(kind, cls)
    if required is not None:
        missing = [m for m in required if not hasattr(cls, m)]
        if missing:
            raise TypeError(f"'{kind}/{name}' 缺少方法: {missing}")
    if kind != "orchestrator":
        _validate_component_setup(cls, kind, name)
    _registry[(kind, name)] = cls
    if kind == "orchestrator" and name not in _orch_attr_cache:
        _orch_attr_cache[name] = _derive_orch_attr_map(cls)
```

### 3.4 新增 API 路由 (`app/routers/plugins.py`)

| 方法 | 路径 | 功能 |
|------|------|------|
| `POST` | `/api/plugins/upload` | 上传一个 .py 文件并注册 |
| `GET` | `/api/plugins/user` | 列出所有用户插件 |
| `GET` | `/api/plugins/user/{id}` | 查看单个插件详情（源码、方法签名） |
| `DELETE` | `/api/plugins/user/{id}` | 删除用户插件 |
| `POST` | `/api/plugins/user/{id}/validate` | 单独验证插件的接口契约 |
| `POST` | `/api/plugins/scaffold` | 根据 kind + 方法列表生成插件代码模板 |

#### `POST /api/plugins/upload` 请求体

```json
{
  "kind": "algorithm",
  "name": "my_transformer_dqn",
  "author": "张三",
  "description": "基于 Transformer 的 DQN 变体",
  "file": "<base64 编码的 .py 文件内容>"
}
```

响应：
```json
{
  "id": "my_transformer_dqn",
  "kind": "algorithm",
  "status": "valid",
  "warnings": [],
  "methods": ["setup", "act"],
  "config_keys": ["lr", "gamma", "hidden_dim"]
}
```

#### `GET /api/plugins/user` 响应

```json
{
  "orchestrator": [
    { "id": "my_orch", "name": "my_orch", "methods": [...], "config_keys": [...] }
  ],
  "algorithm": [
    { "id": "my_dqn", "name": "my_dqn", "status": "valid", ... }
  ]
}
```

### 3.5 与现有装配系统的集成

现有的 `list_modules_with_params()` 函数通过 `list_all(kind)` 获取所有注册插件。用户插件一旦通过 `register_dynamic()` 注册，就会自动出现在 `_registry` 中，从而：

1. **自由装配** — `assembly_iterative()` 自动包含用户插件在 `slots[kind].options` 里
2. **兼容性检查** — `compute_compatibility_cache()` 需在注册后调用 `invalidate_compatibility_cache()` 重建
3. **约束会话** — `ConstraintSession` 对用户插件的维度分析同样适用
4. **配置脚手架** — `scaffold_config()` 通过 `find(kind, name)` 查找类，用户插件同样能被找到

关键：每次上传或删除用户插件后，必须调用 `invalidate_compatibility_cache()` 重建兼容性矩阵。

---

## 四、前端架构

### 4.1 新增页面："自定义插件"

在左侧导航栏新增一项：

```
📦 自定义插件
```

### 4.2 页面布局

```
┌─────────────────────────────────────────────────────────┐
│  📦 自定义插件管理                                       │
│  ─────────                                             │
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │  📤 上传新插件         │  │  📋 我的插件列表           │ │
│  │                      │  │                           │ │
│  │  组件类型: [下拉菜单]  │  │  ┌───┬────┬────┬───────┐ │ │
│  │    algorithm ▼       │  │  │名称│类型│状态│ 操作    │ │ │
│  │                      │  │  ├───┼────┼────┼───────┤ │ │
│  │  插件名称: [______]  │  │  │my  │alg │ ✅ │🗑️查看 │ │ │
│  │                      │  │  │dqn │    │    │       │ │ │
│  │  插件描述: [______]  │  │  └───┴────┴────┴───────┘ │ │
│  │                      │  │                           │ │
│  │  作者(选填): [____]  │  │                           │ │
│  │                      │  │                           │ │
│  │  ┌────────────────┐  │  │                           │ │
│  │  │  拖拽或点击上传  │  │  │                           │ │
│  │  │  仅支持 .py 文件 │  │  │                           │ │
│  │  └────────────────┘  │  │                           │ │
│  │                      │  │                           │ │
│  │  [🚀 上传并验证]      │  │                           │ │
│  └──────────────────────┘  └──────────────────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  📝 代码模板生成器                                    │ │
│  │  类型: [algorithm ▼]  名称: [_____]  [生成模板⬇️]    │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 4.3 插件详情弹窗

点击列表中的"查看"按钮，弹出详情窗口展示：
- 插件名称、类型、状态
- 方法列表与签名（通过 API 返回的 methods 字段）
- 配置参数列表（通过 API 返回的 config_keys）
- 源代码预览（前 50 行，可展开）

### 4.4 与自由装配的联动

自由装配的"模块货架"中，每个分类下原有内置模块的列表现在改为"内置"和"自定义"两个分组：

```
算法 (algorithm)
├── 📌 内置
│   ├── dqn
│   ├── frap
│   └── colight
└── 📦 自定义
    ├── my_transformer_dqn  (用户上传)
    └── my_rule_based       (用户上传)
```

前端通过组合两个 API 数据实现：
1. `GET /api/assemble/modules` — 获取内置模块
2. `GET /api/plugins/user` — 获取用户自定义模块
3. 合并后按类型分组渲染

---

## 五、插件开发规范（面向用户）

### 5.1 最小模板

用户上传的文件必须包含一个类，并使用 `@register(kind, name)` 装饰器注册：

```python
from modutsc.scheduling.registry import register
from typing import Optional, Any
from modutsc.env import Env


@register("algorithm", "my_dqn")
class MyDQN:
    def setup(self, cfg: dict, env: Optional[Env] = None):
        self.lr = cfg.get("lr", 1e-4)
        self.gamma = cfg.get("gamma", 0.99)
        # env 可用于获取拓扑信息
        if env is not None:
            self._num_phase = max(env.phase_count(j) for j in env.ids())

    def act(self, obs, deterministic=False):
        # 返回动作
        return 0
```

### 5.2 各类组件必须实现的方法

| 组件类型 | 必须方法 |
|----------|----------|
| observer | `setup`, `observe` |
| actor | `setup`, `translate` |
| reward | `setup`, `compute` |
| collector | `setup`, `push`, `ready`, `pull`, `size` |
| algorithm | `setup`, `act` |
| orchestrator | `setup`, `warmup`, `episode`, `evaluate`, `save`, `load` |

### 5.3 编写建议

1. **setup 签名**：`def setup(self, cfg: dict, env: Optional[Env] = None)` — `env` 为可选参数，框架通过 `try-except TypeError` 适配新旧签名
2. **拓扑参数**：优先从 `env` 获取 `num_phase`、`max_lanelinks` 等，而非硬编码或从 `cfg` 读取
3. **类型标注**：建议标注输入输出类型（`__input_type__`, `__output_type__`），以支持更好的兼容性推荐
4. **依赖声明**：编排器可声明 `__compatible_plugins__` 字段指定可搭配的组件

---

## 六、数据流与状态管理

### 6.1 用户插件索引 (`user_plugins/index.json`)

```json
{
  "plugins": {
    "my_dqn": {
      "kind": "algorithm",
      "file": "algorithm/my_dqn.py",
      "name": "my_dqn",
      "author": "张三",
      "description": "基于 Transformer 的 DQN 变体",
      "uploaded_at": "2026-05-26T14:30:00",
      "status": "valid",
      "methods": ["setup", "act"],
      "config_keys": ["lr", "gamma", "hidden_dim"]
    },
    "my_orch": {
      "kind": "orchestrator",
      "file": "orchestrator/my_orch.py",
      "name": "my_orch",
      "author": "李四",
      "description": "自定义编排器",
      "uploaded_at": "2026-05-26T15:00:00",
      "status": "valid",
      "methods": ["setup", "warmup", "episode", "evaluate", "save", "load"],
      "config_keys": []
    }
  }
}
```

### 6.2 启动时加载顺序

```
FastAPI 启动
    │
    ├── 1. discover()              ← 内置插件（modutsc.plugins.*）
    │       _registry 填充
    │
    ├── 2. discover_user_plugins() ← 用户插件（user_plugins/**/*.py）
    │       _registry 填充
    │
    ├── 3. compute_compatibility_cache()
    │       _COMPATIBILITY_CACHE 重建
    │
    └── 4. 服务就绪
```

---

## 七、与现有三大功能的融合方式

### 7.1 快速开始

暂不涉及自定义组件，沿用内置预设的 DQN 实验。若后续需要，可在模型选择中增加"自定义算法"分组。

### 7.2 论文查阅

不涉及。

### 7.3 自由装配

**这是主要集成点**。用户上传自定义插件后：

1. 进入"自由装配"页面
2. "模块货架"中每个分类下出现自定义插件（标记 📦 图标）
3. 如同内置插件一样拖拽/点击选择
4. 选择数据集后，约束系统自动分析维度兼容性（用户插件同样参与）
5. 参数配置面板展示用户插件的 `config_keys`
6. 生成配置 → 启动实验

### 7.4 对比分析

已完成实验中的自定义插件正常参与对比，模型名称显示为自定义插件的名称。

---

## 八、错误处理与用户提示

| 场景 | 前端提示 |
|------|----------|
| 上传非 `.py` 文件 | "仅支持 .py 文件" |
| 语法错误 | "文件语法错误：第 X 行 ..." |
| 缺少 `@register` 装饰器 | "文件必须包含 @register(kind, name) 装饰器" |
| 缺少必需方法 | "插件 'my_obs' (observer) 缺少方法：observe" |
| setup 签名不匹配 | "setup 方法的参数必须为 (self, cfg, env=None)" |
| 包含危险的 import | "文件包含禁止的导入：os。仅允许导入 modutsc, torch, numpy 等" |
| 插件名与已有插件冲突 | "插件名 'dqn' 与内置插件冲突，请更换名称" |
| 删除正在实验中使用的插件 | "该插件正被实验 'xxx' 使用，无法删除" |

---

## 九、实施建议与优先级

### Phase 1（核心功能）
1. `dynamic_loader.py` 模块实现
2. `POST /api/plugins/upload` 和 `GET /api/plugins/user` API
3. 前端"自定义插件"页面（上传 + 列表）
4. 与自由装配模块货架的集成

### Phase 2（体验增强）
1. 代码模板生成器（`POST /api/plugins/scaffold`）
2. 插件详情查看（源码预览、方法签名）
3. 插件更新/覆盖上传
4. 危险 import 检查白名单

### Phase 3（高级功能）
1. 插件版本管理
2. 插件市场（插件共享/导入导出）
3. 插件测试沙箱（上传后试运行一个 mini-episode）
4. 插件依赖分析（自动检测对 torch/numpy 的版本需求）

---

## 十、小结

本方案的核心思路是 **"最小侵入，最大复用"**：

- **后端**：扩展现有的 `discover → register → compatibility_cache` 流水线，新增 `user_plugins/` 目录 + 动态加载器，让用户插件无缝融入已有体系
- **前端**：在自由装配的"模块货架"中增加自定义插件入口，其余的参数配置、约束校验、配置生成、实验启动等环节完全复用现有逻辑
- **安全**：AST 级别的静态白名单检查 + 接口契约验证，确保恶意代码无法注入
- **持久化**：插件文件存储在 `user_plugins/`，索引记录在 `index.json`，服务重启自动恢复