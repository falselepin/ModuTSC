# PI-Light 前端 API 参考手册

> 版本 2.1 | 提供给前端开发人员

---

## 目录

1. [概述](#1-概述)
2. [API 函数速查表](#2-api-函数速查表)
3. [场景一：自由装配阶段（迭代缩小）](#3-场景一自由装配阶段迭代缩小)
4. [场景二：配置参数约束阶段（ConstraintSession）](#4-场景二配置参数约束阶段constraintsession)
5. [场景三：代码脚手架（三种模板）](#5-场景三代码脚手架三种模板)
6. [场景四：实验运行与实时日志](#6-场景四实验运行与实时日志)
7. [场景五：数据集与配置文件](#7-场景五数据集与配置文件)
8. [完整的前端交互流程](#8-完整的前端交互流程)
9. [附录：返回值字段速查](#9-附录返回值字段速查)

---

## 1. 概述

`modutsc.api` 提供 **25 个原子化函数和 1 个 `ConstraintSession` 类**，覆盖 PI-Light 前端所需的全部后端计算能力。所有返回值均为可 JSON 序列化的 Python 原生类型（`dict`、`list`、`str`、`int`、`float`、`bool`、`None`）。

**核心设计：**

- **自由装配**：`assembly_iterative()` 支持从编排器出发或从组件出发，每次选择后自动缩小可用范围。`dead_end` 标记死胡同。
- **配置约束**：`ConstraintSession` 一次性构建（基于装配结果），然后反复 `set_value()` 迭代。用户修改约束组内的值时实时返回"哪些组剩一个未知"。
- **代码脚手架**：三种模板函数生成带编写约束注释的 Python 骨架文件。
- **实验运行**：批量运行、流式回调、检查点评估、检查点浏览。
- **拓扑参数透明**：路网拓扑参数（`num_phase`、`max_lanelinks`、`num_tsc` 等）**不直接暴露给用户**。底层约束求解以拓扑参数集合为单位进行兼容性分析，但前端只向用户展示**数据集名称**。用户看到的始终是 `@dataset` 这一个整体条目——"推荐"操作会遍历数据集索引，返回匹配的候选数据集列表。

---

## 2. API 函数速查表

| # | 函数 | 分类 | 输入 | 输出 |
|---|------|------|------|------|
| 1 | `list_kinds()` | 注册表 | — | `["environment","observer",...]` |
| 2 | `list_plugins(kind)` | 注册表 | kind 或 None | `["frap","dqn",...]` 或 `{kind:[...]}` |
| 3 | `list_orchestrators()` | 注册表 | — | `["ma2c","rule","single"]` |
| 4 | `get_orchestrator_kinds(name)` | 注册表 | "single" | `["environment","observer",...]` |
| 5 | `get_orchestrator_wiring(name)` | 注册表 | "single" | `[{from,to},...]` 边列表 |
| 6 | `assembly_iterative(selections)` | 装配 | `{"algorithm":"frap"}` | `{viable_orchestrators,slots,dead_end,...}` |
| 7 | `assembly_requirements(orch)` | 装配 | "single" | `{kinds,slots:{kind:{required_methods,options}}}` |
| 8 | `compatible_orchestrators(kind,name)` | 装配 | "algorithm","frap" | `["single","rule"]` |
| 9 | `get_plugin_config_keys(kind,name)` | 装配 | "observer","frap" | `["features","num_phase","max_lanelinks"]` |
| 10 | `create_constraint_session(orch,sel,ds)` | 约束 | 装配结果+数据集 | `ConstraintSession` 对象 |
| 11 | `session.set_value(kind,key,val)` | 约束 | "observer","features",["num"] | `{notable_groups,resolved_groups,warnings,unknown_count}` |
| 12 | `session.get_state()` | 约束 | — | 完整约束状态 |
| 13 | `session.recommend(kind,key)` | 约束 | "observer","observe_out_dim" | `{kind,key,recommendations:[{candidates,source}]}` |
| 14 | `analyze_constraints(orch,sel,uc,ds)` | 约束 | 一次性分析 | 同 get_state()（兼容旧代码） |
| 15 | `list_datasets(dir)` | 数据集 | "data" | `[{path,num_phase,max_lanelinks,...},...]` |
| 16 | `get_dataset_topo(path)` | 数据集 | "data/LA/roadnet.net.xml" | `{num_phase:5,max_lanelinks:28,...}` |
| 17 | `match_datasets_by_topo(topo,dir)` | 数据集 | `{num_phase:5}` | 匹配的数据集列表 |
| 18 | `read_config_structure(path)` | 配置 | "configs/frap.yaml" | 完整 YAML dict |
| 19 | `scaffold_config(orch,sel,params,path)` | 配置 | 装配+参数 | YAML 字符串 |
| 20 | `scaffold_orchestrator(name,dir)` | 脚手架 | "my_orch","modutsc/orchestration" | Python 代码字符串 |
| 21 | `scaffold_kind_base(name,dir,methods)` | 脚手架 | "predictor","modutsc/plugins/predictors" | Python 代码字符串 |
| 22 | `scaffold_plugin(kind,name,dir)` | 脚手架 | "observer","my_obs","modutsc/plugins/observers" | Python 代码字符串 |
| 23 | `run_experiment(config_path)` | 实验 | "configs/frap.yaml" | `{training,config,elapsed_sec,...}` |
| 24 | `run_experiment_stream(path,on_episode)` | 实验 | 配置+回调 | 同上（实时流式） |
| 25 | `evaluate_checkpoint(config_path,ckpt,steps)` | 实验 | 配置+检查点 | `{metrics,checkpoint}` |
| 26 | `list_checkpoints(dir)` | 实验 | "checkpoints" | `[{path,filename,epoch,size_kb},...]` |

---

### 设计原则：数据集即拓扑参数的整体载体

路由拓扑参数（`num_phase`、`max_lanelinks`、`num_tsc`、`total_incoming_lanes` 等）是仿真器路网分析产出的**物理事实**——用户不能也不应单独修改它们。系统处理时的规则：

1. **底层约束使用拓扑参数集**：约束组中 `@dataset` 展开为一组拓扑键（如 `{num_phase:5, max_lanelinks:28, num_tsc:18}`）。端口探针（如 `observe_out_dim`）依赖这些键，`add_equal` 将其值跨 kind 传播。兼容性匹配的本质是**用数据集对应的拓扑参数值去实例化组件并验证**。

2. **前端只显示 `@dataset`**：`merged_groups` 中的拓扑键被折叠为单个 `@dataset` 条目（`kind="environment", key="@dataset"`），包含子字段 `topo_keys` 列出展开的拓扑细节（仅供开发者调试）。用户看到并交互的只有这一个聚合条目。

3. **推荐返回数据集列表**：用户点击 `@dataset` 的推荐按钮 → `session.recommend("environment", "@dataset")` → 遍历 `datasets_index.yaml` 中的所有数据集记录，返回全部拓扑键组合与已确定参数兼容的数据集候选列表。

4. **用户选数据集 = 一次性注入全部拓扑键**：用户从候选列表中选择一个数据集后，其全部拓扑键值一次性注入所有 kind 的配置命名空间。这与用户选中数据集路径时直接填充的效果完全一致。

| 用户看到的 | 底层实际运作的 |
|-----------|-------------|
| 选择一个数据集 `data/LA/roadnet.net.xml` | `{num_phase:5, max_lanelinks:28, num_tsc:18, ...}` 注入所有 kind |
| 约束组中显示 `@dataset = null` | 拓扑键未知 → probe 无法触发 |
| 点击 `@dataset` 推荐 → 弹出 4 个数据集 | `match_datasets(known_topo)` → 返回 4 个匹配 |
| 约束组中显示 `@dataset = "data/LA/..."` ✅ | 全部拓扑键已知 → probe 可执行 |

---

## 3. 场景一：自由装配阶段（迭代缩小）

### 3.1 核心思想

`assembly_iterative()` 是自由装配的核心。它接收**当前已做的部分选择**，返回**下一步所有可用的选项**。支持两种路径：

- **从编排器出发**：先选编排器 → 填槽位
- **从组件出发**：先选一个组件 → 自动缩小编排器 → 再选组件 → 继续缩小

两种路径都调用同一个函数，只传入当前的 `selections`。

### 3.2 assembly_iterative 详解

#### 调用方式

```python
from modutsc.api import assembly_iterative

# 场景 A：用户什么还没选（初始状态）
r = assembly_iterative({})
# → viable_orchestrators = ["ma2c","rule","single"]
# → dead_end = false
# → all_complete = false

# 场景 B：用户从组件出发，选了 algorithm="frap"
r = assembly_iterative({"algorithm": "frap"})
# → viable_orchestrators = ["ma2c","rule","single"]  （三个都支持 frap）
# → missing_kinds = ["environment","observer","actor","reward","collector","tracker"]

# 场景 C：用户继续选 observer="frap"
r = assembly_iterative({"algorithm": "frap", "observer": "frap"})
# → viable_orchestrators = ["ma2c","rule","single"]  （observer 槽位从 missing_kinds 消失）

# 场景 D：选了 rule 不支持的组合
r = assembly_iterative({"algorithm": "frap", "collector": "ma2c"})
# → viable_orchestrators = ["ma2c"]    ← rule 被排除！它没有 collector="ma2c"
# → slots.collector.options:
#     {"plugin":"ma2c",    "viable_in":["ma2c"],        "missing_in":[]}
#     {"plugin":"replay",  "viable_in":["ma2c"],        "missing_in":["rule"]}

# 场景 E：死胡同
r = assembly_iterative({"algorithm": "frap", "actor": "nonexistent"})
# → dead_end = true, viable_orchestrators = []
# 前端提示：无编排器可用，请回退
```

#### 返回值完整结构

```json
{
  "viable_orchestrators": ["ma2c","rule","single"],
  "slots": {
    "observer": {
      "options": [
        {
          "plugin": "frap",
          "compatible": true,
          "viable_in": ["ma2c","rule","single"],
          "missing_in": []
        },
        {
          "plugin": "standard",
          "compatible": true,
          "viable_in": ["single"],
          "missing_in": ["rule","ma2c"]
        }
      ]
    },
    "...": "每种组件种类同样结构"
  },
  "kinds_order": ["environment","observer","actor","reward","collector","algorithm","tracker"],
  "all_complete": false,
  "dead_end": false,
  "missing_kinds": ["observer","actor","reward","collector","algorithm"]
}
```

#### 字段详解

| 字段 | 类型 | 含义 | 前端用法 |
|------|------|------|---------|
| `viable_orchestrators` | `list[str]` | 当前可选编排器 | 编排器选择面板 |
| `slots.<kind>.options` | `list[dict]` | 每种类的可选插件 | 渲染每个槽位的下拉框 |
| `slots.<kind>.options[].compatible` | `bool` | 在至少一个编排器下兼容 | 绿色=可选项 |
| `slots.<kind>.options[].viable_in` | `list[str]` | 在哪些编排器下可用 | 工具提示："支持 single, rule" |
| `slots.<kind>.options[].missing_in` | `list[str]` | 在哪些编排器下**不可用** | 工具提示："rule 不支持" |
| `all_complete` | `bool` | 所有槽位均已选择 | 可进入下一阶段 |
| `dead_end` | `bool` | 无编排器可用 | 弹出"请重新选择" |
| `missing_kinds` | `list[str]` | 尚未选择的槽位 | 进度指示 |

### 3.3 前端交互伪代码

```javascript
let selections = {};

function onSelect(kind, pluginName) {
  selections = { ...selections, [kind]: pluginName };
  const r = api.assembly_iterative(selections);

  if (r.dead_end) {
    showAlert("无编排器可用，请回退上一步");
    delete selections[kind];
    // 重新计算上一步的可用选项
    const prev = api.assembly_iterative(selections);
    render(prev);
    return;
  }

  // 更新 UI
  updateOrchestratorList(r.viable_orchestrators);
  r.missing_kinds.forEach(k => updateSlotOptions(k, r.slots[k].options));
  updateProgress(r.missing_kinds.length, r.kinds_order.length);
}
```

---

## 4. 场景二：配置参数约束阶段（ConstraintSession）

### 4.1 核心思想

**约束组的结构只取决于装配结果（哪个编排器 + 哪些插件）**，与用户填写了哪些参数值无关。因此：

1. 装配完成后 **创建一次 `ConstraintSession`**
2. 用户填写配置值时 **只调 `set_value()`**，无需重建
3. `set_value()` 返回 `delta`（哪些组刚变成"只剩一个未知"），前端据此点亮推荐按钮
4. 约束组外的参数（如 `lr`、`gamma`）不调 `set_value`，只更新本地状态

### 4.2 完整用法

```python
from modutsc.api import create_constraint_session

# === 步骤 1：装配完成后创建会话 ===
sel = {
    'observer': 'frap', 'actor': 'phase',
    'reward': 'composite', 'collector': 'replay',
    'algorithm': 'frap', 'tracker': 'console',
    'environment': 'sumo',
}

session = create_constraint_session(
    'single', sel,
    'data/LosAngeles/roadnet.net.xml'  # 可选
)

# === 步骤 2：获取初始状态 ===
state = session.get_state()
# state['unknown_count'] = 9
# state['groups_with_one_unknown'] = [{...observer.observe_out_dim...}, ...]
# state['merged_groups'] = [约束组1, 约束组2, ...]

# === 步骤 3：用户填写一个值 ===
delta = session.set_value('observer', 'features', ['num'])
# delta['unknown_count'] = 7
# delta['notable_groups'] = [{members:[...], unknown:{kind:"observer",key:"observe_out_dim"}}]
# delta['resolved_groups'] = [{members:[...所有值已知...]}]
# delta['warnings'] = []

# === 步骤 4：用户点击推荐按钮 ===
rec = session.recommend('observer', 'observe_out_dim')
# rec = {
#   "kind": "observer",
#   "key": "observe_out_dim",
#   "recommendations": [
#     {"candidates": {"observe_out_dim": 32}, "source": "probe"}
#   ]
# }

# === 步骤 5：用户确认推荐或手动填写 ===
delta = session.set_value('observer', 'observe_out_dim', 32)
# 继续迭代直到 unknown_count == 0
```

### 4.3 create_constraint_session 参数

```python
def create_constraint_session(
    orch_name: str,        # 编排器名，如 "single"
    selections: dict,      # {kind: plugin_name} 完整装配
    dataset_path: str      # 数据集路径，可选。不传则拓扑键会出现在推荐中
) -> ConstraintSession
```

### 4.4 set_value 返回值详解

```json
{
  "notable_groups": [
    {
      "members": [
        {"kind":"observer","key":"observe_out_dim","value":null},
        {"kind":"observer","key":"num_phase","value":5},
        {"kind":"observer","key":"features","value":["num"]},
        {"kind":"algorithm","key":"act_in_dim","value":null}
      ],
      "unknown": {
        "kind": "observer",
        "key": "observe_out_dim"
      }
    }
  ],
  "resolved_groups": [
    {
      "members": [
        {"kind":"environment","key":"num_phase","value":5},
        {"kind":"observer","key":"num_phase","value":5}
      ]
    }
  ],
  "warnings": [
    "conflict: {33, '64'}"  — 只在出现冲突时非空
  ],
  "unknown_count": 7
}
```

| 字段 | 用途 |
|------|------|
| `notable_groups` | **"推荐"按钮应该亮起的约束组**。遍历这些组，在 `unknown` 位置显示可点击的推荐按钮 |
| `resolved_groups` | **全部确定了的约束组**。渲染为绿色背景，表示该组所有依赖已满足 |
| `warnings` | **冲突信息**。不为空时在页面顶部显示红色横幅 |
| `unknown_count` | 总未确定参数数。前端显示进度 "7 个参数待确定" |

### 4.5 recommend 返回值详解

**调用：**
```python
rec = session.recommend("observer", "observe_out_dim")  # 普通端口推荐
# 或
rec = session.recommend("environment", "@dataset")       # 数据集推荐
```

**普通端口推荐返回值：**

```json
{
  "kind": "observer",
  "key": "observe_out_dim",
  "recommendations": [
    {
      "candidates": {
        "observe_out_dim": 32
      },
      "source": "probe"
    }
  ]
}
```

| 字段 | 值 | 含义 |
|------|-----|------|
| `source` | `"probe"` | 通过实例化组件实测得到的值（如 `observe_out_dim`） |
| `source` | `"dataset"` | 通过匹配 `datasets_index.yaml` 得到的数据集（仅 `@dataset` 推荐出现） |
| `candidates` | `{key: val, ...}` | 端口推荐通常 1 个键；数据集推荐包含全部拓扑键 |

**`candidates` 的键数量**：
- 端口输出推荐：1 个键（如 `{"observe_out_dim": 32}`）
- 数据集匹配推荐：3+ 个键（如 `{"num_phase":5,"max_lanelinks":28,"num_tsc":18}`）

**`recommendations` 长度**：
- 通常为 0（无法确定）或 1（唯一推荐）
- 多个推荐意味着系统找到了多个可行值，需要用户从列表中选择

### 4.6 get_state 返回值完整结构

```json
{
  "equal_pairs": [
    {"kind_a":"environment","key_a":"num_phase","kind_b":"observer","key_b":"num_phase"}
  ],
  "probe_groups": [
    {"kind":"observer","dim_key":"observe_out_dim","deps":["features","num_phase","max_lanelinks"],"cls":"frap"}
  ],
  "merged_groups": [
    {
      "members": [
        {"kind":"observer","key":"observe_out_dim","value":null},
        {"kind":"observer","key":"num_phase","value":5},
        {"kind":"observer","key":"max_lanelinks","value":28},
        {"kind":"observer","key":"features","value":["num"]},
        {"kind":"algorithm","key":"act_in_dim","value":null}
      ]
    }
  ],
  "values": {
    "observer": {
      "observe_out_dim": null, "num_phase": 5, "max_lanelinks": 28,
      "features": ["num"], "...": "..."
    }
  },
  "unknown_count": 8,
  "groups_with_one_unknown": [
    {"members":[...],"unknown":{"kind":"observer","key":"observe_out_dim"}},
    {"members":[...],"unknown":{"kind":"environment","key":"@dataset"}}
  ],
  "warnings": []
}
```

**`@dataset` 成员结构**（在 `merged_groups[].members` 中）：

```json
{
  "kind": "environment",
  "key": "@dataset",
  "value": "data/LosAngeles/roadnet.net.xml",   // null 表示未选数据集
  "topo_keys": [                                  // 展开的拓扑细节，仅供调试
    {"kind":"environment","key":"num_phase","value":5},
    {"kind":"environment","key":"max_lanelinks","value":28},
    {"kind":"environment","key":"num_tsc","value":18}
  ]
}
```

- `value` 非 null → 用户已选数据集（全部拓扑键已知）
- `value` 为 null → 拓扑键未知，依赖其他参数确定后触发推荐
- `topo_keys` 仅供前端开发者调试用，不需要渲染给最终用户

### 4.7 优化：约束组外的参数不调 set_value

约束组的结构可以从 `get_state().merged_groups` 中获得。用户修改的参数如果不在任何 group 的 members 中，则**不必调 `set_value`**——直接更新本地 `userConfig` 即可。

```javascript
function isInAnyGroup(kind, key, mergedGroups) {
  return mergedGroups.some(group =>
    group.members.some(m => m.kind === kind && m.key === key)
  );
}

function onParamChange(kind, key, value) {
  setUserConfig(prev => ({ ...prev, [kind]: { ...prev[kind], [key]: value } }));

  if (isInAnyGroup(kind, key, constraintState.merged_groups)) {
    const delta = constraintSession.set_value(kind, key, value);
    updateDelta(delta);
  }
}
```

### 4.8 读取已有配置并还原

```python
from modutsc.api import read_config_structure, create_constraint_session

cfg = read_config_structure("configs/frap_resolved.yaml")
components = cfg.get("components", {})
sel = {k: v[0] for k, v in components.items()}

session = create_constraint_session(
    cfg["orchestrator"]["plugin"],
    sel,
    cfg.get("environment", {}).get("config", {}).get("roadnet_file")
)

for kind, section in cfg.items():
    if isinstance(section, dict) and "config" in section:
        for k, v in section["config"].items():
            session.set_value(kind, k, v)

# 如果 unknown_count == 0 → "此配置完整，可直接运行"
```

### 4.9 无数据集时的交互（`@dataset` 推荐）

用户未选数据集时，`@dataset` 条目在所有约束组中显示为 `null`。

```python
session = create_constraint_session("single", sel)  # 无 dataset_path
state = session.get_state()
# state['groups_with_one_unknown'] 中包含 {"kind":"environment","key":"@dataset"}

# 用户填了一些值后，@dataset 可能变成"最后一个未知"
d = session.set_value("observer", "features", ["num"])
# d['notable_groups'] 中包含 @dataset 所在的组

# 用户点击"推荐"按钮
rec = session.recommend("environment", "@dataset")
# → rec = {
#   "recommendations": [
#     {"candidates": {"num_phase":5,"max_lanelinks":28,"num_tsc":18},
#      "source": "dataset"},
#     {"candidates": {"num_phase":2,"max_lanelinks":8,"num_tsc":5},
#      "source": "dataset"},
#     ...
#   ]
# }
# source="dataset" 表示推荐来自数据集索引匹配（而非组件探针）
```

**`source` 说明：**

| source | 含义 | 前端图标 |
|--------|------|---------|
| `"probe"` | 通过实例化组件实测得到 | 🔬 |
| `"dataset"` | 通过匹配 `datasets_index.yaml` 得到 | 🗂️ |

---

### 4.10 手动声明 vs AST 自动分析

所有自动分析点都支持通过类属性**手动声明覆盖**。原则：
- 声明了就优先用声明，不声明则 AST 自动分析
- 声明格式错误 → 打印警告 + 回退 AST
- 两种方式地位相同，互不依赖

#### 五个类属性

| 类属性 | 声明在 | 格式 |
|--------|--------|------|
| `__wiring_edges__` | 编排器类 | `[{"from":{"kind":"observer","method":"observe"},"to":{"kind":"algorithm","method":"act"}}, ...]` |
| `__kind_calls__` | 编排器类 | `{"observer":["observe"], "algorithm":["act","learn","eval"], ...}` |
| `__port_deps__` | 组件类（Observer/Actor 等） | `{"observe":["features","num_phase","max_lanelinks"]}` |
| `__config_keys__` | 任意组件类 | `["features","num_phase","max_lanelinks"]` |
| `__param_deps__` | 任意组件类 | `{"reward_norm":["num_phase"]}` |

#### 编排器手动声明示例

```python
@register("orchestrator", "my_orch")
class MyOrch(Orchestrator):
    __wiring_edges__ = [
        {"from": {"kind": "observer",  "method": "observe"},
         "to":   {"kind": "algorithm", "method": "act"}},
        {"from": {"kind": "algorithm", "method": "act"},
         "to":   {"kind": "actor",     "method": "translate"}},
    ]
    __kind_calls__ = {
        "observer":  ["observe"],
        "algorithm": ["act", "learn", "eval", "sync"],
        "actor":     ["translate"],
        "reward":    ["compute"],
        "collector": ["push", "pull", "pull_for"],
        "tracker":   ["log", "accumulate_step"],
    }
```

#### 组件手动声明示例

```python
@register("observer", "my_obs")
class MyObs(Observer):
    __config_keys__ = ["features", "num_phase", "max_lanelinks"]
    __port_deps__ = {
        "observe": ["features", "num_phase", "max_lanelinks"],
    }
    __param_deps__ = {
        "reward_norm": ["num_phase"],
    }
```

#### 回退行为

```
用户声明了 __port_deps__ 但格式错误（如值为 None）
  → [config] MyObs.__port_deps__ failed (...), falling back to AST
  → 自动分析继续工作，不影响运行
```

---

## 5. 场景三：代码脚手架（三种模板）

### 5.1 scaffold_orchestrator — 编排器模板

```python
from modutsc.api import scaffold_orchestrator
code = scaffold_orchestrator("my_new_orch", "modutsc/orchestration")
```

生成的模板包含：
- `setup()` 方法骨架（接收 env/observer/actor/reward/collector/algorithms/cfg/tracker）
- `warmup()` / `episode()` / `evaluate()` / `save()` / `load()` 方法空实现
- 编写约束注释（不 import 具体类、只从 cfg 读配置）

### 5.2 scaffold_kind_base — 新组件种类 ABC

```python
from modutsc.api import scaffold_kind_base
code = scaffold_kind_base("predictor", "modutsc/plugins/predictors",
    methods={"setup": "cfg: dict", "predict": "env", "evaluate": "env, dataset_path: str"})
```

输出到 `modutsc/plugins/predictors/__init__.py`。`predictor` 成为系统的新 kind。

### 5.3 scaffold_plugin — 组件实现骨架

```python
from modutsc.api import scaffold_plugin
code = scaffold_plugin("observer", "my_tracker", "modutsc/plugins/observers")
```

生成的模板带有约束规则注释：

1. **setup 中只读 `cfg.get("key")`** — 所有键暴露到 YAML
2. **不 import 其他组件的实现类** — 只 import 本 kind 的 ABC
3. **使用 env 时只调用 env ABC 上的方法** — 如 `env.lane_vehicle_count(lid)`
4. **拓扑参数用仿真器产出的键名** — 如 `num_phase`，不自行起别名
5. **如果参数等于某个拓扑参数** — 直接用同一个 `cfg.get()` 键名

### 5.4 用户上传新代码后的集成步骤

1. 用户编写代码，遵守约束规则
2. 放置到对应目录（如 `modutsc/plugins/observers/my_tracker.py`）
3. 重启平台 → `discover()` 自动扫描 → `@register` 自动注册
4. `assembly_iterative({})` 的选项中立即出现新插件
5. **无需修改任何后端代码**

---

## 6. 场景四：实验运行与实时日志

### 6.1 run_experiment — 批量运行

```python
from modutsc.api import run_experiment
result = run_experiment("configs/frap.yaml")
```

**返回：**

```json
{
  "config": { "experiment": {"name":"frap"}, ... },
  "resolved_config": { "observer": {"plugin":"frap","config":{...}}, ... },
  "training": [
    {
      "episode": 1,
      "epoch": 0,
      "ATT": 1304.5,
      "AQL": 17.92,
      "Throughput": 1.05,
      "RealDelay": 1539.8,
      "TripFlow": 0.21,
      "departed": 890,
      "arrived": 754,
      "avg_reward": -62.07,
      "steps": 720,
      "sim_time": 3600,
      "queue": 16.1,
      "epsilon": 0.995,
      "total_loss": 0.42
    }
  ],
  "eval_results": [],
  "elapsed_sec": 45.2
}
```

**`training` 中每个 episode 的所有可能字段：**

| 字段 | 类型 | 始终存在？ | 含义 |
|------|------|-----------|------|
| `episode` | int | ✅ | episode 编号 |
| `epoch` | int | ✅ | 所属 epoch |
| `ATT` | float | ✅ | Average Travel Time |
| `AQL` | float | ✅ | Average Queue Length |
| `Throughput` | float | ✅ | 吞吐量 |
| `RealDelay` | float | ✅ | 实际延迟 |
| `TripFlow` | float | ✅ | 行程流量 |
| `departed` | float | ✅ | 出发车辆数 |
| `arrived` | float | ✅ | 到达车辆数 |
| `avg_reward` | float | ✅ | 平均奖励 |
| `steps` | int | ✅ | 决策步数 |
| `sim_time` | float | ✅ | 仿真时间(s) |
| `queue` | float | ✅ | 队列长度 |
| `epsilon` | float | ✅ | 探索率 |
| `total_loss` | float | ❌ | 总损失（有 learn 时才有） |
| `policy_loss` | float | ❌ | 策略损失 |
| `value_loss` | float | ❌ | 值损失 |
| `entropy` | float | ❌ | 熵 |
| `grad_norm` | float | ❌ | 梯度范数 |

### 6.2 run_experiment_stream — 流式运行（实时推送）

```python
from modutsc.api import run_experiment_stream

# 前端通过 WebSocket/SSE 调用的后端函数
def on_episode(i, metrics):
    push_to_client({"type": "episode_end", "episode": i, "metrics": metrics})

result = run_experiment_stream(
    "configs/frap.yaml",
    on_episode=on_episode,
)
```

`on_episode` 由 `Orchestrator.run()` 在单次 episode 完成的瞬间直接调用，不是运行结束后的迭代，因此可以实现实时推送。如需逐决策步的日志，可使用 tracker 输出的训练日志文件。

### 6.3 evaluate_checkpoint — 加载检查点并评估

```python
from modutsc.api import evaluate_checkpoint
result = evaluate_checkpoint(
    "configs/frap.yaml",
    "checkpoints/frap/ckpt_epoch_10.pkl",
    eval_steps=500   # 可选，不传则从配置读取
)
```

**返回：**

```json
{
  "metrics": { "ATT": ..., "AQL": ..., ... },
  "checkpoint": "checkpoints/frap/ckpt_epoch_10.pkl"
}
```

### 6.4 list_checkpoints — 浏览器检点

```python
from modutsc.api import list_checkpoints
entries = list_checkpoints("checkpoints/frap")
```

**返回：**

```json
[
  {"path": "checkpoints/frap/ckpt_epoch_5.pkl", "filename": "ckpt_epoch_5.pkl", "epoch": 5, "size_kb": 45},
  {"path": "checkpoints/frap/ckpt_epoch_10.pkl", "filename": "ckpt_epoch_10.pkl", "epoch": 10, "size_kb": 46}
]
```

**前端用法**：渲染为可点击的检查点列表，点击后调 `evaluate_checkpoint`。

---

## 7. 场景五：数据集与配置文件

### 7.1 list_datasets

```python
datasets = list_datasets("data")
# → [
#   {"path":"Atlanta/roadnet.net.xml","num_phase":5,"max_lanelinks":28,"num_tsc":18,...},
#   {"path":"grid/grid.net.xml","num_phase":2,...}
# ]
```

### 7.2 get_dataset_topo

```python
topo = get_dataset_topo("data/LosAngeles/roadnet.net.xml")
# → {"num_phase":5,"max_lanelinks":28,"num_tsc":18,"total_incoming_lanes":100,...}
```

### 7.3 match_datasets_by_topo

```python
matches = match_datasets_by_topo({"num_phase":5})
# → 返回所有 num_phase==5 的数据集
```

### 7.4 read_config_structure + scaffold_config

```python
cfg = read_config_structure("configs/frap.yaml")
# → 完整 YAML dict

yaml_str = scaffold_config(
    "single",
    {"observer":"frap","actor":"phase","...":"..."},
    {"observer":{"features":["num"],"observe_out_dim":32}},
    "configs/my_config.yaml"  # 可选
)
# → YAML 字符串（未传 output_path 时只返回字符串）
```

---

## 8. 完整的前端交互流程

### 8.1 全局状态

```javascript
const [phase, setPhase] = useState("assembly");  // "assembly" | "params" | "running" | "results"
const [selections, setSelections] = useState({});
const [assemblyState, setAssemblyState] = useState(null);
const [datasetPath, setDatasetPath] = useState(null);
const [constraintSession, setConstraintSession] = useState(null);
const [constraintState, setConstraintState] = useState(null);
const [userConfig, setUserConfig] = useState({});
const [experimentResult, setExperimentResult] = useState(null);
```

### 8.2 阶段一：自由装配

```javascript
// 初始加载
const r = api.assembly_iterative({});
setAssemblyState(r);

function onSelectKind(kind, pluginName) {
  const newSel = { ...selections, [kind]: pluginName };
  const r = api.assembly_iterative(newSel);

  if (r.dead_end) {
    showAlert("无编排器支持当前选择组合，请回退");
    return;
  }

  setSelections(newSel);
  setAssemblyState(r);

  // 展示选中插件的配置键列表
  const keys = api.get_plugin_config_keys(kind, pluginName);
  showPluginInfo(kind, keys);
}

function onSelectDataset(path) {
  setDatasetPath(path);
  const topo = api.get_dataset_topo(path);
  showTopoPreview(topo);
}

function onAssemblyComplete() {
  // 用户需选择一个编排器（或自动选唯一兼容的）
  const orchName = selectedOrch || assemblyState.viable_orchestrators[0];
  const session = api.create_constraint_session(orchName, selections, datasetPath);
  setConstraintSession(session);
  setConstraintState(session.get_state());
  setPhase("params");
}
```

### 8.3 阶段二：配置参数约束

```javascript
function onParamChange(kind, key, value) {
  setUserConfig(prev => ({ ...prev, [kind]: { ...prev[kind], [key]: value } }));

  // 判断是否在约束组中
  const inGroup = constraintState.merged_groups.some(g =>
    g.members.some(m => m.kind === kind && m.key === key)
  );
  if (!inGroup) return;  // 自由参数，不触发约束刷新

  const delta = constraintSession.set_value(kind, key, value);

  // 渲染变化
  delta.notable_groups.forEach(g => highlightRecommendButton(g.unknown));
  delta.resolved_groups.forEach(g => markGroupResolved(g));
  if (delta.warnings.length) showWarnings(delta.warnings);

  setConstraintState(constraintSession.get_state());
}

function onRecommend(kind, key) {
  const rec = constraintSession.recommend(kind, key);
  if (rec.recommendations.length === 0) {
    showToast("无法确定该值，请手动填写");
    return;
  }
  if (rec.recommendations.length === 1) {
    // 唯一推荐 → 弹出确认
    showConfirm(`推荐: ${JSON.stringify(rec.recommendations[0].candidates)}`, () => {
      const cand = rec.recommendations[0].candidates;
      for (const [k, v] of Object.entries(cand)) {
        onParamChange(kind, k, v);   // 逐个设置（数据集推荐可能有多个键）
      }
    });
  } else {
    // 多个推荐 → 弹出选择器
    showPicker(rec.recommendations, (chosen) => {
      for (const [k, v] of Object.entries(chosen.candidates)) {
        onParamChange(kind, k, v);
      }
    });
  }
}
```

### 8.4 阶段三：配置完成 → 导出/运行

```javascript
function onExport() {
  const yaml = api.scaffold_config(selectedOrch, selections, userConfig);
  downloadFile("my_config.yaml", yaml);
}

async function onRun() {
  setPhase("running");
  const result = api.run_experiment("configs/my_config.yaml");
  setExperimentResult(result);
  setPhase("results");
}

// 或流式运行（需 WebSocket 后端支持）
async function onRunStreaming() {
  setPhase("running");
  api.run_experiment_stream("configs/my_config.yaml",
    (i, m) => pushSSE({type: "episode", index: i, metrics: m}),
    (s, m) => pushSSE({type: "step", step: s, metrics: m})
  );
}
```

### 8.5 阶段四：查看结果 + 管理检查点

```javascript
// 检查点浏览
const ckpts = api.list_checkpoints("checkpoints/frap");
ckpts.forEach(c => {
  renderCheckpointEntry(c.epoch, c.size_kb, () => {
    const evalResult = api.evaluate_checkpoint("configs/frap.yaml", c.path);
    renderEvalResult(evalResult.metrics);
  });
});

// 训练曲线图表
// 从 experimentResult.training 中提取 episode 和 ATT/loss 等指标
// 用 ECharts/Chart.js 渲染折线图
```

---

## 9. 附录：返回值字段速查

### assembly_iterative 返回值

```
viable_orchestrators     list[str]    当前可用编排器
slots.<kind>.options[]   list[dict]   每个插件的选项
  .plugin                str          插件名
  .compatible            bool         是否兼容
  .viable_in             list[str]    在哪些编排器下可用
  .missing_in            list[str]    在哪些编排器下不可用
kinds_order             list[str]    组件种类排序
all_complete            bool         是否全部槽位已选
dead_end                bool         是否无编排器可用
missing_kinds           list[str]    未选槽位
```

### ConstraintSession.get_state / set_value(delta) / analyze_constraints 返回值

```
equal_pairs[]           list[dict]   等式约束
  .kind_a / .key_a / .kind_b / .key_b   str  两端标识
probe_groups[]          list[dict]   探针组
  .kind / .dim_key / .deps / .cls     str  探针信息
merged_groups[]          list[dict]   合并后的约束组
  .members[]             list[dict]   每个成员
    .kind / .key / .value             str  value 可能为 null
groups_with_one_unknown[] list[dict] 只剩一个未知的约束组（仅 get_state）
  .members[] / .unknown  {kind,key}
notable_groups[]        list[dict]   刚变成 1 未知的组（仅 set_value delta）
resolved_groups[]       list[dict]   已全部确定的组（仅 set_value delta）
values.<kind>.<key>     any          所有配置键的当前值
unknown_count           int          总未确定参数数
warnings                list[str]    冲突信息
```

### ConstraintSession.recommend 返回值

```
kind                    str          所属种类
key                     str          键名（"@dataset" 或 "observe_out_dim" 等）
recommendations[]       list[dict]   推荐列表
  .candidates           dict         {key: value, ...} — 端口推荐 1 键，数据集推荐多键
  .source               str          "probe"（组件实测）或 "dataset"（索引匹配）
```

### run_experiment 返回值

```
config                  dict         原始配置
resolved_config         dict         求解后配置
training[]              list[dict]   每个 episode 的指标
  .episode / .epoch / .ATT / .AQL / .Throughput /
  .RealDelay / .TripFlow / .departed / .arrived /
  .avg_reward / .steps / .sim_time / .queue /
  .epsilon / .total_loss / .policy_loss /
  .value_loss / .entropy / .grad_norm
eval_results            list[dict]   评估指标列表
elapsed_sec             float        耗时(秒)
```
