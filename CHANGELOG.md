# PI-Light 开发变更记录

> 记录时间：2026-05-19
> 涉及范围：自由装配维度约束、训练中断、设备自动检测、DRL 算法维度自适应

---

## 一、自由装配维度约束系统

### 1.1 需求背景

自由装配功能允许用户自主选择编排器、算法、观测器等组件进行组合训练。不同组件之间存在维度约束关系（例如：编排器输出维度 = 算法输入维度），用户需要填写各组件的输入/输出维度数，才能保证训练时组件正确对接。

### 1.2 后端接口整理

从 `modutsc/api.py` 中选取与自由装配维度约束相关的接口，整理到 `app/routers/assemble.py`：

| 端点 | 方法 | 功能 | 对应 api.py 函数 |
|------|------|------|-----------------|
| `/api/assemble/modules` | GET | 获取所有组件模块（含默认参数、配置键、维度键） | `list_all` + `scan_setup_cfg_keys` |
| `/api/assemble/recommend` | GET | 根据编排器推荐默认组件搭配 | `recommend_assembly` |
| `/api/assemble/wiring` | GET | 获取编排器数据流边定义 | `get_orch_wiring` |
| `/api/assemble/orchestrator-kinds` | GET | 获取编排器所需组件种类 | `get_orch_kinds` |
| `/api/assemble/iterative` | GET | 迭代式装配选择 | `assembly_iterative` |
| `/api/assemble/config-keys` | GET | 获取插件配置键列表 | `get_plugin_config_keys` |
| `/api/assemble/datasets-topo` | GET | 列出所有数据集及拓扑信息 | `list_datasets` |
| `/api/assemble/dataset-topo` | GET | 获取单个数据集拓扑参数 | `get_dataset_topo` |
| `/api/assemble/constraint-session` | POST | 创建约束求解会话 | `create_constraint_session` |
| `/api/assemble/constraint-session/{id}/state` | GET | 获取约束会话当前状态 | `ConstraintSession.get_state` |
| `/api/assemble/constraint-session/{id}/set-value` | POST | 设置约束值并触发求解 | `ConstraintSession.set_value` |
| `/api/assemble/constraint-session/{id}/recommend` | POST | 生成推荐值 | `ConstraintSession.recommend` |

**新增常量**：

```python
DIMENSION_KEYS = {
    "observer": ["observe_out_dim"],
    "algorithm": ["act_in_dim", "act_out_dim"],
    "actor": ["translate_in_dim", "translate_out_dim"],
    "reward": ["compute_out_dim"],
    "collector": [],
    "orchestrator": [],
}
```

`/modules` 端点现在额外返回每个插件的 `config_keys`（AST 扫描得到的配置键名）和 `dim_keys`（维度相关键名）。

### 1.3 约束组构建逻辑修复

**文件**：`modutsc/api.py`

**问题**：原约束系统会扫描所有 `cfg.get()` 获取的参数并添加到约束组，导致边栏显示大量"待填"参数（如 `reward.metrics`、`collector.batch_size`），但这些参数有默认值，不需要用户填写。

**修复**：约束组只追踪维度参数（以 `_dim` 结尾的键），非维度参数不再出现在约束组中：

```python
dim_deps = [dep for dep in exact_deps if dep.endswith("_dim")]
if dim_deps:
    for dep in dim_deps:
        self._solver.add_group({(kind, dim_key), (kind, dep)})
else:
    self._solver.add_group({(kind, dim_key)})
```

同时修复了独立维度参数不被追踪的问题——即使维度参数没有依赖，也会单独创建约束组。

### 1.4 前端维度配置界面

**文件**：`test3.0.html`

**assembleConfig 页面重构为双栏布局**：

**左侧主配置区**：
- 当前装配标签（颜色区分组件类型）
- 数据集选择 + 车辆数-时间步折线图 + 拓扑信息自动展示
- 每个组件的参数卡片中新增 **输入/输出维度** 配置区：
  - 维度键名左侧有颜色条纹，标识所属约束组
  - 约束组只剩一个未知值时，显示推荐标签和"填入 X"按钮
  - 用户填入值与约束冲突时，显示冲突警告 + "使用推荐值"/"保留我的值"按钮
- 仿真设置 + 生成配置并开始训练

**右侧约束概览侧栏**：
- 约束概览：展示所有约束组及其状态（已满足 / 可自动填充 / N项待填 / 冲突）
- 维度等式：展示等式关系（如 `observer.observe_out_dim == algorithm.act_in_dim`）
- 冲突警告列表
- 待填写维度参数计数

**核心交互流程**：

```
用户选择编排器+组件 → 进入配置页 → 选择数据集
    ↓
后端创建 ConstraintSession → 返回约束组 + 拓扑参数
    ↓
前端展示维度输入框 + 约束概览侧栏
    ↓
用户填写维度值 → 前端通知后端 → 后端约束求解 → 返回更新状态
    ↓
推荐值自动显示 / 冲突告警 → 用户确认或覆盖
    ↓
所有维度约束满足 → 生成配置并启动训练
```

**新增 JavaScript 函数**：

| 函数 | 功能 |
|------|------|
| `initConstraintSession()` | 选择数据集后创建约束求解会话 |
| `onDatasetSelected()` | 选择数据集时获取拓扑 + 初始化约束 |
| `notifyDimChange()` | 维度值变更时通知后端约束求解 |
| `syncDimValuesFromState()` | 从约束求解结果同步维度值到前端 |
| `getDimColor()` | 根据约束组返回颜色 |
| `getDimRecommendedValue()` / `isDimRecommended()` | 推荐值逻辑 |
| `isDimConflicting()` / `getDimExpectedValue()` | 冲突检测 |
| `applyDimRecommend()` / `keepDimOverride()` | 推荐值应用/用户覆盖 |
| `getConstraintGroupStatusIcon/Label/Class()` | 约束组状态判断 |

---

## 二、Orchestrator.run() 调用签名修复

### 2.1 问题

`Orchestrator.run()` 的签名是 `run(self, cfg: dict)`，只接受一个参数。但两处调用传了多余参数：

- `app/routers/experiments.py`：`orch.run(custom_config, stop_event, shared_state)`
- `app/services/experiment_service.py`：`orch.run(full_config, stop_event=stop_event, shared_state=shared_state)`

### 2.2 修复

将 `stop_event` 和 `shared_state` 作为实例属性挂到编排器上：

```python
orch._stop_event = stop_event
orch._shared_state = shared_state
result = orch.run(custom_config)
```

---

## 三、训练中断功能

### 3.1 实现

**文件**：`modutsc/orchestration/__init__.py`

在 `Orchestrator.run()` 的训练循环中，每个 epoch 结束后检查 `_stop_event`：

```python
stop_event = getattr(self, "_stop_event", None)
if stop_event and stop_event.is_set():
    if tr is not None:
        tr.note("[Training] Stopped by user")
    else:
        print("[Training] Stopped by user")
    break
```

用户点击中断按钮时，`stop_event` 被设置，训练在下一个 epoch 结束时优雅停止。

---

## 四、GPU/CPU 自动检测

### 4.1 后端实现

**文件**：`app/routers/experiments.py`、`app/services/experiment_service.py`

新增 `detect_device()` 函数：

```python
def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"
```

在算法初始化时，如果配置中没有指定 `device`，则自动检测并设置：

```python
detected_device = detect_device()
if "device" not in algo_cfg:
    algo_cfg["device"] = detected_device
active_experiments[exp_id]["device"] = detected_device
```

### 4.2 前端展示

**文件**：`test3.0.html`

在运行中实验表格中添加设备列：
- 🟢 GPU（CUDA）
- 🟢 MPS（Apple Silicon GPU）
- ⚪ CPU

---

## 五、数据集可视化

### 5.1 实现

**文件**：`test3.0.html`

在自由装配页面的数据集选择板块，添加了与快速开始页面相同的车辆数-时间步折线图展示。使用已有的 `setChartRef` 函数和 ECharts 进行渲染。

---

## 六、DRL 算法维度自适应修复

### 6.1 问题

FRAP 算法训练时报错：

```
RuntimeError: Sizes of tensors must match except in dimension 2.
Expected size 4 but got size 1 for tensor number 1 in the list.
```

**根本原因**：

1. FRAP 算法在 `setup()` 中用默认值初始化 `num_phase=4, num_lanelink=4`
2. `bind_topology()` 从环境获取实际拓扑后，**没有更新** `self._num_phase` 和 `self._num_lanelink`
3. FRAP 观测器产生的观测维度可能与算法期望不一致
4. `forward()` 中 `phase_feat` 和 `ll_feat` 切片后维度不一致，导致 `torch.cat` 报错

### 6.2 修复方案

#### FRAP 算法（`modutsc/plugins/algorithms/drl/frap.py`）

- **`bind_topology()`**：在获取环境拓扑后，更新 `self._num_phase`、`self._act_dim`、`self._num_lanelink`，然后重建网络
- **`FrapNet.forward()`**：添加维度保护，观测维度不足时零填充，超出时截断

```python
obs_dim = obs.shape[1]
expected_dim = self.num_phase + self.num_lanelink
if obs_dim < expected_dim:
    pad = torch.zeros(B, expected_dim - obs_dim, device=obs.device)
    obs = torch.cat([obs, pad], dim=1)
elif obs_dim > expected_dim:
    obs = obs[:, :expected_dim]
```

#### FRAP 观测器（`modutsc/plugins/observers/frap.py`）

新增 `bind_topology(env)` 方法，根据环境拓扑自动更新 `self._num_phase` 和 `self._max_lanelinks`：

```python
def bind_topology(self, env) -> None:
    ids = env.ids()
    if not ids:
        return
    jid = ids[0]
    gp = env.green_phase_indices(jid)
    self._num_phase = max(len(gp), 1)
    ctrl_links = env.traffic_light_controlled_links(jid)
    if ctrl_links:
        self._max_lanelinks = max(len(ctrl_links), 1)
```

#### DQN 算法（`modutsc/plugins/algorithms/drl/dqn.py`）

- 新增 `bind_topology(env)` 方法，根据环境拓扑更新 `act_dim` 和 `obs_dim`，必要时重建网络
- `MLP.forward()` 添加维度保护

#### CoLight 算法（`modutsc/plugins/algorithms/drl/colight.py`）

- 新增 `bind_topology(env)` 方法，根据环境拓扑更新 `num_phase` 和 `num_lane`，必要时重建网络
- `MultiHeadAttentionNetwork.forward()` 添加维度保护

#### 编排器（`modutsc/orchestration/single.py`、`ma2c.py`）

在算法 `bind_topology()` 之后，调用观测器的 `bind_topology()`（如果存在）：

```python
if hasattr(self._observer, 'bind_topology'):
    self._observer.bind_topology(self._env)
```

---

## 七、修改文件汇总

| 文件 | 修改内容 |
|------|----------|
| `app/routers/assemble.py` | 新增 12 个装配相关 API 端点，含约束会话管理 |
| `app/routers/experiments.py` | 新增 `detect_device()`，修复 `run()` 调用签名，设备自动注入 |
| `app/services/experiment_service.py` | 新增 `detect_device()`，修复 `run()` 调用签名，设备自动注入 |
| `modutsc/api.py` | 约束组只追踪维度参数，独立维度参数单独成组 |
| `modutsc/orchestration/__init__.py` | `run()` 方法添加训练中断检查 |
| `modutsc/orchestration/single.py` | 调用观测器 `bind_topology()` |
| `modutsc/orchestration/ma2c.py` | 调用观测器 `bind_topology()` |
| `modutsc/plugins/algorithms/drl/frap.py` | `bind_topology` 更新维度 + `forward` 维度保护 |
| `modutsc/plugins/algorithms/drl/dqn.py` | 新增 `bind_topology` + `MLP.forward` 维度保护 |
| `modutsc/plugins/algorithms/drl/colight.py` | 新增 `bind_topology` + `forward` 维度保护 |
| `modutsc/plugins/observers/frap.py` | 新增 `bind_topology` 方法 |
| `test3.0.html` | 维度约束 UI、约束概览侧栏、数据集可视化、设备显示、中断功能 |
