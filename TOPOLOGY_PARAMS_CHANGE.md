# 拓扑参数获取方式变更说明

## 概述

ModuTSC 配置系统已更新，**拓扑参数不再从配置文件中读取，而是通过环境运行时实时探测获取**。这一变更提升了配置的准确性和灵活性。

## 核心变化

### 变更前
拓扑参数需要在配置文件中手动配置：
```yaml
observer:
  plugin: frap
  config:
    features: ["num", "waiting"]
    num_phase: 4           # ← 手动配置
    max_lanelinks: 12      # ← 手动配置

algorithm:
  plugin: frap
  config:
    num_phase: 4           # ← 重复配置
    max_lanelinks: 12      # ← 重复配置
```

### 变更后
只需设置路网文件路径，拓扑参数自动获取：
```yaml
environment:
  plugin: sumo
  config:
    roadnet_file: data/Manhattan/roadnet.net.xml  # ← 只需设置这个

observer:
  plugin: frap
  config:
    features: ["num", "waiting"]  # 拓扑参数不再需要

algorithm:
  plugin: frap
  config:
    gamma: 0.95  # 拓扑参数不再需要
```

## 工作原理

### 1. 数据集选择触发环境启动

当设置 `environment.roadnet_file` 时，系统自动执行以下流程：

```
设置 roadnet_file
       │
       ▼
┌─────────────────────────┐
│ 1. 启动 SUMO 环境       │
│    env.launch(path)     │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ 2. 探测拓扑参数         │
│    - num_phase          │
│    - max_lanelinks      │
│    - max_green_phases   │
│    - num_tsc            │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ 3. 注入到所有组件       │
│    通过约束系统传播     │
└─────────────────────────┘
```

### 2. 约束系统自动传播

拓扑参数通过约束组机制自动传播到所有需要的组件：

```python
# ConstraintSession 初始化时建立约束组
self._solver.add_group(
    {("environment", "roadnet_file")}
    | {("environment", k) for k in self._topo_keys}
)

# 对需要环境的端口，建立与拓扑参数的依赖
if needs_env:
    for topo_key in self._topo_keys:
        self._solver.add_group({(kind, dim_key), (kind, topo_key)})
```

### 3. 延迟确定器机制

对于依赖环境的端口维度计算，采用延迟确定器模式：

```python
# 注册延迟探测器
self._deferred_probes.append({
    "kind": kind, "dim_key": dim_key,
    "deps": sorted(exact_deps), "cls": plugin_name,
    "method_name": method_name, "needs_env": needs_env,
})

# 环境启动后注册确定器
for probe in self._deferred_probes:
    resolver = _probe_determiner(
        probe["kind"], cls, probe["method_name"],
        set(probe["deps"]), env=self._env  # 传入 env 实例
    )
    self._solver.add_determiner(probe["kind"], probe["dim_key"], resolver)
```

## 涉及的文件

| 文件 | 作用 |
|------|------|
| `modutsc/api.py` | `ConstraintSession.select_dataset()` - 数据集选择与拓扑探测 |
| `modutsc/api.py` | `ConstraintSession._detect_dimension_conflicts()` - 动态冲突检测 |
| `modutsc/scheduling/launcher.py` | `Launcher.build()` - 环境启动与拓扑校正 |
| `modutsc/scheduling/config_solver.py` | `_method_uses_env()` - 方法环境依赖分析 |

## 组件开发指南

### setup 方法签名

组件的 `setup` 方法现在支持可选的 `env` 参数：

```python
def setup(self, cfg, env=None):
    # env 可用时可调用拓扑接口
    if env is not None:
        ids = env.ids()
        self._num_phase = max(env.phase_count(j) for j in ids)
        self._max_lanelinks = max(
            len(env.traffic_light_controlled_links(j)) for j in ids
        )
    else:
        # 降级处理：使用配置或默认值
        self._num_phase = cfg.get("num_phase", 4)
```

### 端口方法签名

输出端口方法同样支持 `env` 参数：

```python
def observe(self, env=None):
    # 使用 env 获取实时拓扑信息
    if env:
        # 动态计算输出维度
        return self._compute_features(env)
    return self._default_features()
```

### 向后兼容性

系统通过 `try-except TypeError` 自动适配新旧签名：

```python
# Launcher.build() 中的适配逻辑
try:
    obj.setup(kind_cfg, env=self._env)
except TypeError:
    obj.setup(kind_cfg)  # 旧组件无 env 参数
```

## 冲突检测与推荐

系统现在支持智能冲突检测，根据组件是否依赖环境来判断可调整性：

| 场景 | 源锁定 | 目标锁定 | 处理建议 |
|------|--------|----------|----------|
| 双锁定 | True | True | 更换插件实现（结构性冲突） |
| 源可调 | False | True | 调整源组件配置参数 |
| 目标可调 | True | False | 调整目标组件配置参数 |
| 双可调 | False | False | 调整任一组件配置 |

## 配置文件最佳实践

### 推荐格式

```yaml
experiment:
  name: my_experiment
  seed: 42

environment:
  plugin: sumo
  config:
    roadnet_file: data/Manhattan/roadnet.net.xml
    flow_file: data/Manhattan/flow_0.rou.xml
    sim_max_time: 3600
    decision_interval: 5

observer:
  plugin: frap
  config:
    features: ["num", "waiting", "speed"]
    # num_phase, max_lanelinks 不再需要

algorithm:
  plugin: frap
  config:
    gamma: 0.95
    lr: 1e-4
    # num_phase, max_lanelinks 不再需要

actor:
  plugin: phase
  config:
    # max_phase 不再需要

orchestrator:
  plugin: single
  config: {}
```

### 数据集推荐功能

在配置会话中，可通过 `recommend()` 方法获取兼容的数据集列表：

```python
session = create_constraint_session("single", selections)
result = session.recommend("environment", "roadnet_file")
# 返回按兼容性排序的数据集列表
```

## 总结

| 特性 | 变更前 | 变更后 |
|------|--------|--------|
| 拓扑参数来源 | 配置文件手动设置 | 环境运行时探测 |
| 配置复杂度 | 高（多处重复配置） | 低（只需设置路网） |
| 准确性 | 依赖人工配置 | 100% 准确 |
| 灵活性 | 数据集固定后无法修改 | 任意时刻可切换数据集 |
| 冲突检测 | 静态分析 | 动态 + 运行时验证 |

这一变更大幅简化了配置流程，提升了系统的可靠性和易用性。