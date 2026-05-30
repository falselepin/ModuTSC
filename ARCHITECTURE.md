# PI-Light 调度层架构说明

## 目录

1. [站在用户的视角：一次实验如何跑起来](#1-站在用户的视角一次实验如何跑起来)
2. [设计灵魂：配置即接口](#2-设计灵魂配置即接口)
3. [维度检查：端口方法之间的维度等式](#3-维度检查端口方法之间的维度等式)
4. ["最后一个缺口"规则：什么时候平台自动填入](#4-最后一个缺口规则什么时候平台自动填入)
5. [灵敏度分析：找出真正影响维度的配置项](#5-灵敏度分析找出真正影响维度的配置项)
6. [两层筛选：组件兼容性 vs 维度一致性](#6-两层筛选组件兼容性-vs-维度一致性)
7. [配置约束搜索：用已知配置键筛选未知配置项](#7-配置约束搜索用已知配置键筛选未知配置项)
8. [AST 推导：编排器代码即交互契约](#8-ast-推导编排器代码即交互契约)
9. [前端呈现：约束组的可视化](#9-前端呈现约束组的可视化)
10. [附录](#10-附录)

---

## 1. 站在用户的视角：一次实验如何跑起来

### 1.1 我写一个 YAML

```yaml
experiment:
  name: "frap"
  seed: 42

components:                                  # 选哪些组件组合
  observer:     [frap]
  actor:        [phase]
  reward:       [composite]
  collector:    [replay]
  algorithm:    [frap]
  orchestrator: [single]
  environment:  [sumo]
  tracker:      [console]

environment:
  plugin: "sumo"
  config:
    roadnet_file: "data/LosAngeles/roadnet.net.xml"
    flow_file: "data/LosAngeles/flow_0.rou.xml"
    gui: false
    sim_max_time: 3600
    decision_interval: 5

algorithm:
  - plugin: "frap"
    config:
      lr: 0.0005
      gamma: 0.95
      tau: 0.01
```

然后运行：

```
py -m modutsc run configs/frap.yaml
```

### 1.2 配置求解的结果

| 我写的 | 构建时算法实际收到的完整配置 |
|--------|---------------------------|
| `lr: 0.0005` | `lr: 0.0005`（原样保留） |
| `gamma: 0.95` | `gamma: 0.95`（原样保留） |
| *没写 `act_out_dim`* | `act_out_dim: 5`（配置求解器填入） |
| *没写 `act_in_dim`* | `act_in_dim: 33`（配置求解器填入） |
| *没写 `num_phase`* | `num_phase: 5`（从路网拓扑填充） |

**每个配置参数都是我显式写或平台求出的。不存在"隐藏参数"——`act_out_dim` 就是一个实际存在于算法 `setup(cfg)` 中的、由 `cfg.get("act_out_dim")` 读取的配置键。** 我写了就按我的，我没写就由约束求解。

### 1.3 配置参数的自由度

| 场景 | 行为 |
|------|------|
| 我写了 `act_out_dim: 6` | 等于我指定了输出 6 维。这个值通过等式传播到 actor 的 `translate_in_dim` 等 |
| 我写了 `num_phase: 8` | 覆盖拓扑值。`act_out_dim` 自动推导为 8 |
| 我什么都没写 | 拓扑提供 `num_phase=5` → `act_out_dim=5` → 全链路传播 |
| 我写了 `act_out_dim: 6` 同时拓扑 `num_phase=5` | **冲突告警**，但用户值不被覆盖 |

**配置参数不是"平台替我决定"，而是"平台检查约束一致性，当等式序列只剩下一个缺口时自动闭合"。**

---

## 2. 设计灵魂：配置即接口

### 2.1 组件之间只通过配置键名交互

传统方式：

```
Observer 输出 (B,33) → Algorithm.forward 需要知道第一维是 batch、第二维是 33
                        ↑ 硬编码或由调用者直接传递
```

PI-Light 方式：

```
Observer 通过 cfg["num_phase"] 决定 obverse 的输出结构
Algorithm 通过 cfg["act_in_dim"] 初始化网络输入层
            ↑
这两个配置键由 ConfigSolver 通过等式 {observer.observe_out_dim == algorithm.act_in_dim} 保持相等
```

**组件不持有对彼此的引用。它们共同持有的是一组配置键名。约束就是键名之间的等式。**

### 2.2 所有外部需求都是配置键

Algorithm 需要知道的东西，全部来自 `cfg`：

| 需要什么 | 哪个配置键 | 谁提供这个键的值 |
|---------|-----------|----------------|
| 输入维度 | `act_in_dim` | 边等式（从 `observer.observe_out_dim` 传播） |
| 输出维度 | `act_out_dim` | 派生约束（从 `num_phase` 推导，或用户直接写） |
| 网络内部的 lane-link 数 | `num_lanelink` | 派生约束（从 `max_lanelinks` 推导） |
| 学习率 | `lr` | 用户在 yaml 里显式写 |

**Observer 也一样**——它需要的 `num_phase`、`max_lanelinks` 来自拓扑缓存（或用户覆盖），`features` 来自用户 yaml。

### 2.3 三类配置键：不同来源，统一管理

所有组件读取的配置键，按**值的来源**分为三类：

#### 拓扑键（Topology Key）

**定义**：值来源于物理路网本身，由仿真器查询得到。

```
num_phase       — 全路网最大绿灯相位数
max_lanelinks   — 全路网最大进口车道数
num_tsc         — 信号灯总数
```

**来源**：框架启动仿真器（只解析路网、不运行仿真），调用 `env.phase_count(jid)`、`env.incoming_lanes(jid)` 等 Env ABC 标准接口，将结果写入 `datasets_index.yaml` 缓存。

**为什么拓扑键不可从派生约束反向修改**：

拓扑键描述的是物理世界的客观事实。框架不能因为用户写了 `act_out_dim: 100` 就把 `num_phase` 改成 100——路网上根本没有 100 个相位的路口。这和"用户说天空是绿色的但实际是蓝色的"是一个道理。

```
允许: num_phase=5 (路网真实值) → act_out_dim=5 (推导)
禁止: act_out_dim=100 (用户写的) → num_phase=100 (修改物理事实)
```

**用户可以覆盖拓扑键**——这是用户明确声明"我要用一个不匹配这个路网的值"。ConfigSolver 会告警（用户值 ≠ 实测值），但不阻止。这在以下场景有用：用户在真实路网上测试一个小相位子集的算法。

#### 端口键（Port Key）

**定义**：值由 PortEquation **实测**得到。框架实例化组件 → 调用端口方法 → 测量输出维度。

```
observe_out_dim      — Observer.observe() 的输出特征维度
compute_out_dim      — Reward.compute() 的输出维度
step_out_dim         — Env.step() 的返回数据结构维度
```

**来源**：PortEquation 的 `compute()` 方法。当该方法的所有 `involved_cfg_keys`（灵敏度分析确认的影响维度的配置键）全部已知时触发实测。

**用户可写吗**：可以。如果用户写了 `observer.observe_out_dim: 64`，PortEquation 实测为 33 时会告警冲突。用户保留自己的值（标记"用户覆盖"），约束组的其他成员依据用户值传播。

#### 派生键（Derived Key）

**定义**：值从其他已知配置键通过公式计算得到，不需要实例化组件。

```
act_out_dim   = num_phase              （输出维度 = 可用相位数）
num_lanelink  = max_lanelinks          （lane-link 数 = 最大进口道数）
max_lanes     = max_lanelinks          （别名）
```

**来源**：`ConfigSolver.add_internal_constraint(kind, target, dependencies, formula)`。当所有依赖键的值已知时自动触发计算。

**可双向传播**：派生键之间的别名等式是双向的（`max_lanelinks ↔ num_lanelink`），但从拓扑键到派生键只是单向的（`num_phase → act_out_dim`）。

**用户可写吗**：可以。写了就覆盖推导值，和拓扑键冲突的规则一样——告警但不覆盖。

#### 三类键总览

```
拓扑键（物理事实，不可被约束反向修改）
  num_phase ──→ act_out_dim ──→ actor.translate_in_dim
  max_lanelinks ──→ num_lanelink
             └──→ max_lanes（别名）

端口键（PortEquation 实测，可被用户覆盖）
  observer.observe_out_dim == algorithm.act_in_dim
  observer.observe_out_dim：由灵敏度分析确定的依赖键触发实测

派生键（从已知键计算，可双向传播）
  max_lanelinks ↔ max_lanes（别名，双向）
  num_phase → act_out_dim（单向，物理事实不可反推）
```

### 2.4 配置耦合 vs 直接耦合

| 直接耦合 | 配置耦合 |
|---------|---------|
| Observer 改输出结构 → Algorithm 崩溃 | Observer 改输出结构 → `observe_out_dim` 重新实测 → Algorithm 通过等式感知新维度 |
| 换数据集 → 所有维度手动重算 | 换数据集 → 拓扑参数重新查询 → 等式链自动重新求解 |

---

## 3. 维度检查：端口方法之间的维度等式

### 3.1 每个端口方法都有输入/输出维度

编排器调用链中的每个方法，框架都为它提取一个输入维度和一个输出维度：

```
Observer.observe(raw)
  → 接收 raw (List[JunctionState])
  → 输出 List[Obs.features]
  → 输入维度：不直接约束（来自 env.state 的产出已内嵌在 raw 中）
  → 输出维度：observe_out_dim

Algorithm.act(obs)
  → 接收 Obs.features (np.ndarray)
  → 输出 Action
  → 输入维度：act_in_dim
  → 输出维度：act_out_dim（Q 值的维度）

Actor.translate(actions)
  → 接收 List[Action]
  → 输出 Dict[str, int]（信号灯命令）
  → 输入维度：translate_in_dim（动作空间的维度）
  → 输出维度：translate_out_dim（命令的维度）
```

### 3.2 编排器的调用边自动产生等式

AST 扫描编排器源码（见第 8 章）产生方向边，每条边产生一个等式：

```
编排器源码                                    边                          等式
─────────────────────────────────────────────────────────────────────────
obs = self._observer.observe(raw)      observer.observe → [下一个调用]
a = algo.act(obs)                      observer.observe → algorithm.act
                                       ↓
                                       observer.observe_out_dim == algorithm.act_in_dim
─────────────────────────────────────────────────────────────────────────
a = algo.act(obs)                      algorithm.act → [下一个调用]
cmds = self._actor.translate([a])      algorithm.act → actor.translate
                                       ↓
                                       algorithm.act_out_dim == actor.translate_in_dim
```

### 3.3 这是维度一致性检查，不是"自动帮你配"

这些等式的本质是**断言**：

> orchestrator 把 observer 的产出直接传给 algorithm。所以 observer 的输出维度必须等于 algorithm 的输入维度。如果不等于，这个编排就是不可行的。

框架做的事情：

1. 发现等式（从编排器 AST）
2. 当等式两边已知时，检查是否相等（不一致就告警）
3. 当等式一边已知一边未知时，填入未知的一边（因为这是等式的必然结果）
4. 当两边都未知时，不做任何事（等待其他等式或派生约束填其中一边）

---

## 4. "最后一个缺口"规则：什么时候平台自动填入

### 4.1 规则描述

**平台不会"自动配一切"。只有在一个等式（或约束链）中只剩最后一个未知值时，才自动闭合它。**

```
例：端口维度链
  observer.observe_out_dim == algorithm.act_in_dim

  如果 observer.observe_out_dim 已由 PortEquation 实测为 33，
      algorithm.act_in_dim 未知
      → 等式两边只剩一个缺口 → 自动填入 33

  如果两个都未知
      → 等式两边有两个缺口 → 不做任何事，等别的路径填其中一个
```

### 4.2 派生约束也一样

```
约束：act_out_dim = num_phase

  如果 num_phase 已知（拓扑=5），act_out_dim 未知
      → 依赖全部已知，只剩一个缺口 → 自动填入 5

  如果 act_out_dim 已知（用户在 yaml 写了 6），num_phase 未知
      → 等式反向传播，但 ConfigSolver 不会反向写拓扑参数
      → 拓扑参数有其他来源（缓存/用户覆盖），不被等式反向修改
      → 端口键之间可以双向传播，拓扑键只能从源头来
```

### 4.3 正向与反向传播的区别

| 等式类型 | 正向 | 反向 |
|---------|------|------|
| 端口键 ↔ 端口键 | `observe_out_dim=33` → `act_in_dim=33` | `act_in_dim=64` → `observe_out_dim=64` |
| 拓扑键 → 派生键 | `num_phase=5` → `act_out_dim=5` | **不做反向**（拓扑参数不可从派生参数反推） |
| 派生键 ↔ 派生键（别名） | `max_lanelinks=28` → `num_lanelink=28` | `num_lanelink=28` → `max_lanelinks=28` |

**如果是端口键之间的等式，用户从哪个方向写都行——框架往两边传播。**

### 4.4 用户参与的完整示例

```
用户 YAML 写了:
  algorithm.config.act_out_dim: 6

ConfigSolver 求解:
  初始: act_out_dim=6, num_phase=None, act_in_dim=None, observe_out_dim=None

  # 没有反向传播（num_phase 不能从 act_out_dim 反推）
  # 边等式 observe_out_dim == act_in_dim 两边都 None，跳过

  结果: unresolved = ["num_phase", "act_in_dim", "observe_out_dim"]

此时框架提示: "num_phase is unresolved. Please specify it or provide roadnet_file."
```

```
用户 YAML 改为:
  roadnet_file: "data/LosAngeles/..."    # → topology gives num_phase=5
  algorithm.config.act_out_dim: 6

ConfigSolver 求解:
  初始: num_phase=5 (topology), act_out_dim=6 (user), act_in_dim=None, observe_out_dim=None

  派生约束: num_phase=5 已知，但 act_out_dim=6 已由用户设置
    → dim conflict: act_out_dim=6 vs computed from num_phase=5
    → 告警，保留用户值 6

  PortEquation: num_phase=5, max_lanelinks=28 → observer.observe_out_dim=33
  边等式: observe_out_dim=33 → act_in_dim=33

  结果: 全链路闭合（尽管 act_out_dim 和 num_phase 不一致—用户选择了 6，框架告警但不覆盖）
```

---

## 5. 灵敏度分析：找出真正影响维度的配置项

### 5.1 为什么需要灵敏度分析

PortEquation 需要知道它的输出维度受哪些配置键影响，才能在那些配置键全部已知时触发实测。

如果不用灵敏度分析直接"盲测"：Observer 有 20 个配置键，不知道哪些影响维度，要么永远等不全、要么用错误的依赖关系测出错误的维度。

### 5.2 第一步：AST 过近似扫描（trace_port_deps）

扫描组件源码，找出端口方法引用了哪些 `self._xxx`，追溯到 `setup()` 中来自 `cfg.get("key")` 的键。

```python
# FRAP Observer 源码：
def observe(self, raw):
    for li in range(self._max_lanelinks):   # 引用 self._max_lanelinks
        ...
    for i in range(self._num_phase):        # 引用 self._num_phase
        ...

# trace_port_deps 追溯到：
→ {"num_phase", "max_lanelinks", "features"}
```

**这是过近似集合**——包含了方法中所有被访问到的配置键，但不知道"改变这个键的值，输出的维度是否会变"。

### 5.3 第二步：灵敏度精化（filter_deps_by_sensitivity）

对过近似集合中的每个键，分别做：**改变它的值 → 重新构建组件 → 实测输出维度 → 看维度是否变化**。

维度变了 → 这个键确实影响了端口输出维度 → 保留。
维度不变 → 这个键虽然被端口方法使用了，但不影响维度 → 剔除。

```python
base_cfg  = {"num_phase": 5, "max_lanelinks": 28, "features": ["num"]}
dim_ref   = 33   # observe_out_dim 基线

扰动 num_phase (5→6):
  test_cfg = {"num_phase": 6, ...}
  dim_test = 38 ≠ 33 → num_phase 真正影响维度 ✓

扰动 features (["num"]→["num","waiting"]):
  dim_test = 61 ≠ 33 → features 真正影响维度 ✓
```

### 5.4 精化后用于 PortEquation

只有灵敏度确认为"影响维度"的键，才作为 `PortEquation.involved_cfg_keys`。

ConfigSolver 求解时：当该方程的所有 `involved_cfg_keys` 都已知了 → 触发实测 → 实例化组件 → 调端口方法 → 测量维度 → 写入对应的 `xxx_out_dim`。

---

## 6. 两层筛选：组件兼容性 vs 维度一致性

这是两个完全不同的检查，作用于不同的阶段和不同的对象。

### 6.1 第一层：组件兼容性（方法级筛选）

**做什么**：检查一个组件类是否有 Orchestrator 会调用的方法。

**作用对象**：类（插件类型），不是实例。

**怎么做的**：`recommend_assembly(orchestrator_name)` — 扫描 Orchestrator 源码找出对每个 kind 调用了哪些方法，然后筛选注册组件中实现了那些方法的。

```
SingleOrchestrator 对 algorithm 的调用:
  algo.act(obs)     → require method "act"
  algo.learn(batch) → require method "learn"
  algo.train()      → require method "train"
  algo.eval()       → require method "eval"
  algo.sync()       → require method "sync"

筛选：只有同时有 act + learn + train + eval + sync 的算法类才能通过
```

**时机**：可在用户选择组件时提供推荐列表，不强制。

**失败结果**：运行时 AttributeError（你选了一个没有 learn 的算法用在 single 编排器下）。

### 6.2 第二层：维度一致性（端口级检查）

**做什么**：检查编排器调用链中相邻端口方法的维度是否匹配。

**作用对象**：配置参数的值（不是方法存不存在），是实例构建后的实际维度。

**怎么做的**：ConfigSolver — 从 AST 推导的边构造等式 → 检查等式一致性 → 最后一个缺口时自动闭合。

```
等式: observer.observe_out_dim == algorithm.act_in_dim

observe_out_dim = PortEquation 实测 = 33
act_in_dim      = 用户在 yaml 里写 = 64
→ 33 ≠ 64 → dim conflict 告警
```

**时机**：构建阶段，在组件实例化之前。

**失败结果**：告警（dim conflict），不阻止构建，但运行时维度不匹配可能崩溃。

### 6.3 区别总览

| | 组件兼容性筛选 | 维度一致性检查 |
|---|---|---|
| 检查什么 | 方法是否存在 | 维度值是否相等 |
| 检查粒度 | 类级别（方法名列表） | 配置参数级别（数值） |
| 输入 | 编排器的方法调用记录 | 用户 yaml + 拓扑 + 实测维度 |
| 输出 | 可用组件列表 | 完整配置 / 冲突告警 |
| 是否阻止构建 | 否（信息性推荐） | 否（告警，不覆盖用户值） |
| 依赖 AST | 是（扫描调用） | 是（扫描边 → 等式） |

**两者协同但不重复**：方法筛选保证"这个组件能接"，维度检查保证"接了以后维度对得上"。

---

## 7. 配置约束搜索：用已知配置键筛选未知配置项

### 7.1 数据集搜索只是搜索的子集

第 2.3 节中定义了三种配置键。其中**拓扑键的值来自路网，而路网本身也是一个可选的配置项**（`roadnet_file`）。

同理，**任何配置键都可以成为搜索的约束条件**：

| 用户写了什么 | 搜索什么 |
|-------------|---------|
| `num_phase: 4`（拓扑键）且未选路网 | 搜索 `num_phase=4` 的路网文件 |
| `act_out_dim: 6`（派生键）且未选路网 | 它不能反向推导 `num_phase`（物理事实不可改），所以**不能**用于搜索路网 |
| `observer.observe_out_dim: 128`（端口键） | 搜索有哪些 Observer 插件在拓扑条件下能产出 128 维的特征 |
| `collector.batch_size: 256`（超参数，无约束参与） | **不需要搜索**——超参数没有"匹配数据集"的概念 |

**数据集搜索本质上是"在所有注册的路网缓存中，找到拓扑键值等于已知值的那些"——和在任何注册组件中找维度匹配的组件是一样的逻辑。**

### 7.2 两阶段求解（当前流程）

```
阶段 A：ConfigSolver 从用户已知配置出发
  → 拓扑键（num_phase=4）已知且 roadnet_file 未知
  → 调用 match_datasets({"num_phase": 4}, datasets_index)
  → 返回 num_phase=4 的所有路网列表
  → 框架自动选第一个，或提示用户选择

阶段 B：选定路网后
  → 从缓存读取该路网的完整拓扑（max_lanelinks、num_tsc 等）
  → 补入 ConfigSolver，继续求解剩余端口键和派生键
  → 全部闭合 → 构建组件
```

### 7.3 搜索的通用化

搜索不限于路网。同一个 `match_datasets` 逻辑可以扩展到任何"可搜索的配置空间"：

| 搜索空间 | 当前状态 | 搜索方式 |
|---------|---------|---------|
| 路网 | 已实现 | `datasets_index.yaml` + `match_datasets()` |
| Observer（按输出维度筛选） | 可扩展 | 扫描所有注册 observer 的 `PortEquation` 实测值 |
| Algorithm（按要求的输入/输出维度筛选） | 可扩展 | 类似上述 |
| Actor（按动作空间维度筛选） | 可扩展 | 类似上述 |

**前端可以给用户提供一个"搜索框"，输入任意配置键的值，系统返回所有能使该配置键成立的互斥选择（数据集、组件、超参组合）。**

### 7.4 为什么需要拓扑缓存而不是启动仿真器

路网拓扑（`num_phase`）需要在**组件构建前**就知道——ConfigSolver 用它计算 `observe_out_dim`、`act_out_dim`。但仿真器必须先知道用哪个路网才能启动。

```
先有鸡还是先有蛋？

  组件构建需要 num_phase ← 需要知道用什么路网 ← 路网由用户指定或搜索确定

解决方案：缓存预分析
  datasets_index.yaml（预先由仿真器扫描所有路网生成）
  → 不用启动仿真器 → 直接查缓存 → 得到拓扑 → 求解配置 → 启动仿真
```

### 7.5 何时启用搜索

| 用户 yaml | 行为 |
|----------|------|
| 写了 `roadnet_file` | 直接查缓存 → 求解 |
| 没写 roadnet，但写了拓扑键如 `num_phase: 4` | 搜索匹配路网 → 自动选第一个 |
| 什么都没推导出且没写 roadnet | 提示：缺少 num_phase 或指定 roadnet_file |

---

## 8. AST 推导：编排器代码即交互契约

### 8.1 三件自动推导的事

| 推导目标 | 从哪里读 | 例子 |
|---------|---------|------|
| Orchestrator 需要哪些 kind | `setup()` 的形参名 | `def setup(self, env, observer, algorithms, ...)` → `["environment", "observer", "algorithm", ...]` |
| `self._X` 对应哪个 kind | `setup()` 的赋值 | `self._env = env` → `"environment" → ["_env"]` |
| 组件间的调用边 | `episode()` 等方法体的调用语句序列 | 见 8.2 |

### 8.2 边的生成

```
赋值语句（返回值被捕获）:
  raw = self._env.state()         → Assign → (env, state, captured=True)
  obs = self._observer.observe()  → Assign → (observer, observe, captured=True)
  a = algo.act(obs)               → Assign → (algorithm, act, captured=True)
      ↓
  每条赋值语句产生一个"有数据产出"的节点。
  前一个 captured=True 的节点连向后一个节点，形成边：
    (env, state) → (observer, observe)
    (observer, observe) → (algorithm, act)

独立表达式（返回值被丢弃）:
  self._tracker.accumulate_step() → Expr → (tracker, accumulate_step, captured=False)
  algo.train()                    → Expr → (algorithm, train, captured=False)
      ↓
  captured=False → 不出边（数据没有传递给下游调用）
```

### 8.3 边的用途

每条 `(from_kind, from_method) → (to_kind, to_method)` 的边自动产生：

```
{from_kind}.{from_method}_out_dim == {to_kind}.{to_method}_in_dim
```

这是 ConfigSolver 获取跨组件维度等式的**唯一入口**。

### 8.4 编写编排器的约束

| 规则 | 原因 |
|------|------|
| `setup()` 参数名用 kind 名 | 框架据此推断需要哪些组件种类 |
| `self._env = env` | 框架通过 AST 构建 kind → 属性映射 |
| 有返回值的方法调用必须用赋值接住 | `x = self._A.foo()` 出边，`self._A.foo()` 不出边 |
| 对于 `getattr`、闭包等元编程 | AST 无法分析，走保守 fallback，可能多报告警 |

---

## 9. 前端呈现：约束组的可视化

本章专门写给要构建配置界面的前端开发者。

### 9.1 核心概念：约束组

后台求解过程中，一组相互关联的配置键共享一个约束等式或推导链：

```
等式 observer.observe_out_dim == algorithm.act_in_dim
  → 约束组：{observer.observe_out_dim, algorithm.act_in_dim}

派生链 num_phase → act_out_dim → actor.translate_in_dim
  → 约束组：{num_phase, algorithm.act_out_dim, actor.translate_in_dim}
```

一个配置键可以同时属于多个约束组。例如 `act_out_dim` 同时参与：
- 与 `num_phase` 的派生约束
- 与 `actor.translate_in_dim` 的端口维度等式

### 9.2 前端呈现方案：颜色标识 + 独立约束面板

**方案：在"自由装配"配置表单中以颜色带（color stripe）标识约束组，另设"约束概览"面板展示全貌。**

#### 9.2.1 每个配置项左侧加颜色带

```
┌──────────────────────────────────────────┐
│  Algorithm 配置                           │
│                                          │
│  ● act_in_dim          [    33    ]      │ ← 蓝色 = 等式组 A
│  ● act_out_dim         [    5     ]      │ ← 橙色 = 等式组 B + C
│  ● num_lanelink        [    28    ]      │ ← 绿色 = 等式组 D
│  ● lr                  [ 0.0005  ]      │ ← 无色 = 无约束（自由值）
│  ● gamma               [  0.95   ]      │ ← 无色
│  ● tau                 [  0.01   ]      │ ← 无色
└──────────────────────────────────────────┘
```

**颜色规则**：

| 颜色 | 含义 |
|------|------|
| 无色 | 没有参与任何约束等式，用户可任意修改 |
| 同色 | 共享同一个约束组（一个等式或推导链） |
| 多色带（如两条 color stripe） | 该配置项同时属于多个约束组 |

#### 9.2.2 "最后一个缺口"高亮

当一个约束组中除一个成员外全部已填，那个未填的成员在 UI 上应呈现特殊状态：

```
等式组：observer.observe_out_dim == algorithm.act_in_dim

  observe_out_dim = 33  ✓ 已填
  act_in_dim      = [   33   ]  ← 系统推荐值，虚线边框 + 蓝色背景
                   推荐填入 33 [应用] [忽略]
```

**交互行为**：
- 输入框边框变为虚线、背景变为浅蓝色
- 显示推荐值（由后台 ConfigSolver 计算，不可手动编辑推荐值本身）
- 用户点击「应用」→ 填入推荐值 → 约束组闭合
- 用户点击「忽略」→ 留空 → 约束组保持未闭合状态

#### 9.2.3 冲突告警

当用户手动填入的值与约束组的已知值推导出的结果不一致：

```
等式组：observer.observe_out_dim == algorithm.act_in_dim

  observe_out_dim = 33 (实测)
  act_in_dim      = [   64   ]  ← 用户手动填的
                    ⚠ 约束组要求此值为 33，当前为 64
                    [保留我的值] [使用推荐值 33]
```

**交互行为**：
- 输入框边框变为红色实线
- 下方出现告警提示，列出冲突详情
- 两个按钮：保留用户值（保留红色边框，标记为"用户覆盖"）、使用推荐值

### 9.3 "约束概览"面板

除了每个字段上的内联颜色带，还应有一个聚合面板展示全部约束关系。

```json
// 后台构造并返回给前端的约束组数据
{
  "constraint_groups": [
    {
      "id": "eq_observer_observe_algorithm_act",
      "label": "observer.observe 输出 → algorithm.act 输入",
      "type": "port_equal",
      "members": [
        {"kind": "observer", "key": "observe_out_dim", "value": 33, "source": "measured"},
        {"kind": "algorithm", "key": "act_in_dim",    "value": null, "source": null}
      ],
      "status": "one_missing",
      "recommendation": {"kind": "algorithm", "key": "act_in_dim", "value": 33}
    },
    {
      "id": "derive_act_out_dim",
      "label": "act_out_dim = num_phase",
      "type": "derived",
      "members": [
        {"kind": "algorithm",  "key": "act_out_dim", "value": 5, "source": "derived"},
        {"kind": "observer",   "key": "num_phase",   "value": 5, "source": "topology"},
        {"kind": "actor",      "key": "translate_in_dim", "value": 5, "source": "propagated"}
      ],
      "status": "resolved"
    }
  ]
}
```

每个约束组的状态：

| status | 含义 | 前端如何表示 |
|--------|------|------------|
| `resolved` | 所有成员已填且一致 | 全组绿色对勾 |
| `one_missing` | 只剩一个未填 | 未填项高亮 + 显示推荐值 |
| `multiple_missing` | 多个未填 | 组内无色带，等待用户填任意一个 |
| `conflict` | 成员值不一致 | 冲突项红色 + 告警信息 |

### 9.4 是否单独放一个"约束参数"栏目

**不建议。** 原因：

1. 约束参数和自由参数没有天然的视觉分界——`act_out_dim` 在 Algorithm 的语境下和 `lr`、`gamma` 一样，都是"我的配置"。用户关心的是"这个东西是干什么的"，而不关心它是怎么推导的。

2. 如果单独抽到一个栏目，用户填了 `act_out_dim: 5` 却不理解它为什么会改变 `num_phase`（因为这两个参数分别位于不同组件），容易产生困惑。

**建议方案：保持参数在各自组件的位置不变，用颜色带做内联标注。** 另设一个可折叠的"约束概览"侧栏（或底部面板），点击任一带颜色的参数时自动展开对应约束组。

### 9.5 数据集的呈现（配置约束搜索的特例）

数据集选择是配置约束搜索的最典型场景。`roadnet_file` 本身是一个可选的配置项，`num_phase`、`max_lanelinks` 等拓扑键是由它决定的。

用户可以**不选路网，直接指定拓扑键**。系统根据已知拓扑键的值，从缓存中搜索匹配的路网。

```
┌──────────────────────────────────────────┐
│  数据集                                   │
│  roadnet_file: [选择数据集...]        ▼   │
│    → num_phase = 5                        │
│    → max_lanelinks = 28                   │
│    → num_tsc = 18                         │
│                                           │
│  或者：直接指定拓扑值（无需选数据集）      │
│  ● num_phase          [   5    ]  ← 橙色 │
│  ● max_lanelinks      [   28   ]  ← 绿色 │
└──────────────────────────────────────────┘
```

如果用户没有选择数据集，但填写了拓扑键，前端应展示搜索结果：

```
num_phase = 4 约束下的数据集匹配：
  ▸ grid/grid.net.xml         num_phase=2  ← 不符合
  ▸ Atlanta/roadnet.net.xml   num_phase=5  ← 不符合
  ▸ Monaco/roadnet.net.xml    num_phase=5  ← 不符合
  ▸ LargeGrid/roadnet.net.xml num_phase=5  ← 不符合
  ▸ LosAngeles/roadnet.net.xml num_phase=5 ← 不符合

没有找到 num_phase=4 的数据集。请调整 num_phase 或手动指定 roadnet_file。
```

**同样的搜索逻辑也适用于其他配置键**。前端可以提供一个通用的"从此配置项搜索"动作：用户在任何带颜色的配置项上右键 / 长按 → 「搜索匹配项」→ 系统返回所有能使该值成立的互斥选项。

| 搜索入口 | 搜索空间 | 返回 |
|---------|---------|------|
| `num_phase: 4` | 路网缓存 | 匹配的数据集列表 |
| `observe_out_dim: 128` | 注册的 Observer 插件 | 所有能产出 128 维特征的 Observer |
| `act_in_dim: 64` | 注册的 Algorithm 插件 | 所有要求 64 维输入的 Algorithm |
| `translate_in_dim: 4` | 注册的 Actor 插件 | 所有接受 4 维动作空间的 Actor |

### 9.6 三个关键的 UI 状态机

**状态机 1：单个约束组内的参数**

```
            ┌──────────┐
            │ 初始态    │ 所有成员空白
            │ 无色带    │
            └────┬─────┘
                 │ 用户填了第一个值
                 ▼
            ┌──────────┐
            │ 部分填    │ 有些已知，有些未知
            │ 色带      │
            └────┬─────┘
        ┌────────┼────────┐
        ▼                 ▼
  ┌──────────┐    ┌──────────────┐
  │ 只剩一个  │    │ 用户填了冲突值 │
  │ 推荐态    │    │ 冲突态        │
  │ 虚线+浅蓝 │    │ 红色边框      │
  └────┬─────┘    └──────┬───────┘
       │                 │
       ▼                 ▼
  ┌──────────┐    ┌──────────────┐
  │ 闭合态    │    │ 用户确认覆盖  │
  │ 绿色对勾  │    │ 红色边框保留  │
  └──────────┘    │ + 覆盖标记   │
                  └──────────────┘
```

**状态机 2：维度检查 vs 方法筛选的错误提示**

```
用户在 YAML 或 UI 中选了 algorithm: fixed_time

推荐面板:
  ⚠ fixed_time 缺少以下方法: learn, sync, train, eval
  SingleOrchestrator 要求 algorithm 组件实现这些方法。
  建议: 使用 dqn / frap / colight / ma2c_agent 代替，或切换到 rule 编排器。
  [仍然使用 fixed_time] [选择推荐算法]
```

**状态机 3：跨组件参数关联的突出**

当用户选中某个带颜色的参数时，前端应高亮同一约束组中**位于其他组件**的关联参数：

```
用户点击 Algorithm > act_out_dim (橙色)

高亮:
  ▶ Observer > num_phase (橙色) — "act_out_dim 从此推导"
  ▶ Actor > translate_in_dim (橙色) — "与此值相等"
```

### 9.7 后台应提供的 API

前端不需要自己算约束组。后台在 `GET /api/config/constraint-groups` 返回：

```json
{
  "groups": [...],                       // 9.3 节定义的结构
  "dataset_matches": [                   // 如果拓扑键已知但未选数据集
    {"path": "grid/grid.net.xml", "num_phase": 2, "num_tsc": 5, "match": false},
    {"path": "Atlanta/roadnet.net.xml", "num_phase": 5, "num_tsc": 6, "match": true}
  ],
  "method_mismatches": [                 // 组件兼容性问题
    {"kind": "algorithm", "plugin": "fixed_time", "missing": ["learn", "sync"]}
  ]
}
```

此 API 应在每次用户修改任何配置参数后重新返回更新后的约束组状态。求解是幂等且轻量的（不启动仿真器），可高频调用。

---

## 10. 附录

### A：ConfigSolver 三层迭代

```
while 任何值被填入:
  ┌──────────────────────────────────────────┐
  │ 第一层：边等式                             │
  │   observe_out_dim == act_in_dim           │
  │   ...（双向传播，任意一端已知 → 填另一端）│
  ├──────────────────────────────────────────┤
  │ 第二层：派生约束                           │
  │   num_lanelink = max_lanelinks             │
  │   max_lanes = max_lanelinks（别名）        │
  │   ...（依赖全部已知 → 触发计算）           │
  ├──────────────────────────────────────────┤
  │ 第三层：PortEquation 实测                  │
  │   依赖全部已知 → 实例化 → 调用 → 测量     │
  │   ...（写入 observe_out_dim 等）           │
  └──────────────────────────────────────────┘
```

### B：调度层文件职责

```
scheduling/registry.py
  ├── register(kind, name)     — 插件注册装饰器
  ├── discover()               — 递归导入所有模块触发注册
  ├── _derive_orch_attr_map    — self._X → kind 映射
  ├── _scan_orch_edges         — 调用序列 → 数据流边
  ├── get_orch_wiring()        — wiring API
  └── recommend_assembly()     — 组件兼容性筛选

scheduling/config_solver.py
  ├── trace_port_deps()        — AST 扫描端口方法的 cfg 引用
  ├── filter_deps_by_sensitivity() — 扰动+实测，精确化依赖
  ├── PortEquation             — 实例化+测量端口输出维度
  ├── ConfigSolver             — 约束收集+三层迭代求解
  ├── solve_dims_from_probe()  — 从已构建的组件实测维度
  └── _verify_dims()           — 运行时维度一致性验证

scheduling/launcher.py
  ├── Launcher.__init__()      — discover + 读 yaml
  ├── Launcher.build()         — 六段式构建流水线
  ├── _load_env_topo()         — 优先缓存，未命中则仿真器即时查询
  └── _build_orch()            — 按 wiring 装配编排器

scheduling/dataset_index.py
  ├── _query_topo_from_env()   — 仿真器查询单路网拓扑
  ├── rebuild_cache()          — 重建全部数据集拓扑缓存
  ├── match_datasets()         — 按约束筛选数据集
  └── find_topo()              — 缓存查指定路网拓扑
```

### C：用户交互路径汇总

```
路径 1：用户写了 roadnet_file + 超参数
  → 从缓存得拓扑 → 等式传播 + PortEquation 实测 → 全链路闭合

路径 2：用户写了维度参数（如 act_out_dim: 6）
  → 等式反向传播到 act_in_dim 等
  → 派生约束不反向（num_phase 不从 act_out_dim 反推）
  → 如果 num_phase 未定，提示需要 roadnet 或 num_phase

路径 3：用户需要数据集搜索
  → 用户写了拓扑级约束（如 num_phase: 4）且无 roadnet
  → match_datasets 筛选匹配路网
  → 确定路网 → 补充完整拓扑 → 求解

路径 4：新数据集
  → py -m modutsc index --rebuild
  → 仿真器逐个分析 .net.xml → 写入缓存

### D：组件构建阶段的通用性

ConfigSolver 求解完成后，所有组件需要的配置参数都已确定。构建阶段不依赖拓扑顺序。

**构建流程**：

```
assembled = {}

① 环境先构建（物理必要 — 需要启动仿真器）
assembled["environment"] = SumoEnv(); env.launch(cfg)

② 其余组件按任意顺序构建（全部 cfg 已闭合）
for kind in kinds (except environment):
    section = yaml.get(kind)
    if isinstance(section, list):             ← 多实例
        for entry in section:
            cls = find(kind, entry.plugin)
            obj = cls(); obj.setup(final_cfg)
    else:                                      ← 单实例
        cls = find(kind, section.plugin)
        obj = cls(); obj.setup(kind_cfg)
```

**为什么不需要拓扑顺序**：ConfigSolver 已经求解全部配置键。Observer 的 `cfg["act_in_dim"]` 在所有 kind 中都是相同的已求解值，不管 Observer 和 Algorithm 谁先构建。

**通用性保证**：只要 YAML section 遵循 `{plugin, config}` 结构，任何新 kind 无需修改 launcher 代码即可构建。list 型自动走多实例分支，dict 型自动走单实例分支。
```
