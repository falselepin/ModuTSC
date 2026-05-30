"""
ModuTSC 前端 API 模块
=====================
提供原子化接口供前端直接调用，覆盖自由装配、配置约束、数据集、配置文件、脚手架五大场景。

模块分类：
  1. 注册表查询     — list_kinds, list_plugins, list_orchestrators
  2. 自由装配阶段   — assembly_requirements, assembly_iterative, compatible_orchestrators
  3. 配置约束阶段   — ConstraintSession (create_constraint_session, get_state, set_value, recommend)
  4. 数据集与拓扑   — list_datasets, get_dataset_topo, match_datasets_by_topo
  5. 配置文件操作   — read_config_structure, scaffold_config
  6. 实验运行与实时日志 — run_experiment, get_experiment_log
  7. 代码脚手架     — scaffold_orchestrator, scaffold_kind_base, scaffold_plugin
"""

import yaml
import json
import os as _os

from modutsc.scheduling.registry import (
    find, discover, list_all, _all_kinds,
    get_orch_kinds, get_orch_wiring,
)
from modutsc.scheduling.dataset_index import (
    load_index, find_topo, match_datasets,
)
from modutsc.scheduling.config_solver import (
    ConfigSolver, ports_from_wiring,
    trace_port_deps, resolve_port_deps, filter_deps_by_sensitivity,
    _build_and_measure_port, measure_dim,
    scan_setup_cfg_keys, resolve_config_keys, scan_setup_param_deps,
    _self_probe_determiner, _method_uses_env,
)

discover()

_COMPATIBILITY_CACHE = {}


def compute_compatibility_cache():
    """启动时一次性计算所有编排器 × 所有组件的兼容矩阵。

    规则（优先级递减）：
      1. 编排器的 __compatible_plugins__ 手动声明 → 直接使用
      2. 无声明 → AST 扫描必需方法 → hasattr 检查 → 自动推导
    """
    global _COMPATIBILITY_CACHE
    from modutsc.scheduling.registry import _scan_orch_kind_calls as _scan_calls

    _COMPATIBILITY_CACHE.clear()
    for orch_name in sorted(list_all("orchestrator")):
        orch_cls = find("orchestrator", orch_name)
        if orch_cls is None:
            continue
        kinds = get_orch_kinds(orch_name)
        orch_decl = getattr(orch_cls, '__compatible_plugins__', None) or {}
        orch_entry = {}
        for kind in kinds:
            methods = sorted(_scan_calls(orch_cls, kind))
            decl_set = set(orch_decl.get(kind, []))
            options = []
            for plugin_name in sorted(list_all(kind)):
                cls = find(kind, plugin_name)
                if cls is None:
                    continue
                method_ok = all(hasattr(cls, m) for m in methods)
                if decl_set:
                    compatible = method_ok and (plugin_name in decl_set)
                else:
                    compatible = method_ok
                options.append({
                    "plugin": plugin_name,
                    "compatible": compatible,
                    "viable_in": [],
                    "missing_in": {},
                })
            orch_entry[kind] = {
                "required_methods": methods,
                "options": options,
            }
        _COMPATIBILITY_CACHE[orch_name] = orch_entry


def invalidate_compatibility_cache():
    """新模块注册后调用，重建缓存。"""
    compute_compatibility_cache()


compute_compatibility_cache()

# ═══════════════════════════════════════════════════════════════
#  1. 注册表查询
# ═══════════════════════════════════════════════════════════════

def list_kinds():
    return sorted(_all_kinds())


def list_plugins(kind=None):
    if kind is None:
        return {k: sorted(list_all(k)) for k in _all_kinds()}
    return sorted(list_all(kind))


def list_orchestrators():
    return sorted(list_all("orchestrator"))


def get_orchestrator_kinds(orch_name):
    kinds = [k for k in get_orch_kinds(orch_name) if k != "orchestrator"]
    if "environment" not in kinds:
        kinds.append("environment")
    return kinds


def get_orchestrator_wiring(orch_name):
    raw = get_orch_wiring(orch_name)
    return raw.get("edges", [])


# ═══════════════════════════════════════════════════════════════
#  2. 自由装配阶段
# ═══════════════════════════════════════════════════════════════

def _single_orch_requirements(orch_name):
    """内部：单个编排器的装配要求。直接从缓存读取。"""
    if not _COMPATIBILITY_CACHE:
        compute_compatibility_cache()
    entry = _COMPATIBILITY_CACHE.get(orch_name)
    if entry is None:
        return None
    kinds = get_orchestrator_kinds(orch_name)
    return {
        "orchestrator": orch_name,
        "kinds": kinds,
        "slots": {kind: {"required_methods": info["required_methods"],
                          "options": list(info["options"])}
                  for kind, info in entry.items()},
    }


def assembly_requirements(orch_name):
    """给定编排器，返回每个组件槽位的装配要求。"""
    req = _single_orch_requirements(orch_name)
    if req is None:
        raise ValueError(f"orchestrator '{orch_name}' not found")
    return req


def _orch_supports_selection(orch_name, selections):
    """内部：编排器是否支持给定的所有选择。从缓存读取。"""
    entry = _COMPATIBILITY_CACHE.get(orch_name)
    if entry is None:
        return False
    for kind, required_plugin in selections.items():
        opts = entry.get(kind, {}).get("options", [])
        if not any(o["plugin"] == required_plugin and o["compatible"] for o in opts):
            return False
    return True


def assembly_iterative(selections):
    """根据当前部分选择，返回下一步可用的编排器和组件选项。

    支持两种交互路径：
      - 从编排器出发：先选编排器 → 然后逐个选组件
      - 从组件出发：先选一个组件 → 自动缩小编排器 → 再选组件 → 继续缩小

    Args:
        selections: {kind: plugin_name}  当前已做的选择，可为空

    Returns:
        {
            "viable_orchestrators": ["single", "rule"],  — 当前仍可用的编排器
            "slots": {
                "observer": {
                    "options": [  — 在至少一个可用编排器下兼容的插件
                        {"plugin":"frap","compatible":true, "viable_in":["single","rule"],
                         "missing_in":[]},
                        {"plugin":"standard","compatible":true, "viable_in":["single"],
                         "missing_in":["rule"]},
                    ]
                },
                ...
            },
            "kinds_order": ["observer","actor",...],
            "all_complete": false,         — 所有槽位都选满了
            "dead_end": false,             — 无可用编排器，需重新选择
            "missing_kinds": ["algorithm"] — 尚未选择的槽位
        }
    """
    selections = dict(selections) if selections else {}
    all_orchs = sorted(list_all("orchestrator"))

    # filter orchestrators
    if selections:
        viable = [on for on in all_orchs if _orch_supports_selection(on, selections)]
    else:
        viable = all_orchs

    if not viable:
        return {
            "viable_orchestrators": [],
            "slots": {},
            "kinds_order": [],
            "all_complete": False,
            "dead_end": True,
            "missing_kinds": [],
        }

    # union kinds from all viable orchestrators
    all_kinds_set = set()
    for on in viable:
        all_kinds_set.update(get_orchestrator_kinds(on))
    kinds_order = [k for k in get_orchestrator_kinds(viable[0]) if k in all_kinds_set]

    # for each kind, union of options viable in at least 1 orchestrator
    slots = {}
    for kind in kinds_order:
        plugin_viable = {}
        for on in viable:
            entry = _COMPATIBILITY_CACHE.get(on, {})
            for opt in entry.get(kind, {}).get("options", []):
                pn = opt["plugin"]
                if pn not in plugin_viable:
                    plugin_viable[pn] = {
                        "plugin": pn,
                        "viable_in": [],
                        "missing_in": {},
                    }
                if opt["compatible"]:
                    plugin_viable[pn]["viable_in"].append(on)
                else:
                    plugin_viable[pn]["missing_in"][on] = []

        options = []
        for pn, data in sorted(plugin_viable.items()):
            options.append({
                "plugin": pn,
                "compatible": len(data["viable_in"]) > 0,
                "viable_in": sorted(data["viable_in"]),
                "missing_in": sorted(data["missing_in"].keys()),
            })
        slots[kind] = {"options": options}

    already_selected = set(selections.keys())
    missing_kinds = [k for k in kinds_order if k not in already_selected]
    all_complete = len(missing_kinds) == 0

    return {
        "viable_orchestrators": sorted(viable),
        "slots": slots,
        "kinds_order": kinds_order,
        "all_complete": all_complete,
        "dead_end": False,
        "missing_kinds": missing_kinds,
    }


def compatible_orchestrators(kind, plugin_name):
    orch_names = []
    for on, entry in _COMPATIBILITY_CACHE.items():
        for opt in entry.get(kind, {}).get("options", []):
            if opt["plugin"] == plugin_name and opt["compatible"]:
                orch_names.append(on)
                break
    return sorted(orch_names)


def _probe_compatibility_with_env(orch_name, dataset_path, selections):
    wiring = get_orch_wiring(orch_name)
    wiring_edges = wiring.get("edges", [])
    ports = ports_from_wiring(wiring_edges)

    env_cls = find("environment", selections.get("environment", "sumo"))
    if env_cls is None:
        return None

    import time
    import subprocess
    waited = 0
    while waited < 10:
        try:
            result = subprocess.run(
                ["tasklist", "/fi", "imagename eq sumo.exe"],
                capture_output=True, text=True, timeout=5,
                encoding='utf-8', errors='ignore'
            )
            if "sumo.exe" not in result.stdout:
                break
        except Exception:
            pass
        time.sleep(1)
        waited += 1

    env = env_cls()
    try:
        env.launch({
            "roadnet_file": dataset_path,
            "gui": False,
        })
    except Exception:
        try:
            env.close()
        except Exception:
            pass
        return None

    env_topo = {}
    try:
        ids = env.ids()
        if ids:
            env_topo["num_phase"] = max(env.phase_count(j) for j in ids)
            env_topo["max_lanelinks"] = max(
                len(env.traffic_light_controlled_links(j)) for j in ids
            )
            env_topo["max_green_phases"] = max(
                len(env.green_phase_indices(j)) for j in ids
            )
            env_topo["num_tsc"] = len(ids)
    except Exception:
        pass

    compat = {}
    for kind in sorted(ports.keys()):
        compat[kind] = {}
        for direction in ["output", "input"]:
            methods = sorted([
                m for m, d in ports.get(kind, {}).items() if d == direction
            ])
            for method_name in methods:
                cls_list = [
                    (pn, find(kind, pn))
                    for pn in sorted(list_all(kind))
                    if find(kind, pn) is not None
                ]
                compat[kind].setdefault(method_name, {})[direction] = {}

                for pn, cls in cls_list:
                    dim = None
                    try:
                        dim = _build_and_measure_port(
                            cls, method_name, env_topo, env=env
                        )
                    except Exception:
                        pass
                    compat[kind][method_name][direction][pn] = dim

    env.close()

    return {
        "env_topo": env_topo,
        "compat": compat,
        "wiring_edges": wiring_edges,
    }


def _port_is_env_locked(kind, method_name):
    cls = find(kind, list_all(kind)[0]) if list_all(kind) else None
    if cls is None:
        return True
    deps = resolve_port_deps(cls, method_name)
    if not deps:
        return True
    topo_like = {"num_phase", "max_lanelinks", "max_green_phases", "num_tsc"}
    return not bool(deps - topo_like)


def filter_plugins_by_dimension(orch_name, dataset_path, selections=None):
    if selections is None:
        selections = {}
    if "environment" not in selections:
        selections["environment"] = "sumo"

    result = _probe_compatibility_with_env(orch_name, dataset_path, selections)
    if result is None:
        return {"error": "无法启动环境进行兼容性检测"}

    compat = result["compat"]
    wiring_edges = result["wiring_edges"]
    env_topo = result["env_topo"]

    by_kind = {}
    for kind, methods in compat.items():
        entries = []
        for method_name, directions in methods.items():
            dim_key = f"{method_name}_out_dim"

        for pn in sorted(list_all(kind)):
            incompatible_with = []
            configurable_with = []
            for edge in wiring_edges:
                fk = edge.get("from", {}).get("kind")
                fm = edge.get("from", {}).get("method")
                tk = edge.get("to", {}).get("kind")
                tm = edge.get("to", {}).get("method")
                if not all([fk, fm, tk, tm]):
                    continue

                if fk == kind:
                    src_dim = compat.get(fk, {}).get(fm, {}).get("output", {}).get(pn)
                    if src_dim is None:
                        continue
                    for dst_pn in sorted(list_all(tk)):
                        dst_dim = compat.get(tk, {}).get(tm, {}).get("input", {}).get(dst_pn)
                        if dst_dim is None or src_dim == dst_dim:
                            continue
                        to_locked = _port_is_env_locked(tk, tm)
                        if to_locked:
                            incompatible_with.append(f"{tk}/{dst_pn}")
                        else:
                            configurable_with.append({
                                "kind": tk,
                                "plugin": dst_pn,
                                "actual_out": src_dim,
                                "expected_in": dst_dim,
                                "hint": f"将 {tk}.config 中的维度参数调整为适配 {src_dim} 即可兼容",
                            })

                if tk == kind:
                    dst_dim = compat.get(tk, {}).get(tm, {}).get("input", {}).get(pn)
                    if dst_dim is None:
                        continue
                    for src_pn in sorted(list_all(fk)):
                        src_dim = compat.get(fk, {}).get(fm, {}).get("output", {}).get(src_pn)
                        if src_dim is None or src_dim == dst_dim:
                            continue
                        from_locked = _port_is_env_locked(fk, fm)
                        if from_locked:
                            incompatible_with.append(f"{fk}/{src_pn}")
                        else:
                            configurable_with.append({
                                "kind": fk,
                                "plugin": src_pn,
                                "actual_out": src_dim,
                                "expected_in": dst_dim,
                                "hint": f"将 {fk}.config 中的维度参数调整为适配 {dst_dim} 即可兼容",
                            })

            dims = {}
            for method_name, directions in methods.items():
                out_dim_val = directions.get("output", {}).get(pn)
                in_dim_val = directions.get("input", {}).get(pn)
                if out_dim_val is not None:
                    dims["out_dim"] = out_dim_val
                if in_dim_val is not None:
                    dims["in_dim"] = in_dim_val

            entries.append({
                "plugin": pn,
                "dimensions": dims,
                "incompatible_with": sorted(set(incompatible_with)),
                "configurable_with": configurable_with,
            })
        by_kind[kind] = entries

    return {
        "orchestrator": orch_name,
        "dataset_path": dataset_path,
        "env_topo": env_topo,
        "wiring_edges": wiring_edges,
        "by_kind": by_kind,
    }


def get_plugin_config_keys(kind, plugin_name):
    cls = find(kind, plugin_name)
    if cls is None:
        raise ValueError(f"plugin {kind}/{plugin_name} not found")
    return sorted(resolve_config_keys(cls))


def get_plugin_default_params(kind, plugin_name):
    cls = find(kind, plugin_name)
    if cls is None:
        raise ValueError(f"plugin {kind}/{plugin_name} not found")
    return _extract_plugin_params(cls, kind, plugin_name)


def list_modules_with_params():
    _DIMENSION_KEYS = {
        "observer": [{"key": "observe_out_dim", "label": "观测输出维度"}],
        "algorithm": [
            {"key": "act_in_dim", "label": "算法输入维度"},
            {"key": "act_out_dim", "label": "算法输出维度"},
        ],
        "actor": [
            {"key": "translate_in_dim", "label": "动作输入维度"},
            {"key": "translate_out_dim", "label": "动作输出维度"},
        ],
        "reward": [{"key": "compute_out_dim", "label": "奖励输出维度"}],
        "collector": [],
        "orchestrator": [],
    }

    modules = {}
    for kind in ["orchestrator", "actor", "algorithm", "collector", "observer", "reward", "environment"]:
        names = sorted(list_all(kind))
        kind_modules = []
        for name in names:
            params = {}
            cls = find(kind, name)
            if cls:
                try:
                    params = _extract_plugin_params(cls, kind, name)
                except Exception:
                    pass

            if kind == "environment" and not params:
                params = {"sim_max_time": 3600, "decision_interval": 5,
                         "yellow_duration": 3, "min_green": 5,
                         "roadnet_file": "", "flow_file": ""}

            if not params:
                params = _PLUGIN_PARAMS_FALLBACK.get(kind, {}).get(name, {})

            cfg_keys = sorted(scan_setup_cfg_keys(cls)) if cls else []
            dim_keys = _DIMENSION_KEYS.get(kind, [])

            kind_modules.append({
                "id": name,
                "name": name,
                "category": kind,
                "default_params": params,
                "config_keys": cfg_keys,
                "dim_keys": dim_keys,
                "required": kind in ["orchestrator", "algorithm"]
            })
        modules[kind] = kind_modules
    modules["default_training"] = {"warmup_steps": 500, "num_epochs": 20, "episodes_per_epoch": 1}
    modules["default_evaluation"] = {"eval_frequency": 5, "eval_steps": 500}
    return modules


def _extract_plugin_params(cls, kind, name):
    import ast
    import inspect

    params = {}
    try:
        src_file = inspect.getfile(cls)
        with open(src_file, encoding='utf-8') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
                for item in ast.walk(node):
                    if isinstance(item, ast.FunctionDef):
                        method_params = _scan_cfg_calls_ast(item)
                        for k, v in method_params.items():
                            params[k] = v
    except Exception:
        pass

    if not params:
        params = _PLUGIN_PARAMS_FALLBACK.get(kind, {}).get(name, {})
    return params


def _scan_cfg_calls_ast(func_node):
    import ast
    params = {}
    for item in ast.walk(func_node):
        if not isinstance(item, ast.Call):
            continue
        if not (isinstance(item.func, ast.Attribute)
                and isinstance(item.func.value, ast.Name)
                and item.func.value.id == 'cfg'
                and item.func.attr == 'get'):
            continue
        if not item.args or not isinstance(item.args[0], ast.Constant):
            continue
        key = item.args[0].value
        if not isinstance(key, str):
            continue
        default_val = None
        if len(item.args) > 1:
            default_val = _ast_to_value_internal(item.args[1])
        params[key] = default_val
    return params


def _ast_to_value_internal(node):
    import ast
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _ast_to_value_internal(node.operand)
        return -inner if isinstance(inner, (int, float)) else None
    if isinstance(node, ast.List):
        return [_ast_to_value_internal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_ast_to_value_internal(k): _ast_to_value_internal(v)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Name) and node.id == 'None':
        return None
    if isinstance(node, ast.NameConstant):
        return node.value
    return None


# ═══════════════════════════════════════════════════════════════
#  3. 配置约束阶段 — ConstraintSession（可迭代会话）
# ═══════════════════════════════════════════════════════════════

class ConstraintSession:
    """配置约束会话。约束组在初始化时一次性构建，随后可反复 set_value 迭代。

    约束组的结构只取决于"哪个编排器 + 哪些插件"（装配结果），
    不取决于用户填写了哪些参数值。因此无需每次修改值都重建。

    用法:
        session = create_constraint_session("single", selections, "data/LA/roadnet.net.xml")
        state = session.get_state()
        # 用户填写 observer.features = ["num"]
        delta = session.set_value("observer", "features", ["num"])
        # delta = {"notable_groups": [...], "warnings": [...]}
    """

    def __init__(self, orch_name, selections, dataset_path=None):
        kinds = get_orchestrator_kinds(orch_name)
        self._kinds = kinds
        self._orch_name = orch_name
        self._selections = dict(selections) if selections else {}
        self._dataset_path = dataset_path
        self._solver = ConfigSolver(kinds)
        self._ds_index = _ensure_dataset_index()
        self._deferred_probes = []
        self._output_kinds = set()
        self._env = None

        self._topo_keys = {"num_phase", "max_lanelinks", "max_green_phases", "num_tsc"}

        self._solver.add_group(
            {("environment", "roadnet_file")}
            | {("environment", k) for k in self._topo_keys}
        )

        # probe determiners + groups (deferred, registered after env)
        wiring_raw = get_orch_wiring(orch_name)
        wiring = wiring_raw.get("edges", [])
        ports = ports_from_wiring(wiring)
        for kind in kinds:
            if kind == "environment":
                continue
            plugin_name = self._selections.get(kind, "")
            cls = find(kind, plugin_name)
            if cls is None:
                continue
            kind_ports = ports.get(kind, {})
            for method_name, direction in kind_ports.items():
                if direction != "output":
                    continue
                overapprox = resolve_port_deps(cls, method_name)
                base_cfg = {}
                exact_deps = set()
                if overapprox:
                    if getattr(cls, '__port_deps__', None) is not None:
                        exact_deps = overapprox
                    else:
                        exact_deps = filter_deps_by_sensitivity(cls, method_name, overapprox, base_cfg)
                dim_key = f"{method_name}_out_dim"

                _, _, needs_env = _method_uses_env(cls, method_name)

                for dep in exact_deps:
                    self._solver.add_group({(kind, dim_key), (kind, dep)})
                if needs_env:
                    for topo_key in self._topo_keys:
                        self._solver.add_group({(kind, dim_key), (kind, topo_key)})
                self._deferred_probes.append({
                    "kind": kind, "dim_key": dim_key,
                    "deps": sorted(exact_deps), "cls": plugin_name,
                    "method_name": method_name, "needs_env": needs_env,
                })

        for kind in kinds:
            kind_ports = ports.get(kind, {})
            for method_name, direction in kind_ports.items():
                if direction == "output":
                    self._output_kinds.add((kind, method_name))

        for edge in wiring:
            fk, fm = edge["from"]["kind"], edge["from"]["method"]
            tk, tm = edge["to"]["kind"], edge["to"]["method"]
            if (fk, fm) in self._output_kinds:
                self._solver.add_equal(fk, f"{fm}_out_dim", tk, f"{tm}_in_dim")

        # ── 组件内部参数依赖（AST 自动 + __param_deps__ 手动声明） ──
        for kind in kinds:
            if kind == "environment" or kind == "orchestrator":
                continue
            plugin_name = self._selections.get(kind, "")
            cls = find(kind, plugin_name)
            if cls is None:
                continue
            internal_deps = {}
            scanned = scan_setup_param_deps(cls)
            if scanned:
                internal_deps.update(scanned)
            manual = getattr(cls, '__param_deps__', None)
            if isinstance(manual, dict):
                for tk, ds in manual.items():
                    internal_deps.setdefault(tk, set()).update(ds)

            for target_key, dep_keys in internal_deps.items():
                members = {(kind, target_key)} | {(kind, d) for d in dep_keys}
                for d in dep_keys:
                    self._solver.add_group({(kind, target_key), (kind, d)})
                self._solver.add_determiner(
                    kind, target_key,
                    _self_probe_determiner(kind, cls, target_key, dep_keys)
                )

        self._groups = self._solver._merge_groups()
        self._display_groups = self._build_display_groups()
        self._recommendations = []
        self._warnings = []

        if self._dataset_path:
            self.select_dataset(self._dataset_path)

    def _build_display_groups(self):
        """从语义源头构建用于前端展示的约束组。

        三源并行:
          1. topology — 拓扑键集合
          2. port_equation — 每个 output 方法的配置参数依赖
          3. input_match — wiring 等式 (fk.fm_out_dim == tk.tm_in_dim)
          4. internal_deps — 组件内部参数依赖 (scan_setup_param_deps / __param_deps__)

        返回 list of dicts:
        [
          {"label":"拓扑与维度约束", "source":"topology", "members":[...]},
          {"label":"observer.observe 输出依赖", "source":"port_equation", "members":[...]},
          {"label":"actor.translate 输出依赖", "source":"port_equation", "members":[]},
          {"label":"observer.observe→algorithm.act", "source":"input_match",
           "from_kind":"observer", "from_method":"observe", "to_kind":"algorithm", "to_method":"act"},
          {"label":"algorithm 内部参数依赖", "source":"internal_deps", "members":[...]},
        ]
        """
        groups = []

        # 1. topology — dataset 驱动
        topo_members = [("environment", "roadnet_file")]
        topo_members += [("environment", k) for k in sorted(self._topo_keys)]
        groups.append({
            "label": "数据集与拓扑参数",
            "source": "topology",
            "members": topo_members,
        })

        # 2. port_equation — output 端口依赖
        for probe in self._deferred_probes:
            kind = probe["kind"]
            deps = probe["deps"]
            dim_key = probe["dim_key"]
            method_name = dim_key.replace("_out_dim", "")
            groups.append({
                "label": f"{kind}.{method_name} 输出依赖",
                "source": "port_equation",
                "kind": kind,
                "method": method_name,
                "dim_key": dim_key,
                "members": [(kind, d) for d in sorted(deps)],
            })

        # 3. input_match — 每个 wiring 等式作为一个约束组
        for (fk, fm_out, tk, tm_in) in self._solver._equals:
            if not fm_out.endswith("_out_dim"):
                continue
            fm = fm_out.replace("_out_dim", "")
            groups.append({
                "label": f"维度匹配: {fk}.{fm} → {tk}.{tm_in.replace('_in_dim','')}",
                "source": "input_match",
                "from_kind": fk, "from_key": fm_out,
                "to_kind": tk, "to_key": tm_in,
            })

        # 4. internal_deps
        for kind in self._kinds:
            if kind in ("environment", "orchestrator"):
                continue
            plugin_name = self._selections.get(kind, "")
            cls = find(kind, plugin_name)
            if cls is None:
                continue
            internal_deps = {}
            scanned = scan_setup_param_deps(cls)
            if scanned:
                internal_deps.update(scanned)
            manual = getattr(cls, '__param_deps__', None)
            if isinstance(manual, dict):
                for tk, ds in manual.items():
                    internal_deps.setdefault(tk, set()).update(ds)
            for target_key, dep_keys in internal_deps.items():
                all_keys = sorted(set([target_key] + list(dep_keys)))
                groups.append({
                    "label": f"{kind} 内部参数依赖",
                    "source": "internal_deps",
                    "kind": kind,
                    "members": [(kind, k) for k in all_keys],
                })

        return groups

    def select_dataset(self, dataset_path):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

        # 约束会话统一走索引回退，避免占用 TraCI 连接影响实验启动
        self._inject_topo_from_index(dataset_path)

    def _inject_topo_from_index(self, dataset_path):
        fallback_topo = self._ds_index.get(dataset_path, {})
        if not fallback_topo and (dataset_path.startswith("data/") or dataset_path.startswith("data\\")):
            fallback_topo = self._ds_index.get(dataset_path[5:], {})
        if not fallback_topo:
            return
        actual_topo = {}
        for k in ("num_phase", "max_lanelinks", "max_green_phases", "num_tsc"):
            if k in fallback_topo:
                actual_topo[k] = fallback_topo[k]
        if "max_green_phases" not in actual_topo and "num_phase" in actual_topo:
            actual_topo["max_green_phases"] = actual_topo["num_phase"]
        self._dataset_path = dataset_path
        self._solver.set_value("environment", "roadnet_file", dataset_path)

        for kind in self._kinds:
            if kind == "environment":
                continue
            plugin_name = self._selections.get(kind, "")
            cls = find(kind, plugin_name)
            if cls is None:
                continue
            setup_cfg = getattr(cls, '__setup_cfg__', None)
            if setup_cfg:
                for key, default_val in setup_cfg.items():
                    if self._solver._values[kind].get(key) is None:
                        self._solver.set_value(kind, key, default_val)

        for topo_key, actual_val in actual_topo.items():
            for kind in self._kinds:
                if kind == "environment":
                    continue
                self._solver.set_value(kind, topo_key, actual_val)
                if topo_key == "num_phase":
                    self._solver.set_value(kind, "max_phase", actual_val)
            self._solver.set_value("environment", topo_key, actual_val)

        for probe in self._deferred_probes:
            kind = probe["kind"]
            dim_key = probe["dim_key"]
            cls = find(kind, self._selections.get(kind, ""))
            if cls is None:
                continue
            if not probe.get("needs_env"):
                resolver = _probe_determiner(
                    kind, cls, probe["method_name"],
                    set(probe["deps"]), env=None
                )
                self._solver.add_determiner(kind, dim_key, resolver)
            else:
                val = _build_and_measure_port(cls, probe["method_name"], {}, env=None)
                if val is not None:
                    self._solver.set_value(kind, dim_key, val)

        self._solver.reset_phase()
        self._solver.solve()

        for probe in self._deferred_probes:
            kind = probe["kind"]
            dim_key = probe["dim_key"]
            if self._solver._values.get(kind, {}).get(dim_key) is None:
                self._solver.set_value(kind, dim_key, 1)

        self._solver.reset_phase()
        self._solver.solve()

    def _detect_dimension_conflicts(self, values):
        conflicts = []
        wiring_raw = get_orch_wiring(self._orch_name)
        wiring_edges = wiring_raw.get("edges", [])

        for edge in wiring_edges:
            fk = edge.get("from", {}).get("kind")
            fm = edge.get("from", {}).get("method")
            tk = edge.get("to", {}).get("kind")
            tm = edge.get("to", {}).get("method")
            if not all([fk, fm, tk, tm]):
                continue

            fkey = f"{fm}_out_dim"
            tkey = f"{tm}_in_dim"
            fval = values.get(fk, {}).get(fkey)
            tval = values.get(tk, {}).get(tkey)
            if fval is None or tval is None or fval == tval:
                continue

            from_cls = find(fk, self._selections.get(fk, ""))
            to_cls = find(tk, self._selections.get(tk, ""))

            _, _, from_env = _method_uses_env(from_cls, fm) if from_cls else (False, False, False)
            _, _, to_env = _method_uses_env(to_cls, tm) if to_cls else (False, False, False)

            from_deps = resolve_port_deps(from_cls, fm) if from_cls else set()
            to_deps = resolve_port_deps(to_cls, tm) if to_cls else set()

            from_locked = from_env and not from_deps
            to_locked = to_env and not to_deps

            if from_locked and to_locked:
                resolution = "hard"
                hint = (
                    f"{fk}.{fm} 和 {tk}.{tm} 的维度计算都由环境拓扑决定，"
                    f"但两者公式不兼容（{fval} ≠ {tval}）。"
                    f"请更换 {fk} 或 {tk} 的插件实现。"
                )
            elif not from_locked and to_locked:
                resolution = "from_tunable"
                hint = (
                    f"{tk}.{tm} 期望 {tval}，但 {fk}.{fm} 输出 {fval}。"
                    f"请调整 {fk}.config 中的参数使输出维度匹配 {tval}。"
                )
            elif from_locked and not to_locked:
                resolution = "to_tunable"
                hint = (
                    f"{fk}.{fm} 输出 {fval}，但 {tk}.{tm} 期望 {tval}。"
                    f"请调整 {tk}.config 中的参数使输入维度匹配 {fval}。"
                )
            else:
                resolution = "both_tunable"
                hint = (
                    f"{fk}.{fm} 输出 {fval}，但 {tk}.{tm} 期望 {tval}。"
                    f"请调整 {fk}.config 或 {tk}.config 中的参数使其一致。"
                )

            conflicts.append({
                "from_kind": fk, "from_method": fm, "from_value": fval,
                "to_kind": tk, "to_method": tm, "to_value": tval,
                "resolution": resolution, "hint": hint,
            })

        return conflicts

    def set_value(self, kind, key, value):
        """设置单个配置值，返回变化摘要。

        如果设置的是 environment.roadnet_file，会自动启动 env 并注入拓扑。
        """
        if kind == "environment" and key == "roadnet_file":
            self.select_dataset(value)

        self._solver.set_value(kind, key, value)
        values, recs, warnings = self._solver.solve()
        self._recommendations = recs
        self._warnings = warnings

        conflicts = self._detect_dimension_conflicts(values)

        delta = self._build_state(values)
        return {
            "notable_groups": delta["groups_with_one_unknown"],
            "resolved_groups": [g for g in delta["merged_groups"]
                               if all(m["value"] is not None for m in g["members"])],
            "warnings": warnings,
            "conflicts": conflicts,
            "unknown_count": delta["unknown_count"],
        }

    def recommend(self, kind, key):
        """为约束组中只剩一个的未知键生成推荐值。

        Args:
            kind: 组件种类
            key: 键名。特殊键 "@dataset" 会触发数据集搜索
            key 为 "roadnet_file" 且未选数据集时，触发 env-probed 兼容性筛选

        Returns:
            {"kind":..., "key":..., "recommendations": [{"candidates":{...},"source":"..."}]}
        """
        if key in ("@dataset", "roadnet_file") and not self._dataset_path:
            return self._recommend_dataset()

        seen = set()
        unique = []
        values, recs, warnings = self._solver.solve()
        for r in recs:
            if r["kind"] == kind and r["key"] == key:
                key_str = json.dumps(r["candidates"], sort_keys=True)
                if key_str not in seen:
                    seen.add(key_str)
                    unique.append({"candidates": r["candidates"], "source": r["source"]})
        return {"kind": kind, "key": key, "recommendations": unique}

    def _recommend_dataset(self):
        env_cls = find("environment", self._selections.get("environment", "sumo"))
        if env_cls is None or not self._ds_index:
            return {"kind": "environment", "key": "roadnet_file", "recommendations": []}

        candidates = []
        import time
        import subprocess

        user_values, _, _ = self._solver.solve()
        user_filled = {}
        for kd, kv in user_values.items():
            if not isinstance(kv, dict):
                continue
            for k, v in kv.items():
                if v is not None:
                    user_filled.setdefault(kd, {})[k] = v

        for ds_path, ds_topo in list(self._ds_index.items())[:20]:
            waited = 0
            while waited < 10:
                try:
                    result = subprocess.run(
                        ["tasklist", "/fi", "imagename eq sumo.exe"],
                        capture_output=True, text=True, timeout=5
                    )
                    if "sumo.exe" not in result.stdout:
                        break
                except Exception:
                    pass
                time.sleep(1)
                waited += 1

            env = env_cls()
            probe_topo = {}
            probe_dims = {}
            try:
                env.launch({"roadnet_file": ds_path, "gui": False})
                ids = env.ids()
                if ids:
                    probe_topo["num_phase"] = max(env.phase_count(j) for j in ids)
                    probe_topo["max_lanelinks"] = max(
                        len(env.traffic_light_controlled_links(j)) for j in ids
                    )
                    probe_topo["max_green_phases"] = max(
                        len(env.green_phase_indices(j)) for j in ids
                    )
                    probe_topo["num_tsc"] = len(ids)

                for probe in self._deferred_probes:
                    cls = find(probe["kind"], self._selections.get(probe["kind"], ""))
                    if cls is None:
                        continue
                    test_cfg = dict(probe_topo)
                    kind_user = {
                        k: v for k, v in user_filled.get(probe["kind"], {}).items()
                        if k not in probe_topo
                    }
                    test_cfg.update(kind_user)
                    dim = _build_and_measure_port(
                        cls, probe["method_name"], test_cfg, env=env
                    )
                    if dim is not None:
                        probe_dims[f"{probe['kind']}.{probe['dim_key']}"] = dim
            except Exception:
                pass
            finally:
                try:
                    env.close()
                except Exception:
                    pass

            # check wiring compatibility
            incompatible = False
            wiring_raw = get_orch_wiring(self._orch_name)
            wiring_edges = wiring_raw.get("edges", [])
            for edge in wiring_edges:
                fk = edge.get("from", {}).get("kind")
                fm = edge.get("from", {}).get("method")
                tk = edge.get("to", {}).get("kind")
                tm = edge.get("to", {}).get("method")
                if not all([fk, fm, tk, tm]):
                    continue
                fdim = probe_dims.get(f"{fk}.{fm}_out_dim")
                tdim = probe_dims.get(f"{tk}.{tm}_in_dim")
                if fdim is not None and tdim is not None and fdim != tdim:
                    incompatible = True
                    break

            candidates.append({
                "path": ds_path,
                "topo": probe_topo,
                "compatible": not incompatible,
                "dimensions": probe_dims,
            })

        candidates.sort(key=lambda c: (not c["compatible"], c["path"]))

        return {
            "kind": "environment",
            "key": "roadnet_file",
            "recommendations": [
                {"candidates": {"roadnet_file": c["path"], **c["topo"]},
                 "source": "env_probe",
                 "compatible": c["compatible"],
                 "dimensions": c["dimensions"]}
                for c in candidates
            ],
        }

    def get_state(self):
        """返回当前完整状态。"""
        values, _, warnings = self._solver.solve()
        return self._build_state(values)

    def _build_state(self, values):
        equal_pairs = []
        for (ka, key_a, kb, key_b) in self._solver._equals:
            equal_pairs.append({
                "kind_a": ka, "key_a": key_a,
                "kind_b": kb, "key_b": key_b,
            })

        merged_groups = []
        groups_with_one_unknown = []
        unknown_total = 0

        for dg in self._display_groups:
            members = []
            unknown_set = []

            if dg["source"] == "topology":
                topo_members_for_state = []
                topo_unknown = False
                for kind, key in dg["members"]:
                    v = values.get(kind, {}).get(key)
                    topo_members_for_state.append((kind, key, v))
                    if v is None:
                        topo_unknown = True

                topo_entry = {
                    "kind": "environment", "key": "@dataset",
                    "value": self._dataset_path if not topo_unknown else None,
                    "topo_keys": [{"kind": k, "key": kk, "value": v} for k, kk, v in topo_members_for_state],
                }
                members.append(topo_entry)
                if topo_unknown:
                    unknown_set.append({"kind": "environment", "key": "@dataset"})
                    unknown_total += 1

            elif dg["source"] == "input_match":
                fv = values.get(dg["from_kind"], {}).get(dg["from_key"])
                tv = values.get(dg["to_kind"], {}).get(dg["to_key"])
                match_status = "unknown"
                if fv is not None and tv is not None:
                    match_status = "matched" if fv == tv else "conflict"
                members = [
                    {"kind": dg["from_kind"], "key": dg["from_key"], "value": fv},
                    {"kind": dg["to_kind"], "key": dg["to_key"], "value": tv},
                ]
                merged_groups.append({
                    "members": members,
                    "label": dg.get("label", ""),
                    "source": dg.get("source", ""),
                    "from_value": fv,
                    "to_value": tv,
                    "match_status": match_status,
                })
                if fv is None:
                    unknown_total += 1
                if tv is None:
                    unknown_total += 1
                continue  # 跳过下面的通用处理

            else:
                for kind, key in dg["members"]:
                    v = values.get(kind, {}).get(key)
                    members.append({"kind": kind, "key": key, "value": v})
                    if v is None:
                        unknown_set.append({"kind": kind, "key": key})
                        unknown_total += 1

            merged_groups.append({
                "members": members,
                "label": dg.get("label", ""),
                "source": dg.get("source", ""),
            })
            if len(unknown_set) == 1:
                groups_with_one_unknown.append({
                    "members": members,
                    "unknown": unknown_set[0],
                })

        rel_warnings = []
        for w in self._warnings:
            if "conflict:" in str(w):
                rel_warnings.append(w)

        return {
            "equal_pairs": equal_pairs,
            "probe_groups": self._deferred_probes,
            "merged_groups": merged_groups,
            "values": _serialize_values(self._kinds, values),
            "unknown_count": unknown_total,
            "groups_with_one_unknown": groups_with_one_unknown,
            "warnings": rel_warnings,
        }


def create_constraint_session(orch_name, selections, dataset_path=None):
    """创建可迭代的约束会话。装配完成后调用一次，然后反复 set_value。

    Args:
        orch_name: 编排器名
        selections: {kind: plugin_name}  完整的装配选择
        dataset_path: 数据集路径（可选）

    Returns:
        ConstraintSession
    """
    return ConstraintSession(orch_name, selections, dataset_path)


# 向后兼容的快捷函数
def analyze_constraints(orch_name, selections, user_config=None, dataset_path=None):
    """一次性分析（向后兼容）。新代码建议用 ConstraintSession。"""
    s = ConstraintSession(orch_name, selections, dataset_path)
    if user_config:
        for kind, params in user_config.items():
            if isinstance(params, dict):
                for k, v in params.items():
                    s.set_value(kind, k, v)
    return s.get_state()


# ═══════════════════════════════════════════════════════════════
#  4. 数据集与拓扑
# ═══════════════════════════════════════════════════════════════

def list_datasets(data_dir="data"):
    idx = _ensure_dataset_index(data_dir)
    return [{"path": path, **topo} for path, topo in idx.items()]


def get_dataset_topo(dataset_path):
    topo = find_topo(dataset_path)
    return topo or {}


def match_datasets_by_topo(known_topo, data_dir="data"):
    idx = _ensure_dataset_index(data_dir)
    return match_datasets(known_topo, idx)


# ═══════════════════════════════════════════════════════════════
#  5. 配置文件操作
# ═══════════════════════════════════════════════════════════════

def read_config_structure(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_PLUGIN_PARAMS_FALLBACK = {
    "observer": {
        "frap": {
            "features": ["num"],
            "num_phase": 4,
            "max_lanelinks": 1,
            "norm": {},
        },
        "standard": {
            "features": ["num", "waiting"],
            "history": 1,
            "normalize": True,
        },
    },
    "actor": {
        "phase": {
            "max_phase": 4,
        },
    },
    "reward": {
        "composite": {
            "metrics": {"waiting": -1.0},
            "reward_norm": 1.0,
        },
        "queue": {
            "target_flow": 1000.0,
        },
    },
    "collector": {
        "replay": {
            "capacity": 100000,
            "per_agent": False,
            "batch_size": 64,
        },
        "ma2c": {
            "capacity": 100000,
            "batch_size": 64,
        },
    },
    "algorithm": {
        "frap": {
            "num_phase": 4,
            "gamma": 0.95,
            "lr": 1e-4,
            "tau": 0.005,
            "device": "cpu",
            "max_lanelinks": 4,
            "frap_tls_id": None,
            "phase_2_passable_lanelink": None,
        },
        "dqn": {
            "num_phase": 4,
            "gamma": 0.95,
            "lr": 1e-4,
            "tau": 0.005,
            "device": "cpu",
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
        },
        "colight": {
            "num_phase": 4,
            "gamma": 0.95,
            "lr": 1e-4,
            "tau": 0.005,
            "device": "cpu",
            "max_lanelinks": 4,
        },
        "ma2c": {
            "num_phase": 4,
            "gamma": 0.95,
            "lr": 1e-4,
            "device": "cpu",
            "max_lanelinks": 4,
            "num_tsc": 1,
        },
        "fixed": {
            "phase_duration": 30,
        },
        "maxpressure": {
            "min_green": 5,
            "max_green": 60,
        },
    },
    "tracker": {
        "console": {
            "prefix": "",
            "flush": True,
        },
        "tensorboard": {
            "log_dir": "tb_logs/",
        },
    },
}


def scaffold_config(orch_name, selections, config_params=None, output_path=None):
    from modutsc.scheduling.registry import find
    from modutsc.scheduling.scaffold import _extract_config_params, _enrich_config_defaults, _scan_cfg_calls
    import ast
    import inspect
    
    cfg = {"experiment": {"name": "scaffold_exp", "seed": 42}}
    cfg["components"] = {k: [v] for k, v in sorted(selections.items())}
    cfg["orchestrator"] = {"plugin": orch_name, "config": {}}

    ds_index = _ensure_dataset_index()
    topo_params = {}
    if ds_index:
        first_ds = next(iter(ds_index.values()), {})
        topo_params = {k: v for k, v in first_ds.items() if k in ("num_phase", "max_lanelinks", "num_tsc")}

    for kind, plugin_name in sorted(selections.items()):
        params = {}
        
        cls = find(kind, plugin_name)
        if cls:
            try:
                params = _extract_config_params(cls)
                _enrich_config_defaults(cls, params)
                
                src_file = inspect.getfile(cls)
                with open(src_file, encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
                        for item in ast.walk(node):
                            if isinstance(item, ast.FunctionDef):
                                method_params = _scan_cfg_calls(item)
                                for k, v in method_params.items():
                                    params[k] = v
            except Exception:
                pass
        
        fallback = _PLUGIN_PARAMS_FALLBACK.get(kind, {}).get(plugin_name, {})
        for k, v in fallback.items():
            params[k] = v

        if kind == "environment" and topo_params:
            sample_ds = next(iter(ds_index.keys()), "") if ds_index else ""
            if sample_ds:
                params["roadnet_file"] = f"data/{sample_ds}"
                params["flow_file"] = params["roadnet_file"].replace("roadnet.net.xml", "flow_0.rou.xml")
            params["gui"] = False
            params["sim_max_time"] = 3600
            params["decision_interval"] = 5
            params["yellow_duration"] = 3
            params["min_green"] = 5
            for k in ("num_phase", "max_lanelinks"):
                if k in topo_params:
                    params[k] = topo_params[k]

        elif kind in ("observer", "algorithm") and topo_params:
            for k in ("num_phase", "max_lanelinks"):
                if k in topo_params:
                    params[k] = topo_params[k]

        if config_params and kind in config_params:
            params.update(config_params[kind])

        params = {k: v for k, v in params.items() if v is not None or k in ("frap_tls_id", "phase_2_passable_lanelink")}

        section = {"plugin": plugin_name, "config": params}
        if kind == "algorithm":
            cfg[kind] = [section]
        elif kind not in cfg:
            cfg[kind] = section

    cfg["training"] = {
        "warmup_steps": 200,
        "num_epochs": 5,
        "episodes_per_epoch": 2,
    }
    cfg["evaluation"] = {
        "eval_frequency": 5,
        "eval_steps": 200,
        "checkpoint_dir": "checkpoints/",
    }
    cfg["tracker"] = {
        "plugin": "console",
        "config": {"sumo_episode_kpis": True},
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return cfg


# ═══════════════════════════════════════════════════════════════
#  6. 实验运行与实时日志
# ═══════════════════════════════════════════════════════════════

def run_experiment(config_path, stop_event=None, overrides=None, shared_state=None):
    """批量运行实验，返回完整的训练/评估结果。

    内部走 Launcher 完整流水线（resolve → build → run），
    适合"导出配置后一键运行"场景。

    Args:
        config_path: 配置文件路径，如 "configs/frap.yaml"
        stop_event: threading.Event 实例，设置后可优雅中断训练
        overrides: 可选，覆盖 YAML 配置的 dict
        shared_state: 可选，跨线程共享状态的 dict

    Returns:
        {
            "config": {...},
            "resolved_config": {...},
            "training": [{...}],
            "eval_results": [{...}],
            "elapsed_sec": 45.2,
            "stopped": false,
            "error": null           — 非 null 表示运行出错
        }
    """
    import time
    from modutsc.scheduling.launcher import Launcher

    t0 = time.time()
    try:
        launcher = Launcher(config_path)
        orch = launcher.build(overrides=overrides)
        if stop_event is not None:
            orch._stop_event = stop_event
        if shared_state is not None:
            orch._shared_state = shared_state
        try:
            results = orch.run(launcher.config)
        finally:
            try:
                orch.teardown()
            except Exception:
                pass
        elapsed = round(time.time() - t0, 2)
        return {
            "config": launcher.config,
            "resolved_config": launcher.resolved_config,
            "training": results.get("training", []),
            "eval_results": results.get("evaluation", []),
            "elapsed_sec": elapsed,
            "stopped": results.get("stopped", False),
        }
    except Exception as e:
        import traceback
        print(f"[run_experiment] FATAL: {e}")
        traceback.print_exc()
        elapsed = round(time.time() - t0, 2)
        return {
            "config": None,
            "resolved_config": None,
            "training": [],
            "eval_results": [],
            "elapsed_sec": elapsed,
            "error": str(e),
        }


def run_experiment_stream(config_path, on_episode=None, stop_event=None, overrides=None, shared_state=None):
    """流式运行实验，每个 episode 结束时实时回调 on_episode(index, metrics)。

    on_episode 由 Orchestrator.run() 在单次 episode 完成的瞬间直接调用，
    而非运行结束后的迭代 —— 全程实时推送。

    Args:
        config_path: 配置文件路径
        on_episode: fn(episode_index, metrics_dict) 每个 episode 结束时调用
        stop_event: threading.Event 实例，设置后可优雅中断训练

    Returns:
        同 run_experiment()
    """
    from modutsc.scheduling.launcher import Launcher

    try:
        launcher = Launcher(config_path)
        orch = launcher.build(overrides=overrides)
        if stop_event is not None:
            orch._stop_event = stop_event
        if shared_state is not None:
            orch._shared_state = shared_state
        try:
            results = orch.run(launcher.config,
                              on_episode=on_episode)
        finally:
            try:
                orch.teardown()
            except Exception:
                pass
        return {
            "config": launcher.config,
            "resolved_config": launcher.resolved_config,
            "training": results.get("training", []),
            "eval_results": results.get("evaluation", []),
            "stopped": results.get("stopped", False),
        }
    except Exception as e:
        return {
            "config": None,
            "resolved_config": None,
            "training": [],
            "eval_results": [],
            "error": str(e),
        }


def evaluate_checkpoint(config_path, checkpoint_path, eval_steps=None):
    """加载检查点并运行评估。

    Args:
        config_path: 配置文件路径
        checkpoint_path: 检查点路径，如 "checkpoints/frap/ckpt_epoch_10.pkl"
        eval_steps: 评估步数，不传则从配置中读取

    Returns:
        {
            "metrics": {...},          — 评估指标
            "checkpoint": "checkpoints/...",
            "error": null              — 非 null 表示出错
        }
    """
    from modutsc.scheduling.launcher import Launcher

    try:
        launcher = Launcher(config_path)
        orch = launcher.build()

        if eval_steps is None:
            eval_steps = (
                launcher.config.get("evaluation", {})
                .get("eval_steps", 3600)
            )

        try:
            orch.load(checkpoint_path)
            metrics = orch.evaluate(eval_steps)
        finally:
            try:
                orch.teardown()
            except Exception:
                pass

        return {
            "metrics": metrics,
            "checkpoint": checkpoint_path,
        }
    except Exception as e:
        return {
            "metrics": {},
            "checkpoint": checkpoint_path,
            "error": str(e),
        }


def list_checkpoints(checkpoint_dir="checkpoints"):
    """列出检查点目录下的所有检查点文件。

    Returns:
        list[dict]: [{"path":"checkpoints/frap/ckpt_epoch_10.pkl","epoch":10,"size_kb":45}, ...]
    """
    import os as _os
    entries = []
    for root, _, files in _os.walk(checkpoint_dir):
        for f in sorted(files):
            if f.endswith(".pkl"):
                full = _os.path.join(root, f).replace("\\", "/")
                epoch = None
                if "epoch_" in f:
                    try:
                        epoch = int(f.split("epoch_")[1].split(".")[0])
                    except ValueError:
                        pass
                try:
                    size = _os.path.getsize(full) // 1024
                except OSError:
                    size = 0
                entries.append({
                    "path": full, "filename": f,
                    "epoch": epoch, "size_kb": size,
                })
    return entries


# ═══════════════════════════════════════════════════════════════
#  7. 代码脚手架
# ═══════════════════════════════════════════════════════════════

def scaffold_orchestrator(name, output_dir, kinds_order=None):
    """生成编排器模板文件。

    编排器模板包含 setup/warmup/episode/evaluate/save/load 方法骨架。
    用户需要实现数据流编排逻辑。

    编写规则：
      - 所有组件通过 self._env / self._observer 等属性访问
      - 不 import 具体组件类，只 import ABC
      - 配置参数只从 cfg dict 读取

    Args:
        name: 编排器名，如 "my_orch"
        output_dir: 输出目录，如 "modutsc/orchestration"
        kinds_order: 需要的组件种类列表，默认用 single 的
    """
    import os as _os
    if kinds_order is None:
        kinds_order = get_orchestrator_kinds("single")
    kinds_order = [k for k in kinds_order if k not in ("environment", "orchestrator")]

    cls_name = "".join(w.capitalize() for w in name.split("_"))

    lines = [
        '"""{} — 自定义编排器。"""'.format(cls_name),
        "",
        "import random",
        "from typing import Dict, Any, List, Optional",
        "from modutsc.orchestration import Orchestrator",
        "from modutsc.env import Env",
    ]
    kind_imports = {
        "observer": "from modutsc.plugins.observers import Observer",
        "actor": "from modutsc.plugins.actors import Actor",
        "reward": "from modutsc.plugins.rewards import Reward",
        "collector": "from modutsc.plugins.collectors import Collector",
        "algorithm": "from modutsc.plugins.algorithms import Algorithm",
        "tracker": "from modutsc.plugins.trackers import Tracker",
    }
    for k in kinds_order:
        if k in kind_imports:
            lines.append(kind_imports[k])
    lines += [
        "from modutsc.scheduling.registry import register",
        "",
        "",
        '@register("orchestrator", "{}")'.format(name),
        "class {}(Orchestrator):".format(cls_name),
        "",
        "    # ── 可选：声明与各组件的兼容关系 ──",
        "    # 不声明则自动通过 AST 分析方法调用，推导兼容的插件。",
        "    # 声明后可以精确控制哪些插件与编排器兼容。",
        "    __compatible_plugins__ = {",
    ]
    for k in kinds_order:
        examples = {
            "observer": '["frap", "standard", "flat_lane"]',
            "actor": '["phase"]',
            "algorithm": '["frap", "dqn"]',
            "collector": '["replay"]',
            "reward": '["composite"]',
            "tracker": '["console"]',
        }.get(k, "[]")
        lines.append(f'        "{k}": {examples},')
    lines[-1] = lines[-1] + "  # TODO: 按需增减"
    lines += [
        "    }",
        "",
        "    # ── 编写规则 ──",
        '    # 1. 只能调用组件的方法（如 self._observer.observe(env)），',
        "    #    不能直接访问组件内部属性（如 ag._n_lstm、ag._net）",
        "    # 2. 所有组件通过 self._env / self._observer 等属性访问",
        "    # 3. 配置参数只从 cfg dict 读取",
        "    # 4. 不 import 具体组件类，只 import ABC",
        "",
        "    def setup(self, env: Env, observer: Observer,",
        "              actor: Actor, reward: Reward,",
        "              collector: Collector, algorithms: list,",
        "              cfg: dict, tracker=None, **kwargs) -> None:",
        "        self._env = env",
        "        self._observer = observer",
        "        self._actor = actor",
        "        self._reward = reward",
        "        self._collector = collector",
        "        self._algos = algorithms",
        "        self._tracker = tracker",
        "        # TODO: 读取编排器级配置",
        "        # self._max_steps = cfg.get('max_steps', 3600)",
        "",
        "    def warmup(self, steps: int) -> dict:",
        '        """预热阶段。"""',
        "        return {'warmup_steps': 0}",
        "",
        "    def episode(self) -> dict:",
        '        """单次 episode 主循环。"""',
        "        self._env.reset()",
        "        # TODO: 实现 episode 循环",
        "        # self._env.step(actions)",
        '        return {"steps": 0, "sim_time": 0}',
        "",
        "    def evaluate(self, steps: int) -> dict:",
        '        """评估模式。"""',
        "        # TODO: 实现评估逻辑",
        '        return {"eval": "ok"}',
        "",
        "    def save(self, path: str) -> None:",
        '        """保存检查点。"""',
        "        pass",
        "",
        "    def load(self, path: str) -> None:",
        '        """加载检查点。"""',
        "        pass",
        "",
    ]

    code = "\n".join(lines)
    path = _os.path.join(output_dir, name + ".py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return code


def scaffold_kind_base(kind_name, output_dir, methods=None):
    """生成新组件种类的 ABC 骨架。

    创建一个新的 base 类（如 Predictor），继承 ABC，
    定义一组 @abstractmethod 抽象方法。这会自动注册为新 kind。

    编写规则：
      - setup(cfg) 必须接收 dict，所有配置通过 cfg.get("key") 读取
      - 业务方法（如 predict）可通过 env 参数访问仿真器

    Args:
        kind_name: 组件种类名，如 "predictor"
        output_dir: 输出目录
        methods: {"method_name": "params_info", ...} 或 None 使用默认
    """
    import os as _os
    if methods is None:
        methods = {
            "setup": "cfg: dict",
            "predict": "env",
        }

    abcs = []
    impls = []
    cls_base = "".join(w.capitalize() for w in kind_name.split("_"))

    for mname, params in sorted(methods.items()):
        params_str = ", ".join(["self"] + (params.split(", ") if params.strip() else []))
        abcs.append(f"    @abstractmethod")
        abcs.append(f"    def {mname}({params_str}): ...")
        abcs.append("")

    lines = [
        '"""{} ABC — 自定义组件种类的抽象基类。"""'.format(cls_base),
        "",
        "from abc import ABC, abstractmethod",
        "from typing import Dict, List, Any, Optional",
        "",
        "",
        "class {}(ABC):".format(cls_base),
    ] + abcs

    path = _os.path.join(output_dir, "__init__.py")
    _os.makedirs(output_dir, exist_ok=True)
    code = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return code


def scaffold_plugin(kind, name, output_dir, abc_module=None, abc_class=None):
    """生成组件实现骨架文件。

    生成的模板包含：
      - @register 装饰器
      - setup(cfg) 方法（从 cfg.get("key") 读取配置）
      - 所有抽象方法的空实现

    编写约束（请在生成的文件中遵守）：
      1. setup 中的配置键通过 cfg.get("key") 读取，每个键最终都会暴露到 YAML
      2. 不允许 import 其他组件的类，只允许 import 本 kind 的 ABC
      3. 使用 env 接口（如 env.lane_vehicle_count()）时，必须用 env 已有的方法名
      4. 拓扑参数（如 num_phase）必须用仿真器分析产出的键名，不要自己起别名
      5. 如果某个参数值等于某个拓扑参数，直接在 cfg 中读同一个名字

    Args:
        kind: 组件种类名，如 "observer"
        name: 插件名，如 "my_custom"
        output_dir: 输出目录
        abc_module: ABC 所在模块（自定义 kind 时需要）
        abc_class: ABC 类名（自定义 kind 时需要）
    """
    import os as _os
    from modutsc.scheduling.scaffold import _load_abc, _scan_abc

    abc_cls = _load_abc(kind, abc_module, abc_class)

    cls_name = "".join(w.capitalize() for w in name.split("_"))
    cls_name += abc_cls.__name__

    _abc_mod = abc_module or _KNOWN_ABC_MODULE.get(kind, f"modutsc.plugins.{kind}s")
    _abc_name = abc_class or abc_cls.__name__

    methods = _scan_abc(abc_cls)

    lines = [
        '"""{} — {} 组件实现。"""'.format(cls_name, kind),
        "",
        "from typing import List, Dict, Any, Optional",
        "",
        "from {} import {}".format(_abc_mod, _abc_name),
        "from modutsc.scheduling.registry import register",
        "",
        "",
        '@register("{}", "{}")'.format(kind, name),
        "class {}({}):".format(cls_name, _abc_name),
        '    """TODO: {} 组件的具体实现。"""'.format(kind),
        "",
        "    # ── 配置规则 ──",
        "    # 1. setup 中只读 cfg.get('key')，所有键暴露到 YAML",
        "    # 2. 不 import 其他组件的实现类",
        "    # 3. 使用 env 时只调用 env ABC 上的方法",
        "    # 4. 拓扑参数用仿真器产出的名字（如 num_phase）",
        "",
        "    # ── 可选：手动声明配置键和端口依赖（不声明则自动 AST 分析） ──",
        "    # __config_keys__ = [\"features\", \"num_phase\", \"max_lanelinks\"]",
        "    # __port_deps__ = {",
        '    #     "observe": ["features", "num_phase", "max_lanelinks"],',
        "    # }",
        "    # __param_deps__ = {",
        '    #     "reward_norm": ["num_phase"],',
        "    # }",
        "",
        "    def setup(self, cfg: dict) -> None:",
        "        # TODO: 从 cfg 中读取配置参数",
        "        # 例: self._lr = cfg.get('lr', 1e-4)",
        "        pass",
        "",
    ]

    for method_name, args_str, ret_str in methods:
        if method_name == "setup":
            continue
        ret_ann = f"{ret_str}" if ret_str else ""
        sig = f"{args_str})" if args_str else ")"
        lines.append(f"    def {method_name}(self, {sig}{ret_ann}:")
        lines.append("        # TODO: 实现此方法")
        lines.append("        pass")
        lines.append("")

    code = "\n".join(lines)
    _os.makedirs(output_dir, exist_ok=True)
    path = _os.path.join(output_dir, name + ".py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return code


_KNOWN_ABC_MODULE = {
    "observer": "modutsc.plugins.observers",
    "actor": "modutsc.plugins.actors",
    "reward": "modutsc.plugins.rewards",
    "collector": "modutsc.plugins.collectors",
    "algorithm": "modutsc.plugins.algorithms",
    "orchestrator": "modutsc.orchestration",
    "environment": "modutsc.env",
    "tracker": "modutsc.plugins.trackers",
}


# ═══════════════════════════════════════════════════════════════
#  内部辅助
# ═══════════════════════════════════════════════════════════════

_ds_index_cache = None
_ds_index_dir = None


def _ensure_dataset_index(data_dir="data"):
    global _ds_index_cache, _ds_index_dir
    index_path = _os.path.join(data_dir, "datasets_index.yaml")
    if _ds_index_cache is None or _ds_index_dir != data_dir:
        _ds_index_cache = load_index(index_path)
        _ds_index_dir = data_dir
    return _ds_index_cache


def _get_env_topo(dataset_path):
    if not dataset_path:
        return {}
    return find_topo(dataset_path) or {}


def _probe_determiner(kind, cls, method_name, deps, env=None):
    dep_list = sorted(deps)
    def f(known):
        if not all(d in known for d in dep_list):
            return {}
        cfg = {d: known[d] for d in dep_list}
        val = _build_and_measure_port(cls, method_name, cfg, env=env)
        if val is not None:
            return {f"{method_name}_out_dim": val}
        return {}
    return f


def _topo_determiner(key, topo_keys, ds_index):
    def f(known):
        known_topo = {k: v for k, v in known.items() if k in topo_keys}
        if not known_topo:
            return {}
        matches = match_datasets(known_topo, ds_index)
        if len(matches) == 1:
            return {k: matches[0].get(k) for k in topo_keys}
        return {}
    return f


def _serialize_values(kinds, values):
    import numpy as np
    result = {}
    for kind in kinds:
        params = values.get(kind, {})
        result[kind] = _json_safe(params)
    return result


def _json_safe(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
