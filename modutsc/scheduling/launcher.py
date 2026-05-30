import yaml
import inspect

from modutsc.scheduling.registry import (
    find, discover, get_orch_kinds, get_orch_wiring,
    _get_kind_config_params, _get_kind_config, _canonical_kind,
)
from modutsc.scheduling.config_solver import (
    ConfigSolver,
    trace_port_deps, resolve_port_deps, filter_deps_by_sensitivity,
    ports_from_wiring, measure_dim, _build_and_measure_port,
    scan_setup_param_deps, _self_probe_determiner,
    _method_uses_env,
)
from modutsc.orchestration import Orchestrator


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


class Launcher:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)
        self._config_path = config_path
        self._per_junction_topo = {}
        discover()

    def build(self, overrides: dict = None) -> Orchestrator:
        if overrides:
            self._deep_merge(self._cfg, overrides)

        orch_section = _get_kind_config(self._cfg, "orchestrator")
        if orch_section is None:
            raise ValueError("missing 'orchestrator' section in config")
        orch_name = orch_section.get("plugin", "")
        if not orch_name:
            raise ValueError("missing 'plugin' in 'orchestrator' section")

        kinds = get_orch_kinds(orch_name)
        kinds = [k for k in kinds if k != "orchestrator"]
        # environment 始终参与 solver（拓扑校正需要写入），即使编排器 setup 未声明
        if "environment" not in kinds:
            kinds.append("environment")

        wiring = get_orch_wiring(orch_name)
        wiring_edges = wiring.get("edges", [])
        env_topo = self._load_env_topo()

        solver = ConfigSolver(kinds)

        for kind in kinds:
            for k, v in _get_kind_config_params(self._cfg, kind).items():
                solver.set_value(kind, k, v)

        ports = ports_from_wiring(wiring_edges)
        output_kinds = set()
        env_dependent_ports = set()
        deferred_ports = []

        for kind in kinds:
            if kind == "environment":
                continue
            plugin_name = self._plugin_name(kind)
            cls = find(kind, plugin_name)
            if cls is None:
                continue
            kind_ports = ports.get(kind, {})
            for method_name, direction in kind_ports.items():
                if direction != "output":
                    continue
                dim_key = f"{method_name}_out_dim"
                overapprox = resolve_port_deps(cls, method_name)
                base_cfg = dict(_get_kind_config_params(self._cfg, kind))
                base_cfg.update(env_topo)
                base_cfg = {k: v for k, v in base_cfg.items() if k in overapprox}

                exact_deps = set()
                if overapprox:
                    if getattr(cls, '__port_deps__', None) is not None:
                        exact_deps = overapprox
                    else:
                        exact_deps = filter_deps_by_sensitivity(cls, method_name, overapprox, base_cfg)

                for dep in exact_deps:
                    solver.add_group({(kind, dim_key), (kind, dep)})

                _, _, needs_env = _method_uses_env(cls, method_name)
                if needs_env:
                    for topo_key in env_topo:
                        solver.add_group({(kind, dim_key), (kind, topo_key)})

                deferred_ports.append(
                    (kind, cls, method_name, dim_key, exact_deps, needs_env)
                )

                if needs_env:
                    env_dependent_ports.add((kind, method_name, dim_key))

                output_kinds.add((kind, method_name, dim_key))

        for kind in kinds:
            if kind == "environment" or kind == "orchestrator":
                continue
            plugin_name = self._plugin_name(kind)
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
                for d in dep_keys:
                    solver.add_group({(kind, target_key), (kind, d)})
                solver.add_determiner(
                    kind, target_key,
                    _self_probe_determiner(kind, cls, target_key, dep_keys)
                )

        for edge in wiring_edges:
            fk = edge.get("from", {}).get("kind")
            fm = edge.get("from", {}).get("method")
            tk = edge.get("to", {}).get("kind")
            tm = edge.get("to", {}).get("method")
            if not all([fk, fm, tk, tm]):
                continue
            if any(ok == fk and om == fm for ok, om, _ in output_kinds):
                fkey = f"{fm}_out_dim"
                tkey = f"{tm}_in_dim"
                solver.add_equal(fk, fkey, tk, tkey)

        assembled = {}
        self._env = None
        topo_corrections = {}

        try:
            self._env = self._build_env()
            assembled["environment"] = self._env

            ids = self._env.ids()
            if ids:
                actual_num_phase = max(
                    self._env.phase_count(j) for j in ids
                )
                actual_max_links = max(
                    len(self._env.traffic_light_controlled_links(j))
                    for j in ids
                )
                actual_max_green = max(
                    len(self._env.green_phase_indices(j))
                    for j in ids
                )
                actual_tsc = len(ids)

                if actual_num_phase and actual_num_phase != env_topo.get("num_phase"):
                    topo_corrections["num_phase"] = actual_num_phase
                if actual_max_links and actual_max_links != env_topo.get("max_lanelinks"):
                    topo_corrections["max_lanelinks"] = actual_max_links
                if actual_max_green and actual_max_green != env_topo.get("max_green_phases"):
                    topo_corrections["max_green_phases"] = actual_max_green
                if actual_tsc and actual_tsc != env_topo.get("num_tsc"):
                    topo_corrections["num_tsc"] = actual_tsc

                self._per_junction_topo = {}
                for i, jid in enumerate(ids):
                    self._per_junction_topo[i] = {
                        "num_phase": self._env.phase_count(jid),
                        "max_lanelinks": len(self._env.traffic_light_controlled_links(jid)),
                        "max_green_phases": len(self._env.green_phase_indices(jid)),
                    }

                for topo_key, actual_val in topo_corrections.items():
                    old_val = env_topo.get(topo_key)
                    print(f"[config] env拓扑校正: {topo_key} {old_val} -> {actual_val}")
                    for kind in kinds:
                        if kind == "environment":
                            continue
                        solver.clear_dependent_group(kind, topo_key)
                        solver.set_value(kind, topo_key, actual_val)
                    solver.set_value("environment", topo_key, actual_val)

            for kind, cls, method_name, dim_key, exact_deps, needs_env in deferred_ports:
                if not needs_env:
                    continue
                resolver = _probe_determiner(
                    kind, cls, method_name, exact_deps, env=self._env
                )
                solver.add_determiner(kind, dim_key, resolver)

            resolved, recommendations, warnings = solver.solve()
            for w in warnings:
                print(f"[config] {w}")

            for rec in recommendations:
                if len(rec["candidates"]) == 1:
                    k, v = next(iter(rec["candidates"].items()))
                    kind = rec.get("kind", "")
                    if resolved.get(kind, {}).get(k) is None:
                        resolved.setdefault(kind, {})[k] = v
                else:
                    print(f"[config] recommend: {rec['kind']}.{rec['key']} candidates={list(rec['candidates'].values())}")

            self._resolved_cfg = self._build_resolved_yaml(resolved)
            self._save_resolved_config()

            self._check_wiring_compatibility(
                wiring_edges, resolved, topo_corrections, env_topo
            )

            for kind in kinds:
                if kind == "environment":
                    continue

                kind_cfg = dict(resolved.get(kind, {}))
                section = self._cfg.get(kind)

                if isinstance(section, list):
                    per_instance = (
                        self._per_junction_topo if self._per_junction_topo else None
                    )
                    assembled[kind] = self._build_multi(
                        kind, section, kind_cfg, per_instance_topo=per_instance
                    )
                else:
                    plugin_name = self._plugin_name(kind)
                    if not plugin_name:
                        continue
                    cls = find(kind, plugin_name)
                    if cls is None:
                        raise ValueError(f"Plugin {kind}/{plugin_name} not found")
                    obj = cls()
                    try:
                        obj.setup(kind_cfg, env=self._env)
                    except TypeError:
                        obj.setup(kind_cfg)
                    assembled[kind] = obj

            if "tracker" not in assembled:
                tracker_cls = find("tracker", "console")
                if tracker_cls is not None:
                    default_tracker = tracker_cls()
                    default_tracker.setup({"sumo_episode_kpis": True})
                    assembled["tracker"] = default_tracker

            missing = solver.missing()
            if missing:
                print(f"[config] unresolved parameters: {missing}")

            self._verify_dims(assembled, wiring_edges, resolved)
            return self._build_orch(assembled)
        except Exception:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    pass
            raise

    def _build_resolved_yaml(self, resolved):
        # 拓扑参数不需要出现在配置文件中，它们由环境自动探测并注入
        topo_params = {"num_phase", "max_lanelinks", "max_green_phases", "num_tsc", "max_phase"}
        
        cfg = dict(self._cfg)
        for kind, params in resolved.items():
            section = cfg.get(kind)
            if section is None:
                cfg[kind] = {"plugin": self._plugin_name(kind), "config": {}}
            elif isinstance(section, list):
                for item in section:
                    if isinstance(item, dict):
                        item.setdefault("config", {})
                        for k, v in params.items():
                            # 过滤拓扑参数
                            if k in topo_params:
                                continue
                            if v is not None:
                                item["config"][k] = v
            elif isinstance(section, dict):
                section.setdefault("config", {})
                for k, v in params.items():
                    # 过滤拓扑参数
                    if k in topo_params:
                        continue
                    if v is not None:
                        section["config"][k] = v
        return cfg

    def _save_resolved_config(self):
        path = self._config_path.replace(".yaml", "_resolved.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._resolved_cfg, f, allow_unicode=True, sort_keys=False)
        print(f"[config] resolved config saved to {path}")

    @property
    def resolved_config(self) -> dict:
        return getattr(self, "_resolved_cfg", {})

    def _load_env_topo(self):
        env_section = _get_kind_config(self._cfg, "environment")
        if env_section is None:
            return {"num_phase": 4, "max_lanelinks": 12, "num_tsc": 1}
        roadnet = env_section.get("config", {}).get("roadnet_file", "")
        if not roadnet:
            return {"num_phase": 4, "max_lanelinks": 12, "num_tsc": 1}
        from modutsc.scheduling.dataset_index import find_topo, _query_topo_from_env
        topo = find_topo(roadnet)
        if topo:
            return topo
        env_cls = find("environment", env_section.get("plugin", ""))
        if env_cls is None:
            return {"num_phase": 4, "max_lanelinks": 12, "num_tsc": 1}
        try:
            topo = _query_topo_from_env(env_cls, roadnet)
            return topo
        except Exception:
            return {"num_phase": 4, "max_lanelinks": 12, "num_tsc": 1}

    def _plugin_name(self, kind: str) -> str:
        section = _get_kind_config(self._cfg, kind)
        if section is None:
            return ""
        return section.get("plugin", "")

    def _build_env(self):
        env_section = _get_kind_config(self._cfg, "environment")
        if env_section is None:
            raise ValueError("missing 'environment' section in config")
        env_cls = find("environment", env_section.get("plugin", ""))
        if env_cls is None:
            raise ValueError(f"Environment '{env_section.get('plugin', '')}' not found")
        env = env_cls()
        try:
            env.launch(env_section.get("config", {}))
        except Exception:
            try:
                env.close()
            except Exception:
                pass
            raise
        return env

    def _build_multi(self, kind: str, section: list, kind_cfg: dict,
                     per_instance_topo: dict = None):
        items = []
        for i, entry in enumerate(section):
            if not isinstance(entry, dict):
                continue
            cls = find(kind, entry.get("plugin", ""))
            if cls is None:
                raise ValueError(f"{kind}/{entry.get('plugin', '<missing>')} not found")
            final_cfg = dict(kind_cfg)
            final_cfg.update(entry.get("config", {}))
            if per_instance_topo and i in per_instance_topo:
                final_cfg.update(per_instance_topo[i])
            obj = cls()
            try:
                obj.setup(final_cfg, env=self._env)
            except TypeError:
                obj.setup(final_cfg)
            items.append(obj)
        return items

    def _build_orch(self, assembled):
        orch_section = _get_kind_config(self._cfg, "orchestrator")
        if orch_section is None:
            raise ValueError("missing 'orchestrator' section in config")
        orch_cls = find("orchestrator", orch_section.get("plugin", ""))
        if orch_cls is None:
            raise ValueError(f"Orchestrator '{orch_section.get('plugin', '')}' not found")
        orch = orch_cls()

        sig = inspect.signature(orch_cls.setup)
        orch_kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cfg'):
                continue
            if param.kind == param.VAR_KEYWORD:
                continue
            canonical = _canonical_kind(param_name)
            if canonical in assembled:
                value = assembled[canonical]
                if param_name in ('algorithms',) and not isinstance(value, list):
                    value = [value]
                orch_kwargs[param_name] = value

        orch_kwargs["cfg"] = orch_section.get("config", {})
        orch.setup(**orch_kwargs)
        return orch

    def _verify_dims(self, assembled, wiring_edges, resolved):
        env = assembled.get("environment")
        if not env:
            return

        for edge in wiring_edges:
            fk = edge.get("from", {}).get("kind")
            fm = edge.get("from", {}).get("method")
            if not fk or not fm:
                continue
            try:
                obj = assembled[fk]
                try:
                    result = getattr(obj, fm)()
                except TypeError:
                    result = getattr(obj, fm)(env)
                actual = measure_dim(result)
                expected = resolved.get(fk, {}).get(f"{fm}_out_dim")
                if actual is not None and expected is not None and actual != expected:
                    print(f"[verify] {fk}.{fm} dim={actual} cfg={expected}")
            except Exception:
                pass

    def _check_wiring_compatibility(self, wiring_edges, resolved,
                                    topo_corrections, env_topo):
        errors = []
        recommendations = []
        for edge in wiring_edges:
            fk = edge.get("from", {}).get("kind")
            fm = edge.get("from", {}).get("method")
            tk = edge.get("to", {}).get("kind")
            tm = edge.get("to", {}).get("method")
            if not all([fk, fm, tk, tm]):
                continue
            if fk == "environment" or tk == "environment":
                continue

            from_cls = find(fk, self._plugin_name(fk))
            if from_cls is None:
                continue

            from_cfg = dict(resolved.get(fk, {}))
            from_cfg.update(env_topo)
            from_cfg.update(topo_corrections)

            actual_out = None
            try:
                actual_out = _build_and_measure_port(
                    from_cls, fm, from_cfg, env=self._env
                )
            except Exception:
                continue

            if actual_out is None:
                continue

            expected_in = resolved.get(tk, {}).get(f"{tm}_in_dim")
            if expected_in is None:
                continue

            if actual_out == expected_in:
                continue

            from_plugin = self._plugin_name(fk)
            to_plugin = self._plugin_name(tk)
            errors.append(
                f"{fk}[{from_plugin}].{fm} 实际输出维度={actual_out}，"
                f"但 {tk}[{to_plugin}].{tm} 期望输入维度={expected_in}。"
            )

            rec = self._recommend_compatible_config(
                fk, fm, actual_out, tk, tm, resolved,
                env_topo, topo_corrections
            )
            if rec:
                recommendations.append(rec)

        if errors:
            msg = "无法完成实验组装，检测到代码层面的维度不兼容:\n"
            msg += "\n".join(f"  - {e}" for e in errors)
            if recommendations:
                msg += "\n推荐方案:\n"
                msg += "\n".join(f"  + {r}" for r in recommendations)
            raise RuntimeError(msg)

    def _recommend_compatible_config(self, fk, fm, actual_out,
                                      tk, tm, resolved,
                                      env_topo, topo_corrections):
        to_cls = find(tk, self._plugin_name(tk))
        if to_cls is None:
            return None

        port_deps = resolve_port_deps(to_cls, tm)
        if not port_deps:
            return None

        extra_deps = set()
        manual = getattr(to_cls, '__port_deps__', None)
        if isinstance(manual, dict):
            extra_deps = set(manual.get(tm, []))

        all_deps = port_deps | extra_deps
        tunable = {
            d for d in all_deps
            if d not in env_topo and d not in topo_corrections
        }

        if not tunable:
            return None

        base_cfg = dict(resolved.get(tk, {}))
        base_cfg.update(env_topo)
        base_cfg.update(topo_corrections)

        for key in sorted(tunable):
            for attempt in self._enum_cfg_values(key, base_cfg):
                test_cfg = dict(base_cfg)
                test_cfg[key] = attempt
                try:
                    test_dim = _build_and_measure_port(
                        to_cls, tm, test_cfg, env=self._env
                    )
                except Exception:
                    continue
                if test_dim is not None and test_dim == actual_out:
                    return (
                        f"将 {tk}.config.{key} 设置为 {attempt} "
                        f"可使 {tk}.{tm}_in_dim 匹配 {fk}.{fm} 输出维度 {actual_out}"
                    )
        return (
            f"所有候选参数均无法使 {tk}.{tm}_in_dim 匹配 {fk}.{fm} 输出维度 "
            f"{actual_out}。这两个组件的维度计算逻辑存在结构性冲突，无法在同一实验中组合使用。"
            f"建议: 更换 {fk} 或 {tk} 的插件实现。"
        )

    def _enum_cfg_values(self, key, base_cfg):
        current = base_cfg.get(key)
        if isinstance(current, list):
            yield current
            for i in range(len(current), 0, -1):
                yield current[:i]
            return
        if isinstance(current, int):
            for v in [current + d for d in range(-20, 21)]:
                if v > 0:
                    yield v
            for mul in [2, 3, 5]:
                if current * mul != current:
                    yield current * mul
            return
        if isinstance(current, float):
            for mul in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
                yield int(current * mul)
            return
        yield current

    def _deep_merge(self, base: dict, override: dict):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            elif k in base and isinstance(base[k], list) and isinstance(v, list):
                for i in range(min(len(base[k]), len(v))):
                    if isinstance(base[k][i], dict) and isinstance(v[i], dict):
                        self._deep_merge(base[k][i], v[i])
                for i in range(len(base[k]), len(v)):
                    base[k].append(v[i])
            else:
                base[k] = v

    @property
    def config(self) -> dict:
        return self._cfg

    @property
    def resolved_config(self) -> dict:
        return self._resolved_cfg
