import ast
import inspect

from modutsc.scheduling.registry import _get_method_ast


# ── AST scan helpers (unchanged) ──

def scan_setup_cfg_keys(cls):
    try:
        src_file = inspect.getfile(cls)
    except TypeError:
        return set()
    try:
        with open(src_file, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, OSError):
        return set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            keys = set()
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef) and item.name == "setup":
                    keys.update(_scan_cfg_calls(item))
            return keys
    return set()


def resolve_config_keys(cls):
    """统一入口：读 __config_keys__ 类属性（手动声明），否则 AST 自动分析。"""
    manual = getattr(cls, '__config_keys__', None)
    if manual is not None:
        try:
            return set(manual)
        except Exception as e:
            print(f"[config] {cls.__name__}.__config_keys__ failed ({e}), falling back to AST")
    return scan_setup_cfg_keys(cls)


def _scan_cfg_calls(func_node):
    keys = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "cfg"
                and func.attr == "get"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        k = node.args[0].value
        if isinstance(k, str):
            keys.add(k)
    return keys


def _extract_setup_cfg_map(func_node):
    mapping = {}
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Assign):
            continue
        key = _first_cfg_key_in(node.value)
        if not key:
            continue
        for target in node.targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                mapping[target.attr] = key
    return mapping


def _first_cfg_key_in(expr_node):
    """在 AST 表达式（包括 int(cfg.get(...)) 等包装）中提取第一个 cfg.get 的键名。"""
    for node in ast.walk(expr_node):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "cfg"
                and node.func.attr == "get"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        k = node.args[0].value
        if isinstance(k, str):
            return k
    return None


def _find_cfg_get_calls(node):
    """在 AST 节点（包括 int(cfg.get(...)) 等包装）中找到所有 cfg.get 调用。"""
    result = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if (isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "cfg"
                    and child.func.attr == "get"):
                result.append(child)
    return result


def _attr_to_cfg(cls):
    mapping = {}
    try:
        src_file = inspect.getfile(cls)
    except TypeError:
        return mapping
    try:
        with open(src_file, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, OSError):
        return mapping
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "setup":
                    mapping = _extract_setup_cfg_map(item)
                    return mapping
    return mapping


def trace_port_deps(cls, method_name):
    func_node = _get_method_ast(cls, method_name)
    if func_node is None:
        return set()
    attr_to_cfg = _attr_to_cfg(cls)
    if not attr_to_cfg:
        return set()
    port_attrs = set()
    for node in ast.walk(func_node):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            port_attrs.add(node.attr)
    return {attr_to_cfg[a] for a in port_attrs if a in attr_to_cfg}


def resolve_port_deps(cls, method_name):
    """统一入口：读 __port_deps__ 类属性（手动声明），否则 AST 自动分析。

    __port_deps__ 格式: {"observe": ["features","num_phase","max_lanelinks"], ...}
    """
    manual = getattr(cls, '__port_deps__', None)
    if manual is not None:
        try:
            deps = set(manual.get(method_name, []))
            return deps
        except Exception as e:
            print(f"[config] {cls.__name__}.__port_deps__ failed ({e}), falling back to AST")
    return trace_port_deps(cls, method_name)


def scan_setup_param_deps(cls):
    """AST 扫描 setup() 中 cfg.get(key, expr) 的默认表达式引用了其他 self._xxx，
    追溯回 cfg 键，得到组件内部的参数依赖关系。

    例: self._reward_norm = cfg.get("reward_norm", 1.0 / self._num_phase)
        且 self._num_phase = cfg.get("num_phase", 4)
        → {"reward_norm": {"num_phase"}}

    Returns:
        dict: {target_cfg_key: {dep_cfg_key, ...}}
    """
    func_node = _get_method_ast(cls, "setup")
    if func_node is None:
        return {}

    attr_to_cfg = _attr_to_cfg(cls)
    if not attr_to_cfg:
        return {}

    result = {}
    for node in ast.walk(func_node):
        for cfg_node in _find_cfg_get_calls(node):
            if len(cfg_node.args) < 2:
                continue
            target_key = cfg_node.args[0].value
            if not isinstance(target_key, str):
                continue
            default_expr = cfg_node.args[1] if len(cfg_node.args) >= 2 else None
            if default_expr is None:
                continue
            refs = _find_self_refs(default_expr, attr_to_cfg)
            if refs:
                existing = result.setdefault(target_key, set())
                existing.update(refs - {target_key})
    return {k: s for k, s in result.items() if s}


def _find_self_refs(expr_node, attr_to_cfg):
    """在 AST 表达式中找到所有 self._xxx 引用，映射回 cfg 键名。"""
    refs = set()
    for node in ast.walk(expr_node):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            if node.attr in attr_to_cfg:
                refs.add(attr_to_cfg[node.attr])
    return refs


def _self_probe_determiner(kind, cls, target_key, deps):
    """self_probe 确定器：已知依赖键 → 实例化组件 → 读 self._xxx 属性值。"""
    dep_list = sorted(deps)
    def f(known):
        if not all(d in known for d in dep_list):
            return {}
        cfg = {d: known[d] for d in dep_list}
        cfg[target_key] = _build_and_read_attr(cls, target_key, cfg)
        val = cfg.get(target_key)
        if val is not None:
            return {target_key: val}
        return {}
    return f


def _perturb_param(key, current_value):
    if current_value is None:
        return None
    if isinstance(current_value, bool):
        return not current_value
    if isinstance(current_value, int):
        return current_value + 1 if current_value < 1000 else current_value - 1
    if isinstance(current_value, float):
        return current_value * 1.1
    if isinstance(current_value, list):
        if not current_value:
            return ["_probe"]
        return current_value + [current_value[0]]
    if isinstance(current_value, dict):
        d = dict(current_value)
        if d:
            first_key = next(iter(d))
            d[first_key] = (d[first_key] + 1) if isinstance(d[first_key], int) else "_changed"
        return d
    if isinstance(current_value, str):
        return current_value + "_"
    return None


def filter_deps_by_sensitivity(cls, method_name, overapprox_deps, base_cfg):
    if not overapprox_deps:
        return set()
    try:
        dim_ref = _build_and_measure_port(cls, method_name, base_cfg)
    except Exception:
        return overapprox_deps
    if dim_ref is None:
        return overapprox_deps
    real = set()
    for key in sorted(overapprox_deps):
        if key not in base_cfg:
            continue
        perturbed_value = _perturb_param(key, base_cfg[key])
        if perturbed_value is None:
            real.add(key)
            continue
        test_cfg = dict(base_cfg)
        test_cfg[key] = perturbed_value
        try:
            dim_test = _build_and_measure_port(cls, method_name, test_cfg)
        except Exception:
            real.add(key)
            continue
        if dim_test is None:
            real.add(key)
        elif dim_test != dim_ref:
            real.add(key)
    return real


def _build_and_read_attr(cls, target_key, cfg_vals):
    """实例化组件，读 self._xxx 属性值（通过 attr_to_cfg 逆查属性名）。

    注意：会真实执行 setup()。组件编写者应确保 setup() 是无副作用的赋值操作。
    """
    attr_to_cfg = _attr_to_cfg(cls)
    obj = cls()
    try:
        obj.setup(dict(cfg_vals))
    except Exception:
        return cfg_vals.get(target_key)
    try:
        for attr_name, cfg_key in attr_to_cfg.items():
            if cfg_key == target_key and hasattr(obj, attr_name):
                return getattr(obj, attr_name)
    finally:
        if hasattr(obj, 'teardown'):
            try:
                obj.teardown()
            except Exception:
                pass
    return cfg_vals.get(target_key)


def _method_uses_env(cls, method_name):
    func_node = _get_method_ast(cls, method_name)
    if func_node is None:
        return False, False, False

    uses_env_param = False
    env_param_name = None
    try:
        method = getattr(cls, method_name, None)
        if method is not None:
            sig = inspect.signature(method)
            for p_name, p in sig.parameters.items():
                if p_name in ('self', 'cfg'):
                    continue
                if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                    continue
                if p_name in ('env', 'environment'):
                    env_param_name = p_name
                    break
    except Exception:
        pass

    if env_param_name is not None:
        uses_env_param = _ast_body_uses_param(func_node, env_param_name)

    uses_self_env = _ast_body_uses_self_env(func_node)

    return uses_env_param, uses_self_env, (uses_env_param or uses_self_env)


def _ast_body_uses_param(func_node, param_name):
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == param_name):
                return True
        if isinstance(node, ast.Attribute):
            if (isinstance(node.value, ast.Name)
                    and node.value.id == param_name):
                return True
    return False


def _ast_body_uses_self_env(func_node):
    for node in ast.walk(func_node):
        if isinstance(node, ast.Attribute):
            if (isinstance(node.value, ast.Name)
                    and node.value.id == 'self'
                    and node.attr in ('_env', '_environment')):
                return True
    return False


def _build_and_measure_port(cls, method_name, cfg_vals, env=None):
    obj = cls()
    try:
        try:
            obj.setup(dict(cfg_vals), env=env)
        except TypeError:
            obj.setup(dict(cfg_vals))
        try:
            result = getattr(obj, method_name)()
        except TypeError:
            if env is not None:
                result = getattr(obj, method_name)(env)
            elif hasattr(obj, 'dim'):
                return obj.dim()
            else:
                return None
        val = measure_dim(result)
        if val is None and hasattr(obj, 'dim'):
            val = obj.dim()
        return val
    except Exception:
        if hasattr(obj, 'dim'):
            return obj.dim()
        return None
    finally:
        if hasattr(obj, 'teardown'):
            try:
                obj.teardown()
            except Exception:
                pass


def measure_dim(result):
    if result is None:
        return None
    if hasattr(result, "shape"):
        return result.shape[-1]
    if isinstance(result, (list, tuple)):
        if len(result) == 0:
            return None
        return measure_dim(result[0])
    if isinstance(result, dict):
        feats = result.get("features")
        if feats is not None and hasattr(feats, "shape"):
            return feats.shape[-1]
        return None
    return None


def ports_from_wiring(wiring_edges):
    ports = {}
    for edge in wiring_edges:
        fk, fm = edge["from"]["kind"], edge["from"]["method"]
        tk, tm = edge["to"]["kind"], edge["to"]["method"]
        ports.setdefault(fk, {})[fm] = "output"
        ports.setdefault(tk, {})[tm] = "input"
    return ports


# ═══════════════════════════════════════════════════════════════
#  Unified Constraint Solver
# ═══════════════════════════════════════════════════════════════


class ConfigSolver:
    def __init__(self, kinds):
        self._values = {k: {} for k in kinds}
        self._groups = []
        self._equals = []
        self._determiners = {}
        self._recommendations = []
        self._warnings = []

    def set_value(self, kind, key, value):
        if self._values[kind].get(key) == value:
            return
        self._values[kind][key] = value

    def add_group(self, members):
        self._groups.append(frozenset(members))

    def add_determiner(self, kind, key, fn):
        self._determiners.setdefault((kind, key), []).append(("probe", fn))

    def add_equal(self, kind_a, key_a, kind_b, key_b):
        self._equals.append((kind_a, key_a, kind_b, key_b))
        self._groups.append(frozenset({(kind_a, key_a), (kind_b, key_b)}))

    def solve(self):
        groups = self._merge_groups()
        changed = True
        while changed:
            changed = False
            for (kind_a, key_a, kind_b, key_b) in self._equals:
                va = self._values[kind_a].get(key_a)
                vb = self._values[kind_b].get(key_b)
                if va is not None and vb is None:
                    self._values[kind_b][key_b] = va
                    changed = True
                elif vb is not None and va is None:
                    self._values[kind_a][key_a] = vb
                    changed = True
                elif va is not None and vb is not None and va != vb:
                    self._warnings.append(
                        f"conflict: ({kind_a}.{key_a}={va}) != ({kind_b}.{key_b}={vb})"
                    )

        recomputed_groups = True
        while recomputed_groups:
            recomputed_groups = False
            groups = self._merge_groups()
            for group in groups:
                known = {}
                unknown = set()
                for kind, key in group:
                    v = self._values[kind].get(key)
                    if v is not None:
                        known[key] = v
                    else:
                        unknown.add((kind, key))

                if not unknown:
                    continue

                for (kind, key) in list(unknown):
                    for tag, determiner in self._determiners.get((kind, key), []):
                        candidates = determiner(known)
                        if candidates:
                            self._recommendations.append({
                                "kind": kind, "key": key,
                                "candidates": candidates,
                                "source": tag,
                            })
                            for ck, cv in candidates.items():
                                self._values[kind][ck] = cv
                            recomputed_groups = True
                            break

        changed = True
        while changed:
            changed = False
            for (kind_a, key_a, kind_b, key_b) in self._equals:
                va = self._values[kind_a].get(key_a)
                vb = self._values[kind_b].get(key_b)
                if va is not None and vb is None:
                    self._values[kind_b][key_b] = va
                    changed = True
                elif vb is not None and va is None:
                    self._values[kind_a][key_a] = vb
                    changed = True
                elif va is not None and vb is not None and va != vb:
                    self._warnings.append(
                        f"conflict: ({kind_a}.{key_a}={va}) != ({kind_b}.{key_b}={vb})"
                    )

        return self._values, self._recommendations, self._warnings

    def clear_dependent_group(self, kind, key):
        groups = self._merge_groups()
        for group in groups:
            if (kind, key) in group:
                for (k, kk) in group:
                    self._values[k].pop(kk, None)
                return

    def reset_phase(self):
        self._recommendations = []
        self._warnings = []

    def _merge_groups(self):
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for group in self._groups:
            mlist = list(group)
            if not mlist:
                continue
            find(mlist[0])
            for m in mlist[1:]:
                find(m)
                union(mlist[0], m)

        merged = {}
        for group in self._groups:
            mlist = list(group)
            if not mlist:
                continue
            root = find(mlist[0])
            if root not in merged:
                merged[root] = set()
            merged[root].update(group)

        return list(merged.values())

    def missing(self):
        m = {}
        for kind, params in self._values.items():
            unresolved = [k for k, v in params.items() if v is None]
            if unresolved:
                m[kind] = unresolved
        return m
