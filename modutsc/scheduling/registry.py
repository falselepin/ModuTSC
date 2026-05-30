import inspect as _inspect
import ast as _ast
import typing as _typing

_registry = {}
_orch_attr_cache = {}

# ── Kind name canonical form ──

_KIND_CONTRACT = {
    "observer":     ["setup", "observe"],
    "actor":        ["setup", "translate"],
    "reward":       ["setup", "compute"],
    "collector":    ["setup", "push", "ready", "pull", "size"],
    "algorithm":    ["setup", "act"],
    "orchestrator": ["setup", "warmup", "episode", "evaluate", "save", "load"],
    "environment":  ["ids", "phase_count", "launch", "reset", "step",
                     "time", "done", "close"],
    "tracker":      ["log", "close"],
}

_KIND_ALIASES = {
    "env": "environment",
    "algorithms": "algorithm",
    "orch": "orchestrator",
}

_PLURAL_MAP = {
    "observers":     "observer",
    "actors":        "actor",
    "rewards":       "reward",
    "collectors":    "collector",
    "algorithms":    "algorithm",
    "orchestrators": "orchestrator",
    "environments":  "environment",
    "trackers":      "tracker",
}


def _canonical_kind(name):
    if name in _KIND_ALIASES:
        return _KIND_ALIASES[name]
    if name in _KIND_CONTRACT:
        return name
    if name in _PLURAL_MAP:
        return _PLURAL_MAP[name]
    return name


def _get_kind_config(yaml_cfg, kind):
    """安全获取 YAML 中某个 kind 的配置节。
    自动处理 dict/list 差异，始终返回 dict 或 None。
    """
    section = yaml_cfg.get(kind)
    if section is None:
        return None
    if isinstance(section, list):
        section = section[0] if section else {}
    if not isinstance(section, dict):
        return None
    return section


def _get_kind_config_params(yaml_cfg, kind):
    """获取某个 kind 的 config 参数 dict，始终返回 dict"""
    section = _get_kind_config(yaml_cfg, kind)
    if section is None:
        return {}
    return dict(section.get("config", {}))


# ── Registration ──

def register(kind: str, name: str):
    kind = _canonical_kind(kind)

    def decorator(cls):
        required = _get_kind_contract(kind, cls)
        if required is not None:
            missing = [m for m in required if not hasattr(cls, m)]
            if missing:
                raise TypeError(
                    f"'{kind}/{name}' missing methods:\n" +
                    "\n".join(f"    def {m}(self, ...): ..." for m in missing)
                )
        if kind != "orchestrator":
            _validate_component_setup(cls, kind, name)
        _registry[(kind, name)] = cls
        if kind == "orchestrator" and name not in _orch_attr_cache:
            _orch_attr_cache[name] = _derive_orch_attr_map(cls)
        return cls
    return decorator


def _validate_component_setup(cls, kind, name):
    if not hasattr(cls, 'setup'):
        return
    try:
        sig = _inspect.signature(cls.setup)
    except (ValueError, TypeError):
        return
    params = [p for p in sig.parameters.values()
              if p.name not in ('self', 'cfg', 'env')
              and p.kind not in (_inspect.Parameter.VAR_POSITIONAL, _inspect.Parameter.VAR_KEYWORD)]
    if not params:
        return
    raise TypeError(
        f"'{kind}/{name}.setup()' must accept only 'cfg: dict'. "
        f"Found extra parameter: '{params[0].name}'. "
        f"Components must not depend on other components. "
        f"All inter-component data flow is managed by the orchestrator."
    )


def _get_kind_contract(kind, cls):
    if kind in _KIND_CONTRACT:
        return _KIND_CONTRACT[kind]
    methods = [m for m in dir(cls)
               if getattr(getattr(cls, m, None), '__isabstractmethod__', False)]
    if methods:
        _KIND_CONTRACT[kind] = methods
        return methods
    for base in cls.__mro__:
        if base is cls or base is object:
            continue
        methods = [m for m in dir(base)
                   if getattr(getattr(base, m, None), '__isabstractmethod__', False)]
        if methods:
            _KIND_CONTRACT[kind] = methods
            return methods
    return ["setup"]


# ── Orchestrator self._X → kind mapping (auto-derived) ──

def _derive_orch_attr_map(orch_cls):
    sig = _inspect.signature(orch_cls.setup)
    param_to_kind = {}
    for p_name, p in sig.parameters.items():
        if p_name in ('self', 'cfg', 'env'):
            continue
        if p.kind == p.VAR_KEYWORD:
            continue
        param_to_kind[p_name] = _canonical_kind(p_name)

    mapping = {}
    func_node = _get_method_ast(orch_cls, "setup")
    if func_node is None:
        for param_name, kind in param_to_kind.items():
            mapping.setdefault(kind, []).append(f"_{param_name}")
        return mapping

    for node in _ast.walk(func_node):
        if isinstance(node, _ast.Assign):
            for target in node.targets:
                if (_is_self_attr_node(target)
                        and isinstance(node.value, _ast.Name)):
                    kind = param_to_kind.get(node.value.id)
                    if kind:
                        mapping.setdefault(kind, []).append(target.attr)

    for param_name, kind in param_to_kind.items():
        if kind not in mapping:
            mapping.setdefault(kind, []).append(f"_{param_name}")

    return mapping


def _is_self_attr_node(node):
    return (isinstance(node, _ast.Attribute)
            and isinstance(node.value, _ast.Name)
            and node.value.id == 'self')


def _get_orch_attr_map(orch_name):
    if orch_name in _orch_attr_cache:
        return _orch_attr_cache[orch_name]
    orch_cls = find("orchestrator", orch_name)
    if orch_cls is None:
        return {}
    mapping = _derive_orch_attr_map(orch_cls)
    _orch_attr_cache[orch_name] = mapping
    return mapping


# ── Public API ──

def find(kind: str, name: str):
    return _registry.get((_canonical_kind(kind), name))


def list_all(kind: str = None):
    if kind:
        kind = _canonical_kind(kind)
        return [name for (k, name) in _registry if k == kind]
    return list(_registry.keys())


def _all_kinds():
    return sorted(set(k for (k, _) in _registry))


def discover(package_name: str = "modutsc"):
    import pkgutil
    import importlib
    package = importlib.import_module(package_name)
    errors = []
    for module_info in pkgutil.walk_packages(
        package.__path__, prefix=package_name + ".",
        onerror=lambda name: None
    ):
        try:
            importlib.import_module(module_info.name)
        except BaseException as e:
            errors.append((module_info.name, str(e)))
    if errors:
        print("[discover] some modules failed to import:")
        for name, err in errors:
            print(f"  {name}: {err.split(chr(10))[0][:120]}")


def get_orch_kinds(orchestrator_name):
    orch_cls = find("orchestrator", orchestrator_name)
    if orch_cls is None:
        return []
    sig = _inspect.signature(orch_cls.setup)
    return [_canonical_kind(name) for name, p in sig.parameters.items()
            if name not in ('self', 'cfg', 'env')
            and p.kind != p.VAR_KEYWORD]


def get_orch_wiring(orchestrator_name: str) -> dict:
    orch_cls = find("orchestrator", orchestrator_name)
    if orch_cls is None:
        raise ValueError(f"orchestrator '{orchestrator_name}' not found")

    manual = getattr(orch_cls, '__wiring_edges__', None)
    if manual is not None:
        edges = _validate_wiring_edges(manual, orch_cls)
        if edges is not _MANUAL_FAILED:
            return {"orchestrator": orchestrator_name, "edges": edges}

    raw = _scan_orch_edges(orch_cls)
    seen = set()
    edges = []
    for (fk, fm), (tk, tm) in raw:
        key = (fk, fm, tk, tm)
        if key not in seen:
            seen.add(key)
            edges.append({
                "from": {"kind": _canonical_kind(fk), "method": fm},
                "to":   {"kind": _canonical_kind(tk), "method": tm},
            })
    return {"orchestrator": orchestrator_name, "edges": edges}


_MANUAL_FAILED = object()


def _validate_wiring_edges(manual, cls):
    try:
        edges = []
        for edge in manual:
            fk = edge["from"]["kind"]
            fm = edge["from"]["method"]
            tk = edge["to"]["kind"]
            tm = edge["to"]["method"]
            edges.append({
                "from": {"kind": _canonical_kind(fk), "method": fm},
                "to":   {"kind": _canonical_kind(tk), "method": tm},
            })
        return edges
    except Exception as e:
        print(f"[config] {cls.__name__}.__wiring_edges__ failed ({e}), falling back to AST")
        return _MANUAL_FAILED


def get_orch_flowchart(orchestrator_name: str) -> dict:
    orch_cls = find("orchestrator", orchestrator_name)
    if orch_cls is None:
        raise ValueError(f"orchestrator '{orchestrator_name}' not found")
    return {"orchestrator": orchestrator_name, "methods": _scan_orch_flowchart(orch_cls)}


# ── Type checking ──

def check_method_compatibility(kind_a, name_a, kind_b, name_b):
    cls_a = find(kind_a, name_a)
    cls_b = find(kind_b, name_b)
    if cls_a is None or cls_b is None:
        return [], []
    sigs_a = _extract_class_sigs(cls_a)
    sigs_b = _extract_class_sigs(cls_b)
    compatible, incompatible = [], []
    for ma, sa in sigs_a.items():
        if sa["return"] is None:
            continue
        for mb, sb in sigs_b.items():
            first = list(sb["params"].values())[0] if sb["params"] else None
            if first is None:
                continue
            ok = _check_types(sa["return"], first)
            pair = (ma, mb, str(sa["return"]), str(first))
            (compatible if ok else incompatible).append(pair)
    return compatible, incompatible


def _extract_class_sigs(cls):
    sigs = {}
    for m_name in dir(cls):
        if m_name.startswith('_'):
            continue
        m = getattr(cls, m_name, None)
        if m is None or not callable(m):
            continue
        try:
            hints = _typing.get_type_hints(m)
        except Exception:
            continue
        sigs[m_name] = {"params": hints, "return": hints.pop('return', None)}
    return sigs


def _check_types(a, b):
    if a is None or b is None:
        return None
    if a == b:
        return True
    try:
        return issubclass(a, b)
    except TypeError:
        pass
    oa, ob = getattr(a, '__origin__', None), getattr(b, '__origin__', None)
    if oa is list and ob is list:
        aa, ab = getattr(a, '__args__', ()), getattr(b, '__args__', ())
        if aa and ab:
            return _check_types(aa[0], ab[0])
    return False


def validate_orch_wiring(orch_name, selections):
    wiring = get_orch_wiring(orch_name)
    issues = []
    for edge in wiring["edges"]:
        fk, fm = edge["from"]["kind"], edge["from"]["method"]
        tk, tm = edge["to"]["kind"], edge["to"]["method"]
        if fk not in selections or tk not in selections:
            continue
        fc, tc = find(fk, selections[fk]), find(tk, selections[tk])
        if fc is None or tc is None:
            continue
        if not hasattr(fc, fm):
            issues.append(f"{fk}/{selections[fk]} missing method {fm}")
            continue
        if not hasattr(tc, tm):
            issues.append(f"{tk}/{selections[tk]} missing method {tm}")
            continue
        try:
            fh = _typing.get_type_hints(getattr(fc, fm))
            th = _typing.get_type_hints(getattr(tc, tm))
        except Exception:
            continue
        fr = fh.pop('return', None) if fh else None
        tf = list(th.values())[0] if th else None
        if _check_types(fr, tf) is False:
            issues.append(f"type mismatch: {fk}.{fm}->{fr} vs {tk}.{tm}<-{tf}")
    return len(issues) == 0, issues


# ── Recommendations ──

def suggest_compatible(kind: str, name: str) -> dict:
    cls = find(kind, name)
    if cls is None:
        return {}
    wi, wo = getattr(cls, '__input_type__', None), getattr(cls, '__output_type__', None)
    result = {}
    for ok in _all_kinds():
        if ok == kind:
            continue
        names = list_all(ok)
        if not names:
            continue
        comp = []
        for n in names:
            o = find(ok, n)
            if o is None:
                continue
            if wi is not None and getattr(o, '__output_type__', None) is wi:
                comp.append(n)
            elif wo is not None and getattr(o, '__input_type__', None) is wo:
                comp.append(n)
        result[ok] = comp if comp else names
    return result


def filter_orchestrators_by_plugin(kind: str, name: str) -> list:
    return [on for on in list_all("orchestrator")
            if name in recommend_assembly(on)["by_kind"].get(kind, {}).get("options", [])]


def recommend_assembly(orchestrator_name: str) -> dict:
    orch_cls = find("orchestrator", orchestrator_name)
    if orch_cls is None:
        raise ValueError(f"orchestrator '{orchestrator_name}' not found")
    order = get_orch_kinds(orchestrator_name)
    by_kind = {}
    for kind in order:
        if kind == "orchestrator":
            continue
        options = list_all(kind)
        required = _scan_orch_kind_calls(orch_cls, kind)
        if required:
            options = [n for n in options
                       if find(kind, n) and all(hasattr(find(kind, n), m) for m in required)]
        by_kind[kind] = {"required": True, "options": options, "depends_on": []}
    return {"orchestrator": orchestrator_name, "required_kinds": order, "order": order, "by_kind": by_kind}


def compat_matrix(kind: str, name: str) -> dict:
    cls = find(kind, name)
    if cls is None:
        return {}
    return {dep: list_all(dep) for dep in getattr(cls, '__kind_deps__', [])}


# ── AST analysis: wiring edges ──


def _resolve_call(call_node, loop_bindings=None, orch_attr_map=None):
    if not isinstance(call_node.func, _ast.Attribute):
        return None
    method = call_node.func.attr
    target = call_node.func.value
    am = orch_attr_map

    if isinstance(target, _ast.Attribute) and _is_self_attr_node(target):
        for kind, attrs in am.items():
            if target.attr in attrs:
                return (kind, method)

    if isinstance(target, _ast.Subscript):
        if (isinstance(target.value, _ast.Attribute)
                and isinstance(target.value.value, _ast.Name)
                and target.value.value.id == 'self'):
            for kind, attrs in am.items():
                if target.value.attr in attrs:
                    return (kind, method)

    if isinstance(target, _ast.Name) and loop_bindings:
        k = loop_bindings.get(target.id)
        if k:
            return (k, method)

    return None


def _walk_body_in_order(stmt_list, loop_bindings=None, orch_attr_map=None):
    if loop_bindings is None:
        loop_bindings = {}
    for stmt in stmt_list:
        yield from _walk_stmt(stmt, loop_bindings, orch_attr_map)


def _walk_stmt(node, loop_bindings, orch_attr_map=None):
    am = orch_attr_map

    if isinstance(node, _ast.Expr):
        for kind, method in _walk_expr(node.value, loop_bindings, am):
            yield (kind, method, False)
    elif isinstance(node, _ast.Assign):
        _track_binding(node, loop_bindings, am)
        for kind, method in _walk_expr(node.value, loop_bindings, am):
            yield (kind, method, True)
    elif isinstance(node, _ast.For):
        inner = dict(loop_bindings)
        _collect_loop_binding(node, inner, am)
        for s in node.body:
            yield from _walk_stmt(s, inner, am)
    elif isinstance(node, _ast.If):
        if _is_hasattr_guard(node.test):
            pass
        else:
            for s in node.body:
                yield from _walk_stmt(s, loop_bindings, am)
            for s in node.orelse:
                yield from _walk_stmt(s, loop_bindings, am)
    elif isinstance(node, _ast.While):
        for s in node.body:
            yield from _walk_stmt(s, loop_bindings, am)


def _track_binding(assign_node, bindings, am):
    for target in assign_node.targets:
        if isinstance(target, _ast.Name):
            bindings[target.id] = _kind_from_expr(assign_node.value, am)


def _kind_from_expr(expr, am):
    if isinstance(expr, _ast.Subscript) and isinstance(expr.value, _ast.Attribute):
        if isinstance(expr.value.value, _ast.Name) and expr.value.value.id == 'self':
            for kind, attrs in am.items():
                if expr.value.attr in attrs:
                    return kind
    if isinstance(expr, _ast.IfExp):
        return _kind_from_expr(expr.body, am) or _kind_from_expr(expr.orelse, am)
    return None


def _is_hasattr_guard(test_node):
    if isinstance(test_node, _ast.Call) and isinstance(test_node.func, _ast.Name):
        if test_node.func.id == 'hasattr' and len(test_node.args) >= 2:
            arg0 = test_node.args[0]
            if isinstance(arg0, _ast.Attribute) and _is_self_attr_node(arg0):
                return True
    return False


def _collect_loop_binding(for_node, bindings, am=None):
    iter_val = for_node.iter
    if isinstance(iter_val, _ast.Call) and isinstance(iter_val.func, _ast.Name):
        if iter_val.func.id in ("enumerate", "zip") and iter_val.args:
            iter_val = iter_val.args[0]
    if not (isinstance(iter_val, _ast.Attribute)
            and isinstance(iter_val.value, _ast.Name)
            and iter_val.value.id == 'self'):
        return
    for kind, attrs in am.items():
        if iter_val.attr in attrs:
            if isinstance(for_node.target, _ast.Name):
                bindings[for_node.target.id] = kind
            elif isinstance(for_node.target, _ast.Tuple):
                for elt in for_node.target.elts:
                    if isinstance(elt, _ast.Name):
                        bindings[elt.id] = kind


def _walk_expr(node, loop_bindings, am):
    if isinstance(node, _ast.Call):
        for arg in node.args:
            yield from _walk_expr(arg, loop_bindings, am)
        for kw in node.keywords:
            yield from _walk_expr(kw.value, loop_bindings, am)
        r = _resolve_call(node, loop_bindings, am)
        if r:
            yield r
    elif isinstance(node, _ast.List):
        for elt in node.elts:
            yield from _walk_expr(elt, loop_bindings, am)
    elif isinstance(node, _ast.ListComp):
        yield from _walk_expr(node.elt, loop_bindings, am)
        for gen in node.generators:
            yield from _walk_expr(gen.iter, loop_bindings, am)
    elif isinstance(node, _ast.GeneratorExp):
        yield from _walk_expr(node.elt, loop_bindings, am)
        for gen in node.generators:
            yield from _walk_expr(gen.iter, loop_bindings, am)
    elif isinstance(node, _ast.Dict):
        for k in node.keys:
            if k is not None:
                yield from _walk_expr(k, loop_bindings, am)
        for v in node.values:
            yield from _walk_expr(v, loop_bindings, am)
    elif isinstance(node, _ast.Tuple):
        for elt in node.elts:
            yield from _walk_expr(elt, loop_bindings, am)


def _get_orch_registered_name(orch_cls):
    for (kind, name), cls in _registry.items():
        if kind == "orchestrator" and cls is orch_cls:
            return name
    return orch_cls.__name__.lower()


def _scan_orch_kind_calls(orchestrator_cls, kind):
    orch_name = _get_orch_registered_name(orchestrator_cls)

    manual = getattr(orchestrator_cls, '__kind_calls__', None)
    if manual is not None:
        try:
            calls = set(manual.get(kind, []))
            return calls
        except Exception as e:
            print(f"[config] {orchestrator_cls.__name__}.__kind_calls__ failed ({e}), falling back to AST")

    am = _get_orch_attr_map(orch_name)
    attrs = set(am.get(kind, []))
    if not attrs:
        return set()
    calls = set()
    try:
        src = _inspect.getfile(orchestrator_cls)
    except TypeError:
        return calls
    try:
        with open(src, encoding="utf-8") as f:
            tree = _ast.parse(f.read())
    except (SyntaxError, OSError):
        return calls
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == orchestrator_cls.__name__:
            for item in node.body:
                if isinstance(item, _ast.FunctionDef):
                    for kn, m, _ in _walk_body_in_order(item.body, orch_attr_map=am):
                        if kn == kind:
                            calls.add(m)
    return calls


def _scan_orch_edges(orch_cls):
    orch_name = _get_orch_registered_name(orch_cls)
    am = _get_orch_attr_map(orch_name)
    try:
        src = _inspect.getfile(orch_cls)
    except TypeError:
        return []
    try:
        with open(src, encoding="utf-8") as f:
            tree = _ast.parse(f.read())
    except (SyntaxError, OSError):
        return []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == orch_cls.__name__:
            edges = []
            for item in node.body:
                if isinstance(item, _ast.FunctionDef):
                    prev = None
                    prev_captured = False
                    for kind, method, captured in _walk_body_in_order(item.body, orch_attr_map=am):
                        cur = (kind, method)
                        if prev is not None and prev_captured:
                            edges.append((prev, cur))
                        prev = cur
                        prev_captured = captured
            return edges
    return []


def _scan_orch_flowchart(orch_cls):
    orch_name = _get_orch_registered_name(orch_cls)
    am = _get_orch_attr_map(orch_name)
    try:
        src = _inspect.getfile(orch_cls)
    except TypeError:
        return {}
    try:
        with open(src, encoding="utf-8") as f:
            tree = _ast.parse(f.read())
    except (SyntaxError, OSError):
        return {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == orch_cls.__name__:
            result = {}
            for item in node.body:
                if isinstance(item, _ast.FunctionDef) and item.name != '__init__':
                    chart = _build_flowchart(item.body, orch_attr_map=am)
                    if chart:
                        result[item.name] = chart
            return result
    return {}


def _build_flowchart(stmt_list, loop_bindings=None, orch_attr_map=None):
    if loop_bindings is None:
        loop_bindings = {}
    nodes = []
    for stmt in stmt_list:
        n = _build_flowchart_node(stmt, loop_bindings, orch_attr_map)
        if n is not None:
            nodes.append(n)
    return nodes


def _build_flowchart_node(stmt, loop_bindings, orch_attr_map=None):
    am = orch_attr_map
    if isinstance(stmt, _ast.Expr):
        calls = list(_walk_expr(stmt.value, loop_bindings, am))
        if calls:
            return {"type": "call", "kind": calls[0][0], "method": calls[0][1]}
    elif isinstance(stmt, _ast.Assign):
        _track_binding(stmt, loop_bindings, am)
        calls = list(_walk_expr(stmt.value, loop_bindings, am))
        if calls:
            if len(calls) == 1:
                return {"type": "call", "kind": calls[0][0], "method": calls[0][1]}
            return {"type": "sequence", "nodes": [{"kind": k, "method": m, "type": "call"} for k, m in calls]}
    elif isinstance(stmt, _ast.For):
        inner = dict(loop_bindings)
        _collect_loop_binding(stmt, inner, am)
        ti = _ast.unparse(stmt.target) if hasattr(_ast, 'unparse') else '...'
        ii = _ast.unparse(stmt.iter) if hasattr(_ast, 'unparse') else '...'
        return {"type": "for", "label": f"for {ti} in {ii}", "body": _build_flowchart(stmt.body, inner, am)}
    elif isinstance(stmt, _ast.While):
        ts = _ast.unparse(stmt.test) if hasattr(_ast, 'unparse') else '...'
        return {"type": "while", "label": ts, "body": _build_flowchart(stmt.body, loop_bindings, am)}
    elif isinstance(stmt, _ast.If):
        ts = _ast.unparse(stmt.test) if hasattr(_ast, 'unparse') else '...'
        r = {"type": "if", "label": ts, "body": _build_flowchart(stmt.body, loop_bindings, am)}
        if stmt.orelse:
            r["else"] = _build_flowchart(stmt.orelse, loop_bindings, am)
        return r
    return None


def _get_method_ast(cls, method_name):
    try:
        src = _inspect.getfile(cls)
    except TypeError:
        return None
    try:
        with open(src, encoding="utf-8") as f:
            tree = _ast.parse(f.read())
    except (SyntaxError, OSError):
        return None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == cls.__name__:
            for item in node.body:
                if isinstance(item, _ast.FunctionDef) and item.name == method_name:
                    return item
    return None


# ── Runtime dynamic registration ──

def register_dynamic(kind: str, name: str, cls):
    """运行时动态注册一个组件类（无需 @register 装饰器）。

    用于用户上传插件后即时注册，无需重启服务。
    """
    kind = _canonical_kind(kind)
    required = _get_kind_contract(kind, cls)
    if required is not None:
        missing = [m for m in required if not hasattr(cls, m)]
        if missing:
            raise TypeError(
                f"'{kind}/{name}' 缺少方法:\n" +
                "\n".join(f"    def {m}(self, ...): ..." for m in missing)
            )
    if kind != "orchestrator":
        _validate_component_setup(cls, kind, name)
    _registry[(kind, name)] = cls
    if kind == "orchestrator" and name not in _orch_attr_cache:
        _orch_attr_cache[name] = _derive_orch_attr_map(cls)


def discover_user_plugins():
    """扫描 user_plugins/ 目录，注册所有用户插件。

    委托给 dynamic_loader 模块，避免循环导入。
    """
    from modutsc.scheduling.dynamic_loader import discover_user_plugins as _discover
    return _discover()


def register_component_kind(kind: str):
    """注册一个新的组件类型。
    
    仅在内存中注册，不持久化。
    """
    kind = _canonical_kind(kind)
    if kind not in _KIND_CONTRACT:
        _KIND_CONTRACT[kind] = ["setup"]


def unregister_component_kind(kind: str):
    """取消注册一个组件类型。"""
    kind = _canonical_kind(kind)
    if kind in _KIND_CONTRACT:
        del _KIND_CONTRACT[kind]
