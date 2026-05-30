"""Scaffold a new plugin implementation from an ABC.

Usage:
  py -m modutsc run <config_path>                                 (launch experiment)
  py -m modutsc.scheduling.scaffold --yaml <output_path>           (generate config)
      --env <name> --observer <name> --actor <name>
      --reward <name> --collector <name> --algorithm <name>
      --orchestrator <name>
  py -m modutsc.scheduling.scaffold --init-kind <kind>            (create new ABC)
  py -m modutsc.scheduling.scaffold <kind> <name>                 (create impl)
  py -m modutsc.scheduling.scaffold <kind> <name>                 (custom ABC)
      --abc-module <module> --abc-class <class>

Examples:
  py -m modutsc run configs/dqn_monaco.yaml
  py -m modutsc.scheduling.scaffold --yaml configs/my_exp.yaml \\
      --env sumo --observer standard --actor phase \\
      --reward composite --collector replay --algorithm dqn \\
      --orchestrator single
  py -m modutsc.scheduling.scaffold --init-kind predictor
  py -m modutsc.scheduling.scaffold observer my_lidar
  py -m modutsc.scheduling.scaffold predictor lstm
      --abc-module modutsc.plugins.predictors
      --abc-class TrafficPredictor
"""

import ast
import inspect
import importlib
import sys
from pathlib import Path

_KNOWN_ABC = {
    "observer":     ("modutsc.plugins.observers", "Observer"),
    "actor":        ("modutsc.plugins.actors", "Actor"),
    "reward":       ("modutsc.plugins.rewards", "Reward"),
    "collector":    ("modutsc.plugins.collectors", "Collector"),
    "algorithm":    ("modutsc.plugins.algorithms", "Algorithm"),
    "orchestrator": ("modutsc.orchestration", "Orchestrator"),
    "environment":  ("modutsc.env", "Env"),
    "tracker":      ("modutsc.plugins.trackers", "Tracker"),
}

_KNOWN_OUTDIR = {
    "observer":     "modutsc/plugins/observers",
    "actor":        "modutsc/plugins/actors",
    "reward":       "modutsc/plugins/rewards",
    "collector":    "modutsc/plugins/collectors",
    "algorithm":    "modutsc/plugins/algorithms",
    "orchestrator": "modutsc/orchestration",
    "environment":  "modutsc/env",
    "tracker":      "modutsc/plugins/trackers",
}

_TYPING_IMPORTS = {"List", "Dict", "Any", "Optional", "Tuple", "Union"}


def _load_abc(kind, abc_module, abc_class):
    if abc_module and abc_class:
        mod = importlib.import_module(abc_module)
        return getattr(mod, abc_class)

    if kind in _KNOWN_ABC:
        mod_path, cls_name = _KNOWN_ABC[kind]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)

    raise ValueError(
        f"Unknown kind '{kind}'. For custom kinds, provide --abc-module and --abc-class."
    )


def _scan_abc(abc_cls):
    methods = []
    for name, value in abc_cls.__dict__.items():
        if not getattr(value, '__isabstractmethod__', False):
            continue
        func = getattr(value, '__func__', value)
        args_str = ""
        ret_str = ""
        try:
            sig = inspect.signature(func)
            params = []
            for p_name, p in sig.parameters.items():
                if p_name == 'self':
                    continue
                if p.kind == inspect.Parameter.VAR_POSITIONAL:
                    params.append(f'*{p_name}')
                elif p.kind == inspect.Parameter.VAR_KEYWORD:
                    params.append(f'**{p_name}')
                elif p.default is inspect.Parameter.empty:
                    params.append(p_name)
                else:
                    params.append(f"{p_name}={repr(p.default)}")
            args_str = ', '.join(params)
            ret_str = _format_return(sig.return_annotation)
        except Exception:
            ret_str = ' -> None'
        methods.append((name, args_str, ret_str))
    return methods


def _format_return(annotation) -> str:
    if annotation is None or annotation is inspect.Parameter.empty:
        return ''
    if annotation is type(None):
        return ' -> None'
    origin = getattr(annotation, '__origin__', None)
    if origin is not None:
        args = getattr(annotation, '__args__', ())
        if origin.__name__ in ('Union',) and len(args) == 2 and type(None) in args:
            inner = args[0] if args[0] is not type(None) else args[1]
            return f" -> Optional[{_type_basename(inner)}]"
        n = origin.__name__
        if n in ('dict',):
            return ' -> dict'
        if n in ('list', 'set', 'tuple', 'frozenset'):
            inner = ', '.join(_type_basename(a) for a in args)
            return f" -> {n}[{inner}]"
        args_str = ', '.join(_type_basename(a) for a in args)
        return f" -> {origin.__name__}[{args_str}]"
    name = getattr(annotation, '__name__', None)
    if name:
        return f' -> {name}'
    return ' -> None'


def _type_basename(t):
    if t is type(None):
        return 'None'
    name = getattr(t, '__name__', None)
    return name if name else str(t)


def _generate_code(kind, name, abc_cls, methods):
    cls_name = ''.join(w.capitalize() for w in name.split('_'))
    cls_name += abc_cls.__name__
    abc_mod = abc_cls.__module__
    abc_name = abc_cls.__name__

    lines = []
    lines.append("from typing import List, Dict, Any, Optional")
    lines.append("")
    lines.append(f"from {abc_mod} import {abc_name}")
    lines.append("from modutsc.scheduling.registry import register")
    lines.append("")
    lines.append("")
    lines.append(f'@register("{kind}", "{name}")')
    lines.append(f"class {cls_name}({abc_name}):")
    lines.append(f'    """TODO: implement this {kind}"""')
    lines.append("")

    for method_name, args_str, ret_str in methods:
        ret_ann = f"{ret_str}" if ret_str else ""
        sig = f"{args_str})" if args_str else ")"
        lines.append(f"    def {method_name}(self, {sig}{ret_ann}:")
        lines.append("        pass")
        lines.append("")

    return '\n'.join(lines)


def _output_path(kind, name, project_root, abc_module):
    if kind in _KNOWN_OUTDIR:
        rel = _KNOWN_OUTDIR[kind]
    else:
        rel = abc_module.replace('.', '/') if abc_module else f"modutsc/plugins/{kind}s"
    return project_root / rel / f"{name}.py"


def _parse_args(argv):
    args = {"abc_module": None, "abc_class": None}
    i = 0
    positional = []
    while i < len(argv):
        a = argv[i]
        if a == "--abc-module":
            i += 1
            args["abc_module"] = argv[i] if i < len(argv) else None
        elif a == "--abc-class":
            i += 1
            args["abc_class"] = argv[i] if i < len(argv) else None
        else:
            positional.append(a)
        i += 1
    if len(positional) < 2:
        raise SystemExit(
            "Usage: py -m modutsc.scheduling.scaffold <kind> <name> "
            "[--abc-module <module>] [--abc-class <class>]"
        )
    return positional[0], positional[1], args


def _init_kind(kind, project_root):
    if kind in _KNOWN_ABC:
        raise SystemExit(
            f"'{kind}' ????????????????? ABC ??????:\n"
            f"  {_KNOWN_ABC[kind][0]} -> {_KNOWN_ABC[kind][1]}\n"
            f"  ???: py -m modutsc.scheduling.scaffold {kind} <name>\n"
            f"  ????????????????"
        )

    folder_name = kind + "s"
    out_dir = project_root / "modutsc" / "plugins" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "__init__.py"
    cls_name = ''.join(w.capitalize() for w in kind.split('_'))

    if out_path.exists():
        raise SystemExit(
            f"ABC ????????: {out_path}\n"
            f"  ????????????????????:\n"
            f"  py -m modutsc.scheduling.scaffold {kind} <name> \\\n"
            f"      --abc-module modutsc.plugins.{folder_name} \\\n"
            f"      --abc-class {cls_name}"
        )

    lines = [
        "from abc import ABC, abstractmethod",
        "from typing import List, Dict, Any",
        "",
        "",
        f"class {cls_name}(ABC):",
        f'    """TODO: ???? {kind} ????????"""',
        "",
        "    @abstractmethod",
        "    def setup(self, cfg: dict, env=None) -> None:",
        '        """?? YAML config ?????, env ?? SUMO ???? """',
        "        ...",
        "",
        "    # TODO: ???????? @abstractmethod ???????????",
        "    # ??:",
        "    # @abstractmethod",
        "    # def predict(self, state) -> list: ...",
        "",
    ]
    code = '\n'.join(lines)
    out_path.write_text(code, encoding="utf-8")
    print(f"? ??????: {out_path}")
    print()
    print("   ?????:")
    print("   1. ??????????? @abstractmethod ???????????")
    print("   2. ???????????????????:")

    abc_module = f"modutsc.plugins.{folder_name}"
    print(f"      py -m modutsc.scheduling.scaffold {kind} <name> \\")
    print(f"          --abc-module {abc_module} \\")
    print(f"          --abc-class {cls_name}")


def _ast_to_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_ast_to_value(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for k, v in zip(node.keys, node.values):
            key = _ast_to_value(k) if k else None
            result[key] = _ast_to_value(v)
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant):
            return -node.operand.value
    try:
        return f"<{ast.unparse(node)}>"
    except Exception:
        return "<expr>"


def _scan_cfg_calls(func_node):
    params = {}
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'cfg'
                and node.func.attr == 'get'):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        key = node.args[0].value
        if not isinstance(key, str):
            continue
        if len(node.args) > 1:
            default = _ast_to_value(node.args[1])
            if default == '<expr>':
                default = None
        else:
            default = None
        params[key] = default
    return params


def _extract_config_params(cls):
    try:
        src_file = inspect.getfile(cls)
    except TypeError:
        return {}
    try:
        with open(src_file, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, OSError):
        return {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            params = {}
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef):
                    params.update(_scan_cfg_calls(item))
            return params
    return {}


def _enrich_config_defaults(cls, params):
    defaults = getattr(cls, '__config_defaults__', None)
    if defaults is None:
        return
    if callable(defaults):
        try:
            defaults = defaults()
        except Exception:
            return
    if not isinstance(defaults, dict):
        return
    features = params.get("features")
    if isinstance(features, list):
        filtered = {f: defaults[f] for f in features if f in defaults}
        if filtered:
            params["norm"] = filtered
    else:
        for rk, rv in defaults.items():
            if rk not in params:
                params[rk] = rv


def _format_yaml_value(v, indent=0):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if v == int(v) and isinstance(v, float) and v != float('inf'):
            return repr(v)
        return repr(v) if isinstance(v, float) else str(v)
    if isinstance(v, str) and v.startswith('<'):
        return repr(v) + "  # ??????"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(isinstance(x, (int, float, str, bool)) for x in v):
            items = ", ".join(_format_yaml_value(x) for x in v)
            return f"[{items}]"
        inner = "\n".join(
            f"{'  ' * (indent + 1)}- {_format_yaml_value(x, indent + 1)}"
            for x in v
        )
        return f"\n{inner}"
    if isinstance(v, dict):
        if not v:
            return "{}"
        inner = "\n".join(
            f"{'  ' * (indent + 1)}{k}: {_format_yaml_value(val, indent + 1)}"
            for k, val in v.items()
        )
        return f"\n{inner}"
    return str(v)


def _params_to_yaml(params, base_indent):
    lines = []
    for key, value in params.items():
        val_str = _format_yaml_value(value, base_indent + 1)
        if val_str.startswith('\n'):
            lines.append(f"{key}:{val_str}")
        else:
            lines.append(f"{key}: {val_str}")
    return lines


def _generate_yaml(output_path, selections, project_root):
    from modutsc.scheduling.registry import find, discover
    from modutsc.scheduling.dataset_index import load_index
    sys.path.insert(0, str(project_root))
    discover()

    lines = []
    lines.append(f"experiment:")
    lines.append(f"  name: \"{output_path.stem}\"")
    lines.append(f"  seed: 42")
    lines.append("")

    components = {}
    kinds = []
    for kind, name in selections.values():
        components[kind] = [name]
        kinds.append((kind, name))
    lines.append("components:")
    for k, v in components.items():
        lines.append(f"  {k}: [{', '.join(v)}]")
    lines.append("")

    topo_params = {}
    ds_index_path = project_root / "data" / "datasets_index.yaml"
    if ds_index_path.exists():
        ds_index = load_index(str(ds_index_path))
        if ds_index:
            first_ds = next(iter(ds_index.values()), {})
            topo_params = {k: v for k, v in first_ds.items() if k in ("num_phase", "max_lanelinks", "num_tsc")}
            if topo_params:
                print(f"[scaffold] Using topology from datasets_index.yaml: {topo_params}")

    env_config = {}
    env_kind = selections.get("environment")
    if env_kind and topo_params:
        sample_ds = next(iter(ds_index.keys()), "") if ds_index else ""
        if sample_ds:
            env_config["roadnet_file"] = f"data/{sample_ds}"
        env_config.update({k: v for k, v in topo_params.items() if k != "num_tsc"})

    for kind, name in kinds:
        cls = find(kind, name)
        if cls is None:
            lines.append(f"  # WARNING: {kind}/{name} not found in registry")
            continue
        params = _extract_config_params(cls)
        _enrich_config_defaults(cls, params)

        if kind == "environment":
            params.update(env_config)
            if "roadnet_file" in params:
                params["flow_file"] = params["roadnet_file"].replace("roadnet.net.xml", "flow_0.rou.xml")
                params["gui"] = False
                params["sim_max_time"] = 3600
                params["decision_interval"] = 5
                params["yellow_duration"] = 3
                params["min_green"] = 5

        elif kind in ("observer", "algorithm") and topo_params:
            for k in ("num_phase", "max_lanelinks"):
                if k in topo_params and k not in params:
                    params[k] = topo_params[k]

        if kind == "algorithm":
            lines.append("algorithm:")
            lines.append(f"  - plugin: \"{name}\"")
            if params:
                lines.append("    config:")
                for pline in _params_to_yaml(params, 2):
                    lines.append(f"      {pline}")
            else:
                lines.append("    config: {}")
        else:
            lines.append(f"{kind}:")
            lines.append(f"  plugin: \"{name}\"")
            if params:
                lines.append(f"  config:")
                for pline in _params_to_yaml(params, 1):
                    lines.append(f"    {pline}")
            else:
                lines.append(f"  config: {{}}")
        lines.append("")

    lines.append("training:")
    lines.append("  warmup_steps: 200")
    lines.append("  num_epochs: 5")
    lines.append("  episodes_per_epoch: 2")
    lines.append("  # hooks:                         # ????????????????????")
    lines.append("  #   after_episode:")
    lines.append("  #     - method: custom_log")
    lines.append("  #       every: 50")
    lines.append("  #   after_epoch:")
    lines.append("  #     - method: meta_update")
    lines.append("  #       every: 3")
    lines.append("")
    lines.append("evaluation:")
    lines.append("  eval_frequency: 5")
    lines.append("  eval_steps: 200")
    lines.append("  checkpoint_dir: \"checkpoints/\"")
    lines.append("")

    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding="utf-8")
    print(f"? ??????: {output_path}")
    print(f"   ???? {len(selections)} ???????????")


def _print_config_section(cls, kind, name):
    params = _extract_config_params(cls)
    _enrich_config_defaults(cls, params)
    if not params:
        return
    print()
    print(f"   ???????? (???? YAML ?? {kind}.config):")
    for pname, pval in params.items():
        val_str = _format_yaml_value(pval, 0)
        print(f"     {pname}: {val_str}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        raise SystemExit(
            "Usage: py -m modutsc.scheduling.scaffold <kind> <name>\n"
            "       py -m modutsc.scheduling.scaffold --init-kind <kind>\n"
            "       py -m modutsc.scheduling.scaffold --yaml <output> "
            "--env ... --observer ... --actor ... --reward ... --collector ... "
            "--algorithm ... --orchestrator ..."
        )

    if "--init-kind" in argv:
        idx = argv.index("--init-kind")
        kind = argv[idx + 1] if idx + 1 < len(argv) else None
        if not kind:
            raise SystemExit("Usage: py -m modutsc.scheduling.scaffold --init-kind <kind>")
        project_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))
        _init_kind(kind, project_root)
        return

    if "--yaml" in argv:
        _cmd_yaml(argv)
        return

    kind, name, opts = _parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

    abc_cls = _load_abc(kind, opts["abc_module"], opts["abc_class"])
    methods = _scan_abc(abc_cls)

    if not methods:
        raise SystemExit(f"'{abc_cls.__name__}' ???????????")

    code = _generate_code(kind, name, abc_cls, methods)
    out_path = _output_path(kind, name, project_root, opts["abc_module"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(code, encoding="utf-8")
    print(f"? ??????: {out_path}")
    print(f"   kind: {kind}  name: {name}  inherits: {abc_cls.__name__}")
    print(f"   ??????:")
    for m_name, m_args, m_ret in methods:
        print(f"     def {m_name}(self, {m_args}){m_ret}")

    if kind in _KNOWN_ABC:
        from modutsc.scheduling.registry import find
        cls = find(kind, name)
        if cls:
            _print_config_section(cls, kind, name)


def _cmd_yaml(argv):
    from modutsc.scheduling.registry import find, discover
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    discover()

    selections = {}
    flags = {
        "--env": "environment", "--observer": "observer",
        "--actor": "actor", "--reward": "reward",
        "--collector": "collector", "--algorithm": "algorithm",
        "--orchestrator": "orchestrator",
    }
    output_path = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--yaml":
            i += 1
            output_path = Path(argv[i]) if i < len(argv) else None
        elif a == "--plugin":
            i += 1; k = argv[i] if i < len(argv) else ""
            i += 1; name = argv[i] if i < len(argv) else ""
            if k and name:
                selections[k] = (k, name)
        elif a in flags:
            i += 1
            k = flags[a]
            name = argv[i] if i < len(argv) else ""
            if k not in selections:
                selections[k] = (k, name)
        i += 1

    if not output_path:
        raise SystemExit("Usage: py -m modutsc.scheduling.scaffold --yaml <output_path> ...")
    if not selections:
        raise SystemExit("???????????????????: --env <name> --observer <name> ...")

    _generate_yaml(output_path, selections, project_root)


if __name__ == "__main__":
    main()
