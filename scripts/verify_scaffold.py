"""Run this script to verify scaffold can scan all existing ABCs.
   Usage: py scripts/verify_scaffold.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modutsc.scheduling.scaffold import _KNOWN_ABC, _load_abc, _scan_abc

all_ok = True

print("=" * 60)
print("  1. Scanning all ABCs: abstract methods detected")
print("=" * 60)

for kind, (mod_path, cls_name) in sorted(_KNOWN_ABC.items()):
    try:
        abc_cls = _load_abc(kind, None, None)
        methods = _scan_abc(abc_cls)
        print(f"\n  [{kind}]  {cls_name}  ({mod_path})")
        print(f"  abstract methods: {len(methods)}")
        for name, args, ret in methods:
            sig = f"def {name}(self, {args})"
            if ret:
                sig += ret
            print(f"    {sig}")
    except Exception as e:
        print(f"\n  ? [{kind}] FAILED: {e}")
        all_ok = False

if not all_ok:
    print("\n\n? Some ABCs failed to scan.")
    sys.exit(1)

print("\n\n" + "=" * 60)
print("  2. Registry contract vs ABC scan (contract must be subset)")
print("=" * 60)

from modutsc.scheduling.registry import _KIND_CONTRACT

for kind, contract_methods in sorted(_KIND_CONTRACT.items()):
    expected = set(contract_methods)
    mod_path, cls_name = _KNOWN_ABC[kind]
    abc_cls = _load_abc(kind, None, None)
    scanned = {m[0] for m in _scan_abc(abc_cls)}

    missing_from_abc = expected - scanned
    if missing_from_abc:
        print(f"  ? [{kind}] contract lists methods not in ABC: {missing_from_abc}")
        all_ok = False
    else:
        print(f"  ? [{kind}] contract {len(expected)} methods, ABC has {len(scanned)} total")

if all_ok:
    print("\n? All checks passed.")
    print("   - scaffold can scan every ABC and extract abstract method signatures.")
    print("   - registry contract methods are a valid subset of ABC abstract methods.")
else:
    print("\n? Some checks failed. Review _KIND_CONTRACT or ABC definitions.")
