"""Restore prop34_cmp (and the other hop sets) to the propagation dependency list.

Run from the repo root:  python3 fix_prop34.py

src/features.py defines prop34_cmp but does not list it in NEEDS_PROPAGATED, so
load_prepared never attaches the p2_/p3_/p4_ columns and the set silently falls back to
162 raw+component features instead of 639. The notebook builds these columns directly and
is unaffected; this only concerns the staged pipeline in src/.
"""
import pathlib
import sys

p = pathlib.Path("src/features.py")
if not p.exists():
    sys.exit("run this from the repository root")
s = p.read_text()

WANT = ["prop3_cmp", "prop23_cmp", "prop4_cmp", "prop34_cmp", "prop4only_cmp"]
if "NEEDS_PROPAGATED" not in s:
    sys.exit("NEEDS_PROPAGATED not found in src/features.py")

missing = [w for w in WANT if f'"{w}"' not in s.split("NEEDS_PROPAGATED")[1].split("}")[0]]
if not missing:
    print("nothing to do: all hop sets already listed")
else:
    anchor = "NEEDS_PROPAGATED"
    line_start = s.index(anchor)
    line_end = s.index("}", line_start) + 1
    old = s[line_start:line_end]
    add = ", ".join(f'"{w}"' for w in missing)
    new = old[:-1].rstrip() + (", " if old[:-1].rstrip().endswith(('"', "'")) else "") + add + "}"
    s = s.replace(old, new, 1)
    p.write_text(s)
    print("added to NEEDS_PROPAGATED:", ", ".join(missing))
