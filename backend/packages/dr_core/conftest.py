"""Make dr_core (and the sibling deerflow harness) importable under a bare `uv run pytest`.

On this machine `uv` marks the editable-install `.pth` files UF_HIDDEN, and CPython's
`site.py` skips hidden `.pth` files, so neither `dr_core` nor `deerflow` lands on `sys.path`
via the editable install. Inserting the source roots here lets `pytest packages/dr_core/`
collect without a `PYTHONPATH=` prefix. This file lives inside our additive `dr_core`
package, so it is NOT a DeerFlow core edit (no FORK_DELTA entry).
"""

import sys
from pathlib import Path

_dr_core_pkg_root = Path(__file__).resolve().parent  # packages/dr_core (holds dr_core/)
_harness_root = _dr_core_pkg_root.parent / "harness"  # packages/harness (holds deerflow/)

for _p in (_dr_core_pkg_root, _harness_root):
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)
