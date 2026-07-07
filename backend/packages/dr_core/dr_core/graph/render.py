"""Render node — STUB (D5 walking skeleton).

# TODO(phase2-logic): call dr_core.render.render_report over the frozen gated ledger.
"""

from __future__ import annotations

import os
import tempfile


def render_node(state) -> dict:
    """Write a marker file proving the render node ran; no state change.

    Resolves a run directory from ``state["dr_run"]["run_dir"]`` when present
    (the real render node will resolve the actual run directory the same
    way); falls back to a temp directory so the stub is exercisable without a
    live run.
    """
    dr_run = state.get("dr_run") or {}
    run_dir = dr_run.get("run_dir") or tempfile.gettempdir()
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "dr_render_marker.txt"), "w") as f:
        f.write("dr_render stub ran\n")
    return {}
