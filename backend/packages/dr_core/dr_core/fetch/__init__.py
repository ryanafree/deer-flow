"""dr_core.fetch — portable structured-connector fetch layer.

Phase 1 port of the DeepResearch harness's data_fetch.py (edgar/fred/courtlistener).
Schema-independent: no claim-ledger types, importable standalone. Not wired into
DeerFlow's tool registry yet — that is Phase 3 (connectors.yaml -> extension config).
"""

from dr_core.fetch.structured import env, fetch_courtlistener, fetch_edgar, fetch_fred

__all__ = ["env", "fetch_courtlistener", "fetch_edgar", "fetch_fred"]
