"""D13 (DECISIONS.md) -- append-only Source.snapshots storage, ratifying
Option 1 of build-logs/fable-options-memo-claim-grounding-snapshots-2026-07-18.md:
deterministic snapshot id, append-only union merge (idempotent on identical
content, raises on divergent content under the same id).

HERMETIC: pure model + reducer tests, no network, no graph execution.
"""

from datetime import UTC, datetime

import pytest
from dr_core.graph.state import merge_ledger
from dr_core.models import Snapshot, Source, compute_snapshot_id


class TestComputeSnapshotId:
    def test_deterministic_for_same_inputs(self):
        a = compute_snapshot_id("src1", "call1", "normalized text")
        b = compute_snapshot_id("src1", "call1", "normalized text")
        assert a == b

    def test_differs_when_any_input_differs(self):
        base = compute_snapshot_id("src1", "call1", "text")
        assert compute_snapshot_id("src2", "call1", "text") != base
        assert compute_snapshot_id("src1", "call2", "text") != base
        assert compute_snapshot_id("src1", "call1", "other text") != base


class TestSourceSnapshotsDefaultEmpty:
    def test_source_without_snapshots_arg_dumps_empty_list(self):
        source = Source(id="s1", url_or_id="https://example.com", source_system="web", authority_tier=3, retrieved_at=datetime.now(UTC))
        dumped = source.model_dump(mode="json")
        assert dumped["snapshots"] == []


class TestSnapshotMergeUnion:
    """Exercises the REAL merge_ledger reducer (graph/state.py), which is
    what the graph folds C2-hook updates through -- the same path
    test_dr_source_hook.py's TestCrossMessageUrlDedup drives for web
    sources."""

    _FIXED_RETRIEVED_AT = datetime(2026, 7, 18, tzinfo=UTC)

    def _source_dict(self, source_id: str, snapshots: list[dict]) -> dict:
        # A fixed retrieved_at so re-building the base Source dict for
        # "existing" vs "incoming" never itself looks like a divergent write
        # on that field -- only `snapshots` is meant to vary between calls.
        source = Source(
            id=source_id,
            url_or_id="wrds://crsp-compustat/AAPL/2023",
            source_system="wrds",
            authority_tier=1,
            retrieved_at=self._FIXED_RETRIEVED_AT,
        )
        dumped = source.model_dump(mode="json")
        dumped["snapshots"] = snapshots
        return dumped

    def _snapshot(self, source_id: str, tool_call_id: str, data_ref: dict) -> dict:
        import json

        normalized = json.dumps(data_ref, sort_keys=True)
        snap = Snapshot(
            snapshot_id=compute_snapshot_id(source_id, tool_call_id, normalized),
            source_id=source_id,
            tool_call_id=tool_call_id,
            data_ref=data_ref,
            retrieved_at=datetime.now(UTC),
        )
        return snap.model_dump(mode="json")

    def test_second_sighting_with_new_snapshot_appends_not_replaces(self):
        source_id = "src1"
        snap1 = self._snapshot(source_id, "call1", {"period": "2023"})
        snap2 = self._snapshot(source_id, "call2", {"period": "2024"})

        existing = {source_id: self._source_dict(source_id, [snap1])}
        incoming = {source_id: self._source_dict(source_id, [snap2])}

        merged = merge_ledger(existing, incoming)

        assert len(merged[source_id]["snapshots"]) == 2
        ids = {s["snapshot_id"] for s in merged[source_id]["snapshots"]}
        assert ids == {snap1["snapshot_id"], snap2["snapshot_id"]}

    def test_identical_snapshot_reapplied_is_idempotent_no_op(self):
        source_id = "src1"
        snap1 = self._snapshot(source_id, "call1", {"period": "2023"})

        existing = {source_id: self._source_dict(source_id, [snap1])}
        incoming = {source_id: self._source_dict(source_id, [snap1])}

        merged = merge_ledger(existing, incoming)

        assert merged[source_id]["snapshots"] == [snap1]

    def test_same_snapshot_id_divergent_content_raises(self):
        source_id = "src1"
        snap1 = self._snapshot(source_id, "call1", {"period": "2023"})
        # Same id, but forge different content (should be structurally
        # impossible in real code since the id is a content hash -- this
        # simulates a data-integrity bug and confirms the reducer catches it).
        tampered = dict(snap1)
        tampered["data_ref"] = {"period": "2099"}

        existing = {source_id: self._source_dict(source_id, [snap1])}
        incoming = {source_id: self._source_dict(source_id, [tampered])}

        with pytest.raises(ValueError, match="divergent snapshot content"):
            merge_ledger(existing, incoming)

    def test_source_with_no_prior_snapshots_key_accepts_first_write(self):
        # Simulates a Source dict that predates snapshots (or omitted the
        # key entirely) merging against an incoming record that has one --
        # old.get(key, _MISSING) path in _merge_record.
        source_id = "src1"
        snap1 = self._snapshot(source_id, "call1", {"period": "2023"})

        old_no_snapshots = self._source_dict(source_id, [])
        del old_no_snapshots["snapshots"]
        existing = {source_id: old_no_snapshots}
        incoming = {source_id: self._source_dict(source_id, [snap1])}

        merged = merge_ledger(existing, incoming)

        assert merged[source_id]["snapshots"] == [snap1]
