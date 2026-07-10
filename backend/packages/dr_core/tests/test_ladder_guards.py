"""Direct unit tests for the CitationStatus / DataProvenance advance-only ladder
guards (D2), the siblings of assert_verification_transition_allowed added for B5
(see test_publication_status.py for the verification guard's own direct tests).
Exercises the full 3x3 transition table for each ladder.
"""

from __future__ import annotations

import pytest
from dr_core.models.derive import (
    assert_citation_transition_allowed,
    assert_provenance_transition_allowed,
)
from dr_core.models.enums import CitationStatus, DataProvenance

_CITATION_LEGAL = {
    (CitationStatus.UNRESOLVED, CitationStatus.UNRESOLVED),
    (CitationStatus.UNRESOLVED, CitationStatus.RESOLVED),
    (CitationStatus.UNRESOLVED, CitationStatus.NOT_FOUND),
    (CitationStatus.RESOLVED, CitationStatus.RESOLVED),
    (CitationStatus.NOT_FOUND, CitationStatus.NOT_FOUND),
}

_PROVENANCE_LEGAL = {
    (DataProvenance.UNAUDITED, DataProvenance.UNAUDITED),
    (DataProvenance.UNAUDITED, DataProvenance.MATCHED),
    (DataProvenance.UNAUDITED, DataProvenance.MISMATCH),
    (DataProvenance.MATCHED, DataProvenance.MATCHED),
    (DataProvenance.MISMATCH, DataProvenance.MISMATCH),
}


@pytest.mark.parametrize("current", list(CitationStatus))
@pytest.mark.parametrize("new", list(CitationStatus))
def test_citation_transition_full_table(current: CitationStatus, new: CitationStatus):
    if (current, new) in _CITATION_LEGAL:
        assert_citation_transition_allowed(current, new)  # must not raise
    else:
        with pytest.raises(ValueError):
            assert_citation_transition_allowed(current, new)


@pytest.mark.parametrize("current", list(DataProvenance))
@pytest.mark.parametrize("new", list(DataProvenance))
def test_provenance_transition_full_table(current: DataProvenance, new: DataProvenance):
    if (current, new) in _PROVENANCE_LEGAL:
        assert_provenance_transition_allowed(current, new)  # must not raise
    else:
        with pytest.raises(ValueError):
            assert_provenance_transition_allowed(current, new)
