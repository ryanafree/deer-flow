"""D11 item 3 (DECISIONS.md): deterministic value/unit/period comparison
backing the provenance audit's MATCHED decision.

Binding guard (D11 item 3 amendment, mirroring D10's refute-requires-
contradiction clamp): MISMATCH requires an AFFIRMATIVE, deterministic
contradiction after unit/scale normalization against this module's explicit
tolerance table -- normalized units must match exactly, and values compare
within a stated relative tolerance for rounding/scale artifacts (e.g. a claim
written as "$1.5 million" against a data_ref raw value of "1500000" must
normalize to the same quantity, not read as a contradiction). A claim value
-- or a data_ref value -- that cannot be parsed or normalized never produces
MISMATCH; callers must fall back to UNAUDITED. MISMATCH -> EXCLUDED is
terminal (D8), so a false MISMATCH from a scale artifact would reintroduce
the D10 failure shape as deterministic over-kill -- this module exists to
prevent that.

Pure, network-free, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Explicit scale-word/abbreviation table (magnitude multipliers) -- not
# inline magic numbers. "k"/"mm"/"bn"/"tn" are the common financial
# shorthand for thousand/million/billion/trillion; single-letter "m"/"b"/"t"
# are deliberately excluded (too ambiguous outside a financial-shorthand
# context, e.g. "m" for meters).
SCALE_MULTIPLIERS: dict[str, float] = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mm": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
}

# Explicit unit-label aliases for an EXPLICIT data_ref "unit" key (e.g. an
# EDGAR XBRL fact's "unit"). Only "percent" is distinguished as its own
# comparable unit; every other label (including "USD", missing, or
# unrecognized) normalizes to the default "count" unit that a bare parsed
# number also gets -- currency symbols are stripped as noise (per instruction),
# not used to build per-currency unit codes, since the guard's job is
# catching SCALE artifacts and percent-vs-raw-number contradictions, not FX.
_PERCENT_UNIT_ALIASES: frozenset[str] = frozenset({"%", "percent", "pct"})

DEFAULT_RELATIVE_TOLERANCE = 0.005  # 0.5%: rounding/scale-artifact slack (D11 item 3)

_CURRENCY_SYMBOLS = "$€£¥"  # $ € £ ¥ -- stripped, not unit-forming

_QUANTITY_RE = re.compile(
    r"(?P<currency>[" + re.escape(_CURRENCY_SYMBOLS) + r"])?\s*"
    r"(?P<number>-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>thousand|million|billion|trillion|mm|bn|tn|k)?"
    r"(?P<percent>%)?",
    re.IGNORECASE,
)


class Comparison(Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    INCONCLUSIVE = "inconclusive"  # unparseable, unnormalizable, or incomparable units


@dataclass(frozen=True)
class NormalizedQuantity:
    value: float
    unit: str  # "percent" | "count"


def normalize_unit_label(raw: object) -> str | None:
    """Normalize an EXPLICIT unit label (e.g. a data_ref "unit" key) to
    "percent", or None if it is not a recognized percent spelling (callers
    treat None as "no override" rather than forcing a unit)."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return "percent" if key in _PERCENT_UNIT_ALIASES else None


def parse_quantity(raw: object) -> NormalizedQuantity | None:
    """Parse a numeric value + unit out of free text (a claim's own prose)
    or a raw scalar (a data_ref's "value" field). Returns None -- never
    raises -- when no confident numeric parse is possible; callers must
    treat that as UNAUDITED, never MISMATCH (D11 item 3's binding guard).

    Recognizes an optional leading currency symbol (stripped, not
    unit-forming), a decimal number (comma thousands-separators tolerated),
    an optional scale word/abbreviation (see SCALE_MULTIPLIERS), and an
    optional trailing "%". Among multiple numeric tokens in free text (e.g.
    a claim that also mentions a fiscal year), a token carrying a currency
    symbol, scale word, or "%" is preferred over a bare number -- a bare
    number is used only when nothing in the text carries such a marker --
    so a stray "FY2024" does not get mistaken for the asserted figure when
    the real figure ("$383.285 billion") is also present.
    """
    if raw is None:
        return None
    text = str(raw)
    candidates = [m for m in _QUANTITY_RE.finditer(text) if m.group("number")]
    if not candidates:
        return None
    marked = [m for m in candidates if m.group("currency") or m.group("scale") or m.group("percent")]
    match = marked[0] if marked else candidates[0]
    try:
        number = float(match.group("number").replace(",", ""))
    except ValueError:
        return None
    scale = match.group("scale")
    if scale:
        number *= SCALE_MULTIPLIERS[scale.lower()]
    unit = "percent" if match.group("percent") else "count"
    return NormalizedQuantity(value=number, unit=unit)


def compare_quantities(claim_quantity: NormalizedQuantity | None, record_quantity: NormalizedQuantity | None, *, relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE) -> Comparison:
    """Pure comparison over two already-parsed quantities. Never raises.
    Different normalized units (e.g. a percent claim against a raw-count
    record) are INCONCLUSIVE, not MISMATCH -- the guard only affirms a
    contradiction when the two figures are the same KIND of quantity."""
    if claim_quantity is None or record_quantity is None:
        return Comparison.INCONCLUSIVE
    if claim_quantity.unit != record_quantity.unit:
        return Comparison.INCONCLUSIVE
    if record_quantity.value == 0:
        return Comparison.MATCH if claim_quantity.value == 0 else Comparison.MISMATCH
    relative_diff = abs(claim_quantity.value - record_quantity.value) / abs(record_quantity.value)
    return Comparison.MATCH if relative_diff <= relative_tolerance else Comparison.MISMATCH


def compare_claim_to_data_ref(claim_text: str, data_ref: dict, *, relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE) -> Comparison:
    """Compare a claim's own asserted figure (parsed from its text) against
    a data_ref's structured "value" (+ optional explicit "unit" override).
    INCONCLUSIVE when the record carries no parseable value at all (no
    structured record to compare against) or the claim's text carries no
    parseable figure -- both fall back to UNAUDITED in the caller, never
    MATCHED or MISMATCH."""
    record_quantity = parse_quantity(data_ref.get("value"))
    if record_quantity is not None:
        unit_override = normalize_unit_label(data_ref.get("unit"))
        if unit_override is not None:
            record_quantity = NormalizedQuantity(value=record_quantity.value, unit=unit_override)
    claim_quantity = parse_quantity(claim_text)
    return compare_quantities(claim_quantity, record_quantity, relative_tolerance=relative_tolerance)
