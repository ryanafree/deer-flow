"""tiers.py -- the single owner of dr_core's model-tier env knobs.

Three env vars select the model used by each phase of a run: ``DR_PLAN_MODEL``
(extraction/mapping), ``DR_VERIFY_MODEL`` (verify votes and the relation
review), and ``DR_SYNTH_MODEL`` (the D12 synthesize node). No other module in
``dr_core`` should read these env vars directly -- resolve through this module
instead, so the three knobs never drift into a second copy.

Defaults: ``or-sonnet`` for plan (q.16-3, moved off gpt-4o-mini), ``claude-verify``
for verify (unchanged, q.5), ``claude-top`` for synth (lands ahead of the D12
synthesize node per q.16-2). See ``dr_core/accounting.py`` for the pricing row
each name resolves to.
"""

from __future__ import annotations

import os

_DEFAULT_PLAN_MODEL = "or-sonnet"
_DEFAULT_VERIFY_MODEL = "claude-verify"
_DEFAULT_SYNTH_MODEL = "claude-top"


def plan_model_name() -> str:
    """Resolve the plan-tier model name from ``DR_PLAN_MODEL`` (default ``or-sonnet``)."""
    return os.environ.get("DR_PLAN_MODEL", _DEFAULT_PLAN_MODEL)


def verify_model_name() -> str:
    """Resolve the verify-tier model name from ``DR_VERIFY_MODEL`` (default ``claude-verify``)."""
    return os.environ.get("DR_VERIFY_MODEL", _DEFAULT_VERIFY_MODEL)


def synth_model_name() -> str:
    """Resolve the synth-tier model name from ``DR_SYNTH_MODEL`` (default ``claude-top``)."""
    return os.environ.get("DR_SYNTH_MODEL", _DEFAULT_SYNTH_MODEL)


def plan_model():
    """Construct the plan-tier chat model via ``deerflow.models.create_chat_model``."""
    from deerflow.models import create_chat_model

    return create_chat_model(plan_model_name())


def verify_model():
    """Construct the verify-tier chat model via ``deerflow.models.create_chat_model``."""
    from deerflow.models import create_chat_model

    return create_chat_model(verify_model_name())


def synth_model():
    """Construct the synth-tier chat model via ``deerflow.models.create_chat_model``."""
    from deerflow.models import create_chat_model

    return create_chat_model(synth_model_name())
