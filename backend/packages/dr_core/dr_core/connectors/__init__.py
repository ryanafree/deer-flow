"""connectors/ — dr_core's connector registry, config generator, and reachability probe.

Source of truth for the closed-enum connector rows is the vendored
``connectors.yaml`` in this package (see its PROVENANCE header); the frozen
upstream copy at ``~/Documents/Projects/DeepResearch/harness/connectors.yaml``
is mined but never edited by this code.
"""

from dr_core.connectors.registry import Connector, ConnectorValidationError, load_connectors
from dr_core.connectors.tool_map import TOOL_TO_CONNECTORS, connectors_for_tool, register_tool

__all__ = [
    "Connector",
    "ConnectorValidationError",
    "load_connectors",
    "TOOL_TO_CONNECTORS",
    "connectors_for_tool",
    "register_tool",
]
