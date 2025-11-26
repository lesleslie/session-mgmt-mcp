"""Session Management MCP Server.

Provides comprehensive session management, conversation memory,
and quality monitoring for Claude Code projects.
"""

from contextlib import suppress

# Phase 2 Decomposition: New modular architecture
# These imports expose the decomposed server components
with suppress(ImportError):
    from .advanced_features import (
        AdvancedFeaturesHub,
    )
    from .core.permissions import (
        SessionPermissionsManager,
    )
    # QualityScoreResult is not directly exposed from quality_engine as it doesn't exist
    # MCPServerCore does not exist in server_core
from .utils.logging import SessionLogger

__version__ = "0.7.4"

__all__ = [
    # Advanced features
    "AdvancedFeaturesHub",
    # Core components are not directly exposed
    "SessionLogger",
    "SessionPermissionsManager",
    # Package metadata
    "__version__",
]
