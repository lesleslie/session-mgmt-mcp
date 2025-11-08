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
    from .quality_engine import (
        QualityEngine,
        QualityScoreResult,
    )
    from .server_core import (
        MCPServerCore,
        SessionLogger,  # type: ignore[attr-defined]
        SessionPermissionsManager,
    )

__version__ = "0.7.4"

__all__ = [
    # Advanced features
    "AdvancedFeaturesHub",
    # Core components
    "MCPServerCore",
    # Quality engine
    "QualityEngine",
    "QualityScoreResult",
    "SessionLogger",
    "SessionPermissionsManager",
    # Package metadata
    "__version__",
]
