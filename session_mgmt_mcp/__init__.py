"""Session Management MCP Server.

Provides comprehensive session management, conversation memory,
and quality monitoring for Claude Code projects.
"""

# Phase 2 Decomposition: New modular architecture
# These imports expose the decomposed server components
try:
    from .advanced_features import (
        AdvancedFeaturesHub,
    )
    from .quality_engine import (
        QualityEngine,
        QualityScoreResult,
    )
    from .server_core import (
        MCPServerCore,
        SessionLogger,
        SessionPermissionsManager,
    )
except ImportError:
    # Modules not yet fully implemented - skeletons only
    pass

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
