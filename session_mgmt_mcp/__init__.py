"""Session Management MCP Server.

Provides comprehensive session management, conversation memory,
and quality monitoring for Claude Code projects.
"""

# Phase 2 Decomposition: New modular architecture
# These imports expose the decomposed server components
try:
    from .server_core import (
        MCPServerCore,
        SessionLogger,
        SessionPermissionsManager,
    )
    from .quality_engine import (
        QualityEngine,
        QualityScoreResult,
    )
    from .advanced_features import (
        AdvancedFeaturesHub,
    )
except ImportError:
    # Modules not yet fully implemented - skeletons only
    pass

__version__ = "0.7.4"

__all__ = [
    # Core components
    "MCPServerCore",
    "SessionLogger",
    "SessionPermissionsManager",
    # Quality engine
    "QualityEngine",
    "QualityScoreResult",
    # Advanced features
    "AdvancedFeaturesHub",
    # Package metadata
    "__version__",
]
