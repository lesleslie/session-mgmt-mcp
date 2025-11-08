from __future__ import annotations

import tempfile
import typing as t
from contextlib import suppress
from datetime import datetime
from pathlib import Path

from acb.depends import depends

if t.TYPE_CHECKING:
    from session_mgmt_mcp.core import (
        SessionLifecycleManager as SessionLifecycleManagerT,  # noqa: F401
    )
    from session_mgmt_mcp.server_core import (
        SessionPermissionsManager as SessionPermissionsManagerT,  # noqa: F401
    )
    from session_mgmt_mcp.utils.logging import (  # type: ignore[attr-defined]
        SessionLogger as SessionLoggerT,  # noqa: F401
    )

from .config import SessionPaths
from .constants import CLAUDE_DIR_KEY, COMMANDS_DIR_KEY, LOGS_DIR_KEY

_configured = False


def configure(*, force: bool = False) -> None:
    """Register default dependencies for the session-mgmt MCP stack.

    This function sets up the dependency injection container with type-safe
    configuration and singleton instances for the session management system.

    Args:
        force: If True, re-registers all dependencies even if already configured.
               Used primarily for testing to reset singleton state.

    Example:
        >>> from session_mgmt_mcp.di import configure
        >>> configure()  # First call registers dependencies
        >>> configure()  # Subsequent calls are no-ops unless force=True
        >>> configure(force=True)  # Re-registers all dependencies

    """
    global _configured
    if _configured and not force:
        return

    # Register type-safe path configuration
    paths = SessionPaths.from_home()
    paths.ensure_directories()
    depends.set(SessionPaths, paths)

    # Register services with type-safe path access
    _register_logger(paths.logs_dir, force)
    _register_session_logger(paths.logs_dir, force)  # Register SessionLogger
    _register_permissions_manager(paths.claude_dir, force)
    _register_lifecycle_manager(force)

    # Register ACB adapters (Phase 2.7: DuckDB migration)
    _register_vector_adapter(paths, force)
    _register_graph_adapter(paths, force)  # Fixed: ACB ssl_enabled bug resolved

    _configured = True


def reset() -> None:
    """Reset dependencies to defaults."""
    # Reset singleton instances that have class-level state
    with suppress(ImportError, AttributeError):
        from session_mgmt_mcp.server_core import SessionPermissionsManager

        SessionPermissionsManager.reset_singleton()

    configure(force=True)


def _register_logger(logs_dir: Path, force: bool) -> None:
    """Register ACB logger adapter with the given logs directory.

    Args:
        logs_dir: Directory for session log files
        force: If True, re-registers even if already registered

    Note:
        Uses ACB's logger adapter system which automatically selects
        the best available logger (loguru, logly, or structlog).

    """
    from acb.adapters import import_adapter

    # Import ACB's Logger class (returns the class, not instance)
    logger_class = import_adapter("logger")

    if not force:
        with suppress(KeyError, AttributeError, RuntimeError):
            # RuntimeError: when adapter requires async (re-register)
            existing = depends.get_sync(logger_class)
            # Only skip if we already have a Logger instance (not just the module name string)
            if isinstance(existing, logger_class):
                return

    # Create logger instance (ACB logger takes no init args)
    logger_instance = logger_class()

    # Configure logger with file sink, falling back to a temp directory when
    # the default home-based location is not writable (e.g., sandboxed tests).
    log_file = logs_dir / f"session_management_{datetime.now().strftime('%Y%m%d')}.log"
    try:
        logger_instance.add(
            str(log_file),
            level="INFO",
            rotation="1 day",
            retention="7 days",
            compression="gz",
        )
    except Exception:
        # Fallback: use a temp logs directory under the system temp path
        tmp_logs = Path(tempfile.gettempdir()) / "session-mgmt-mcp" / "logs"
        tmp_logs.mkdir(parents=True, exist_ok=True)
        depends.set(LOGS_DIR_KEY, tmp_logs)
        log_file = (
            tmp_logs / f"session_management_{datetime.now().strftime('%Y%m%d')}.log"
        )
        logger_instance.add(
            str(log_file),
            level="INFO",
            rotation="1 day",
            retention="7 days",
            compression="gz",
        )

    # Register the instance using the class we imported
    depends.set(logger_class, logger_instance)


def _register_session_logger(logs_dir: Path, force: bool) -> None:
    """Register SessionLogger with the given logs directory.

    Args:
        logs_dir: Directory for session log files
        force: If True, re-registers even if already registered

    """
    from session_mgmt_mcp.utils.logging import SessionLogger

    if not force:
        with suppress(Exception):
            existing = depends.get_sync(SessionLogger)
            if isinstance(existing, SessionLogger):
                return

    # Create SessionLogger instance with fallback to temp logs if needed
    try:
        session_logger = SessionLogger(logs_dir)
    except Exception:
        tmp_logs = Path(tempfile.gettempdir()) / "session-mgmt-mcp" / "logs"
        tmp_logs.mkdir(parents=True, exist_ok=True)
        depends.set(LOGS_DIR_KEY, tmp_logs)
        session_logger = SessionLogger(tmp_logs)
    depends.set(SessionLogger, session_logger)


def _register_vector_adapter(paths: SessionPaths, force: bool) -> None:
    """Register ACB vector adapter with DuckDB backend.

    Args:
        paths: Session paths configuration with data directory
        force: If True, re-registers even if already registered

    Note:
        Uses ACB's vector adapter with VSS extension for semantic search.
        Configured for 384-dimensional embeddings (all-MiniLM-L6-v2 model).

    """
    # Import ACB's Vector adapter and settings
    from acb.adapters.vector.duckdb import Vector, VectorSettings
    from acb.config import Config

    if not force:
        with suppress(KeyError, AttributeError, RuntimeError):
            existing = depends.get_sync(Vector)
            if isinstance(existing, Vector):
                return

    # Get Config singleton and ensure it's initialized
    config = depends.get_sync(Config)
    config.ensure_initialized()

    # Create settings for vector adapter
    settings = VectorSettings(
        database_path=str(paths.data_dir / "reflection.duckdb"),
        default_dimension=384,  # all-MiniLM-L6-v2 embeddings
        default_distance_metric="cosine",
        enable_vss=True,  # Enable DuckDB VSS extension
        threads=4,
        memory_limit="2GB",
    )

    # Register settings in Config object
    config.vector = settings  # type: ignore[attr-defined]

    # Create adapter instance and manually set config (ACB test pattern)
    vector_adapter = Vector()
    vector_adapter.config = config  # Override _DependencyMarker with actual Config

    # Set logger from DI (adapters need a logger instance)
    try:
        from acb.adapters import import_adapter

        logger_class = import_adapter("logger")
        logger_instance = depends.get_sync(logger_class)
        vector_adapter.logger = logger_instance
    except Exception:
        # If logger not available, create a minimal print-based logger
        import logging

        vector_adapter.logger = logging.getLogger("acb.vector")

    # Initialize adapter to create vectors schema (ACB requirement)
    # Note: Initialization is deferred - schema creation happens on first use
    # This avoids event loop conflicts when configure() is called from async contexts
    depends.set(Vector, vector_adapter)

    # Mark that initialization is needed
    vector_adapter._schema_initialized = False


def _register_graph_adapter(paths: SessionPaths, force: bool) -> None:
    """Register ACB graph adapter with DuckDB PGQ backend.

    Args:
        paths: Session paths configuration with data directory
        force: If True, re-registers even if already registered

    Note:
        Uses ACB's graph adapter with DuckPGQ extension for property graphs.
        Maintains compatibility with existing kg_entities/kg_relationships schema.

    """
    # Import ACB's Graph adapter and settings
    from acb.adapters.graph.duckdb_pgq import DuckDBPGQSettings, Graph
    from acb.config import Config

    if not force:
        with suppress(KeyError, AttributeError, RuntimeError):
            existing = depends.get_sync(Graph)
            if isinstance(existing, Graph):
                return

    # Get Config singleton and ensure it's initialized
    config = depends.get_sync(Config)
    config.ensure_initialized()

    # Create settings for graph adapter
    settings = DuckDBPGQSettings(
        database_url=f"duckdb:///{paths.data_dir}/knowledge_graph.duckdb",
        graph_name="session_mgmt_graph",
        nodes_table="kg_entities",
        edges_table="kg_relationships",
        install_extensions=["duckpgq"],
    )

    # Register settings in Config object
    config.graph = settings  # type: ignore[attr-defined]

    # Create adapter instance and manually set config (ACB test pattern)
    graph_adapter = Graph()
    graph_adapter.config = config  # Override _DependencyMarker with actual Config
    depends.set(Graph, graph_adapter)


def _register_permissions_manager(claude_dir: Path, force: bool) -> None:
    """Register SessionPermissionsManager with the given Claude directory.

    Args:
        claude_dir: Root Claude directory for session data
        force: If True, re-registers even if already registered

    Note:
        Accepts Path directly instead of resolving from string keys,
        following ACB's type-based dependency injection pattern.

    """
    # Import deferred to avoid circular dependency at module load time
    # SessionPermissionsManager will be imported when needed
    # Registration happens lazily when first accessed


def _register_lifecycle_manager(force: bool) -> None:
    """Register SessionLifecycleManager lazily.

    Note:
        Import deferred to avoid circular dependency at module load time.
        SessionLifecycleManager will be registered when first accessed.

    """
    # Registration happens lazily in session_tools.py _get_session_manager()


__all__ = [
    # Legacy string keys (deprecated - use SessionPaths instead)
    "CLAUDE_DIR_KEY",
    "COMMANDS_DIR_KEY",
    "LOGS_DIR_KEY",
    "SessionPaths",
    "configure",
    "reset",
]
