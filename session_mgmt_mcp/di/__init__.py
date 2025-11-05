from __future__ import annotations

import typing as t
from contextlib import suppress
from datetime import datetime

from acb.depends import depends

if t.TYPE_CHECKING:
    from pathlib import Path

    from session_mgmt_mcp.core import (
        SessionLifecycleManager as SessionLifecycleManagerT,  # noqa: F401
    )
    from session_mgmt_mcp.server_core import (
        SessionPermissionsManager as SessionPermissionsManagerT,  # noqa: F401
    )
    from session_mgmt_mcp.utils.logging import (
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
    _register_permissions_manager(paths.claude_dir, force)
    _register_lifecycle_manager(force)

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

    # Import ACB's Logger class
    Logger = import_adapter("logger")

    if not force:
        with suppress(KeyError, AttributeError, RuntimeError):
            # RuntimeError: when adapter requires async (re-register)
            existing = depends.get_sync(Logger)
            # Only skip if we already have a Logger instance (not just the module name string)
            if isinstance(existing, Logger):
                return

    # Create logger instance (ACB logger takes no init args)
    logger_instance = Logger()

    # Configure logger with file sink
    log_file = logs_dir / f"session_management_{datetime.now().strftime('%Y%m%d')}.log"
    logger_instance.add(
        str(log_file),
        level="INFO",
        rotation="1 day",
        retention="7 days",
        compression="gz",
    )

    # Register the instance
    depends.set(Logger, logger_instance)


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
    pass  # Registration happens lazily when first accessed


def _register_lifecycle_manager(force: bool) -> None:
    """Register SessionLifecycleManager lazily.

    Note:
        Import deferred to avoid circular dependency at module load time.
        SessionLifecycleManager will be registered when first accessed.
    """
    pass  # Registration happens lazily in session_tools.py _get_session_manager()


__all__ = [
    # Legacy string keys (deprecated - use SessionPaths instead)
    "CLAUDE_DIR_KEY",
    "COMMANDS_DIR_KEY",
    "LOGS_DIR_KEY",
    "SessionPaths",
    "configure",
    "reset",
]
