from __future__ import annotations

import typing as t
from contextlib import suppress

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
    """Register SessionLogger with the given logs directory.

    Args:
        logs_dir: Directory for session log files
        force: If True, re-registers even if already registered

    Note:
        Accepts Path directly instead of resolving from string keys,
        following ACB's type-based dependency injection pattern.

    """
    from session_mgmt_mcp.utils.logging import SessionLogger

    if not force:
        with suppress(KeyError, AttributeError, RuntimeError):
            # RuntimeError: when adapter requires async (re-register)
            depends.get_sync(SessionLogger)
            return

    logger = SessionLogger(logs_dir)
    depends.set(SessionLogger, logger)


def _register_permissions_manager(claude_dir: Path, force: bool) -> None:
    """Register SessionPermissionsManager with the given Claude directory.

    Args:
        claude_dir: Root Claude directory for session data
        force: If True, re-registers even if already registered

    Note:
        Accepts Path directly instead of resolving from string keys,
        following ACB's type-based dependency injection pattern.

    """
    from session_mgmt_mcp.server_core import SessionPermissionsManager

    if not force:
        with suppress(KeyError, AttributeError, RuntimeError):
            # RuntimeError: when adapter requires async (re-register)
            depends.get_sync(SessionPermissionsManager)
            return

    manager = SessionPermissionsManager(claude_dir)
    depends.set(SessionPermissionsManager, manager)


def _register_lifecycle_manager(force: bool) -> None:
    from session_mgmt_mcp.core import SessionLifecycleManager

    if not force:
        with suppress(KeyError, AttributeError, RuntimeError):
            # RuntimeError: when adapter requires async (re-register)
            depends.get_sync(SessionLifecycleManager)
            return
    manager = SessionLifecycleManager()
    depends.set(SessionLifecycleManager, manager)


__all__ = [
    # Legacy string keys (deprecated - use SessionPaths instead)
    "CLAUDE_DIR_KEY",
    "COMMANDS_DIR_KEY",
    "LOGS_DIR_KEY",
    "SessionPaths",
    "configure",
    "reset",
]
