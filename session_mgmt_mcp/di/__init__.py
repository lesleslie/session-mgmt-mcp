from __future__ import annotations

import typing as t
from contextlib import suppress
from pathlib import Path

from acb.depends import depends

if t.TYPE_CHECKING:
    from session_mgmt_mcp.core import (
        SessionLifecycleManager as SessionLifecycleManagerT,  # noqa: F401
    )
    from session_mgmt_mcp.server_core import (
        SessionPermissionsManager as SessionPermissionsManagerT,  # noqa: F401
    )
    from session_mgmt_mcp.utils.logging import (
        SessionLogger as SessionLoggerT,  # noqa: F401
    )

from .constants import CLAUDE_DIR_KEY, COMMANDS_DIR_KEY, LOGS_DIR_KEY

_configured = False


def configure(*, force: bool = False) -> None:
    """Register default dependencies for the session-mgmt MCP stack."""
    global _configured
    if _configured and not force:
        return

    claude_dir = Path.home() / ".claude"
    _register_path(CLAUDE_DIR_KEY, claude_dir, force)
    _register_path(LOGS_DIR_KEY, claude_dir / "logs", force)
    _register_path(COMMANDS_DIR_KEY, claude_dir / "commands", force)
    _register_logger(force)
    _register_permissions_manager(force)
    _register_lifecycle_manager(force)

    _configured = True


def reset() -> None:
    """Reset dependencies to defaults."""
    configure(force=True)


def _register_path(key: str, path: Path, force: bool) -> None:
    """Register path dependency with optional override."""
    if not force:
        with suppress(Exception):
            existing = depends.get_sync(key)
            if isinstance(existing, Path):
                return
    path.mkdir(parents=True, exist_ok=True)
    depends.set(key, path)


def _resolve_path(key: str, default: Path) -> Path:
    with suppress(Exception):
        resolved = depends.get_sync(key)
        if isinstance(resolved, Path):
            return resolved
    return default


def _register_logger(force: bool) -> None:
    from session_mgmt_mcp.utils.logging import SessionLogger

    if not force:
        with suppress(Exception):
            depends.get_sync(SessionLogger)
            return
    logs_dir = _resolve_path(LOGS_DIR_KEY, Path.home() / ".claude" / "logs")
    logger = SessionLogger(logs_dir)
    depends.set(SessionLogger, logger)


def _register_permissions_manager(force: bool) -> None:
    from session_mgmt_mcp.server_core import SessionPermissionsManager

    if not force:
        with suppress(Exception):
            depends.get_sync(SessionPermissionsManager)
            return
    claude_dir = _resolve_path(CLAUDE_DIR_KEY, Path.home() / ".claude")
    manager = SessionPermissionsManager(claude_dir)
    depends.set(SessionPermissionsManager, manager)


def _register_lifecycle_manager(force: bool) -> None:
    from session_mgmt_mcp.core import SessionLifecycleManager

    if not force:
        with suppress(Exception):
            depends.get_sync(SessionLifecycleManager)
            return
    manager = SessionLifecycleManager()
    depends.set(SessionLifecycleManager, manager)


__all__ = [
    "CLAUDE_DIR_KEY",
    "COMMANDS_DIR_KEY",
    "LOGS_DIR_KEY",
    "configure",
    "reset",
]
