"""Instance managers for MCP server singletons.

This module provides lazy initialization and access to global singleton instances
for application monitoring, LLM providers, and serverless session management.

Extracted from server.py Phase 2.6 to reduce cognitive complexity.
"""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from acb.depends import depends
from bevy import get_container
from session_mgmt_mcp.di.constants import CLAUDE_DIR_KEY

if TYPE_CHECKING:
    from session_mgmt_mcp.app_monitor import ApplicationMonitor
    from session_mgmt_mcp.interruption_manager import InterruptionManager
    from session_mgmt_mcp.llm_providers import LLMManager
    from session_mgmt_mcp.reflection_tools import ReflectionDatabase
    from session_mgmt_mcp.serverless_mode import ServerlessSessionManager


async def get_app_monitor() -> ApplicationMonitor | None:
    """Resolve application monitor via DI, creating it on demand."""
    try:
        from session_mgmt_mcp.app_monitor import ApplicationMonitor
    except ImportError:
        return None

    with suppress(Exception):
        monitor = depends.get(ApplicationMonitor)
        if isinstance(monitor, ApplicationMonitor):
            return monitor

    data_dir = _resolve_claude_dir() / "data" / "app_monitoring"
    working_dir = Path(os.environ.get("PWD", str(Path.cwd())))
    project_paths = [str(working_dir)] if working_dir.exists() else []

    monitor = ApplicationMonitor(str(data_dir), project_paths)
    depends.set(ApplicationMonitor, monitor)
    return monitor


async def get_llm_manager() -> LLMManager | None:
    """Resolve LLM manager via DI, creating it on demand."""
    try:
        from session_mgmt_mcp.llm_providers import LLMManager
    except ImportError:
        return None

    with suppress(Exception):
        manager = depends.get(LLMManager)
        if isinstance(manager, LLMManager):
            return manager

    config_path = _resolve_claude_dir() / "data" / "llm_config.json"
    manager = LLMManager(str(config_path) if config_path.exists() else None)
    depends.set(LLMManager, manager)
    return manager


async def get_serverless_manager() -> ServerlessSessionManager | None:
    """Resolve serverless session manager via DI, creating it on demand."""
    try:
        from session_mgmt_mcp.serverless_mode import (
            ServerlessConfigManager,
            ServerlessSessionManager,
        )
    except ImportError:
        return None

    with suppress(Exception):
        manager = depends.get(ServerlessSessionManager)
        if isinstance(manager, ServerlessSessionManager):
            return manager

    claude_dir = _resolve_claude_dir()
    config_path = claude_dir / "data" / "serverless_config.json"
    config = ServerlessConfigManager.load_config(
        str(config_path) if config_path.exists() else None,
    )
    storage_backend = ServerlessConfigManager.create_storage_backend(config)
    manager = ServerlessSessionManager(storage_backend)
    depends.set(ServerlessSessionManager, manager)
    return manager


async def get_reflection_database() -> ReflectionDatabase | None:
    """Resolve reflection database via DI, creating it on demand."""
    try:
        from session_mgmt_mcp.reflection_tools import (
            ReflectionDatabase,
        )
        from session_mgmt_mcp.reflection_tools import (
            get_reflection_database as load_reflection_database,
        )
    except ImportError:
        return None

    with suppress(Exception):
        db = depends.get(ReflectionDatabase)
        if isinstance(db, ReflectionDatabase):
            return db

    db = await load_reflection_database()
    depends.set(ReflectionDatabase, db)
    return db


async def get_interruption_manager() -> InterruptionManager | None:
    """Resolve interruption manager via DI, creating it on demand."""
    try:
        from session_mgmt_mcp.interruption_manager import InterruptionManager
    except ImportError:
        return None

    with suppress(Exception):
        manager = depends.get(InterruptionManager)
        if isinstance(manager, InterruptionManager):
            return manager

    manager = InterruptionManager()
    depends.set(InterruptionManager, manager)
    return manager


def reset_instances() -> None:
    """Reset registered instances in the DI container."""
    container = get_container()
    for dependency in _iter_dependencies():
        with suppress(Exception):
            container.instances.pop(dependency, None)


def _resolve_claude_dir() -> Path:
    with suppress(Exception):
        claude_dir = depends.get(CLAUDE_DIR_KEY)
        if isinstance(claude_dir, Path):
            claude_dir.mkdir(parents=True, exist_ok=True)
            return claude_dir

    default_dir = Path.home() / ".claude"
    default_dir.mkdir(parents=True, exist_ok=True)
    depends.set(CLAUDE_DIR_KEY, default_dir)
    return default_dir


def _iter_dependencies() -> list[type[Any]]:
    deps: list[type[Any]] = []
    with suppress(ImportError):
        from session_mgmt_mcp.app_monitor import ApplicationMonitor

        deps.append(ApplicationMonitor)
    with suppress(ImportError):
        from session_mgmt_mcp.llm_providers import LLMManager

        deps.append(LLMManager)
    with suppress(ImportError):
        from session_mgmt_mcp.interruption_manager import InterruptionManager

        deps.append(InterruptionManager)
    with suppress(ImportError):
        from session_mgmt_mcp.serverless_mode import ServerlessSessionManager

        deps.append(ServerlessSessionManager)
    with suppress(ImportError):
        from session_mgmt_mcp.reflection_tools import ReflectionDatabase

        deps.append(ReflectionDatabase)
    return deps
