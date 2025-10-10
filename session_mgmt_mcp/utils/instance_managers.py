"""Instance managers for MCP server singletons.

This module provides lazy initialization and access to global singleton instances
for application monitoring, LLM providers, and serverless session management.

Extracted from server.py Phase 2.6 to reduce cognitive complexity.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from session_mgmt_mcp.app_monitor import ApplicationMonitor
    from session_mgmt_mcp.llm_providers import LLMManager
    from session_mgmt_mcp.serverless_mode import ServerlessSessionManager


# Global singleton instances
_app_monitor: ApplicationMonitor | None = None
_llm_manager: LLMManager | None = None
_serverless_manager: ServerlessSessionManager | None = None


async def get_app_monitor() -> ApplicationMonitor | None:
    """Get or initialize application monitor singleton.

    Returns:
        ApplicationMonitor instance if available, None otherwise.
    """
    global _app_monitor

    # Check if feature is available
    try:
        from session_mgmt_mcp.app_monitor import ApplicationMonitor
    except ImportError:
        return None

    if _app_monitor is None:
        data_dir = Path.home() / ".claude" / "data" / "app_monitoring"
        working_dir = os.environ.get("PWD", str(Path.cwd()))
        project_paths = [working_dir] if Path(working_dir).exists() else []
        _app_monitor = ApplicationMonitor(str(data_dir), project_paths)

    return _app_monitor


async def get_llm_manager() -> LLMManager | None:
    """Get or initialize LLM manager singleton.

    Returns:
        LLMManager instance if available, None otherwise.
    """
    global _llm_manager

    # Check if feature is available
    try:
        from session_mgmt_mcp.llm_providers import LLMManager
    except ImportError:
        return None

    if _llm_manager is None:
        config_path = Path.home() / ".claude" / "data" / "llm_config.json"
        _llm_manager = LLMManager(str(config_path) if config_path.exists() else None)

    return _llm_manager


async def get_serverless_manager() -> ServerlessSessionManager | None:
    """Get or initialize serverless session manager singleton.

    Returns:
        ServerlessSessionManager instance if available, None otherwise.
    """
    global _serverless_manager

    # Check if feature is available
    try:
        from session_mgmt_mcp.serverless_mode import (
            ServerlessConfigManager,
            ServerlessSessionManager,
        )
    except ImportError:
        return None

    if _serverless_manager is None:
        config_path = Path.home() / ".claude" / "data" / "serverless_config.json"
        config = ServerlessConfigManager.load_config(
            str(config_path) if config_path.exists() else None,
        )
        storage_backend = ServerlessConfigManager.create_storage_backend(config)
        _serverless_manager = ServerlessSessionManager(storage_backend)

    return _serverless_manager


def reset_instances() -> None:
    """Reset all singleton instances (useful for testing)."""
    global _app_monitor, _llm_manager, _serverless_manager
    _app_monitor = None
    _llm_manager = None
    _serverless_manager = None
