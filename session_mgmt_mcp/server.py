#!/usr/bin/env python3
"""Claude Session Management MCP Server - FastMCP Version.

A dedicated MCP server that provides session management functionality
including initialization, checkpoints, and cleanup across all projects.

This server can be included in any project's .mcp.json file to provide
automatic access to /session-init, /session-checkpoint,
and /session-end slash commands.
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Suppress transformers warnings about PyTorch/TensorFlow for cleaner CLI output
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")

# Phase 2.5: Import core infrastructure from server_core
from session_mgmt_mcp.server_core import (
    # Classes
    SessionLogger,
    SessionPermissionsManager,
    # Configuration functions
    _detect_other_mcp_servers,
    _generate_server_guidance,
    _load_mcp_config,
    # Session lifecycle handler
    session_lifecycle as _session_lifecycle_impl,
    # Initialization functions
    auto_setup_git_working_directory,
    initialize_new_features as _initialize_new_features_impl,
    analyze_project_context,
    # Health & status functions
    health_check as _health_check_impl,
    _add_basic_status_info,
    _add_health_status_info,
    _get_project_context_info,
    # Quality & formatting functions
    _format_quality_results,
    _perform_git_checkpoint,
    _format_conversation_summary,
    # Utility functions
    _should_retry_search,
    # Phase 2.6: Feature detection
    get_feature_flags,
)

# Initialize logger
claude_dir = Path.home() / ".claude"
session_logger = SessionLogger(claude_dir / "logs")

# Import FastMCP with test environment fallback
try:
    from fastmcp import FastMCP

    MCP_AVAILABLE = True
except ImportError:
    if "pytest" in sys.modules or "test" in sys.argv[0].lower():
        from tests.conftest import MockFastMCP

        FastMCP = MockFastMCP  # type: ignore[no-redef,misc]
        MCP_AVAILABLE = False
    else:
        print("FastMCP not available. Install with: uv add fastmcp", file=sys.stderr)
        sys.exit(1)

# Phase 2.6: Get all feature flags from centralized detector
_features = get_feature_flags()
SESSION_MANAGEMENT_AVAILABLE = _features["SESSION_MANAGEMENT_AVAILABLE"]
REFLECTION_TOOLS_AVAILABLE = _features["REFLECTION_TOOLS_AVAILABLE"]
ENHANCED_SEARCH_AVAILABLE = _features["ENHANCED_SEARCH_AVAILABLE"]
UTILITY_FUNCTIONS_AVAILABLE = _features["UTILITY_FUNCTIONS_AVAILABLE"]
MULTI_PROJECT_AVAILABLE = _features["MULTI_PROJECT_AVAILABLE"]
ADVANCED_SEARCH_AVAILABLE = _features["ADVANCED_SEARCH_AVAILABLE"]
CONFIG_AVAILABLE = _features["CONFIG_AVAILABLE"]
AUTO_CONTEXT_AVAILABLE = _features["AUTO_CONTEXT_AVAILABLE"]
MEMORY_OPTIMIZER_AVAILABLE = _features["MEMORY_OPTIMIZER_AVAILABLE"]
APP_MONITOR_AVAILABLE = _features["APP_MONITOR_AVAILABLE"]
LLM_PROVIDERS_AVAILABLE = _features["LLM_PROVIDERS_AVAILABLE"]
SERVERLESS_MODE_AVAILABLE = _features["SERVERLESS_MODE_AVAILABLE"]
CRACKERJACK_INTEGRATION_AVAILABLE = _features["CRACKERJACK_INTEGRATION_AVAILABLE"]

# Global feature instances (initialized on-demand)
multi_project_coordinator: Any = None
advanced_search_engine: Any = None
app_config: Any = None
current_project: str | None = None

# Create global permissions manager instance
permissions_manager = SessionPermissionsManager(claude_dir)

# Import required components for automatic lifecycle
from session_mgmt_mcp.core import SessionLifecycleManager
from session_mgmt_mcp.utils.git_operations import get_git_root, is_git_repository

# Phase 2.2: Import utility and formatting functions from server_helpers
from session_mgmt_mcp.utils.server_helpers import (
    # Formatting functions (26)
    _format_advanced_search_results,
    _format_basic_worktree_info,
    _format_common_patterns_section,
    _format_current_worktree_info,
    _format_detached_head_warning,
    _format_git_worktree_header,
    _format_interruption_statistics,
    _format_metrics_summary,
    _format_no_reminders_message,
    _format_other_branches_info,
    _format_project_activity_section,
    _format_project_insights,
    _format_project_maturity_section,
    _format_reminder_basic_info,
    _format_reminders_header,
    _format_reminders_list,
    _format_session_info,
    _format_session_summary,
    _format_single_reminder,
    _format_single_worktree,
    _format_snapshot_statistics,
    _format_worktree_count_info,
    _format_worktree_list_header,
    _format_worktree_status,
    _format_worktree_status_display,
    _format_worktree_suggestions,
    # Helper functions (14)
    _add_basic_tools_info,
    _add_configuration_info,
    _add_crackerjack_integration_info,
    _add_current_session_context,
    _add_feature_status_info,
    _add_final_summary,
    _add_permissions_and_tools_summary,
    _add_permissions_info,
    _add_session_health_insights,
    _handle_uv_operations,
    _run_uv_sync_and_compile,
    _setup_claude_directory,
    _setup_session_management,
    _setup_uv_dependencies,
)

# Global session manager for lifespan handlers
lifecycle_manager = SessionLifecycleManager()


# Lifespan handler wrapper for FastMCP
@asynccontextmanager
async def session_lifecycle(app: Any) -> AsyncGenerator[None]:
    """Automatic session lifecycle for git repositories only (wrapper)."""
    async with _session_lifecycle_impl(app, lifecycle_manager, session_logger):
        yield


# Load configuration and initialize FastMCP 2.0 server with lifespan
_mcp_config = _load_mcp_config()

# Initialize MCP server with lifespan
mcp = FastMCP("session-mgmt-mcp", lifespan=session_lifecycle)

# Register extracted tool modules following crackerjack architecture patterns
from .tools import (
    register_crackerjack_tools,
    register_llm_tools,
    register_monitoring_tools,
    register_prompt_tools,
    register_search_tools,
    register_serverless_tools,
    register_session_tools,
    register_team_tools,
)

# Import utility functions
from .utils import (
    _analyze_quality_trend,
    _build_search_header,
    _cleanup_session_logs,
    _cleanup_temp_files,
    _cleanup_uv_cache,
    _extract_quality_scores,
    _format_efficiency_metrics,
    _format_no_data_message,
    _format_search_results,
    _format_statistics_header,
    _generate_quality_trend_recommendations,
    _get_intelligence_error_result,
    _get_time_based_recommendations,
    _optimize_git_repository,
    validate_claude_directory,
)

# Register all extracted tool modules
register_search_tools(mcp)
register_crackerjack_tools(mcp)
register_llm_tools(mcp)
register_monitoring_tools(mcp)
register_prompt_tools(mcp)
register_serverless_tools(mcp)
register_session_tools(mcp)
register_team_tools(mcp)


# Wrapper for initialize_new_features that manages global state
async def initialize_new_features() -> None:
    """Initialize multi-project coordination and advanced search features (wrapper)."""
    global multi_project_coordinator, advanced_search_engine, app_config
    await _initialize_new_features_impl(
        session_logger,
        multi_project_coordinator,
        advanced_search_engine,
        app_config,
    )


# Phase 2.3: Import quality engine functions
from session_mgmt_mcp.quality_engine import (
    # Main quality functions (5)
    should_suggest_compact,
    perform_strategic_compaction,
    monitor_proactive_quality,
    generate_session_intelligence,
    analyze_context_usage,
    # Context analysis (4)
    summarize_current_conversation,
    _generate_basic_insights,
    _add_project_context_insights,
    _generate_session_tags,
    # Token & conversation analysis (5)
    analyze_token_usage_patterns,
    analyze_conversation_flow,
    analyze_memory_patterns,
    analyze_project_workflow_patterns,
    analyze_advanced_context_metrics,
    # Quality analysis & recommendations (6)
    _perform_quality_analysis,
    _get_quality_error_result,
    _analyze_token_usage_recommendations,
    _analyze_conversation_flow_recommendations,
    _analyze_memory_recommendations,
    _analyze_quality_monitoring_recommendations,
    # Intelligence & insights (3)
    _capture_intelligence_insights,
    _analyze_reflection_based_intelligence,
    _analyze_project_workflow_recommendations,
    _analyze_session_intelligence_recommendations,
    _add_fallback_recommendations,
    # Helper functions
    _generate_quality_recommendations,
    _check_workflow_drift,
    _optimize_reflection_database,
    _analyze_context_compaction,
    _store_context_summary,
    _perform_quality_assessment,
    # Quality score calculation
    calculate_quality_score,
)


# Wrapper for health_check that provides required parameters
async def health_check() -> dict[str, Any]:
    """Comprehensive health check for MCP server and toolkit availability (wrapper)."""
    return await _health_check_impl(
        session_logger, permissions_manager, validate_claude_directory
    )


# Phase 2.4: Import advanced feature tools from advanced_features module
from session_mgmt_mcp.advanced_features import (
    # Natural Language Scheduling Tools (5 MCP tools)
    create_natural_reminder,
    list_user_reminders,
    cancel_user_reminder,
    start_reminder_service,
    stop_reminder_service,
    # Interruption Management Tools (1 MCP tool)
    get_interruption_statistics,
    # Multi-Project Coordination Tools (4 MCP tools)
    create_project_group,
    add_project_dependency,
    search_across_projects,
    get_project_insights,
    # Advanced Search Tools (3 MCP tools)
    advanced_search,
    search_suggestions,
    get_search_metrics,
    # Git Worktree Management Tools (3 MCP tools)
    git_worktree_add,
    git_worktree_remove,
    git_worktree_switch,
    # Session Welcome Tool (1 MCP tool)
    session_welcome,
    set_connection_info,
)

# Register all 17 advanced MCP tools
mcp.tool()(create_natural_reminder)
mcp.tool()(list_user_reminders)
mcp.tool()(cancel_user_reminder)
mcp.tool()(start_reminder_service)
mcp.tool()(stop_reminder_service)
mcp.tool()(get_interruption_statistics)
mcp.tool()(create_project_group)
mcp.tool()(add_project_dependency)
mcp.tool()(search_across_projects)
mcp.tool()(get_project_insights)
mcp.tool()(advanced_search)
mcp.tool()(search_suggestions)
mcp.tool()(get_search_metrics)
mcp.tool()(git_worktree_add)
mcp.tool()(git_worktree_remove)
mcp.tool()(git_worktree_switch)
mcp.tool()(session_welcome)


def main(http_mode: bool = False, http_port: int | None = None) -> None:
    """Main entry point for the MCP server."""
    # Initialize new features on startup
    with suppress(Exception):
        asyncio.run(initialize_new_features())

    # Get host and port from config
    host = _mcp_config.get("http_host", "127.0.0.1")
    port = http_port or _mcp_config.get("http_port", 8678)

    # Check configuration and command line flags
    config_http_enabled = _mcp_config.get("http_enabled", False)
    use_http = http_mode or config_http_enabled

    if use_http:
        print(
            f"Starting Session Management MCP HTTP Server on http://{host}:{port}/mcp",
            file=sys.stderr,
        )
        print(
            f"WebSocket Monitor: {_mcp_config.get('websocket_monitor_port', 8677)}",
            file=sys.stderr,
        )
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
            path="/mcp",
            stateless_http=True,
        )
    else:
        print("Starting Session Management MCP Server in STDIO mode", file=sys.stderr)
        mcp.run(stateless_http=True)


def _ensure_default_recommendations(priority_actions: list[str]) -> list[str]:
    """Ensure we always have default recommendations available."""
    if not priority_actions:
        return [
            "Run quality checks with `crackerjack lint`",
            "Check test coverage with `pytest --cov`",
            "Review recent git commits for patterns",
        ]
    return priority_actions


def _has_statistics_data(
    sessions: list[dict[str, Any]],
    interruptions: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> bool:
    """Check if we have any statistics data to display."""
    return bool(sessions or interruptions or snapshots)


if __name__ == "__main__":
    import sys

    # Check for HTTP mode flags
    http_mode = "--http" in sys.argv
    http_port = None

    if "--http-port" in sys.argv:
        port_idx = sys.argv.index("--http-port")
        if port_idx + 1 < len(sys.argv):
            http_port = int(sys.argv[port_idx + 1])

    main(http_mode, http_port)
