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
    from collections.abc import AsyncGenerator, Callable

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
)

# Initialize logger
claude_dir = Path.home() / ".claude"
session_logger = SessionLogger(claude_dir / "logs")

try:
    from fastmcp import FastMCP

    MCP_AVAILABLE = True
except ImportError:
    # Check if we're in a test environment
    if "pytest" in sys.modules or "test" in sys.argv[0].lower():
        print(
            "Warning: FastMCP not available in test environment, using mock",
            file=sys.stderr,
        )

        # Create a minimal mock FastMCP for testing
        class MockFastMCP:
            def __init__(self, name: str) -> None:
                self.name = name
                self.tools: dict[str, Any] = {}
                self.prompts: dict[str, Any] = {}

            def tool(
                self, *args: Any, **kwargs: Any
            ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                    return func

                return decorator

            def prompt(
                self, *args: Any, **kwargs: Any
            ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                    return func

                return decorator

            def run(self, *args: Any, **kwargs: Any) -> None:
                pass

        FastMCP = MockFastMCP  # type: ignore[no-redef]
        MCP_AVAILABLE = False
    else:
        print("FastMCP not available. Install with: uv add fastmcp", file=sys.stderr)
        sys.exit(1)

# Import session management core
try:
    from session_mgmt_mcp.core.session_manager import SessionLifecycleManager

    SESSION_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    print(f"Session management core import failed: {e}", file=sys.stderr)
    SESSION_MANAGEMENT_AVAILABLE = False

# Import reflection tools
try:
    from session_mgmt_mcp.reflection_tools import (
        ReflectionDatabase,
        get_current_project,
        get_reflection_database,
    )

    REFLECTION_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Reflection tools import failed: {e}", file=sys.stderr)
    REFLECTION_TOOLS_AVAILABLE = False

# Import enhanced search tools
try:
    # EnhancedSearchEngine will be imported when needed
    import session_mgmt_mcp.search_enhanced

    ENHANCED_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced search import failed: {e}", file=sys.stderr)
    ENHANCED_SEARCH_AVAILABLE = False

# Import utility functions
try:
    from session_mgmt_mcp.tools.search_tools import _optimize_search_results_impl
    from session_mgmt_mcp.utils.format_utils import _format_session_statistics

    UTILITY_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Utility functions import failed: {e}", file=sys.stderr)
    UTILITY_FUNCTIONS_AVAILABLE = False

# Global feature instances (initialized on-demand)
multi_project_coordinator: Any = None
advanced_search_engine: Any = None
app_config: Any = None
current_project: str | None = None

# Import multi-project coordination tools
try:
    from session_mgmt_mcp.multi_project_coordinator import MultiProjectCoordinator

    MULTI_PROJECT_AVAILABLE = True
except ImportError as e:
    print(f"Multi-project coordinator import failed: {e}", file=sys.stderr)
    MULTI_PROJECT_AVAILABLE = False

# Import advanced search engine
try:
    from session_mgmt_mcp.advanced_search import AdvancedSearchEngine

    ADVANCED_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Advanced search engine import failed: {e}", file=sys.stderr)
    ADVANCED_SEARCH_AVAILABLE = False

# Import configuration management
try:
    from session_mgmt_mcp.settings import get_settings

    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Configuration management import failed: {e}", file=sys.stderr)
    CONFIG_AVAILABLE = False

# Import auto-context loading tools
try:
    # AutoContextLoader will be imported when needed
    import session_mgmt_mcp.context_manager

    AUTO_CONTEXT_AVAILABLE = True
except ImportError as e:
    print(f"Auto-context loading import failed: {e}", file=sys.stderr)
    AUTO_CONTEXT_AVAILABLE = False

# Import memory optimization tools
try:
    # MemoryOptimizer will be imported when needed
    import session_mgmt_mcp.memory_optimizer

    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"Memory optimizer import failed: {e}", file=sys.stderr)
    MEMORY_OPTIMIZER_AVAILABLE = False

# Import application monitoring tools
try:
    from session_mgmt_mcp.app_monitor import ApplicationMonitor

    APP_MONITOR_AVAILABLE = True
except ImportError as e:
    print(f"Application monitoring import failed: {e}", file=sys.stderr)
    APP_MONITOR_AVAILABLE = False

# Import LLM providers
try:
    from session_mgmt_mcp.llm_providers import LLMManager

    LLM_PROVIDERS_AVAILABLE = True
except ImportError as e:
    print(f"LLM providers import failed: {e}", file=sys.stderr)
    LLM_PROVIDERS_AVAILABLE = False

# Import serverless mode
try:
    from session_mgmt_mcp.serverless_mode import (
        ServerlessConfigManager,
        ServerlessSessionManager,
    )

    SERVERLESS_MODE_AVAILABLE = True
except ImportError as e:
    print(f"Serverless mode import failed: {e}", file=sys.stderr)
    SERVERLESS_MODE_AVAILABLE = False

# Import Crackerjack integration tools
try:
    # CrackerjackIntegration will be imported when needed
    import session_mgmt_mcp.crackerjack_integration

    CRACKERJACK_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Crackerjack integration import failed: {e}", file=sys.stderr)
    CRACKERJACK_INTEGRATION_AVAILABLE = False


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
    # Delegate to the extracted implementation with required parameters
    async with _session_lifecycle_impl(app, lifecycle_manager, session_logger):
        yield


# Load configuration and initialize FastMCP 2.0 server with lifespan
_mcp_config = _load_mcp_config()

# Initialize MCP server with lifespan
mcp = FastMCP("session-mgmt-mcp", lifespan=session_lifecycle)

# Register extracted tool modules following crackerjack architecture patterns
# Import session command definitions
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

# Register slash commands as MCP prompts (not resources!)


# Wrapper for initialize_new_features that manages global state
async def initialize_new_features() -> None:
    """Initialize multi-project coordination and advanced search features (wrapper)."""
    global multi_project_coordinator, advanced_search_engine, app_config

    # Delegate to the extracted implementation
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
    # Quality score calculation (fixed bug)
    calculate_quality_score,
)

# Wrapper for health_check that provides required parameters
async def health_check() -> dict[str, Any]:
    """Comprehensive health check for MCP server and toolkit availability (wrapper)."""
    return await _health_check_impl(
        session_logger, permissions_manager, validate_claude_directory
    )


# Token Optimization Tools


# Enhanced Search Tools (Phase 1)


async def get_app_monitor() -> ApplicationMonitor | None:
    """Get or initialize application monitor."""
    global _app_monitor
    if not APP_MONITOR_AVAILABLE:
        return None

    if _app_monitor is None:
        data_dir = Path.home() / ".claude" / "data" / "app_monitoring"
        working_dir = os.environ.get("PWD", str(Path.cwd()))
        project_paths = [working_dir] if Path(working_dir).exists() else []
        _app_monitor = ApplicationMonitor(str(data_dir), project_paths)

    return _app_monitor


# Global instances
_llm_manager = None
_app_monitor = None


async def get_llm_manager() -> LLMManager | None:
    """Get or initialize LLM manager."""
    global _llm_manager
    if not LLM_PROVIDERS_AVAILABLE:
        return None

    if _llm_manager is None:
        config_path = Path.home() / ".claude" / "data" / "llm_config.json"
        _llm_manager = LLMManager(str(config_path) if config_path.exists() else None)

    return _llm_manager


# Global serverless session manager
_serverless_manager = None


async def get_serverless_manager() -> ServerlessSessionManager | None:
    """Get or initialize serverless session manager."""
    global _serverless_manager
    if not SERVERLESS_MODE_AVAILABLE:
        return None

    if _serverless_manager is None:
        config_path = Path.home() / ".claude" / "data" / "serverless_config.json"
        config = ServerlessConfigManager.load_config(
            str(config_path) if config_path.exists() else None,
        )
        storage_backend = ServerlessConfigManager.create_storage_backend(config)
        _serverless_manager = ServerlessSessionManager(storage_backend)

    return _serverless_manager


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


# =====================================
# Crackerjack Integration MCP Tools
# =====================================


def main(http_mode: bool = False, http_port: int | None = None) -> None:
    """Main entry point for the MCP server."""
    # Initialize new features on startup
    import asyncio
    from contextlib import suppress

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
