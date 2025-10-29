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
import importlib.util
import os
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# Suppress transformers warnings about PyTorch/TensorFlow for cleaner CLI output
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")

from acb.depends import depends
from session_mgmt_mcp.di import configure as configure_di

# Phase 2.5: Import core infrastructure from server_core
from session_mgmt_mcp.server_core import (
    SessionPermissionsManager,
    _load_mcp_config,
    get_feature_flags,
)
from session_mgmt_mcp.server_core import (
    # Health & status functions
    health_check as _health_check_impl,
)
from session_mgmt_mcp.server_core import (
    initialize_new_features as _initialize_new_features_impl,
)
from session_mgmt_mcp.server_core import (
    # Session lifecycle handler
    session_lifecycle as _session_lifecycle_impl,
)
from session_mgmt_mcp.utils.logging import SessionLogger, get_session_logger

configure_di()

try:
    session_logger = depends.get_sync(SessionLogger)
except Exception:
    session_logger = get_session_logger()

# Check token optimizer availability (Phase 3.3 M2: improved pattern)
TOKEN_OPTIMIZER_AVAILABLE = (
    importlib.util.find_spec("session_mgmt_mcp.token_optimizer") is not None
)

if TOKEN_OPTIMIZER_AVAILABLE:
    from session_mgmt_mcp.token_optimizer import (
        get_cached_chunk,
        get_token_usage_stats,
        optimize_search_response,
        track_token_usage,
    )
else:
    # Fallback implementations when token optimizer unavailable
    TOKEN_OPTIMIZER_AVAILABLE = False

    async def optimize_search_response(
        results: list[dict[str, Any]],
        strategy: str = "prioritize_recent",
        max_tokens: int = 4000,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return results, {}

    async def track_token_usage(
        operation: str,
        request_tokens: int,
        response_tokens: int,
        optimization_applied: str | None = None,
    ) -> None:
        return None

    async def get_cached_chunk(
        cache_key: str, chunk_index: int
    ) -> dict[str, Any] | None:
        return None

    async def get_token_usage_stats(hours: int = 24) -> dict[str, Any]:
        return {"status": "token optimizer unavailable"}

    async def optimize_memory_usage(
        strategy: str = "auto",
        max_age_days: int = 30,
        dry_run: bool = True,
    ) -> str:
        return "❌ Token optimizer not available."


# Import FastMCP with test environment fallback
try:
    from fastmcp import FastMCP

    MCP_AVAILABLE = True
except ImportError:
    if "pytest" in sys.modules or "test" in sys.argv[0].lower():
        from tests.conftest import MockFastMCP

        FastMCP = MockFastMCP  # type: ignore[no-redef,misc]
        MCP_AVAILABLE = False
    elif EXCEPTIONS_AVAILABLE:
        raise DependencyMissingError(
            message="FastMCP is required but not installed",
            dependency="fastmcp",
            install_command="uv add fastmcp",
        )
    else:
        # Fallback to sys.exit if exceptions unavailable
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
try:
    permissions_manager = depends.get_sync(SessionPermissionsManager)
except Exception:
    claude_root = session_logger.log_dir.parent
    permissions_manager = SessionPermissionsManager(claude_root)
    depends.set(SessionPermissionsManager, permissions_manager)

# Import required components for automatic lifecycle
from session_mgmt_mcp.core import SessionLifecycleManager
from session_mgmt_mcp.reflection_tools import get_reflection_database

# Check mcp-common ServerPanels availability (Phase 3.3 M2: improved pattern)
SERVERPANELS_AVAILABLE = importlib.util.find_spec("mcp_common.ui") is not None

# Check mcp-common security availability (Phase 3.3 M2: improved pattern)
SECURITY_AVAILABLE = importlib.util.find_spec("mcp_common.security") is not None

# Check FastMCP rate limiting middleware availability (Phase 3.3 M2: improved pattern)
RATE_LIMITING_AVAILABLE = (
    importlib.util.find_spec("fastmcp.server.middleware.rate_limiting") is not None
)

# Check mcp-common exceptions availability (Phase 3.3 M3: custom exceptions)
EXCEPTIONS_AVAILABLE = importlib.util.find_spec("mcp_common.exceptions") is not None

if EXCEPTIONS_AVAILABLE:
    pass

# Phase 2.2: Import utility and formatting functions from server_helpers

# Global session manager for lifespan handlers
try:
    lifecycle_manager = depends.get_sync(SessionLifecycleManager)
except Exception:
    lifecycle_manager = SessionLifecycleManager()
    depends.set(SessionLifecycleManager, lifecycle_manager)


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

# Add rate limiting middleware (Phase 3 Security Hardening)
if RATE_LIMITING_AVAILABLE:
    from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware

    rate_limiter = RateLimitingMiddleware(
        max_requests_per_second=10.0,  # Sustainable rate for session management operations
        burst_capacity=30,  # Allow bursts for checkpoint/status operations
        global_limit=True,  # Protect the session management server globally
    )
    # Use public API (Phase 3.1 C1 fix: standardize middleware access)
    mcp.add_middleware(rate_limiter)
    session_logger.info("Rate limiting enabled: 10 req/sec, burst 30")

# Register extracted tool modules following crackerjack architecture patterns
# Import LLM provider validation (Phase 3 Security Hardening)
from .llm_providers import validate_llm_api_keys_at_startup
from .tools import (
    register_crackerjack_tools,
    register_knowledge_graph_tools,
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
    _format_search_results,
    validate_claude_directory,
)

# Register all extracted tool modules
register_search_tools(mcp)
register_crackerjack_tools(mcp)
register_knowledge_graph_tools(mcp)  # DuckPGQ knowledge graph tools
register_llm_tools(mcp)
register_monitoring_tools(mcp)
register_prompt_tools(mcp)
register_serverless_tools(mcp)
register_session_tools(mcp)
register_team_tools(mcp)


async def reflect_on_past(
    query: str,
    limit: int = 5,
    min_score: float = 0.7,
    project: str | None = None,
    optimize_tokens: bool = True,
    max_tokens: int = 4000,
) -> str:
    """Search past conversations with optional token optimization."""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: uv sync --extra embeddings"

    try:
        db = await get_reflection_database()
    except Exception as exc:  # pragma: no cover - defensive logging
        session_logger.exception(
            "Failed to initialize reflection database", exc_info=exc
        )
        return f"❌ Error searching conversations: {exc}"

    if not db:
        return "❌ Reflection system not available. Install optional dependencies with `uv sync --extra embeddings`"

    try:
        async with db:
            results = await db.search_conversations(
                query=query,
                project=project or current_project,
                limit=limit,
                min_score=min_score,
            )
    except Exception as exc:
        session_logger.exception("Reflection search failed", extra={"query": query})
        return f"❌ Error searching conversations: {exc}"

    if not results:
        return (
            f"🔍 No relevant conversations found for query: '{query}'\n"
            "💡 Try adjusting the search terms or lowering min_score."
        )

    optimization_info: dict[str, Any] = {}
    if optimize_tokens and TOKEN_OPTIMIZER_AVAILABLE:
        try:
            optimized_results, optimization_info = await optimize_search_response(
                results,
                strategy="prioritize_recent",
                max_tokens=max_tokens,
            )
            if optimized_results:
                results = optimized_results

            token_savings = optimization_info.get("token_savings", {})
            await track_token_usage(
                operation="reflect_on_past",
                request_tokens=max_tokens,
                response_tokens=max_tokens - token_savings.get("tokens_saved", 0),
                optimization_applied=optimization_info.get("strategy"),
            )
        except Exception as exc:
            session_logger.warning(
                "Token optimization failed for reflect_on_past",
                extra={"error": str(exc)},
            )
            optimization_info = {}

    output_lines = [
        f"🔍 **Search Results for: '{query}'**",
        "",
        f"📊 Found {len(results)} relevant conversations",
        "",
    ]

    token_savings = (
        optimization_info.get("token_savings")
        if isinstance(optimization_info, dict)
        else None
    )
    if token_savings and token_savings.get("savings_percentage") is not None:
        output_lines.append(
            f"⚡ Token optimization: {token_savings.get('savings_percentage')}% saved"
        )
        output_lines.append("")

    output_lines.extend(_format_search_results(results))
    return "\n".join(output_lines)


# Wrapper for initialize_new_features that manages global state
async def initialize_new_features() -> None:
    """Initialize multi-project coordination and advanced search features (wrapper)."""
    global multi_project_coordinator, advanced_search_engine, app_config

    # Get the initialized instances from the implementation
    (
        multi_project_coordinator,
        advanced_search_engine,
        app_config,
    ) = await _initialize_new_features_impl(
        session_logger,
        multi_project_coordinator,
        advanced_search_engine,
        app_config,
    )


# Phase 2.3: Import quality engine functions
from session_mgmt_mcp.quality_engine import (
    calculate_quality_score as _calculate_quality_score_impl,
)


# Expose quality scoring function for external use
async def calculate_quality_score(project_dir: Path | None = None) -> dict[str, Any]:
    """Calculate session quality score using V2 algorithm.

    This function provides a consistent interface for calculating quality scores
    across the system.

    Args:
        project_dir: Path to the project directory. If not provided, will use current directory.

    Returns:
        Dict with quality score and breakdown information.

    """
    if project_dir is None:
        project_dir = Path(os.environ.get("PWD", Path.cwd()))

    return await _calculate_quality_score_impl(project_dir=project_dir)


# Wrapper for health_check that provides required parameters
async def health_check() -> dict[str, Any]:
    """Comprehensive health check for MCP server and toolkit availability (wrapper)."""
    return await _health_check_impl(
        session_logger, permissions_manager, validate_claude_directory
    )


# Phase 2.4: Import advanced feature tools from advanced_features module
from session_mgmt_mcp.advanced_features import (
    add_project_dependency,
    # Advanced Search Tools (3 MCP tools)
    advanced_search,
    cancel_user_reminder,
    # Natural Language Scheduling Tools (5 MCP tools)
    create_natural_reminder,
    # Multi-Project Coordination Tools (4 MCP tools)
    create_project_group,
    # Interruption Management Tools (1 MCP tool)
    get_interruption_statistics,
    get_project_insights,
    get_search_metrics,
    # Git Worktree Management Tools (3 MCP tools)
    git_worktree_add,
    git_worktree_remove,
    git_worktree_switch,
    list_user_reminders,
    search_across_projects,
    search_suggestions,
    # Session Welcome Tool (1 MCP tool)
    session_welcome,
    start_reminder_service,
    stop_reminder_service,
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
    # Validate LLM API keys at startup (Phase 3 Security Hardening)
    # Phase 3.2 H3 fix: Replace broad exception suppression with specific handling
    if LLM_PROVIDERS_AVAILABLE:
        try:
            validate_llm_api_keys_at_startup()
        except (ImportError, ValueError) as e:
            logger.warning(f"LLM API key validation skipped (optional feature): {e}")
        except Exception as e:
            logger.error(f"Unexpected error during LLM validation: {e}", exc_info=True)

    # Initialize new features on startup
    # Phase 3.2 H3 fix: Replace broad exception suppression with specific handling
    try:
        asyncio.run(initialize_new_features())
    except (ImportError, RuntimeError) as e:
        logger.warning(f"Feature initialization skipped (optional): {e}")
    except Exception as e:
        logger.error(f"Unexpected error during feature init: {e}", exc_info=True)

    # Get host and port from config
    host = _mcp_config.get("http_host", "127.0.0.1")
    port = http_port or _mcp_config.get("http_port", 8678)

    # Check configuration and command line flags
    config_http_enabled = _mcp_config.get("http_enabled", False)
    use_http = http_mode or config_http_enabled

    if use_http:
        # Use ServerPanels for beautiful startup UI
        if SERVERPANELS_AVAILABLE:
            from mcp_common.ui import ServerPanels

            # Build features list with optional security and rate limiting features
            features = [
                "Session Lifecycle Management",
                "Memory & Reflection System",
                "Crackerjack Quality Integration",
                "Knowledge Graph (DuckPGQ)",
                "LLM Provider Management",
            ]
            if SECURITY_AVAILABLE:
                features.append("🔒 API Key Validation (OpenAI/Gemini)")
            if RATE_LIMITING_AVAILABLE:
                features.append("⚡ Rate Limiting (10 req/sec, burst 30)")

            ServerPanels.startup_success(
                server_name="Session Management MCP",
                version="2.0.0",
                features=features,
                endpoint=f"http://{host}:{port}/mcp",
                websocket_monitor=str(_mcp_config.get("websocket_monitor_port", 8677)),
                transport="HTTP (streamable)",
            )
        else:
            # Fallback to simple print
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
        # Use ServerPanels for STDIO mode
        if SERVERPANELS_AVAILABLE:
            from mcp_common.ui import ServerPanels

            # Build features list with optional security and rate limiting features
            features = [
                "Session Lifecycle Management",
                "Memory & Reflection System",
                "Crackerjack Quality Integration",
                "Knowledge Graph (DuckPGQ)",
                "LLM Provider Management",
            ]
            if SECURITY_AVAILABLE:
                features.append("🔒 API Key Validation (OpenAI/Gemini)")
            if RATE_LIMITING_AVAILABLE:
                features.append("⚡ Rate Limiting (10 req/sec, burst 30)")

            ServerPanels.startup_success(
                server_name="Session Management MCP",
                version="2.0.0",
                features=features,
                transport="STDIO",
                mode="Claude Desktop",
            )
        else:
            # Fallback to simple print
            print(
                "Starting Session Management MCP Server in STDIO mode", file=sys.stderr
            )

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
