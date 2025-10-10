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
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import warnings
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

# Suppress transformers warnings about PyTorch/TensorFlow for cleaner CLI output
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")

try:
    import tomli
except ImportError:
    tomli = None  # type: ignore[assignment]


# Configure structured logging
class SessionLogger:
    """Structured logging for session management with context."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (
            log_dir / f"session_management_{datetime.now().strftime('%Y%m%d')}.log"
        )

        # Configure logger
        self.logger = logging.getLogger("session_management")
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            # File handler with structured format
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)

            # Console handler for errors
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.ERROR)

            # Structured formatter
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def debug(self, message: str, **context: Any) -> None:
        """Log debug with optional context."""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.debug(message)

    def info(self, message: str, **context: Any) -> None:
        """Log info with optional context."""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.info(message)

    def warning(self, message: str, **context: Any) -> None:
        """Log warning with optional context."""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.warning(message)

    def error(self, message: str, **context: Any) -> None:
        """Log error with optional context."""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.error(message)

    def exception(self, message: str, **context: Any) -> None:
        """Log exception with optional context."""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.exception(message)


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


class SessionPermissionsManager:
    """Manages session permissions to avoid repeated prompts for trusted operations."""

    _instance: SessionPermissionsManager | None = None
    _session_id: str | None = None
    _initialized: bool = False

    def __new__(cls, claude_dir: Path) -> Self:  # type: ignore[misc]
        """Singleton pattern to ensure consistent session ID across tool calls."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        # Type checker knows this is Self from the annotation above
        return cls._instance  # type: ignore[return-value]

    def __init__(self, claude_dir: Path) -> None:
        if self._initialized:
            return
        self.claude_dir = claude_dir
        self.permissions_file = claude_dir / "sessions" / "trusted_permissions.json"
        self.permissions_file.parent.mkdir(exist_ok=True)
        self.trusted_operations: set[str] = set()
        # Use class-level session ID to persist across instances
        if SessionPermissionsManager._session_id is None:
            SessionPermissionsManager._session_id = self._generate_session_id()
        self.session_id = SessionPermissionsManager._session_id
        self._load_permissions()
        self._initialized = True

    def _generate_session_id(self) -> str:
        """Generate unique session ID based on current time and working directory."""
        session_data = f"{datetime.now().isoformat()}_{Path.cwd()}"
        return hashlib.md5(session_data.encode(), usedforsecurity=False).hexdigest()[
            :12
        ]

    def _load_permissions(self) -> None:
        """Load previously granted permissions."""
        if self.permissions_file.exists():
            try:
                with self.permissions_file.open() as f:
                    data = json.load(f)
                    self.trusted_operations.update(data.get("trusted_operations", []))
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_permissions(self) -> None:
        """Save current trusted permissions."""
        data = {
            "trusted_operations": list(self.trusted_operations),
            "last_updated": datetime.now().isoformat(),
            "session_id": self.session_id,
        }
        with self.permissions_file.open("w") as f:
            json.dump(data, f, indent=2)

    def is_operation_trusted(self, operation: str) -> bool:
        """Check if an operation is already trusted."""
        return operation in self.trusted_operations

    def trust_operation(self, operation: str, description: str = "") -> None:
        """Mark an operation as trusted to avoid future prompts."""
        self.trusted_operations.add(operation)
        self._save_permissions()

    def get_permission_status(self) -> dict[str, Any]:
        """Get current permission status."""
        return {
            "session_id": self.session_id,
            "trusted_operations_count": len(self.trusted_operations),
            "trusted_operations": list(self.trusted_operations),
            "permissions_file": str(self.permissions_file),
        }

    def revoke_all_permissions(self) -> None:
        """Revoke all trusted permissions (for security reset)."""
        self.trusted_operations.clear()
        if self.permissions_file.exists():
            self.permissions_file.unlink()

    # Common trusted operations
    TRUSTED_UV_OPERATIONS = "uv_package_management"
    TRUSTED_GIT_OPERATIONS = "git_repository_access"
    TRUSTED_FILE_OPERATIONS = "project_file_access"
    TRUSTED_SUBPROCESS_OPERATIONS = "subprocess_execution"
    TRUSTED_NETWORK_OPERATIONS = "network_access"
    # TRUSTED_WORKSPACE_OPERATIONS removed - no longer needed


# Create global permissions manager instance
permissions_manager = SessionPermissionsManager(claude_dir)


# Utility Functions
def _detect_other_mcp_servers() -> dict[str, bool]:
    """Detect availability of other MCP servers by checking common paths and processes."""
    detected = {}

    # Check for crackerjack MCP server
    try:
        # Try to import crackerjack to see if it's available
        result = subprocess.run(
            ["crackerjack", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        detected["crackerjack"] = result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        detected["crackerjack"] = False

    return detected


def _generate_server_guidance(detected_servers: dict[str, bool]) -> list[str]:
    """Generate guidance messages based on detected servers."""
    guidance = []

    if detected_servers.get("crackerjack", False):
        guidance.extend(
            [
                "💡 CRACKERJACK INTEGRATION DETECTED:",
                "   Enhanced commands available for better development experience:",
                "   • Use /session-mgmt:crackerjack-run instead of /crackerjack:run",
                "   • Gets memory, analytics, and intelligent insights automatically",
                "   • View trends with /session-mgmt:crackerjack-history",
                "   • Analyze patterns with /session-mgmt:crackerjack-patterns",
            ],
        )

    return guidance


def _load_mcp_config() -> dict[str, Any]:
    """Load MCP server configuration from pyproject.toml."""
    # Look for pyproject.toml in the current project directory
    pyproject_path = Path.cwd() / "pyproject.toml"

    # If not found in cwd, look in parent directories (up to 3 levels)
    if not pyproject_path.exists():
        for parent in Path.cwd().parents[:3]:
            potential_path = parent / "pyproject.toml"
            if potential_path.exists():
                pyproject_path = potential_path
                break

    if not pyproject_path.exists() or not tomli:
        return {
            "http_port": 8678,
            "http_host": "127.0.0.1",
            "websocket_monitor_port": 8677,
            "http_enabled": False,
        }

    try:
        with pyproject_path.open("rb") as f:
            pyproject_data = tomli.load(f)

        session_config = pyproject_data.get("tool", {}).get("session-mgmt-mcp", {})

        return {
            "http_port": session_config.get("mcp_http_port", 8678),
            "http_host": session_config.get("mcp_http_host", "127.0.0.1"),
            "websocket_monitor_port": session_config.get(
                "websocket_monitor_port", 8677
            ),
            "http_enabled": session_config.get("http_enabled", False),
        }
    except Exception as e:
        print(
            f"Warning: Failed to load MCP config from pyproject.toml: {e}",
            file=sys.stderr,
        )
        return {
            "http_port": 8678,
            "http_host": "127.0.0.1",
            "websocket_monitor_port": 8677,
            "http_enabled": False,
        }


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


# Lifespan handler for automatic session management
@asynccontextmanager
async def session_lifecycle(app: Any) -> AsyncGenerator[None]:
    """Automatic session lifecycle for git repositories only."""
    current_dir = Path.cwd()

    # Only auto-initialize for git repositories
    if is_git_repository(current_dir):
        try:
            git_root = get_git_root(current_dir)
            session_logger.info(f"Git repository detected at {git_root}")

            # Run the same logic as the start tool but with connection notification
            result = await lifecycle_manager.initialize_session(str(current_dir))
            if result["success"]:
                session_logger.info("✅ Auto-initialized session for git repository")

                # Store connection info for display via tools (use imported function)
                connection_info = {
                    "connected_at": "just connected",
                    "project": result["project"],
                    "quality_score": result["quality_score"],
                    "previous_session": result.get("previous_session"),
                    "recommendations": result["quality_data"].get(
                        "recommendations", []
                    ),
                }
                set_connection_info(connection_info)
            else:
                session_logger.warning(f"Auto-init failed: {result['error']}")
        except Exception as e:
            session_logger.warning(f"Auto-init failed (non-critical): {e}")
    else:
        # Not a git repository - no auto-initialization
        session_logger.debug("Non-git directory - skipping auto-initialization")

    yield  # Server runs normally

    # On disconnect - cleanup for git repos only
    if is_git_repository(current_dir):
        try:
            result = await lifecycle_manager.end_session()
            if result["success"]:
                session_logger.info("✅ Auto-ended session for git repository")
            else:
                session_logger.warning(f"Auto-cleanup failed: {result['error']}")
        except Exception as e:
            session_logger.warning(f"Auto-cleanup failed (non-critical): {e}")


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


async def auto_setup_git_working_directory() -> None:
    """Auto-detect and setup git working directory for enhanced DX."""
    try:
        # Get current working directory
        current_dir = Path.cwd()

        # Import git utilities
        from session_mgmt_mcp.utils.git_operations import (
            get_git_root,
            is_git_repository,
        )

        # Try to find git root from current directory
        git_root = None
        if is_git_repository(current_dir):
            git_root = get_git_root(current_dir)

        if git_root and git_root.exists():
            # Log the auto-setup action for Claude to see
            session_logger.info(f"🔧 Auto-detected git repository: {git_root}")
            session_logger.info(
                f"💡 Recommend: Use `mcp__git__git_set_working_dir` with path='{git_root}'"
            )

            # Also log to stderr for immediate visibility
            print(f"📍 Git repository detected: {git_root}", file=sys.stderr)
            print(
                f"💡 Tip: Auto-setup git working directory with: git_set_working_dir('{git_root}')",
                file=sys.stderr,
            )
        else:
            session_logger.debug(
                "No git repository detected in current directory - skipping auto-setup"
            )

    except Exception as e:
        # Graceful fallback - don't break server startup
        session_logger.debug(f"Git auto-setup failed (non-critical): {e}")


# Register init prompt
async def initialize_new_features() -> None:
    """Initialize multi-project coordination and advanced search features."""
    global multi_project_coordinator, advanced_search_engine, app_config

    # Auto-setup git working directory for enhanced DX
    await auto_setup_git_working_directory()

    # Load configuration
    if CONFIG_AVAILABLE:
        app_config = get_settings()

    # Initialize reflection database for new features
    if REFLECTION_TOOLS_AVAILABLE:
        from contextlib import suppress

        with suppress(Exception):
            db = await get_reflection_database()

            # Initialize multi-project coordinator
            if MULTI_PROJECT_AVAILABLE:
                multi_project_coordinator = MultiProjectCoordinator(db)

            # Initialize advanced search engine
            if ADVANCED_SEARCH_AVAILABLE:
                advanced_search_engine = AdvancedSearchEngine(db)


async def analyze_project_context(project_dir: Path) -> dict[str, bool]:
    """Analyze project structure and context with enhanced error handling."""
    try:
        # Ensure project_dir exists and is accessible
        if not project_dir.exists():
            return {
                "python_project": False,
                "git_repo": False,
                "has_tests": False,
                "has_docs": False,
                "has_requirements": False,
                "has_uv_lock": False,
                "has_mcp_config": False,
            }

        return {
            "python_project": (project_dir / "pyproject.toml").exists(),
            "git_repo": (project_dir / ".git").exists(),
            "has_tests": any(project_dir.glob("test*"))
            or any(project_dir.glob("**/test*")),
            "has_docs": (project_dir / "README.md").exists()
            or any(project_dir.glob("docs/**")),
            "has_requirements": (project_dir / "requirements.txt").exists(),
            "has_uv_lock": (project_dir / "uv.lock").exists(),
            "has_mcp_config": (project_dir / ".mcp.json").exists(),
        }
    except (OSError, PermissionError) as e:
        # Log error but return safe defaults
        print(
            f"Warning: Could not analyze project context for {project_dir}: {e}",
            file=sys.stderr,
        )
        return {
            "python_project": False,
            "git_repo": False,
            "has_tests": False,
            "has_docs": False,
            "has_requirements": False,
            "has_uv_lock": False,
            "has_mcp_config": False,
        }



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

async def _format_quality_results(
    quality_score: int,
    quality_data: dict[str, Any],
    checkpoint_result: dict[str, Any] | None = None,
) -> list[str]:
    """Format quality assessment results for display."""
    output = []

    # Quality status with version indicator
    version = quality_data.get("version", "1.0")
    if quality_score >= 80:
        output.append(
            f"✅ Session quality: EXCELLENT (Score: {quality_score}/100) [V{version}]"
        )
    elif quality_score >= 60:
        output.append(
            f"✅ Session quality: GOOD (Score: {quality_score}/100) [V{version}]"
        )
    else:
        output.append(
            f"⚠️ Session quality: NEEDS ATTENTION (Score: {quality_score}/100) [V{version}]",
        )

    # Quality breakdown - V2 format (actual code quality metrics)
    output.append("\n📈 Quality breakdown (code health metrics):")
    breakdown = quality_data["breakdown"]
    output.append(f"   • Code quality: {breakdown['code_quality']:.1f}/40")
    output.append(f"   • Project health: {breakdown['project_health']:.1f}/30")
    output.append(f"   • Dev velocity: {breakdown['dev_velocity']:.1f}/20")
    output.append(f"   • Security: {breakdown['security']:.1f}/10")

    # Trust score (separate from quality)
    if "trust_score" in quality_data:
        trust = quality_data["trust_score"]
        output.append(f"\n🔐 Trust score: {trust['total']:.0f}/100 (separate metric)")
        output.append(
            f"   • Trusted operations: {trust['breakdown']['trusted_operations']:.0f}/40"
        )
        output.append(
            f"   • Session features: {trust['breakdown']['session_availability']:.0f}/30"
        )
        output.append(
            f"   • Tool ecosystem: {trust['breakdown']['tool_ecosystem']:.0f}/30"
        )

    # Recommendations
    recommendations = quality_data["recommendations"]
    if recommendations:
        output.append("\n💡 Recommendations:")
        for rec in recommendations[:3]:
            output.append(f"   • {rec}")

    # Session management specific results
    if checkpoint_result:
        strengths = checkpoint_result.get("strengths", [])
        if strengths:
            output.append("\n🌟 Session strengths:")
            for strength in strengths[:3]:
                output.append(f"   • {strength}")

        session_stats = checkpoint_result.get("session_stats", {})
        if session_stats:
            output.append("\n⏱️ Session progress:")
            output.append(
                f"   • Duration: {session_stats.get('duration_minutes', 0)} minutes",
            )
            output.append(
                f"   • Checkpoints: {session_stats.get('total_checkpoints', 0)}",
            )
            output.append(
                f"   • Success rate: {session_stats.get('success_rate', 0):.1f}%",
            )

    return output


async def _perform_git_checkpoint(
    current_dir: Path, quality_score: int, project_name: str
) -> list[str]:
    """Handle git operations for checkpoint commit."""
    output = []
    output.append("\n" + "=" * 50)
    output.append("📦 Git Checkpoint Commit")
    output.append("=" * 50)

    # Use the proper checkpoint commit function from git_operations
    from session_mgmt_mcp.utils.git_operations import create_checkpoint_commit

    success, result, commit_output = create_checkpoint_commit(
        current_dir, project_name, quality_score
    )

    # Add the commit output to our output
    output.extend(commit_output)

    if success and result != "clean":
        output.append(f"✅ Checkpoint commit created: {result}")
    elif not success:
        output.append(f"⚠️ Failed to stage files: {result}")

    return output


async def health_check() -> dict[str, Any]:
    """Comprehensive health check for MCP server and toolkit availability."""
    health_status: dict[str, Any] = {
        "overall_healthy": True,
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    # MCP Server health
    try:
        # Test FastMCP availability
        health_status["checks"]["mcp_server"] = "✅ Active"
    except Exception as e:
        health_status["checks"]["mcp_server"] = "❌ Error"
        health_status["errors"].append(f"MCP server issue: {e}")
        health_status["overall_healthy"] = False

    # Session management toolkit health
    health_status["checks"]["session_toolkit"] = (
        "✅ Available" if SESSION_MANAGEMENT_AVAILABLE else "⚠️ Limited"
    )
    if not SESSION_MANAGEMENT_AVAILABLE:
        health_status["warnings"].append(
            "Session management toolkit not fully available",
        )

    # UV package manager health
    uv_available = shutil.which("uv") is not None
    health_status["checks"]["uv_manager"] = (
        "✅ Available" if uv_available else "❌ Missing"
    )
    if not uv_available:
        health_status["warnings"].append("UV package manager not found")

    # Claude directory health
    validate_claude_directory()
    health_status["checks"]["claude_directory"] = "✅ Valid"

    # Permissions system health
    try:
        permissions_status = permissions_manager.get_permission_status()
        health_status["checks"]["permissions_system"] = "✅ Active"
        health_status["checks"]["session_id"] = (
            f"Active ({permissions_status['session_id']})"
        )
    except Exception as e:
        health_status["checks"]["permissions_system"] = "❌ Error"
        health_status["errors"].append(f"Permissions system issue: {e}")
        health_status["overall_healthy"] = False

    # Crackerjack integration health
    health_status["checks"]["crackerjack_integration"] = (
        "✅ Available" if CRACKERJACK_INTEGRATION_AVAILABLE else "⚠️ Not Available"
    )
    if not CRACKERJACK_INTEGRATION_AVAILABLE:
        health_status["warnings"].append(
            "Crackerjack integration not available - quality monitoring disabled",
        )

    # Log health check results
    session_logger.info(
        "Health check completed",
        overall_healthy=health_status["overall_healthy"],
        warnings_count=len(health_status["warnings"]),
        errors_count=len(health_status["errors"]),
    )

    return health_status


async def _add_basic_status_info(output: list[str], current_dir: Path) -> None:
    """Add basic status information to output."""
    global current_project
    current_project = current_dir.name

    output.append(f"📁 Current project: {current_project}")
    output.append(f"🗂️ Working directory: {current_dir}")
    output.append("🌐 MCP server: Active (Claude Session Management)")


async def _add_health_status_info(output: list[str]) -> None:
    """Add health check information to output."""
    health_status = await health_check()

    output.append(
        f"\n🏥 System Health: {'✅ HEALTHY' if health_status['overall_healthy'] else '⚠️ ISSUES DETECTED'}",
    )

    # Display health check results
    for check_name, status in health_status["checks"].items():
        friendly_name = check_name.replace("_", " ").title()
        output.append(f"   • {friendly_name}: {status}")

    # Show warnings and errors
    if health_status["warnings"]:
        output.append("\n⚠️ Health Warnings:")
        for warning in health_status["warnings"][:3]:  # Limit to 3 warnings
            output.append(f"   • {warning}")

    if health_status["errors"]:
        output.append("\n❌ Health Errors:")
        for error in health_status["errors"][:3]:  # Limit to 3 errors
            output.append(f"   • {error}")


async def _get_project_context_info(
    current_dir: Path,
) -> tuple[dict[str, Any], int, int]:
    """Get project context information and scores."""
    project_context = await analyze_project_context(current_dir)
    context_score = sum(1 for detected in project_context.values() if detected)
    max_score = len(project_context)
    return project_context, context_score, max_score


def _should_retry_search(error: Exception) -> bool:
    """Determine if a search error warrants a retry with cleanup."""
    # Retry for database connection issues or temporary errors
    error_msg = str(error).lower()
    retry_conditions = [
        "database is locked",
        "connection failed",
        "temporary failure",
        "timeout",
        "index not found",
    ]
    return any(condition in error_msg for condition in retry_conditions)


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


# Clean Command Aliases
async def _format_conversation_summary() -> list[str]:
    """Format the conversation summary section."""
    output = []
    from contextlib import suppress

    with suppress(Exception):
        conversation_summary = await summarize_current_conversation()
        if conversation_summary["key_topics"]:
            output.append("\n💬 Current Session Focus:")
            for topic in conversation_summary["key_topics"][:3]:
                output.append(f"   • {topic}")

        if conversation_summary["decisions_made"]:
            output.append("\n✅ Key Decisions:")
            for decision in conversation_summary["decisions_made"][:2]:
                output.append(f"   • {decision}")
    return output


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
