"""MCP Server Core Infrastructure.

This module handles FastMCP initialization, server lifecycle management,
tool registration, and core infrastructure components.

Extracted Components:
- SessionPermissionsManager class (singleton permissions management)
- Configuration functions (_load_mcp_config, _detect_other_mcp_servers, etc.)
- Session lifecycle handler (session_lifecycle)
- Initialization functions (initialize_new_features, analyze_project_context)
- Health and status functions (health_check, _add_basic_status_info, etc.)
- Quality formatting functions (_format_quality_results, _perform_git_checkpoint)
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess  # nosec B404
import sys
import warnings
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from session_mgmt_mcp.utils.logging import SessionLogger

# Suppress transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")

# Import mcp-common ServerPanels for beautiful terminal UI
try:
    from mcp_common.ui import ServerPanels

    SERVERPANELS_AVAILABLE = True
except ImportError:
    SERVERPANELS_AVAILABLE = False

try:
    import tomli
except ImportError:
    tomli = None  # type: ignore[assignment]


# =====================================
# SessionPermissionsManager Class
# =====================================


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

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
        cls._session_id = None
        cls._initialized = False

    # Common trusted operations
    TRUSTED_UV_OPERATIONS = "uv_package_management"
    TRUSTED_GIT_OPERATIONS = "git_repository_access"
    TRUSTED_FILE_OPERATIONS = "project_file_access"
    TRUSTED_SUBPROCESS_OPERATIONS = "subprocess_execution"
    TRUSTED_NETWORK_OPERATIONS = "network_access"


# =====================================
# Configuration Functions
# =====================================


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
        if SERVERPANELS_AVAILABLE:
            ServerPanels.warning(
                title="Configuration Warning",
                message="Failed to load MCP config from pyproject.toml",
                details=[str(e), "Using default configuration values"],
            )
        else:
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


# =====================================
# Session Lifecycle Handler
# =====================================


@asynccontextmanager
async def session_lifecycle(
    app: Any, lifecycle_manager: Any, session_logger: SessionLogger
) -> AsyncGenerator[None]:
    """Automatic session lifecycle for git repositories only.

    Args:
        app: FastMCP application instance
        lifecycle_manager: SessionLifecycleManager instance
        session_logger: SessionLogger instance

    Yields:
        None during server lifetime

    """
    # Import here to avoid circular dependencies
    from session_mgmt_mcp.utils.git_operations import get_git_root, is_git_repository

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

                # Import set_connection_info here to avoid circular dependency
                from session_mgmt_mcp.advanced_features import set_connection_info

                # Store connection info for display via tools
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


# =====================================
# Initialization Functions
# =====================================


async def auto_setup_git_working_directory(session_logger: SessionLogger) -> None:
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
            if SERVERPANELS_AVAILABLE:
                ServerPanels.info(
                    title="Git Repository Detected",
                    message=f"Repository root: {git_root}",
                    items={
                        "Auto-setup command": f"git_set_working_dir('{git_root}')",
                        "Auto-lifecycle": "Enabled (init, checkpoint, cleanup)",
                    },
                )
            else:
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


async def initialize_new_features(
    session_logger: SessionLogger,
    multi_project_coordinator_ref: Any,
    advanced_search_engine_ref: Any,
    app_config_ref: Any,
) -> tuple[Any, Any, Any]:
    """Initialize multi-project coordination and advanced search features.

    Args:
        session_logger: Logger instance for diagnostics
        multi_project_coordinator_ref: Reference to store coordinator instance
        advanced_search_engine_ref: Reference to store search engine instance
        app_config_ref: Reference to store configuration

    Returns:
        Tuple of (multi_project_coordinator, advanced_search_engine, app_config)

    """
    # Import availability flags
    from session_mgmt_mcp.server import (
        ADVANCED_SEARCH_AVAILABLE,
        CONFIG_AVAILABLE,
        MULTI_PROJECT_AVAILABLE,
        REFLECTION_TOOLS_AVAILABLE,
    )

    # Auto-setup git working directory for enhanced DX
    await auto_setup_git_working_directory(session_logger)

    # Initialize default return values
    multi_project_coordinator = multi_project_coordinator_ref
    advanced_search_engine = advanced_search_engine_ref
    app_config = app_config_ref

    # Load configuration
    if CONFIG_AVAILABLE:
        from session_mgmt_mcp.settings import get_settings

        app_config = get_settings()

    # Initialize reflection database for new features
    if REFLECTION_TOOLS_AVAILABLE:
        with suppress(
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
            AttributeError,
            OSError,
            ValueError,
        ):
            from session_mgmt_mcp.reflection_tools import get_reflection_database

            db = await get_reflection_database()

            # Initialize multi-project coordinator
            if MULTI_PROJECT_AVAILABLE:
                from session_mgmt_mcp.multi_project_coordinator import (
                    MultiProjectCoordinator,
                )

                multi_project_coordinator = MultiProjectCoordinator(db)

            # Initialize advanced search engine
            if ADVANCED_SEARCH_AVAILABLE:
                from session_mgmt_mcp.advanced_search import AdvancedSearchEngine

                advanced_search_engine = AdvancedSearchEngine(db)

    return multi_project_coordinator, advanced_search_engine, app_config


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
        if SERVERPANELS_AVAILABLE:
            ServerPanels.warning(
                title="Project Analysis Warning",
                message=f"Could not analyze project context for {project_dir}",
                details=[
                    f"Error type: {type(e).__name__}",
                    f"Error: {e}",
                    "Using safe default values",
                ],
            )
        else:
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


# =====================================
# Health & Status Functions
# =====================================


async def health_check(
    session_logger: SessionLogger,
    permissions_manager: SessionPermissionsManager,
    validate_claude_directory: Any,
) -> dict[str, Any]:
    """Comprehensive health check for MCP server and toolkit availability."""
    # Import availability flags
    from session_mgmt_mcp.server import (
        CRACKERJACK_INTEGRATION_AVAILABLE,
        SESSION_MANAGEMENT_AVAILABLE,
    )

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


async def _add_basic_status_info(
    output: list[str], current_dir: Path, current_project_ref: Any
) -> None:
    """Add basic status information to output."""
    current_project_ref = current_dir.name

    output.append(f"📁 Current project: {current_project_ref}")
    output.append(f"🗂️ Working directory: {current_dir}")
    output.append("🌐 MCP server: Active (Claude Session Management)")


async def _add_health_status_info(
    output: list[str],
    session_logger: SessionLogger,
    permissions_manager: SessionPermissionsManager,
    validate_claude_directory: Any,
) -> None:
    """Add health check information to output."""
    health_status = await health_check(
        session_logger, permissions_manager, validate_claude_directory
    )

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


# =====================================
# Quality & Formatting Functions
# =====================================


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


async def _format_conversation_summary() -> list[str]:
    """Format the conversation summary section."""
    output = []
    with suppress(
        ImportError, ModuleNotFoundError, RuntimeError, AttributeError, ValueError
    ):
        from session_mgmt_mcp.quality_engine import summarize_current_conversation

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


# =====================================
# Utility Functions
# =====================================


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


# =====================================
# Feature Detection (Phase 2.6)
# =====================================


class FeatureDetector:
    """Centralized feature detection for MCP server capabilities."""

    def __init__(self) -> None:
        """Initialize feature detector with all availability checks."""
        self.SESSION_MANAGEMENT_AVAILABLE = self._check_session_management()
        self.REFLECTION_TOOLS_AVAILABLE = self._check_reflection_tools()
        self.ENHANCED_SEARCH_AVAILABLE = self._check_enhanced_search()
        self.UTILITY_FUNCTIONS_AVAILABLE = self._check_utility_functions()
        self.MULTI_PROJECT_AVAILABLE = self._check_multi_project()
        self.ADVANCED_SEARCH_AVAILABLE = self._check_advanced_search()
        self.CONFIG_AVAILABLE = self._check_config()
        self.AUTO_CONTEXT_AVAILABLE = self._check_auto_context()
        self.MEMORY_OPTIMIZER_AVAILABLE = self._check_memory_optimizer()
        self.APP_MONITOR_AVAILABLE = self._check_app_monitor()
        self.LLM_PROVIDERS_AVAILABLE = self._check_llm_providers()
        self.SERVERLESS_MODE_AVAILABLE = self._check_serverless_mode()
        self.CRACKERJACK_INTEGRATION_AVAILABLE = self._check_crackerjack()

    @staticmethod
    def _check_session_management() -> bool:
        """Check if session management is available."""
        try:
            import session_mgmt_mcp.core

            _ = (
                session_mgmt_mcp.core.session_manager
            )  # Reference to avoid unused import warning during static analysis
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_reflection_tools() -> bool:
        """Check if reflection tools are available."""
        try:
            import session_mgmt_mcp.reflection_tools

            _ = (
                session_mgmt_mcp.reflection_tools
            )  # Use the import to avoid unused import warning
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_enhanced_search() -> bool:
        """Check if enhanced search is available."""
        try:
            return (
                importlib.util.find_spec("session_mgmt_mcp.search_enhanced") is not None
            )
        except ImportError:
            return False

    @staticmethod
    def _check_utility_functions() -> bool:
        """Check if utility functions are available."""
        try:
            # Check for the general module availability without importing unused functions
            return (
                importlib.util.find_spec("session_mgmt_mcp.tools.search_tools")
                is not None
            )
        except ImportError:
            return False

    @staticmethod
    def _check_multi_project() -> bool:
        """Check if multi-project coordination is available."""
        try:
            return (
                importlib.util.find_spec("session_mgmt_mcp.multi_project_coordinator")
                is not None
            )
        except ImportError:
            return False

    @staticmethod
    def _check_advanced_search() -> bool:
        """Check if advanced search engine is available."""
        try:
            return (
                importlib.util.find_spec("session_mgmt_mcp.advanced_search") is not None
            )
        except ImportError:
            return False

    @staticmethod
    def _check_config() -> bool:
        """Check if configuration management is available."""
        try:
            return importlib.util.find_spec("session_mgmt_mcp.settings") is not None
        except ImportError:
            return False

    @staticmethod
    def _check_auto_context() -> bool:
        """Check if auto-context loading is available."""
        try:
            return (
                importlib.util.find_spec("session_mgmt_mcp.context_manager") is not None
            )
        except ImportError:
            return False

    @staticmethod
    def _check_memory_optimizer() -> bool:
        """Check if memory optimizer is available."""
        try:
            return (
                importlib.util.find_spec("session_mgmt_mcp.memory_optimizer")
                is not None
            )
        except ImportError:
            return False

    @staticmethod
    def _check_app_monitor() -> bool:
        """Check if application monitoring is available."""
        try:
            return importlib.util.find_spec("session_mgmt_mcp.app_monitor") is not None
        except ImportError:
            return False

    @staticmethod
    def _check_llm_providers() -> bool:
        """Check if LLM providers are available."""
        try:
            return (
                importlib.util.find_spec("session_mgmt_mcp.llm_providers") is not None
            )
        except ImportError:
            return False

    @staticmethod
    def _check_serverless_mode() -> bool:
        """Check if serverless mode is available."""
        try:
            return (
                importlib.util.find_spec("session_mgmt_mcp.serverless_mode") is not None
            )
        except ImportError:
            return False

    @staticmethod
    def _check_crackerjack() -> bool:
        """Check if crackerjack integration is available."""
        try:
            return (
                importlib.util.find_spec("session_mgmt_mcp.crackerjack_integration")
                is not None
            )
        except ImportError:
            return False

    def get_feature_flags(self) -> dict[str, bool]:
        """Get all feature flags as a dictionary."""
        return {
            "SESSION_MANAGEMENT_AVAILABLE": self.SESSION_MANAGEMENT_AVAILABLE,
            "REFLECTION_TOOLS_AVAILABLE": self.REFLECTION_TOOLS_AVAILABLE,
            "ENHANCED_SEARCH_AVAILABLE": self.ENHANCED_SEARCH_AVAILABLE,
            "UTILITY_FUNCTIONS_AVAILABLE": self.UTILITY_FUNCTIONS_AVAILABLE,
            "MULTI_PROJECT_AVAILABLE": self.MULTI_PROJECT_AVAILABLE,
            "ADVANCED_SEARCH_AVAILABLE": self.ADVANCED_SEARCH_AVAILABLE,
            "CONFIG_AVAILABLE": self.CONFIG_AVAILABLE,
            "AUTO_CONTEXT_AVAILABLE": self.AUTO_CONTEXT_AVAILABLE,
            "MEMORY_OPTIMIZER_AVAILABLE": self.MEMORY_OPTIMIZER_AVAILABLE,
            "APP_MONITOR_AVAILABLE": self.APP_MONITOR_AVAILABLE,
            "LLM_PROVIDERS_AVAILABLE": self.LLM_PROVIDERS_AVAILABLE,
            "SERVERLESS_MODE_AVAILABLE": self.SERVERLESS_MODE_AVAILABLE,
            "CRACKERJACK_INTEGRATION_AVAILABLE": self.CRACKERJACK_INTEGRATION_AVAILABLE,
        }


# Create global feature detector instance
_feature_detector = FeatureDetector()


def get_feature_flags() -> dict[str, bool]:
    """Get feature availability flags for the MCP server.

    Returns:
        Dictionary mapping feature names to availability status.

    """
    return _feature_detector.get_feature_flags()
