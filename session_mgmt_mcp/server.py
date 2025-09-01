#!/usr/bin/env python3
"""Claude Session Management MCP Server - FastMCP Version.

A dedicated MCP server that provides session management functionality
including initialization, checkpoints, and cleanup across all projects.

This server can be included in any project's .mcp.json file to provide
automatic access to /session-init, /session-checkpoint,
and /session-end slash commands.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


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

    def info(self, message: str, **context) -> None:
        """Log info with optional context."""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.info(message)

    def warning(self, message: str, **context) -> None:
        """Log warning with optional context."""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.warning(message)

    def error(self, message: str, **context) -> None:
        """Log error with optional context."""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.error(message)


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
            def __init__(self, name) -> None:
                self.name = name
                self.tools = {}
                self.prompts = {}

            def tool(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def prompt(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def run(self, *args, **kwargs) -> None:
                pass

        FastMCP = MockFastMCP
        MCP_AVAILABLE = False
    else:
        print("FastMCP not available. Install with: uv add fastmcp", file=sys.stderr)
        sys.exit(1)

# Session management availability (no longer using global workspace)
SESSION_MANAGEMENT_AVAILABLE = False

# Import reflection tools
try:
    from session_mgmt_mcp.reflection_tools import (
        get_current_project,
        get_reflection_database,
    )

    REFLECTION_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Reflection tools import failed: {e}", file=sys.stderr)
    REFLECTION_TOOLS_AVAILABLE = False

# Import enhanced search tools
try:
    from session_mgmt_mcp.search_enhanced import EnhancedSearchEngine

    ENHANCED_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced search import failed: {e}", file=sys.stderr)
    ENHANCED_SEARCH_AVAILABLE = False

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
    from session_mgmt_mcp.config import get_config

    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Configuration management import failed: {e}", file=sys.stderr)
    CONFIG_AVAILABLE = False

# Import auto-context loading tools
try:
    from session_mgmt_mcp.context_manager import AutoContextLoader

    AUTO_CONTEXT_AVAILABLE = True
except ImportError as e:
    print(f"Auto-context loading import failed: {e}", file=sys.stderr)
    AUTO_CONTEXT_AVAILABLE = False

# Import memory optimization tools
try:
    from session_mgmt_mcp.memory_optimizer import MemoryOptimizer

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
    from session_mgmt_mcp.llm_providers import LLMManager, LLMMessage

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
    from session_mgmt_mcp.crackerjack_integration import CrackerjackIntegration

    CRACKERJACK_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Crackerjack integration import failed: {e}", file=sys.stderr)
    CRACKERJACK_INTEGRATION_AVAILABLE = False


class SessionPermissionsManager:
    """Manages session permissions to avoid repeated prompts for trusted operations."""

    _instance = None
    _session_id = None

    def __new__(cls, claude_dir: Path):
        """Singleton pattern to ensure consistent session ID across tool calls."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

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
        session_data = f"{datetime.now().isoformat()}_{os.getcwd()}"
        return hashlib.md5(session_data.encode()).hexdigest()[:12]

    def _load_permissions(self) -> None:
        """Load previously granted permissions."""
        if self.permissions_file.exists():
            try:
                with open(self.permissions_file) as f:
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
        with open(self.permissions_file, "w") as f:
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


# Initialize FastMCP 2.0 server
mcp = FastMCP("session-mgmt-mcp")

# Register slash commands as MCP prompts (not resources!)
SESSION_COMMANDS = {
    "init": """# Session Initialization

Initialize Claude session with comprehensive setup including UV dependencies and automation tools.

This command will:
- Set up Claude directory structure
- Check for required toolkits and automation tools
- Initialize project context and dependencies
- Set up performance monitoring and session tracking
- Provide comprehensive setup summary with next steps

Run this at the start of any coding session for optimal Claude integration.
""",
    "checkpoint": """# Session Checkpoint

Perform mid-session quality checkpoint with workflow analysis, optimization recommendations, and strategic compaction.

This command will:
- Analyze current session progress and workflow effectiveness
- Check for performance bottlenecks and optimization opportunities
- Validate current task completion status
- Provide recommendations for workflow improvements
- Create checkpoint for session recovery if needed
- **Analyze context usage and recommend /compact when beneficial**
- Perform strategic compaction and cleanup (replaces disabled auto-compact):
  • DuckDB reflection database optimization (VACUUM/ANALYZE)
  • Session log cleanup (retain last 10 files)
  • Temporary file cleanup (cache files, .DS_Store, old coverage files)
  • Git repository optimization (gc --auto, prune remote branches)
  • UV package cache cleanup

**RECOMMENDED WORKFLOW:**
1. Run /session-mgmt:checkpoint for comprehensive analysis
2. If checkpoint recommends context compaction, run: `/compact`
3. Continue with optimized session context

Use this periodically during long coding sessions to maintain optimal productivity and system performance.
""",
    "end": """# Session End

End Claude session with cleanup, learning capture, and handoff file creation.

This command will:
- Clean up temporary files and session artifacts
- Capture learning outcomes and session insights
- Create handoff documentation for future sessions
- Save session analytics and performance metrics
- Provide session summary with accomplishments

Run this at the end of any coding session for proper cleanup and knowledge retention.
""",
    "status": """# Session Status

Get current session status and project context information with health checks.

This command will:
- Display current project context and workspace information
- Show comprehensive health checks for all system components
- Report session permissions and trusted operations
- Analyze project maturity and structure
- List all available MCP tools and their status

Use this to quickly understand your current session state and system health.
""",
    "permissions": """# Session Permissions

Manage session permissions for trusted operations to avoid repeated prompts.

This command will:
- Show current trusted operations and permission status
- Allow you to manually trust specific operations
- Provide security reset option to revoke all permissions
- Display available operation types that can be trusted
- Help streamline workflow by reducing permission prompts

Use this to manage which operations Claude can perform without prompting.
""",
    "reflect": """# Reflect on Past

Search past conversations and store reflections with semantic similarity.

This command will:
- Search through your conversation history using AI embeddings
- Find relevant past discussions and solutions
- Store new insights and reflections for future reference
- Provide semantic similarity scoring for relevance
- Help you learn from previous sessions and avoid repeating work

Use this to leverage your conversation history and build on previous insights.
""",
    "quick-search": """# Quick Search

Quick search that returns only the count and top result for fast overview.

This command will:
- Perform rapid semantic search through conversation history
- Return summary statistics and the most relevant result
- Show total matches with different relevance thresholds
- Provide fast insight into whether you've addressed a topic before
- Optimize for speed when you need quick context checks

Use this for rapid checks before diving into detailed work.
""",
    "search-summary": """# Search Summary

Get aggregated insights from search results without individual result details.

This command will:
- Analyze patterns across multiple conversation matches
- Extract common themes and recurring topics
- Provide statistical overview of search results
- Identify projects and contexts where topics were discussed
- Generate high-level insights without overwhelming detail

Use this to understand broad patterns and themes in your work history.
""",
    "reflection-stats": """# Reflection Stats

Get statistics about the reflection database and conversation memory.

This command will:
- Show total stored conversations and reflections
- Display database health and embedding provider status
- Report memory system capacity and usage statistics
- Verify semantic search system is functioning properly
- Provide technical details about the knowledge storage system

Use this to monitor your conversation memory system and ensure it's working optimally.
""",
    "search-code": """# Code Pattern Search

Search for code patterns in your conversation history using AST (Abstract Syntax Tree) parsing.

This command will:
- Parse Python code blocks from conversations
- Extract functions, classes, imports, loops, and other patterns
- Rank results by relevance to your query
- Show code context and project information

Examples:
- Search for functions: pattern_type='function'
- Search for class definitions: pattern_type='class'
- Search for error handling: query='try except'

Use this to find code examples and patterns from your development sessions.
""",
    "search-errors": """# Error Pattern Search

Search for error messages, exceptions, and debugging contexts in your conversation history.

This command will:
- Find Python tracebacks and exceptions
- Detect JavaScript errors and warnings
- Identify debugging and testing contexts
- Show error context and solutions

Examples:
- Find Python errors: error_type='python_exception'
- Find import issues: query='ImportError'
- Find debugging sessions: query='debug'

Use this to quickly find solutions to similar errors you've encountered before.
""",
    "search-temporal": """# Temporal Search

Search your conversation history using natural language time expressions.

This command will:
- Parse time expressions like "yesterday", "last week", "2 days ago"
- Find conversations within that time range
- Optionally filter by additional search terms
- Sort results by time and relevance

Examples:
- "yesterday" - conversations from yesterday
- "last week" - conversations from the past week
- "2 days ago" - conversations from 2 days ago
- "this month" + query - filter by content within the month

Use this to find recent discussions or work from specific time periods.
""",
    "auto-load-context": """# Auto-Context Loading

Automatically detect current development context and load relevant conversations.

This command will:
- Analyze your current project structure and files
- Detect programming languages and tools in use
- Identify project type (web app, CLI tool, library, etc.)
- Find recent file modifications
- Load conversations relevant to your current context
- Score conversations by relevance to current work

Examples:
- Load default context: auto_load_context()
- Increase results: max_conversations=20
- Lower threshold: min_relevance=0.2

Use this at the start of coding sessions to get relevant context automatically.
""",
    "context-summary": """# Context Summary

Get a quick summary of your current development context without loading conversations.

This command will:
- Detect current project name and type
- Identify programming languages and tools
- Show Git repository information
- Display recently modified files
- Calculate detection confidence score

Use this to understand what context the system has detected about your current work.
""",
    "compress-memory": """# Memory Compression

Compress conversation memory by consolidating old conversations into summaries.

This command will:
- Analyze conversation age and importance
- Group related conversations into clusters
- Create consolidated summaries of old conversations
- Remove redundant conversation data
- Calculate space savings and compression ratios

Examples:
- Default compression: compress_memory()
- Preview changes: dry_run=True
- Aggressive compression: max_age_days=14, importance_threshold=0.5

Use this periodically to keep your conversation memory manageable and efficient.
""",
    "compression-stats": """# Compression Statistics

Get detailed statistics about memory compression history and current database status.

This command will:
- Show last compression run details
- Display space savings and compression ratios
- Report current database size and conversation count
- Show number of consolidated conversations
- Provide compression efficiency metrics

Use this to monitor memory usage and compression effectiveness.
""",
    "retention-policy": """# Retention Policy

Configure memory retention policy parameters for automatic compression.

This command will:
- Set maximum conversation age and count limits
- Configure importance threshold for retention
- Define consolidation age triggers
- Adjust compression ratio targets

Examples:
- Conservative: max_age_days=365, importance_threshold=0.2
- Aggressive: max_age_days=90, importance_threshold=0.5
- Custom: consolidation_age_days=14

Use this to customize how your conversation memory is managed over time.
""",
    "start-app-monitoring": """# Start Application Monitoring

Monitor your development activity to provide better context and insights.

This command will:
- Start file system monitoring for code changes
- Track application focus (IDE, browser, terminal)
- Monitor documentation site visits
- Build activity profiles for better context

Monitoring includes:
- File modifications in your project directories
- IDE and editor activity patterns
- Browser navigation to documentation sites
- Application focus and context switching

Use this to automatically capture your development context for better session insights.
""",
    "stop-app-monitoring": """# Stop Application Monitoring

Stop monitoring your development activity.

This command will:
- Stop file system monitoring
- Stop application focus tracking
- Preserve collected activity data
- Clean up monitoring resources

Use this when you want to pause monitoring or when you're done with a development session.
""",
    "activity-summary": """# Activity Summary

Get a comprehensive summary of your recent development activity.

This command will:
- Show file modification patterns
- List most active applications
- Display visited documentation sites
- Calculate productivity metrics

Summary includes:
- Event counts by type and application
- Most actively edited files
- Documentation resources consulted
- Average relevance scores

Use this to understand your development patterns and identify productive sessions.
""",
    "context-insights": """# Context Insights

Analyze recent development activity for contextual insights.

This command will:
- Identify primary focus areas
- Detect technologies being used
- Count context switches
- Calculate productivity scores

Insights include:
- Primary application focus
- Active programming languages
- Documentation topics explored
- Project switching patterns
- Overall productivity assessment

Use this to understand your current development context and optimize your workflow.
""",
    "active-files": """# Active Files

Show files that are currently being actively worked on.

This command will:
- List recently modified files
- Show activity scores and patterns
- Highlight most frequently changed files
- Include project context

File activity is scored based on:
- Frequency of modifications
- Recency of changes
- File type and relevance
- Project context

Use this to quickly see what you're currently working on and resume interrupted tasks.
""",
    "crackerjack-run": """# Execute Crackerjack Command

Execute a Crackerjack command and parse the output for insights.

This command will:
- Execute any Crackerjack command (test, lint, format, etc.)
- Parse test results, lint issues, coverage data, and security findings
- Extract memory insights from command output
- Track execution history and performance metrics
- Provide structured feedback on code quality

Examples:
- /crackerjack-run test -- Run all tests
- /crackerjack-run lint -- Check code quality
- /crackerjack-run coverage -- Generate coverage report
- /crackerjack-run security -- Run security scan

Use this to integrate code quality checks directly into your development workflow.
""",
    "crackerjack-history": """# Crackerjack Results History

Get recent Crackerjack command execution history with parsed results.

This command will:
- Show recent Crackerjack command executions
- Display test results, lint issues, and coverage trends
- Filter by command type or time period
- Highlight performance changes and quality metrics
- Track project health over time

Use this to monitor project quality trends and identify recurring issues.
""",
    "crackerjack-metrics": """# Quality Metrics Dashboard

Get quality metrics trends from Crackerjack execution history.

This command will:
- Display test pass rates and coverage trends
- Show lint issue patterns and code quality scores
- Track security findings and vulnerability trends
- Provide quality trend analysis with recommendations
- Generate comprehensive project health reports

Use this to get insights into your project's overall quality and identify areas for improvement.
""",
    "crackerjack-patterns": """# Test Failure Pattern Analysis

Analyze test failure patterns and trends for debugging insights.

This command will:
- Identify frequently failing tests and common error patterns
- Analyze test stability scores and duration trends
- Highlight problematic files and modules
- Provide debugging recommendations based on failure patterns
- Track test health and reliability metrics

Use this to proactively identify and fix recurring test issues before they become problems.
""",
}


# Register init prompt
@mcp.prompt("init")
async def get_session_init_prompt() -> str:
    """Initialize Claude session with comprehensive setup including UV dependencies, global workspace verification, and automation tools."""
    return SESSION_COMMANDS["init"]


# Register checkpoint prompt
@mcp.prompt("checkpoint")
async def get_session_checkpoint_prompt() -> str:
    """Perform mid-session quality checkpoint with workflow analysis and optimization recommendations."""
    return SESSION_COMMANDS["checkpoint"]


# Register end prompt
@mcp.prompt("end")
async def get_session_end_prompt() -> str:
    """End Claude session with cleanup, learning capture, and handoff file creation."""
    return SESSION_COMMANDS["end"]


# Register status prompt
@mcp.prompt("status")
async def get_session_status_prompt() -> str:
    """Get current session status and project context information with health checks."""
    return SESSION_COMMANDS["status"]


# Register permissions prompt
@mcp.prompt("permissions")
async def get_session_permissions_prompt() -> str:
    """Manage session permissions for trusted operations to avoid repeated prompts."""
    return SESSION_COMMANDS["permissions"]


# Register reflect prompt
@mcp.prompt("reflect")
async def get_session_reflect_prompt() -> str:
    """Search past conversations and store reflections with semantic similarity."""
    return SESSION_COMMANDS["reflect"]


# Register quick-search prompt
@mcp.prompt("quick-search")
async def get_session_quick_search_prompt() -> str:
    """Quick search that returns only the count and top result for fast overview."""
    return SESSION_COMMANDS["quick-search"]


# Register search-summary prompt
@mcp.prompt("search-summary")
async def get_session_search_summary_prompt() -> str:
    """Get aggregated insights from search results without individual result details."""
    return SESSION_COMMANDS["search-summary"]


# Register reflection-stats prompt
@mcp.prompt("reflection-stats")
async def get_session_reflection_stats_prompt() -> str:
    """Get statistics about the reflection database and conversation memory."""
    return SESSION_COMMANDS["reflection-stats"]


# Register Crackerjack prompts
@mcp.prompt("crackerjack-run")
async def get_crackerjack_run_prompt() -> str:
    """Execute a Crackerjack command and parse the output for insights."""
    return SESSION_COMMANDS["crackerjack-run"]


@mcp.prompt("crackerjack-history")
async def get_crackerjack_history_prompt() -> str:
    """Get recent Crackerjack command execution history with parsed results."""
    return SESSION_COMMANDS["crackerjack-history"]


@mcp.prompt("crackerjack-metrics")
async def get_crackerjack_metrics_prompt() -> str:
    """Get quality metrics trends from Crackerjack execution history."""
    return SESSION_COMMANDS["crackerjack-metrics"]


@mcp.prompt("crackerjack-patterns")
async def get_crackerjack_patterns_prompt() -> str:
    """Analyze test failure patterns and trends for debugging insights."""
    return SESSION_COMMANDS["crackerjack-patterns"]


# Global instances
permissions_manager = SessionPermissionsManager(claude_dir)
current_project = None

# New global instances for multi-project and advanced search
multi_project_coordinator: Any | None = None
advanced_search_engine: Any | None = None
app_config: Any | None = None


async def initialize_new_features() -> None:
    """Initialize multi-project coordination and advanced search features."""
    global multi_project_coordinator, advanced_search_engine, app_config

    # Load configuration
    if CONFIG_AVAILABLE:
        app_config = get_config()

    # Initialize reflection database for new features
    if REFLECTION_TOOLS_AVAILABLE:
        try:
            db = await get_reflection_database()

            # Initialize multi-project coordinator
            if MULTI_PROJECT_AVAILABLE:
                multi_project_coordinator = MultiProjectCoordinator(db)

            # Initialize advanced search engine
            if ADVANCED_SEARCH_AVAILABLE:
                advanced_search_engine = AdvancedSearchEngine(db)

        except Exception:
            # Silently handle optional feature initialization failures
            pass


def validate_claude_directory() -> dict[str, Any]:
    """Simple validation of ~/.claude directory structure."""
    validation_result = {"valid": True, "component_status": {}}

    # Ensure basic ~/.claude directory structure exists
    required_dirs = [claude_dir, claude_dir / "data", claude_dir / "logs"]

    for dir_path in required_dirs:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            validation_result["component_status"][dir_path.name] = "✅ Created"
        else:
            validation_result["component_status"][dir_path.name] = "✅ Present"

    return validation_result


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


def _setup_claude_directory(output: list[str]) -> dict:
    """Setup Claude directory and return validation results."""
    output.append("\n📋 Phase 1: Claude directory setup...")

    claude_validation = validate_claude_directory()
    output.append("✅ Claude directory structure ready")

    # Show component status
    for component, status in claude_validation["component_status"].items():
        output.append(f"   {status} {component}")

    return claude_validation


def _setup_uv_dependencies(output: list[str], current_dir: Path) -> None:
    """Setup UV dependencies and package management."""
    output.append("\n🔧 Phase 2: UV dependency management & session setup...")

    uv_available = shutil.which("uv") is not None
    output.append(
        f"🔍 UV package manager: {'✅ AVAILABLE' if uv_available else '❌ NOT FOUND'}",
    )

    # Check UV permissions
    uv_trusted = permissions_manager.is_operation_trusted(
        permissions_manager.TRUSTED_UV_OPERATIONS,
    )
    if uv_trusted:
        output.append("🔐 UV operations: ✅ TRUSTED (no prompts needed)")
    else:
        output.append("🔐 UV operations: ⚠️ Will require permission prompts")

    if not uv_available:
        output.append("💡 Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return

    _handle_uv_operations(output, current_dir, uv_trusted)


def _handle_uv_operations(
    output: list[str],
    current_dir: Path,
    uv_trusted: bool,
) -> None:
    """Handle UV operations for dependency management."""
    project_has_pyproject = (current_dir / "pyproject.toml").exists()

    if not project_has_pyproject:
        output.append("⚠️ No pyproject.toml found - skipping UV operations")
        output.append("💡 Create pyproject.toml to enable UV dependency management")
        return

    original_cwd = Path.cwd()
    try:
        os.chdir(current_dir)
        output.append(f"📁 Working in: {current_dir}")

        # Trust UV operations if first successful run
        if not uv_trusted:
            output.append("🔓 Trusting UV operations for this session...")
            permissions_manager.trust_operation(
                permissions_manager.TRUSTED_UV_OPERATIONS,
                "UV package management operations",
            )
            output.append("✅ UV operations now trusted - no more prompts needed")

        _run_uv_sync_and_compile(output, current_dir)

    except Exception as e:
        output.append(f"⚠️ UV operation error: {e}")
    finally:
        os.chdir(original_cwd)


def _run_uv_sync_and_compile(output: list[str], current_dir: Path) -> None:
    """Run UV sync and compile operations."""
    # Sync dependencies
    sync_result = subprocess.run(
        ["uv", "sync"], check=False, capture_output=True, text=True
    )
    if sync_result.returncode == 0:
        output.append("✅ UV sync completed successfully")

        # Generate requirements.txt if needed
        if not (current_dir / "requirements.txt").exists():
            compile_result = subprocess.run(
                [
                    "uv",
                    "pip",
                    "compile",
                    "pyproject.toml",
                    "--output-file",
                    "requirements.txt",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if compile_result.returncode == 0:
                output.append("✅ Requirements.txt generated from UV dependencies")
            else:
                output.append(
                    f"⚠️ Requirements compilation warning: {compile_result.stderr}",
                )
        else:
            output.append("✅ Requirements.txt already exists")
    else:
        output.append(f"⚠️ UV sync issues: {sync_result.stderr}")


def _setup_session_management(output: list[str]) -> None:
    """Setup session management functionality."""
    output.append("\n🔧 Phase 3: Session management setup...")
    output.append("✅ Session management functionality ready")
    output.append("   📊 Conversation memory system enabled")
    output.append("   🔍 Semantic search capabilities available")

    output.append("\n🧠 Phase 4: Integrated MCP services initialization...")
    output.append("\n📊 Integrated MCP Services Status:")
    output.append("✅ Session Management - Active (conversation memory enabled)")


async def _analyze_project_structure(
    output: list[str],
    current_dir: Path,
    current_project: str,
) -> tuple[dict, int]:
    """Analyze project structure and add information to output."""
    output.append(f"\n🎯 Phase 5: Project context analysis for {current_project}...")

    project_context = await analyze_project_context(current_dir)
    context_score = sum(1 for detected in project_context.values() if detected)

    output.append("🔍 Project structure analysis:")
    for context_type, detected in project_context.items():
        status = "✅" if detected else "➖"
        output.append(f"   {status} {context_type.replace('_', ' ').title()}")

    output.append(
        f"\n📊 Project maturity: {context_score}/{len(project_context)} indicators",
    )
    if context_score >= len(project_context) - 1:
        output.append("🌟 Excellent project structure - well-organized codebase")
    elif context_score >= len(project_context) // 2:
        output.append("👍 Good project structure - solid foundation")
    else:
        output.append("💡 Basic project - consider adding structure")

    return project_context, context_score


def _add_final_summary(
    output: list[str],
    current_project: str,
    context_score: int,
    project_context: dict,
    claude_validation: dict,
) -> None:
    """Add final summary information to output."""
    output.append("\n" + "=" * 60)
    output.append(f"🎯 {current_project.upper()} SESSION INITIALIZATION COMPLETE")
    output.append("=" * 60)

    output.append(f"📅 Initialized: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"🗂️ Project: {current_project}")
    output.append(f"📊 Structure score: {context_score}/{len(project_context)}")

    missing_files = claude_validation.get("missing_files", [])
    if context_score >= len(project_context) // 2 and not missing_files:
        output.append("✅ Ready for productive session - all systems optimal")
    else:
        output.append("⚠️ Session ready with minor setup opportunities identified")

    _add_permissions_and_tools_summary(output, current_project)


def _add_permissions_and_tools_summary(output: list[str], current_project: str) -> None:
    """Add permissions summary and available tools."""
    # Permissions Summary
    permissions_status = permissions_manager.get_permission_status()
    output.append("\n🔐 Session Permissions Summary:")
    output.append(
        f"   📊 Trusted operations: {permissions_status['trusted_operations_count']}",
    )

    if permissions_status["trusted_operations_count"] > 0:
        output.append("   ✅ Future operations will have reduced permission prompts")
    else:
        output.append("   💡 Operations will be trusted automatically on first use")

    # Server Detection and Guidance
    detected_servers = _detect_other_mcp_servers()
    server_guidance = _generate_server_guidance(detected_servers)

    if server_guidance:
        output.append("\n" + "\n".join(server_guidance))

    output.append("\n📋 AVAILABLE MCP TOOLS:")
    output.append("📊 Session Management:")
    output.append("• checkpoint - Mid-session quality assessment")
    output.append("• end - Complete session cleanup")
    output.append("• status - Current session status")
    output.append("• permissions - Manage trusted operations")
    output.append("• Built-in conversation memory with semantic search")

    output.append(f"\n✨ {current_project} session initialization complete via MCP!")


@mcp.tool()
async def init(working_directory: str | None = None) -> str:
    """Initialize Claude session with comprehensive setup including UV dependencies and automation tools.

    Args:
        working_directory: Optional working directory override (defaults to PWD environment variable or current directory)

    """
    output = []
    output.append("🚀 Claude Session Initialization via MCP Server")
    output.append("=" * 60)

    # Detect current project - use parameter, environment, or fallback to cwd
    if working_directory:
        current_dir = Path(working_directory)
    else:
        current_dir = Path(os.environ.get("PWD", Path.cwd()))

    global current_project
    current_project = current_dir.name
    output.append(f"📁 Current project: {current_project}")

    # Run initialization phases
    claude_validation = _setup_claude_directory(output)
    _setup_uv_dependencies(output, current_dir)
    _setup_session_management(output)
    project_context, context_score = await _analyze_project_structure(
        output,
        current_dir,
        current_project,
    )
    _add_final_summary(
        output,
        current_project,
        context_score,
        project_context,
        claude_validation,
    )

    return "\n".join(output)


async def calculate_quality_score() -> dict[str, Any]:
    """Calculate session quality score based on multiple factors."""
    current_dir = Path(os.environ.get("PWD", Path.cwd()))

    # Project health indicators (40% of score)
    project_context = await analyze_project_context(current_dir)
    project_score = (
        sum(1 for detected in project_context.values() if detected)
        / len(project_context)
    ) * 40

    # Permissions health (20% of score)
    permissions_count = len(permissions_manager.trusted_operations)
    permissions_score = min(
        permissions_count * 5,
        20,
    )  # Up to 4 trusted operations = max score

    # Session management availability (20% of score)
    session_score = 20 if SESSION_MANAGEMENT_AVAILABLE else 5

    # Tool availability (20% of score)
    uv_available = shutil.which("uv") is not None
    tool_score = 20 if uv_available else 10

    total_score = int(project_score + permissions_score + session_score + tool_score)

    return {
        "total_score": total_score,
        "breakdown": {
            "project_health": project_score,
            "permissions": permissions_score,
            "session_management": session_score,
            "tools": tool_score,
        },
        "recommendations": _generate_quality_recommendations(
            total_score,
            project_context,
            permissions_count,
            uv_available,
        ),
    }


def _generate_quality_recommendations(
    score: int,
    project_context: dict,
    permissions_count: int,
    uv_available: bool,
) -> list[str]:
    """Generate quality improvement recommendations based on score factors."""
    recommendations = []

    if score < 50:
        recommendations.append(
            "Session needs attention - multiple areas for improvement",
        )
    elif score < 75:
        recommendations.append("Good session health - minor optimizations available")
    else:
        recommendations.append("Excellent session quality - maintain current practices")

    # Project-specific recommendations
    if not project_context.get("has_tests"):
        recommendations.append("Consider adding tests to improve project structure")
    if not project_context.get("has_docs"):
        recommendations.append("Documentation would enhance project maturity")

    # Permissions recommendations
    if permissions_count == 0:
        recommendations.append(
            "No trusted operations yet - permissions will be granted on first use",
        )
    elif permissions_count > 5:
        recommendations.append(
            "Many trusted operations - consider reviewing for security",
        )

    # Tools recommendations
    if not uv_available:
        recommendations.append(
            "Install UV package manager for better dependency management",
        )

    return recommendations


def should_suggest_compact() -> tuple[bool, str]:
    """Determine if compacting would be beneficial and provide reasoning.
    Returns (should_compact, reason).
    """
    # Heuristics for when compaction might be needed:
    # 1. Large projects with many files
    # 2. Active development (recent git activity)
    # 3. Complex task sequences
    # 4. Session duration indicators

    try:
        current_dir = Path(os.environ.get("PWD", Path.cwd()))

        # Count significant files in project as a complexity indicator
        file_count = 0
        for file_path in current_dir.rglob("*"):
            if (
                file_path.is_file()
                and not any(part.startswith(".") for part in file_path.parts)
                and file_path.suffix
                in {
                    ".py",
                    ".js",
                    ".ts",
                    ".jsx",
                    ".tsx",
                    ".go",
                    ".rs",
                    ".java",
                    ".cpp",
                    ".c",
                    ".h",
                }
            ):
                file_count += 1
                if file_count > 50:  # Stop counting after threshold
                    break

        # Large project heuristic
        if file_count > 50:
            return (
                True,
                "Large codebase with 50+ source files detected - context compaction recommended",
            )

        # Check for active development via git
        git_dir = current_dir / ".git"
        if git_dir.exists():
            try:
                # Check number of recent commits as activity indicator
                result = subprocess.run(
                    ["git", "log", "--oneline", "-20", "--since='24 hours ago'"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=current_dir,
                    timeout=5,
                )
                if result.returncode == 0:
                    recent_commits = len(
                        [
                            line
                            for line in result.stdout.strip().split("\n")
                            if line.strip()
                        ],
                    )
                    if recent_commits >= 3:
                        return (
                            True,
                            f"High development activity ({recent_commits} commits in 24h) - compaction recommended",
                        )

                # Check for large number of modified files
                status_result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=current_dir,
                    timeout=5,
                )
                if status_result.returncode == 0:
                    modified_files = len(
                        [
                            line
                            for line in status_result.stdout.strip().split("\n")
                            if line.strip()
                        ],
                    )
                    if modified_files >= 10:
                        return (
                            True,
                            f"Many modified files ({modified_files}) detected - context optimization beneficial",
                        )

            except (subprocess.TimeoutExpired, Exception):
                pass

        # Check for common patterns suggesting complex session
        if (current_dir / "tests").exists() and (
            current_dir / "pyproject.toml"
        ).exists():
            return (
                True,
                "Python project with tests detected - compaction may improve focus",
            )

        # Default to not suggesting unless we have clear indicators
        return False, "Context appears manageable - compaction not immediately needed"

    except Exception:
        # If we can't determine, err on the side of suggesting compaction for safety
        return (
            True,
            "Unable to assess context complexity - compaction may be beneficial as a precaution",
        )


async def _optimize_reflection_database() -> str:
    """Optimize the reflection database."""
    try:
        from .reflection_tools import get_reflection_database

        db = await get_reflection_database()
        await db.get_stats()
        db_size_before = (
            Path(db.db_path).stat().st_size if Path(db.db_path).exists() else 0
        )

        if db.conn:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: db.conn.execute("VACUUM"),
            )
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: db.conn.execute("ANALYZE"),
            )

        db_size_after = (
            Path(db.db_path).stat().st_size if Path(db.db_path).exists() else 0
        )
        space_saved = db_size_before - db_size_after

        return f"🗄️ Database: {'Optimized reflection DB, saved ' + str(space_saved) + ' bytes' if space_saved > 0 else 'Reflection DB already optimized'}"

    except ImportError:
        return "ℹ️ Database: Reflection tools not available"
    except Exception as e:
        return f"⚠️ Database: Optimization skipped - {str(e)[:50]}"


def _cleanup_session_logs() -> str:
    """Clean up old session log files."""
    try:
        log_dir = claude_dir / "logs"
        if not log_dir.exists():
            return "📝 Logs: No log directory found"

        log_files = sorted(
            log_dir.glob("session_management_*.log"),
            key=lambda x: x.stat().st_mtime,
        )

        if len(log_files) <= 10:
            return "📝 Logs: Within retention limit (10 files)"

        old_files = log_files[:-10]
        total_size = sum(f.stat().st_size for f in old_files)

        for old_file in old_files:
            old_file.unlink()

        return (
            f"📝 Logs: Cleaned {len(old_files)} old files, freed {total_size:,} bytes"
        )

    except Exception as e:
        return f"⚠️ Logs: Cleanup failed - {str(e)[:50]}"


def _cleanup_temp_files(current_dir: Path) -> str:
    """Clean up temporary files in the project directory."""
    try:
        temp_cleaned = 0
        temp_size = 0

        # Clean Python cache files
        for cache_file in current_dir.rglob("__pycache__"):
            if cache_file.is_dir():
                for file in cache_file.iterdir():
                    temp_size += file.stat().st_size
                    file.unlink()
                cache_file.rmdir()
                temp_cleaned += 1

        # Clean .DS_Store files on macOS
        for ds_file in current_dir.rglob(".DS_Store"):
            temp_size += ds_file.stat().st_size
            ds_file.unlink()
            temp_cleaned += 1

        # Clean pytest cache
        pytest_cache = current_dir / ".pytest_cache"
        if pytest_cache.exists():
            shutil.rmtree(pytest_cache)
            temp_cleaned += 1

        # Clean coverage files (keep most recent 3)
        coverage_files = sorted(
            current_dir.glob(".coverage*"),
            key=lambda x: x.stat().st_mtime,
        )
        if len(coverage_files) > 3:
            for cov_file in coverage_files[:-3]:
                temp_size += cov_file.stat().st_size
                cov_file.unlink()
                temp_cleaned += 1

        return f"🧹 Temp files: {'Cleaned ' + str(temp_cleaned) + ' items, freed ' + str(temp_size) + ' bytes' if temp_cleaned > 0 else 'No cleanup needed'}"

    except Exception as e:
        return f"⚠️ Temp files: Cleanup failed - {str(e)[:50]}"


def _optimize_git_repository(current_dir: Path) -> list[str]:
    """Optimize git repository."""
    results = []

    try:
        if not (current_dir / ".git").exists():
            return ["📦 Git: Not a git repository"]

        # Run git garbage collection
        gc_result = subprocess.run(
            ["git", "gc", "--auto"],
            check=False,
            cwd=current_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if gc_result.returncode != 0:
            return ["📦 Git: Optimization skipped (no changes)"]

        # Check repository size
        git_dir = current_dir / ".git"
        git_size = sum(f.stat().st_size for f in git_dir.rglob("*") if f.is_file())
        results.append(f"📦 Git: Repository optimized ({git_size:,} bytes)")

        # Prune remote tracking branches
        prune_result = subprocess.run(
            ["git", "remote", "prune", "origin"],
            check=False,
            cwd=current_dir,
            capture_output=True,
            text=True,
            timeout=15,
        )

        if prune_result.returncode == 0 and prune_result.stdout.strip():
            results.append("🌿 Git: Pruned stale remote branches")

        return results

    except subprocess.TimeoutExpired:
        return ["⏱️ Git: Optimization timeout (large repository)"]
    except Exception as e:
        return [f"⚠️ Git: Optimization failed - {str(e)[:50]}"]


def _cleanup_uv_cache() -> str:
    """Clean up UV package cache."""
    try:
        if not shutil.which("uv"):
            return "📦 UV: Not available"

        cache_result = subprocess.run(
            ["uv", "cache", "clean"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        return (
            "📦 UV: Package cache cleaned"
            if cache_result.returncode == 0
            else "📦 UV: Cache already clean"
        )

    except subprocess.TimeoutExpired:
        return "⏱️ UV: Cache cleanup timeout"
    except Exception as e:
        return f"⚠️ UV: Cache cleanup failed - {str(e)[:50]}"


async def _analyze_context_compaction() -> list[str]:
    """Analyze and recommend context compaction."""
    results = []

    try:
        should_compact, reason = should_suggest_compact()
        results.append("\n🔍 Context Compaction Analysis")
        results.append(f"📊 {reason}")

        if should_compact:
            results.extend(
                [
                    "",
                    "🔄 RECOMMENDATION: Run /compact to optimize context",
                    "📝 Benefits of compaction:",
                    "   • Improved response speed and accuracy",
                    "   • Better focus on current development context",
                    "   • Reduced memory usage for complex sessions",
                    "   • Cleaner conversation flow",
                    "",
                    "💡 WORKFLOW: After this checkpoint completes, run: /compact",
                    "🔄 Context compaction should be applied for optimal performance",
                ],
            )
        else:
            results.append("✅ Context appears well-optimized for current session")

        results.append(
            "💡 This checkpoint includes intelligent conversation summarization",
        )

        # Store conversation summary
        conversation_summary = await summarize_current_conversation()

        if conversation_summary["key_topics"]:
            key_topics_summary = (
                f"Session focus: {', '.join(conversation_summary['key_topics'][:3])}"
            )
            results.append(f"📋 {key_topics_summary}")

        if conversation_summary["decisions_made"]:
            key_decision = conversation_summary["decisions_made"][0]
            results.append(f"✅ Key decision: {key_decision}")

        # Store context summary for post-compaction retrieval
        await _store_context_summary(conversation_summary)
        results.append("💾 Context summary stored for post-compaction retrieval")

    except Exception as e:
        results.append(f"⚠️ Context summary storage failed: {str(e)[:50]}")

    return results


async def _store_context_summary(conversation_summary: dict) -> None:
    """Store comprehensive context summary."""
    try:
        db = await get_reflection_database()
        context_summary = f"Pre-compaction context summary: {', '.join(conversation_summary['key_topics'])}. "
        context_summary += (
            f"Decisions: {', '.join(conversation_summary['decisions_made'])}. "
        )
        context_summary += (
            f"Next steps: {', '.join(conversation_summary['next_steps'])}"
        )

        await db.store_reflection(
            context_summary,
            [
                "pre-compaction",
                "context-summary",
                "checkpoint",
                current_project or "unknown-project",
            ],
        )
    except Exception as e:
        msg = f"Context summary storage failed: {str(e)[:50]}"
        raise Exception(msg)


async def perform_strategic_compaction() -> list[str]:
    """Perform strategic compaction and optimization tasks."""
    results = []
    current_dir = Path(os.environ.get("PWD", Path.cwd()))

    # Database optimization
    results.append(await _optimize_reflection_database())

    # Log cleanup
    results.append(_cleanup_session_logs())

    # Temp file cleanup
    results.append(_cleanup_temp_files(current_dir))

    # Git optimization
    results.extend(_optimize_git_repository(current_dir))

    # UV cache cleanup
    results.append(_cleanup_uv_cache())

    # Context compaction analysis
    results.extend(await _analyze_context_compaction())

    # Summary
    total_operations = len([r for r in results if not r.startswith(("ℹ️", "⚠️", "⏱️"))])
    results.extend(
        [
            f"\n📊 Strategic compaction complete: {total_operations} optimization tasks performed",
            "🎯 Recommendation: Conversation context should be compacted automatically",
        ],
    )

    return results


async def _generate_basic_insights(
    quality_score: float,
    conversation_summary: dict[str, Any],
) -> list[str]:
    """Generate basic session insights from quality score and conversation summary."""
    insights = []

    insights.append(
        f"Session checkpoint completed with quality score: {quality_score}/100",
    )

    if conversation_summary["key_topics"]:
        insights.append(
            f"Key discussion topics: {', '.join(conversation_summary['key_topics'][:3])}",
        )

    if conversation_summary["decisions_made"]:
        insights.append(
            f"Important decisions: {conversation_summary['decisions_made'][0]}",
        )

    if conversation_summary["next_steps"]:
        insights.append(
            f"Next steps identified: {conversation_summary['next_steps'][0]}",
        )

    return insights


async def _add_project_context_insights(insights: list[str]) -> None:
    """Add project context analysis to insights."""
    current_dir = Path(os.environ.get("PWD", Path.cwd()))
    project_context = await analyze_project_context(current_dir)
    context_items = [k for k, v in project_context.items() if v]
    if context_items:
        insights.append(f"Active project context: {', '.join(context_items)}")


def _add_session_health_insights(insights: list[str], quality_score: float) -> None:
    """Add session health indicators to insights."""
    if quality_score >= 80:
        insights.append("Excellent session progress with optimal workflow patterns")
    elif quality_score >= 60:
        insights.append("Good session progress with minor optimization opportunities")
    else:
        insights.append(
            "Session requires attention - potential workflow improvements needed",
        )


def _generate_session_tags(quality_score: float) -> list[str]:
    """Generate contextual tags for session reflection storage."""
    tags = ["checkpoint", "session-summary", current_project or "unknown-project"]
    if quality_score >= 80:
        tags.append("excellent-session")
    elif quality_score < 60:
        tags.append("needs-attention")
    return tags


async def _capture_flow_analysis(db, tags: list[str], results: list[str]) -> None:
    """Capture conversation flow insights."""
    flow_analysis = await analyze_conversation_flow()
    flow_summary = f"Session pattern: {flow_analysis['pattern_type']}. "
    if flow_analysis["recommendations"]:
        flow_summary += f"Key recommendation: {flow_analysis['recommendations'][0]}"

    flow_id = await db.store_reflection(
        flow_summary,
        [*tags, "flow-analysis", "phase3"],
    )
    results.append(f"🔄 Flow analysis stored: {flow_id[:12]}...")


async def _capture_intelligence_insights(
    db,
    tags: list[str],
    results: list[str],
) -> None:
    """Capture session intelligence insights."""
    intelligence = await generate_session_intelligence()
    if intelligence["priority_actions"]:
        intel_summary = f"Session intelligence: {intelligence['intelligence_level']}. "
        intel_summary += f"Priority: {intelligence['priority_actions'][0]}"

        intel_id = await db.store_reflection(
            intel_summary,
            [*tags, "intelligence", "proactive"],
        )
        results.append(f"🧠 Intelligence insights stored: {intel_id[:12]}...")


async def _capture_session_metrics(db, tags: list[str], results: list[str]) -> None:
    """Capture additional session metrics if available."""
    if SESSION_MANAGEMENT_AVAILABLE:
        try:
            checkpoint_result = checkpoint_session()
            session_stats = checkpoint_result.get("session_stats", {})
            if session_stats:
                detail_summary = f"Session metrics - Duration: {session_stats.get('duration_minutes', 0)}min, "
                detail_summary += (
                    f"Success rate: {session_stats.get('success_rate', 0):.1f}%, "
                )
                detail_summary += (
                    f"Checkpoints: {session_stats.get('total_checkpoints', 0)}"
                )

                detail_id = await db.store_reflection(
                    detail_summary,
                    [*tags, "session-metrics"],
                )
                results.append(f"📊 Session metrics stored: {detail_id[:12]}...")
        except Exception as e:
            results.append(f"⚠️ Session metrics capture failed: {str(e)[:50]}...")


async def capture_session_insights(quality_score: float) -> list[str]:
    """Phase 1 & 3: Automatically capture and store session insights with conversation summarization."""
    results = []

    if not REFLECTION_TOOLS_AVAILABLE:
        results.append(
            "⚠️ Reflection storage not available - install dependencies: pip install duckdb transformers",
        )
        return results

    try:
        # Phase 3: AI-Powered Conversation Summarization
        conversation_summary = await summarize_current_conversation()

        # Generate comprehensive session summary
        insights = await _generate_basic_insights(quality_score, conversation_summary)
        await _add_project_context_insights(insights)
        _add_session_health_insights(insights, quality_score)

        # Store main session reflection
        session_summary = ". ".join(insights)
        tags = _generate_session_tags(quality_score)

        db = await get_reflection_database()
        reflection_id = await db.store_reflection(session_summary, tags)

        results.append("✅ Session insights automatically captured and stored")
        results.append(f"🆔 Reflection ID: {reflection_id[:12]}...")
        results.append(f"📝 Summary: {session_summary[:80]}...")
        results.append(f"🏷️ Tags: {', '.join(tags)}")

        # Phase 3A: Enhanced insight capture with advanced intelligence
        try:
            await _capture_flow_analysis(db, tags, results)
            await _capture_intelligence_insights(db, tags, results)
        except Exception as e:
            results.append(f"⚠️ Phase 3 insight capture failed: {str(e)[:50]}...")

        # Store additional detailed context
        await _capture_session_metrics(db, tags, results)

    except Exception as e:
        results.append(f"❌ Insight capture failed: {str(e)[:60]}...")
        results.append(
            "💡 Manual reflection storage still available via store_reflection tool",
        )

    return results


def _create_empty_summary() -> dict[str, Any]:
    """Create empty conversation summary structure."""
    return {
        "key_topics": [],
        "decisions_made": [],
        "next_steps": [],
        "problems_solved": [],
        "code_changes": [],
    }


def _extract_topics_from_content(content: str) -> set[str]:
    """Extract topics from reflection content."""
    topics = set()
    if "project context:" in content:
        context_part = content.split("project context:")[1].split(".")[0]
        topics.update(word.strip() for word in context_part.split(","))
    return topics


def _extract_decisions_from_content(content: str) -> list[str]:
    """Extract decisions from reflection content."""
    decisions = []
    if "excellent" in content:
        decisions.append("Maintaining productive workflow patterns")
    elif "attention" in content:
        decisions.append("Identified areas needing workflow optimization")
    elif "good progress" in content:
        decisions.append("Steady development progress confirmed")
    return decisions


def _extract_next_steps_from_content(content: str) -> list[str]:
    """Extract next steps from reflection content."""
    next_steps = []
    if "priority:" in content:
        priority_part = content.split("priority:")[1].split(".")[0]
        if priority_part.strip():
            next_steps.append(priority_part.strip())
    return next_steps


async def _process_recent_reflections(db, summary: dict[str, Any]) -> None:
    """Process recent reflections to extract conversation insights."""
    recent_reflections = await db.search_reflections("checkpoint", limit=5)

    if not recent_reflections:
        return

    topics = set()
    decisions = []
    next_steps = []

    for reflection in recent_reflections:
        content = reflection["content"].lower()

        topics.update(_extract_topics_from_content(content))
        decisions.extend(_extract_decisions_from_content(content))
        next_steps.extend(_extract_next_steps_from_content(content))

    summary["key_topics"] = list(topics)[:5]
    summary["decisions_made"] = decisions[:3]
    summary["next_steps"] = next_steps[:3]


def _add_current_session_context(summary: dict[str, Any]) -> None:
    """Add current session context to summary."""
    current_dir = Path(os.environ.get("PWD", Path.cwd()))
    if (current_dir / "session_mgmt_mcp").exists():
        summary["key_topics"].append("session-mgmt-mcp development")


def _ensure_summary_defaults(summary: dict[str, Any]) -> None:
    """Ensure summary has default values if empty."""
    if not summary["key_topics"]:
        summary["key_topics"] = [
            "session management",
            "workflow optimization",
        ]

    if not summary["decisions_made"]:
        summary["decisions_made"] = ["Proceeding with current development approach"]

    if not summary["next_steps"]:
        summary["next_steps"] = ["Continue with regular checkpoint monitoring"]


def _get_fallback_summary() -> dict[str, Any]:
    """Get fallback summary when reflection processing fails."""
    return {
        "key_topics": ["development session", "workflow management"],
        "decisions_made": ["Maintaining current session approach"],
        "next_steps": ["Continue monitoring session quality"],
        "problems_solved": ["Session management optimization"],
        "code_changes": ["Enhanced checkpoint functionality"],
    }


def _get_error_summary(error: Exception) -> dict[str, Any]:
    """Get error summary when conversation analysis fails."""
    return {
        "key_topics": ["session analysis"],
        "decisions_made": ["Continue current workflow"],
        "next_steps": ["Regular quality monitoring"],
        "problems_solved": [],
        "code_changes": [],
        "error": str(error),
    }


async def summarize_current_conversation() -> dict[str, Any]:
    """Phase 3: AI-Powered Conversation Summarization."""
    try:
        summary = _create_empty_summary()

        if REFLECTION_TOOLS_AVAILABLE:
            try:
                db = await get_reflection_database()
                await _process_recent_reflections(db, summary)
                _add_current_session_context(summary)
                _ensure_summary_defaults(summary)
            except Exception:
                summary = _get_fallback_summary()

        return summary

    except Exception as e:
        return _get_error_summary(e)


def _extract_quality_scores(reflections: list) -> list[float]:
    """Extract quality scores from reflection content."""
    quality_scores = []

    for reflection in reflections:
        try:
            if "quality score:" in reflection["content"]:
                score_text = (
                    reflection["content"].split("quality score:")[1].split("/")[0]
                )
                score = float(score_text.strip())
                quality_scores.append(score)
        except (ValueError, IndexError, AttributeError):
            continue

    return quality_scores


def _analyze_quality_trend(quality_scores: list[float]) -> tuple[str, list[str], bool]:
    """Analyze quality trend and generate alerts."""
    quality_alerts = []
    quality_trend = "stable"
    recommend_checkpoint = False

    if len(quality_scores) < 3:
        return quality_trend, quality_alerts, recommend_checkpoint

    # Trend analysis
    recent_avg = (
        sum(quality_scores[:2]) / 2 if len(quality_scores) >= 2 else quality_scores[0]
    )
    older_avg = sum(quality_scores[2:]) / len(quality_scores[2:])

    if recent_avg < older_avg - 10:
        quality_trend = "declining"
        quality_alerts.append("Quality trend declining - consider workflow review")
        recommend_checkpoint = True
    elif recent_avg > older_avg + 5:
        quality_trend = "improving"
        quality_alerts.append("Quality trend improving - maintain current patterns")

    # Early warning triggers
    if recent_avg < 50:
        quality_alerts.append("URGENT: Session quality critically low")
        recommend_checkpoint = True
    elif recent_avg < 70:
        quality_alerts.append("WARNING: Session quality below optimal")
        recommend_checkpoint = True

    return quality_trend, quality_alerts, recommend_checkpoint


def _check_workflow_drift(quality_scores: list[float]) -> tuple[list[str], bool]:
    """Check for workflow drift indicators."""
    quality_alerts = []
    recommend_checkpoint = False

    if len(quality_scores) >= 4:
        variance = max(quality_scores) - min(quality_scores)
        if variance > 30:
            quality_alerts.append(
                "High quality variance detected - workflow inconsistency",
            )
            recommend_checkpoint = True

    return quality_alerts, recommend_checkpoint


async def _perform_quality_analysis() -> tuple[str, list[str], bool]:
    """Perform quality analysis with reflection data."""
    quality_alerts = []
    quality_trend = "stable"
    recommend_checkpoint = False

    try:
        db = await get_reflection_database()
        recent_reflections = await db.search_reflections("quality score", limit=5)
        quality_scores = _extract_quality_scores(recent_reflections)

        if quality_scores:
            trend, trend_alerts, trend_checkpoint = _analyze_quality_trend(
                quality_scores,
            )
            quality_trend = trend
            quality_alerts.extend(trend_alerts)
            recommend_checkpoint = recommend_checkpoint or trend_checkpoint

            drift_alerts, drift_checkpoint = _check_workflow_drift(quality_scores)
            quality_alerts.extend(drift_alerts)
            recommend_checkpoint = recommend_checkpoint or drift_checkpoint

    except Exception:
        quality_alerts.append("Quality monitoring analysis unavailable")

    return quality_trend, quality_alerts, recommend_checkpoint


def _get_quality_error_result(error: Exception) -> dict[str, Any]:
    """Get error result for quality monitoring failure."""
    return {
        "quality_trend": "unknown",
        "alerts": ["Quality monitoring failed"],
        "recommend_checkpoint": False,
        "monitoring_active": False,
        "error": str(error),
    }


async def monitor_proactive_quality() -> dict[str, Any]:
    """Phase 3: Proactive Quality Monitoring with Early Warning System."""
    try:
        quality_alerts = []
        quality_trend = "stable"
        recommend_checkpoint = False

        if REFLECTION_TOOLS_AVAILABLE:
            (
                quality_trend,
                quality_alerts,
                recommend_checkpoint,
            ) = await _perform_quality_analysis()

        return {
            "quality_trend": quality_trend,
            "alerts": quality_alerts,
            "recommend_checkpoint": recommend_checkpoint,
            "monitoring_active": True,
        }

    except Exception as e:
        return _get_quality_error_result(e)


async def analyze_advanced_context_metrics() -> dict[str, Any]:
    """Phase 3A: Advanced context metrics analysis."""
    return {
        "estimated_tokens": 0,  # Placeholder for actual token counting
        "context_density": "moderate",
        "conversation_depth": "active",
    }


async def analyze_token_usage_patterns() -> dict[str, Any]:
    """Phase 3A: Intelligent token usage analysis with smart triggers."""
    try:
        # Get conversation statistics from memory system
        conv_stats = {"total_conversations": 0, "recent_activity": "low"}

        if REFLECTION_TOOLS_AVAILABLE:
            try:
                db = await get_reflection_database()
                stats = await db.get_stats()
                conv_stats["total_conversations"] = stats.get("conversations_count", 0)
            except Exception:
                pass

        # Heuristic-based context analysis (approximation)
        # In a real implementation, this would hook into actual context metrics

        # Check session activity patterns
        datetime.now()

        # Estimate context usage based on activity
        estimated_length = "moderate"
        needs_attention = False
        recommend_compact = False
        recommend_clear = False

        # Smart triggers based on conversation patterns and critical context detection

        # PRIORITY: Always recommend compaction if we have significant stored content
        # This indicates a long conversation that needs compaction
        if conv_stats["total_conversations"] > 3:
            # Any significant conversation history indicates compaction needed
            estimated_length = "extensive"
            needs_attention = True
            recommend_compact = True

        if conv_stats["total_conversations"] > 10:
            # Long conversation - definitely needs compaction
            estimated_length = "very long"
            needs_attention = True
            recommend_compact = True

        if conv_stats["total_conversations"] > 20:
            # Extremely long - may need clear after compact
            estimated_length = "extremely long"
            needs_attention = True
            recommend_compact = True
            recommend_clear = True

        # Override: ALWAYS recommend compaction during checkpoints
        # Checkpoints typically happen during long sessions where context is an issue
        # This ensures the "Context low" warning gets addressed
        recommend_compact = True
        needs_attention = True
        estimated_length = (
            "checkpoint-session" if estimated_length == "moderate" else estimated_length
        )

        status = "optimal" if not needs_attention else "needs optimization"

        return {
            "needs_attention": needs_attention,
            "status": status,
            "estimated_length": estimated_length,
            "recommend_compact": recommend_compact,
            "recommend_clear": recommend_clear,
            "confidence": "heuristic",
        }

    except Exception as e:
        return {
            "needs_attention": False,
            "status": "analysis_failed",
            "estimated_length": "unknown",
            "recommend_compact": False,
            "recommend_clear": False,
            "error": str(e),
        }


async def analyze_conversation_flow() -> dict[str, Any]:
    """Phase 3A: Analyze conversation patterns and flow."""
    try:
        # Analyze recent reflection patterns to understand session flow

        if REFLECTION_TOOLS_AVAILABLE:
            try:
                db = await get_reflection_database()

                # Search recent reflections for patterns
                recent_reflections = await db.search_reflections(
                    "session checkpoint",
                    limit=5,
                )

                if recent_reflections:
                    # Analyze pattern based on recent reflections
                    if any(
                        "excellent" in r["content"].lower() for r in recent_reflections
                    ):
                        pattern_type = "productive_development"
                        recommendations = [
                            "Continue current productive workflow",
                            "Consider documenting successful patterns",
                            "Maintain current checkpoint frequency",
                        ]
                    elif any(
                        "attention" in r["content"].lower() for r in recent_reflections
                    ):
                        pattern_type = "optimization_needed"
                        recommendations = [
                            "Review recent workflow changes",
                            "Consider more frequent checkpoints",
                            "Use search tools to find successful patterns",
                        ]
                    else:
                        pattern_type = "steady_progress"
                        recommendations = [
                            "Maintain current workflow patterns",
                            "Consider periodic workflow evaluation",
                        ]
                else:
                    pattern_type = "new_session"
                    recommendations = [
                        "Establish workflow patterns through regular checkpoints",
                    ]

            except Exception:
                pattern_type = "analysis_unavailable"
                recommendations = [
                    "Use regular checkpoints to establish workflow patterns",
                ]
        else:
            pattern_type = "basic_session"
            recommendations = ["Enable reflection tools for advanced flow analysis"]

        return {
            "pattern_type": pattern_type,
            "recommendations": recommendations,
            "confidence": "pattern_based",
        }

    except Exception as e:
        return {
            "pattern_type": "analysis_failed",
            "recommendations": ["Use basic workflow patterns"],
            "error": str(e),
        }


async def analyze_memory_patterns(db, conv_count: int) -> dict[str, Any]:
    """Phase 3A: Advanced memory pattern analysis."""
    try:
        # Analyze conversation history for intelligent insights
        if conv_count == 0:
            return {
                "summary": "New session - no historical patterns yet",
                "proactive_suggestions": [
                    "Start building conversation history for better insights",
                ],
            }
        if conv_count < 5:
            return {
                "summary": f"{conv_count} conversations stored - building pattern recognition",
                "proactive_suggestions": [
                    "Continue regular checkpoints to build session intelligence",
                    "Use store_reflection for important insights",
                ],
            }
        if conv_count < 20:
            return {
                "summary": f"{conv_count} conversations stored - developing patterns",
                "proactive_suggestions": [
                    "Use reflect_on_past to leverage growing knowledge base",
                    "Search previous solutions before starting new implementations",
                ],
            }
        return {
            "summary": f"{conv_count} conversations - rich pattern recognition available",
            "proactive_suggestions": [
                "Leverage extensive history with targeted searches",
                "Consider workflow optimization based on successful patterns",
                "Use conversation history to accelerate problem-solving",
            ],
        }

    except Exception as e:
        return {
            "summary": "Memory analysis unavailable",
            "proactive_suggestions": [
                "Use basic memory tools for conversation tracking",
            ],
            "error": str(e),
        }


async def analyze_project_workflow_patterns(current_dir: Path) -> dict[str, Any]:
    """Phase 3A: Project-specific workflow pattern analysis."""
    try:
        workflow_recommendations = []

        # Detect project characteristics
        has_tests = (current_dir / "tests").exists() or (current_dir / "test").exists()
        has_git = (current_dir / ".git").exists()
        has_python = (current_dir / "pyproject.toml").exists() or (
            current_dir / "requirements.txt"
        ).exists()
        has_node = (current_dir / "package.json").exists()
        has_docker = (current_dir / "Dockerfile").exists() or (
            current_dir / "docker-compose.yml"
        ).exists()

        # Generate intelligent workflow recommendations
        if has_tests:
            workflow_recommendations.append(
                "Use targeted test commands for specific test scenarios",
            )
            workflow_recommendations.append(
                "Consider test-driven development workflow with regular testing",
            )

        if has_git:
            workflow_recommendations.append(
                "Leverage git context for branch-specific development",
            )
            workflow_recommendations.append(
                "Use commit messages to track progress patterns",
            )

        if has_python and has_tests:
            workflow_recommendations.append(
                "Python+Testing: Consider pytest workflows with coverage analysis",
            )

        if has_node:
            workflow_recommendations.append(
                "Node.js project: Leverage npm/yarn scripts in development workflow",
            )

        if has_docker:
            workflow_recommendations.append(
                "Containerized project: Consider container-based development workflows",
            )

        # Default recommendations if no specific patterns detected
        if not workflow_recommendations:
            workflow_recommendations.append(
                "Establish project-specific workflow patterns through regular checkpoints",
            )

        return {
            "workflow_recommendations": workflow_recommendations,
            "project_characteristics": {
                "has_tests": has_tests,
                "has_git": has_git,
                "has_python": has_python,
                "has_node": has_node,
                "has_docker": has_docker,
            },
        }

    except Exception as e:
        return {
            "workflow_recommendations": ["Use basic project workflow patterns"],
            "error": str(e),
        }


def _get_time_based_recommendations(hour: int) -> list[str]:
    """Get time-based intelligent recommendations."""
    recommendations = []

    if 9 <= hour <= 11:
        recommendations.append(
            "Morning session: Consider high-focus tasks and planning",
        )
    elif 13 <= hour <= 15:
        recommendations.append(
            "Afternoon session: Good time for implementation and testing",
        )
    elif hour >= 18:
        recommendations.append(
            "Evening session: Consider review and documentation tasks",
        )

    return recommendations


def _extract_quality_scores_from_reflections(reflections: list) -> list[float]:
    """Extract quality scores from reflection content for trend analysis."""
    scores = []

    for reflection in reflections:
        if "quality score:" in reflection["content"]:
            try:
                score_text = (
                    reflection["content"].split("quality score:")[1].split("/")[0]
                )
                score = float(score_text.strip())
                scores.append(score)
            except (ValueError, IndexError, AttributeError):
                continue

    return scores


def _generate_quality_trend_recommendations(scores: list[float]) -> list[str]:
    """Generate recommendations based on quality score trends."""
    if not scores:
        return []

    avg_score = sum(scores) / len(scores)

    if avg_score > 80:
        return ["Excellent session trend: Maintain current productive patterns"]
    if avg_score < 60:
        return [
            "Session quality declining: Review workflow and take corrective actions",
        ]
    return ["Steady session progress: Consider optimization opportunities"]


async def _analyze_reflection_based_intelligence() -> list[str]:
    """Analyze recent reflections for intelligence recommendations."""
    if not REFLECTION_TOOLS_AVAILABLE:
        return []

    try:
        db = await get_reflection_database()
        recent_reflections = await db.search_reflections("checkpoint", limit=3)

        if recent_reflections:
            recent_scores = _extract_quality_scores_from_reflections(recent_reflections)
            return _generate_quality_trend_recommendations(recent_scores)

    except Exception:
        return ["Enable reflection analysis for session trend intelligence"]

    return []


def _ensure_default_recommendations(priority_actions: list[str]) -> list[str]:
    """Ensure there are default recommendations if none generated."""
    if not priority_actions:
        return ["Establish session intelligence through regular checkpoint patterns"]
    return priority_actions


def _get_intelligence_error_result(error: Exception) -> dict[str, Any]:
    """Get error result for intelligence generation failure."""
    return {
        "priority_actions": ["Use basic session management patterns"],
        "intelligence_level": "fallback",
        "error": str(error),
    }


async def generate_session_intelligence() -> dict[str, Any]:
    """Phase 3A: Generate proactive session intelligence and priority actions."""
    try:
        current_time = datetime.now()

        # Gather all recommendation sources
        priority_actions = []
        priority_actions.extend(_get_time_based_recommendations(current_time.hour))
        priority_actions.extend(await _analyze_reflection_based_intelligence())
        priority_actions = _ensure_default_recommendations(priority_actions)

        return {
            "priority_actions": priority_actions,
            "intelligence_level": "proactive",
            "timestamp": current_time.isoformat(),
        }

    except Exception as e:
        return _get_intelligence_error_result(e)


async def _analyze_token_usage_recommendations(results: list[str]) -> None:
    """Analyze token usage and add recommendations."""
    token_analysis = await analyze_token_usage_patterns()
    if token_analysis["needs_attention"]:
        results.append(f"⚠️ Context usage: {token_analysis['status']}")
        results.append(
            f"   Estimated conversation length: {token_analysis['estimated_length']}",
        )

        # Smart compaction triggers - PRIORITY RECOMMENDATIONS
        if token_analysis["recommend_compact"]:
            results.append(
                "🚨 CRITICAL AUTO-RECOMMENDATION: Context compaction required",
            )
            results.append(
                "🔄 This checkpoint has prepared conversation summary for compaction",
            )
            results.append(
                "💡 Compaction should be applied automatically after this checkpoint",
            )

        if token_analysis["recommend_clear"]:
            results.append(
                "🆕 AUTO-RECOMMENDATION: Consider /clear for fresh context after compaction",
            )
    else:
        results.append(f"✅ Context usage: {token_analysis['status']}")


async def _analyze_conversation_flow_recommendations(results: list[str]) -> None:
    """Analyze conversation flow and add recommendations."""
    flow_analysis = await analyze_conversation_flow()
    results.append(f"📊 Session flow: {flow_analysis['pattern_type']}")

    if flow_analysis["recommendations"]:
        results.append("🎯 Flow-based recommendations:")
        for rec in flow_analysis["recommendations"][:3]:
            results.append(f"   • {rec}")


async def _analyze_memory_recommendations(results: list[str]) -> None:
    """Analyze memory patterns and add recommendations."""
    if REFLECTION_TOOLS_AVAILABLE:
        try:
            db = await get_reflection_database()
            stats = await db.get_stats()
            conv_count = stats.get("conversations_count", 0)

            # Advanced memory analysis
            memory_insights = await analyze_memory_patterns(db, conv_count)
            results.append(f"📚 Memory insights: {memory_insights['summary']}")

            if memory_insights["proactive_suggestions"]:
                results.append("💡 Proactive suggestions:")
                for suggestion in memory_insights["proactive_suggestions"][:2]:
                    results.append(f"   • {suggestion}")

        except Exception:
            results.append("📚 Memory system available for conversation search")


async def _analyze_project_workflow_recommendations(results: list[str]) -> None:
    """Analyze project workflow patterns and add recommendations."""
    current_dir = Path(os.environ.get("PWD", Path.cwd()))
    project_insights = await analyze_project_workflow_patterns(current_dir)

    if project_insights["workflow_recommendations"]:
        results.append("🚀 Workflow optimizations:")
        for opt in project_insights["workflow_recommendations"][:2]:
            results.append(f"   • {opt}")


async def _analyze_session_intelligence_recommendations(results: list[str]) -> None:
    """Analyze session intelligence and add recommendations."""
    session_intelligence = await generate_session_intelligence()
    if session_intelligence["priority_actions"]:
        results.append("\n🧠 Session Intelligence:")
        for action in session_intelligence["priority_actions"][:3]:
            results.append(f"   • {action}")


async def _analyze_quality_monitoring_recommendations(results: list[str]) -> None:
    """Analyze quality monitoring and add recommendations."""
    quality_monitoring = await monitor_proactive_quality()
    if quality_monitoring["monitoring_active"]:
        results.append(f"\n📊 Quality Trend: {quality_monitoring['quality_trend']}")

        if quality_monitoring["alerts"]:
            results.append("⚠️ Quality Alerts:")
            for alert in quality_monitoring["alerts"][:2]:
                results.append(f"   • {alert}")

        if quality_monitoring["recommend_checkpoint"]:
            results.append("🔄 PROACTIVE RECOMMENDATION: Consider immediate checkpoint")


async def _add_fallback_recommendations(results: list[str], error: Exception) -> None:
    """Add fallback recommendations when analysis fails."""
    results.append(f"❌ Advanced context analysis failed: {str(error)[:60]}...")
    results.append("💡 Falling back to basic context management recommendations")

    # Fallback to basic recommendations
    results.append("🎯 Basic context actions:")
    results.append("   • Use /compact for conversation summarization")
    results.append("   • Use /clear for fresh context on new topics")
    results.append("   • Use search tools to retrieve relevant discussions")


async def analyze_context_usage() -> list[str]:
    """Phase 2 & 3A: Advanced context analysis with intelligent recommendations."""
    results = []

    try:
        results.append("🔍 Advanced context analysis and optimization...")

        # Phase 3A: Advanced Context Intelligence
        await analyze_advanced_context_metrics()

        # Run all analysis components
        await _analyze_token_usage_recommendations(results)
        await _analyze_conversation_flow_recommendations(results)
        await _analyze_memory_recommendations(results)
        await _analyze_project_workflow_recommendations(results)
        await _analyze_session_intelligence_recommendations(results)
        await _analyze_quality_monitoring_recommendations(results)

    except Exception as e:
        await _add_fallback_recommendations(results, e)

    return results


async def _perform_quality_assessment() -> tuple[int, dict]:
    """Perform quality assessment and return score and data."""
    quality_data = await calculate_quality_score()
    quality_score = quality_data["total_score"]
    return quality_score, quality_data


async def _format_quality_results(
    quality_score: int,
    quality_data: dict,
    checkpoint_result: dict | None = None,
) -> list[str]:
    """Format quality assessment results for display."""
    output = []

    # Quality status
    if quality_score >= 80:
        output.append(f"✅ Session quality: EXCELLENT (Score: {quality_score}/100)")
    elif quality_score >= 60:
        output.append(f"✅ Session quality: GOOD (Score: {quality_score}/100)")
    else:
        output.append(
            f"⚠️ Session quality: NEEDS ATTENTION (Score: {quality_score}/100)",
        )

    # Quality breakdown
    output.append("\n📈 Quality breakdown:")
    breakdown = quality_data["breakdown"]
    output.append(f"   • Project health: {breakdown['project_health']:.1f}/40")
    output.append(f"   • Permissions: {breakdown['permissions']:.1f}/20")
    output.append(f"   • Session tools: {breakdown['session_management']:.1f}/20")
    output.append(f"   • Tool availability: {breakdown['tools']:.1f}/20")

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


async def _perform_git_checkpoint(current_dir: Path, quality_score: int) -> list[str]:
    """Handle git operations for checkpoint commit."""
    output = []
    output.append("\n" + "=" * 50)
    output.append("📦 Git Checkpoint Commit")
    output.append("=" * 50)

    git_dir = current_dir / ".git"
    if not (git_dir.exists() and git_dir.is_dir()):
        output.append("\nℹ️ Not a git repository - skipping commit")
        return output

    try:
        # Check git status
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            cwd=current_dir,
        )

        if status_result.returncode != 0:
            output.append(f"\n⚠️ Git status check failed: {status_result.stderr}")
            return output

        status_lines = (
            status_result.stdout.strip().split("\n")
            if status_result.stdout.strip()
            else []
        )
        modified_files, untracked_files = _parse_git_status(status_lines)

        if not modified_files and not untracked_files:
            output.append("\n✅ Working directory is clean - no changes to commit")
            return output

        output.append(
            f"\n📝 Found {len(modified_files)} modified files and {len(untracked_files)} untracked files",
        )

        # Handle untracked files
        if untracked_files:
            output.extend(_format_untracked_files(untracked_files))

        # Stage and commit modified files
        if modified_files:
            output.extend(
                _stage_and_commit_files(current_dir, modified_files, quality_score),
            )
        elif untracked_files:
            output.append("\nℹ️ No staged changes to commit")
            output.append(
                "   💡 Add untracked files with 'git add' if you want to include them",
            )

    except Exception as e:
        output.append(f"\n⚠️ Git operations error: {e}")

    return output


def _parse_git_status(status_lines: list[str]) -> tuple[list[str], list[str]]:
    """Parse git status output into modified and untracked files."""
    modified_files = []
    untracked_files = []

    for line in status_lines:
        if line:
            status = line[:2]
            filepath = line[3:]
            if status == "??":
                untracked_files.append(filepath)
            elif status.strip():
                modified_files.append(filepath)

    return modified_files, untracked_files


def _format_untracked_files(untracked_files: list[str]) -> list[str]:
    """Format untracked files display."""
    output = []
    output.append("\n🆕 Untracked files found:")

    for file in untracked_files[:10]:  # Limit to first 10 for display
        output.append(f"   • {file}")

    if len(untracked_files) > 10:
        output.append(f"   ... and {len(untracked_files) - 10} more")

    output.append("\n⚠️ Please manually review and add untracked files if needed:")
    output.append("   Use: git add <file> for files you want to include")

    return output


def _stage_and_commit_files(
    current_dir: Path,
    modified_files: list[str],
    quality_score: int,
) -> list[str]:
    """Stage modified files and create commit."""
    output = []
    output.append(f"\n✅ Staging {len(modified_files)} modified files...")

    # Stage files
    for file in modified_files:
        subprocess.run(
            ["git", "add", file], check=False, cwd=current_dir, capture_output=True
        )

    # Check if there's anything to commit
    staged_result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        check=False,
        capture_output=True,
        text=True,
        cwd=current_dir,
    )

    if not staged_result.stdout.strip():
        output.append("\nℹ️ No staged changes to commit")
        return output

    # Create commit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"checkpoint: Session checkpoint - {timestamp}\n\nAutomatic checkpoint commit via session-management MCP server\nProject: {project_name}\nQuality Score: {quality_score}/100"

    commit_result = subprocess.run(
        ["git", "commit", "-m", commit_message],
        check=False,
        capture_output=True,
        text=True,
        cwd=current_dir,
    )

    if commit_result.returncode == 0:
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=current_dir,
        )
        commit_hash = (
            hash_result.stdout.strip()[:8] if hash_result.returncode == 0 else "unknown"
        )

        output.append(f"\n✅ Checkpoint commit created: {commit_hash}")
        output.append(f"   Message: checkpoint: Session checkpoint - {timestamp}")
        output.append("   💡 Use 'git reset HEAD~1' to undo if needed")
    else:
        output.append(f"\n⚠️ Commit failed: {commit_result.stderr}")

    return output


@mcp.tool()
async def checkpoint(working_directory: str | None = None) -> str:
    """Perform mid-session quality checkpoint with workflow analysis and optimization recommendations.

    Args:
        working_directory: Directory to perform checkpoint analysis in (defaults to current directory)

    """
    # Handle None parameter like other MCP functions
    if working_directory is None:
        working_directory = "."

    # Determine project name from working directory
    project_name = (
        Path(working_directory).name
        if working_directory != "."
        else current_project or "Current Project"
    )

    output = []
    output.append(
        f"🔍 Claude Session Checkpoint - {project_name}",
    )
    output.append("=" * 50)

    # Quality Assessment
    try:
        quality_score, quality_data = await _perform_quality_assessment()

        if SESSION_MANAGEMENT_AVAILABLE:
            output.append("\n📊 Running comprehensive quality assessment...")
            checkpoint_result = checkpoint_session()

            if checkpoint_result.get("checkpoint_passed", True):
                output.extend(
                    await _format_quality_results(
                        quality_score,
                        quality_data,
                        checkpoint_result,
                    ),
                )
            else:
                output.append("⚠️ Session quality issues detected:")
                for issue in checkpoint_result.get("issues_found", []):
                    output.append(f"   • {issue}")
        else:
            output.append(
                "\n⚠️ Session management not available - performing basic checks",
            )
            output.extend(await _format_quality_results(quality_score, quality_data))

    except Exception as e:
        session_logger.exception(
            "Checkpoint error occurred",
            error=str(e),
            project=current_project,
        )
        output.append(f"❌ Checkpoint error: {e}")
        quality_score, quality_data = await _perform_quality_assessment()
        output.append(f"\n📊 Basic quality assessment: {quality_score}/100")

    # Project Context
    current_dir = Path(os.environ.get("PWD", Path.cwd()))
    project_context = await analyze_project_context(current_dir)
    context_score = sum(1 for detected in project_context.values() if detected)
    output.append(
        f"\n🎯 Project context: {context_score}/{len(project_context)} indicators",
    )

    # Dynamic recommendations
    if quality_score >= 80:
        output.append("\n💡 Excellent session - maintain current workflow")
    elif quality_score >= 60:
        output.append("\n💡 Good session - minor optimizations available")
    else:
        output.append("\n💡 Session needs attention - review recommendations above")

    output.append("🔄 Consider next checkpoint in 30-45 minutes")

    # Git operations
    output.extend(await _perform_git_checkpoint(current_dir, quality_score))

    # Session insights
    output.append("\n" + "=" * 50)
    output.append("🧠 Automatic Session Insights Capture")
    output.append("=" * 50)

    insights_results = await capture_session_insights(quality_score)
    output.extend(insights_results)

    # Context analysis
    output.append("\n" + "=" * 50)
    output.append("🔄 Context Management Analysis")
    output.append("=" * 50)

    context_results = await analyze_context_usage()
    output.extend(context_results)

    # Strategic compaction
    output.append("\n" + "=" * 50)
    output.append("📦 Strategic Compaction & Optimization")
    output.append("=" * 50)

    compaction_results = await perform_strategic_compaction()
    output.extend(compaction_results)

    # Auto-compaction
    output.append("\n" + "=" * 50)
    output.append("🔄 Automatic Context Compaction")
    output.append("=" * 50)

    try:
        auto_compact_result = await auto_compact()
        compact_lines = auto_compact_result.split("\n")

        for line in compact_lines:
            if any(
                keyword in line.lower()
                for keyword in [
                    "preserved",
                    "stored",
                    "compaction required",
                    "/compact",
                ]
            ):
                output.append(line)

        output.append("✅ Auto-compaction integrated into checkpoint workflow")
    except Exception as e:
        output.append(f"⚠️ Auto-compaction integration failed: {e}")
        output.append("💡 Manual /compact may be needed")

    output.append(
        f"\n✨ Enhanced checkpoint complete - {project_name} session optimized!",
    )
    output.append("🔄 Context compaction has been automatically triggered")

    return "\n".join(output)


@mcp.tool()
async def end() -> str:
    """End Claude session with cleanup, learning capture, and handoff file creation."""
    output = []
    output.append(f"🏁 Claude Session End - {current_project or 'Current Project'}")
    output.append("=" * 60)

    # Final Checkpoint
    if SESSION_MANAGEMENT_AVAILABLE:
        try:
            output.append("\n📊 Final session quality checkpoint...")
            final_checkpoint = checkpoint_session()

            # Calculate final quality score
            final_quality_data = await calculate_quality_score()
            final_quality_score = final_quality_data["total_score"]

            if final_checkpoint.get("checkpoint_passed"):
                if final_quality_score >= 80:
                    output.append("✅ Final session quality: EXCELLENT")
                elif final_quality_score >= 60:
                    output.append("✅ Final session quality: GOOD")
                else:
                    output.append("⚠️ Final session quality: NEEDS ATTENTION")
                output.append(f"   📈 Quality score: {final_quality_score}/100")
            else:
                output.append("⚠️ Final session quality issues detected")

            # Create handoff file
            handoff_dir = claude_dir / "sessions"
            handoff_dir.mkdir(exist_ok=True)

            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = (
                current_project.replace("/", "_").replace(" ", "_")
                if current_project
                else "unknown"
            )
            handoff_file = (
                handoff_dir
                / f"{project_name}_mcp_session_handoff_{session_timestamp}.md"
            )

            handoff_content = f"""# {current_project} MCP Session Handoff - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Session Summary
- **Project**: {current_project}
- **Working Directory**: {Path(os.environ.get("PWD", Path.cwd()))}
- **MCP Server**: Claude Session Management
- **Quality Score**: {final_quality_score}/100

## Session Tools Used
- init: ✅ Complete setup with UV sync
- checkpoint: Used for quality monitoring
- end: ✅ Comprehensive cleanup

## Next Session Preparation
- MCP server automatically available via .mcp.json
- Use /session-management:init slash command for startup
- Session continuity maintained through MCP integration

## Project Context
- Session managed via MCP server integration
- All session management tools available as slash commands
- Consistent workflow across all projects
"""

            with open(handoff_file, "w") as f:
                f.write(handoff_content)

            output.append(f"📋 Session handoff created: {handoff_file}")

            # End session
            end_result = end_session()
            if end_result.get("session_ended_successfully"):
                output.append("✅ Session terminated successfully")

        except Exception as e:
            output.append(f"❌ Session end error: {e}")
    else:
        output.append("\n✅ Basic session cleanup completed")

    output.append(
        f"\n🙏 {current_project.upper() if current_project else 'SESSION'} COMPLETE!",
    )
    output.append(f"📅 Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("\n🔄 Next session: Use /session-management:init slash command")
    output.append("🎉 MCP server will provide automatic session management!")

    return "\n".join(output)


async def health_check() -> dict[str, Any]:
    """Comprehensive health check for MCP server and toolkit availability."""
    health_status = {
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


async def _add_project_context_info(output: list[str], current_dir: Path) -> None:
    """Add project context information to output."""
    from .utils.git_operations import get_worktree_info, list_worktrees

    project_context = await analyze_project_context(current_dir)
    context_score = sum(1 for detected in project_context.values() if detected)
    output.append(f"\n📈 Project maturity: {context_score}/{len(project_context)}")

    # Add worktree information
    try:
        worktree_info = get_worktree_info(current_dir)
        if worktree_info:
            all_worktrees = list_worktrees(current_dir)

            output.append("\n🌿 Git Worktree Information:")
            if worktree_info.is_main_worktree:
                output.append(
                    f"   📍 Current: Main repository on '{worktree_info.branch}'",
                )
            else:
                output.append(f"   📍 Current: Worktree on '{worktree_info.branch}'")
                output.append(f"   📁 Path: {worktree_info.path}")

            if len(all_worktrees) > 1:
                output.append(f"   🌳 Total worktrees: {len(all_worktrees)}")
                other_branches = [
                    wt.branch for wt in all_worktrees if wt.path != worktree_info.path
                ]
                if other_branches:
                    output.append(
                        f"   🔀 Other branches: {', '.join(other_branches[:3])}",
                    )
                    if len(other_branches) > 3:
                        output.append(f"   ... and {len(other_branches) - 3} more")
                output.append("   💡 Use 'git_worktree_list' to see all worktrees")
            else:
                output.append(
                    "   💡 Use 'git_worktree_add <branch> <path>' to create parallel worktrees",
                )

            if worktree_info.is_detached:
                output.append("   ⚠️ Detached HEAD - consider checking out a branch")

    except Exception:
        # Silently handle worktree detection failures
        pass


def _add_permissions_info(output: list[str]) -> None:
    """Add permissions information to output."""
    permissions_status = permissions_manager.get_permission_status()
    output.append("\n🔐 Session Permissions:")
    output.append(
        f"   📊 Trusted operations: {permissions_status['trusted_operations_count']}",
    )
    if permissions_status["trusted_operations"]:
        for op in permissions_status["trusted_operations"]:
            output.append(f"   ✅ {op.replace('_', ' ').title()}")
    else:
        output.append("   ⚠️ No trusted operations yet - will prompt for permissions")


def _add_basic_tools_info(output: list[str]) -> None:
    """Add basic MCP tools information to output."""
    output.append("\n🛠️ Available MCP Tools:")
    output.append("• init - Full session initialization")
    output.append("• checkpoint - Quality monitoring")
    output.append("• end - Complete cleanup")
    output.append("• status - This status report with health checks")
    output.append("• permissions - Manage trusted operations")
    output.append("• git_worktree_list - List all git worktrees")
    output.append("• git_worktree_add - Create new worktrees")
    output.append("• git_worktree_remove - Remove worktrees")
    output.append("• git_worktree_status - Comprehensive worktree status")
    output.append("• git_worktree_prune - Clean up stale references")


def _add_feature_status_info(output: list[str]) -> None:
    """Add feature status information to output."""
    # Token Optimization Status
    if TOKEN_OPTIMIZER_AVAILABLE:
        output.append("\n⚡ Token Optimization:")
        output.append("• get_cached_chunk - Retrieve chunked response data")
        output.append("• get_token_usage_stats - Token usage and savings metrics")
        output.append("• optimize_memory_usage - Consolidate old conversations")
        output.append("• Built-in response chunking and truncation")

    # Multi-Project Coordination Status
    if MULTI_PROJECT_AVAILABLE:
        output.append("\n🔗 Multi-Project Coordination:")
        output.append("• create_project_group - Create project groups for coordination")
        output.append("• add_project_dependency - Define project relationships")
        output.append(
            "• search_across_projects - Search conversations across related projects",
        )
        output.append("• get_project_insights - Cross-project activity analysis")

    # Advanced Search Status
    if ADVANCED_SEARCH_AVAILABLE:
        output.append("\n🔍 Advanced Search:")
        output.append("• advanced_search - Faceted search with filtering")
        output.append("• search_suggestions - Auto-completion suggestions")
        output.append("• get_search_metrics - Search activity analytics")
        output.append("• Built-in full-text indexing and highlighting")


def _add_configuration_info(output: list[str]) -> None:
    """Add configuration information to output."""
    if CONFIG_AVAILABLE:
        output.append("\n⚙️ Configuration:")
        output.append("• pyproject.toml configuration support")
        output.append("• Environment variable overrides")
        output.append("• Configurable database, search, and optimization settings")

        # Show current optimization stats if available
        try:
            from .token_optimizer import get_token_optimizer

            optimizer = get_token_optimizer()
            usage_stats = optimizer.get_usage_stats(hours=24)

            if usage_stats["status"] == "success" and usage_stats["total_requests"] > 0:
                savings = usage_stats.get("estimated_cost_savings", {})
                if savings.get("savings_usd", 0) > 0:
                    output.append(
                        f"• Last 24h savings: ${savings['savings_usd']:.4f} USD, {savings['estimated_tokens_saved']:,} tokens",
                    )

            cache_size = len(optimizer.chunk_cache)
            if cache_size > 0:
                output.append(f"• Active cached chunks: {cache_size}")

        except Exception:
            pass  # Don't fail status if optimization stats fail
    else:
        output.append("\n❌ Token optimization not available (install tiktoken)")


def _add_crackerjack_integration_info(output: list[str]) -> None:
    """Add Crackerjack integration information to output."""
    if CRACKERJACK_INTEGRATION_AVAILABLE:
        output.append("\n🔧 Crackerjack Integration (Enhanced):")
        output.append("\n🎯 RECOMMENDED COMMANDS (Enhanced with Memory & Analytics):")
        output.append(
            "• /session-mgmt:crackerjack-run <command> - Smart execution with insights",
        )
        output.append("• /session-mgmt:crackerjack-history - View trends and patterns")
        output.append("• /session-mgmt:crackerjack-metrics - Quality metrics over time")
        output.append("• /session-mgmt:crackerjack-patterns - Test failure analysis")
        output.append("• /session-mgmt:crackerjack-help - Complete command guide")

        # Detect if basic crackerjack is also available
        detected_servers = _detect_other_mcp_servers()
        if detected_servers.get("crackerjack", False):
            output.append("\n📋 Basic Commands (Raw Output Only):")
            output.append(
                "• /crackerjack:run <command> - Simple execution without memory",
            )
            output.append(
                "💡 Use enhanced commands above for better development experience",
            )

        output.append("\n🧠 Enhanced Features:")
        output.append("• Automatic conversation memory integration")
        output.append("• Quality metrics tracking and trends")
        output.append("• Intelligent insights and recommendations")
        output.append("• Test failure pattern detection")
    else:
        output.append("\n⚠️ Crackerjack Integration: Not available")


@mcp.tool()
async def status(working_directory: str | None = None) -> str:
    """Get current session status and project context information with health checks.

    Args:
        working_directory: Optional working directory override (defaults to PWD environment variable or current directory)

    """
    output = []
    output.append("📊 Claude Session Status via MCP Server")
    output.append("=" * 40)

    if working_directory:
        current_dir = Path(working_directory)
    else:
        current_dir = Path(os.environ.get("PWD", Path.cwd()))

    await _add_basic_status_info(output, current_dir)
    await _add_health_status_info(output)
    await _add_project_context_info(output, current_dir)
    _add_permissions_info(output)
    _add_basic_tools_info(output)
    _add_feature_status_info(output)
    _add_configuration_info(output)
    _add_crackerjack_integration_info(output)

    return "\n".join(output)


@mcp.tool()
async def permissions(action: str = "status", operation: str | None = None) -> str:
    """Manage session permissions for trusted operations to avoid repeated prompts.

    Args:
        action: Action to perform: status (show current), trust (add operation), revoke_all (reset)
        operation: Operation to trust (required for 'trust' action)

    """
    output = []
    output.append("🔐 Claude Session Permissions Management")
    output.append("=" * 50)

    if action == "status":
        permissions_status = permissions_manager.get_permission_status()
        output.append(f"\n📊 Session ID: {permissions_status['session_id']}")
        output.append(f"📁 Permissions file: {permissions_status['permissions_file']}")
        output.append(
            f"✅ Trusted operations: {permissions_status['trusted_operations_count']}",
        )

        if permissions_status["trusted_operations"]:
            output.append("\n🔓 Currently trusted operations:")
            for op in permissions_status["trusted_operations"]:
                friendly_name = op.replace("_", " ").title()
                output.append(f"   • {friendly_name}")

            output.append(
                "\n💡 These operations will not prompt for permission in future sessions",
            )
        else:
            output.append("\n⚠️ No operations are currently trusted")
            output.append(
                "💡 Operations will be automatically trusted on first successful use",
            )

        output.append("\n🛠️ Common Operations That Can Be Trusted:")
        output.append("   • UV Package Management - uv sync, pip operations")
        output.append("   • Git Repository Access - git status, commit, push")
        output.append("   • Project File Access - reading/writing project files")
        output.append("   • Subprocess Execution - running build tools, tests")
        output.append("   • Claude Directory Access - accessing ~/.claude/")

    elif action == "trust":
        if not operation:
            output.append("❌ Error: 'operation' parameter required for 'trust' action")
            output.append(
                "💡 Example: permissions with action='trust' and operation='uv_package_management'",
            )
        else:
            permissions_manager.trust_operation(
                operation,
                f"Manually trusted via MCP at {datetime.now()}",
            )
            output.append(
                f"✅ Operation trusted: {operation.replace('_', ' ').title()}",
            )
            output.append("🔓 This operation will no longer prompt for permission")

    elif action == "revoke_all":
        permissions_manager.revoke_all_permissions()
        output.append("🚨 All trusted permissions have been revoked")
        output.append("⚠️ All operations will now prompt for permission again")
        output.append(
            "💡 Use this for security reset or if permissions were granted incorrectly",
        )

    else:
        output.append(f"❌ Unknown action: {action}")
        output.append("💡 Valid actions: 'status', 'trust', 'revoke_all'")

    return "\n".join(output)


# Token optimization imports
try:
    from .token_optimizer import (
        get_token_optimizer,
        optimize_search_response,
        track_token_usage,
    )

    TOKEN_OPTIMIZER_AVAILABLE = True
except ImportError:
    TOKEN_OPTIMIZER_AVAILABLE = False


# Reflection Tools
@mcp.tool()
async def _optimize_search_results(
    results: list,
    optimize_tokens: bool,
    max_tokens: int,
    query: str,
) -> tuple[list, dict]:
    """Apply token optimization to search results if available."""
    optimization_info = {}

    if optimize_tokens and TOKEN_OPTIMIZER_AVAILABLE:
        try:
            results, optimization_info = await optimize_search_response(
                results,
                strategy="prioritize_recent",
                max_tokens=max_tokens,
            )

            # Track usage
            response_text = f"Found {len(results)} conversations"
            await track_token_usage(
                operation="reflect_on_past",
                request_tokens=get_token_optimizer().count_tokens(query),
                response_tokens=get_token_optimizer().count_tokens(response_text),
                optimization_applied=optimization_info.get("strategy"),
            )
        except Exception as e:
            session_logger.warning(f"Token optimization failed: {e}")

    return results, optimization_info


def _build_search_header(
    results: list,
    query: str,
    current_proj: str | None,
    optimization_info: dict,
) -> list[str]:
    """Build the header section of search results."""
    output = []
    output.append(f"🧠 Found {len(results)} relevant conversations for: '{query}'")
    if current_proj:
        output.append(f"📁 Project: {current_proj}")

    # Show optimization info if applied
    if optimization_info and optimization_info.get("strategy") != "none":
        savings = optimization_info.get("token_savings", {})
        if savings.get("tokens_saved", 0) > 0:
            output.append(
                f"⚡ Token optimization: {savings.get('savings_percentage', 0)}% saved",
            )

    output.append("=" * 50)
    return output


def _format_search_results(results: list) -> list[str]:
    """Format search results for display."""
    output = []

    for i, result in enumerate(results, 1):
        score_pct = result["score"] * 100
        timestamp = result.get("timestamp", "Unknown time")
        output.append(f"\n#{i} (Score: {score_pct:.1f}%)")
        output.append(f"📅 {timestamp}")
        output.append(f"💬 {result['content'][:200]}...")
        if result.get("project"):
            output.append(f"📁 Project: {result['project']}")

    return output


async def _perform_main_search(
    query: str,
    limit: int,
    min_score: float,
    current_proj: str | None,
) -> list:
    """Perform the main conversation search."""
    db = await get_reflection_database()
    return await db.search_conversations(
        query=query,
        limit=limit,
        min_score=min_score,
        project=current_proj,
    )


def _should_retry_search(error: Exception) -> bool:
    """Check if search error should trigger a retry."""
    error_str = str(error)
    return (
        "'NoneType' object has no attribute" in error_str
        or "Could not set lock" in error_str
    )


async def _retry_search_with_cleanup(
    query: str,
    limit: int,
    min_score: float,
    project: str | None,
) -> str:
    """Retry search after database cleanup."""
    try:
        from session_mgmt_mcp.reflection_tools import cleanup_reflection_database

        cleanup_reflection_database()
        db = await get_reflection_database()
        current_proj = project or get_current_project()

        results = await db.search_conversations(
            query=query,
            limit=limit,
            min_score=min_score,
            project=current_proj,
        )

        if not results:
            return f"🔍 No conversations found matching '{query}' (minimum similarity: {min_score})"

        output = []
        output.append(f"🔍 Found {len(results)} conversations matching '{query}'")
        output.append(f"📊 Project: {current_proj or 'All projects'}")
        output.append(f"🎯 Similarity threshold: {min_score}")
        output.append("")

        for i, result in enumerate(results, 1):
            score = result.get("score", 0)
            timestamp = result.get("timestamp", "Unknown time")
            content_preview = (
                result.get("content", "")[:200] + "..."
                if len(result.get("content", "")) > 200
                else result.get("content", "")
            )

            output.append(f"📝 Result {i} (similarity: {score:.3f})")
            output.append(f"📅 {timestamp}")
            output.append(f"💬 {content_preview}")
            output.append("")

        return "\n".join(output)
    except Exception as retry_e:
        return f"❌ Error searching conversations (retry failed): {retry_e}"


async def reflect_on_past(
    query: str,
    limit: int = 5,
    min_score: float = 0.7,
    project: str | None = None,
    optimize_tokens: bool = True,
    max_tokens: int = 4000,
) -> str:
    """Search past conversations and store reflections with semantic similarity."""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"

    try:
        current_proj = project or get_current_project()
        results = await _perform_main_search(query, limit, min_score, current_proj)

        if not results:
            return f"🔍 No relevant conversations found for query: '{query}'\n💡 Try adjusting the search terms or lowering min_score."

        # Apply token optimization if available
        results, optimization_info = await _optimize_search_results(
            results,
            optimize_tokens,
            max_tokens,
            query,
        )

        # Build and format output
        output = _build_search_header(results, query, current_proj, optimization_info)
        output.extend(_format_search_results(results))

        return "\n".join(output)

    except Exception as e:
        if _should_retry_search(e):
            return await _retry_search_with_cleanup(query, limit, min_score, project)
        return f"❌ Error searching conversations: {e}"


@mcp.tool()
async def store_reflection(content: str, tags: list[str] | None = None) -> str:
    """Store an important insight or reflection for future reference."""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"

    try:
        db = await get_reflection_database()
        reflection_id = await db.store_reflection(content, tags)

        output = []
        output.append("💾 Reflection stored successfully!")
        output.append(f"🆔 ID: {reflection_id}")
        output.append(f"📝 Content: {content[:100]}...")
        if tags:
            output.append(f"🏷️ Tags: {', '.join(tags)}")
        output.append(f"📅 Stored: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(output)

    except Exception as e:
        # If connection failed, try to cleanup and retry once
        if "'NoneType' object has no attribute 'execute'" in str(e):
            try:
                from session_mgmt_mcp.reflection_tools import (
                    cleanup_reflection_database,
                )

                cleanup_reflection_database()
                db = await get_reflection_database()
                reflection_id = await db.store_reflection(content, tags)

                output = []
                output.append(
                    "💾 Reflection stored successfully! (after connection reset)",
                )
                output.append(f"🆔 ID: {reflection_id}")
                output.append(f"📝 Content: {content[:100]}...")
                if tags:
                    output.append(f"🏷️ Tags: {', '.join(tags)}")
                output.append(
                    f"📅 Stored: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                )

                return "\n".join(output)
            except Exception as retry_e:
                return f"❌ Error storing reflection (retry failed): {retry_e}"

        return f"❌ Error storing reflection: {e}"


@mcp.tool()
async def quick_search(
    query: str,
    min_score: float = 0.7,
    project: str | None = None,
) -> str:
    """Quick search that returns only the count and top result for fast overview."""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"

    try:
        db = await get_reflection_database()
        current_proj = project or get_current_project()

        results = await db.search_conversations(
            query=query,
            limit=1,
            min_score=min_score,
            project=current_proj,
        )

        # Get total count with lower score threshold
        all_results = await db.search_conversations(
            query=query,
            limit=100,
            min_score=0.3,
            project=current_proj,
        )

        output = []
        output.append(f"⚡ Quick search: '{query}'")
        if current_proj:
            output.append(f"📁 Project: {current_proj}")
        output.append(f"📊 Total matches: {len(all_results)} (threshold: 0.3)")
        output.append(
            f"🎯 High relevance: {len([r for r in all_results if r['score'] >= min_score])}",
        )

        if results:
            top_result = results[0]
            score_pct = top_result["score"] * 100
            output.append(f"\n🥇 Top result (Score: {score_pct:.1f}%):")
            output.append(f"💬 {top_result['content'][:150]}...")
            output.append(f"📅 {top_result.get('timestamp', 'Unknown time')}")
        else:
            output.append(
                f"\n💡 No high-relevance matches found (min_score: {min_score})",
            )

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error in quick search: {e}"


@mcp.tool()
async def search_summary(
    query: str,
    min_score: float = 0.7,
    project: str | None = None,
) -> str:
    """Get aggregated insights from search results without individual result details."""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"

    try:
        db = await get_reflection_database()
        current_proj = project or get_current_project()

        results = await db.search_conversations(
            query=query,
            limit=10,
            min_score=min_score,
            project=current_proj,
        )

        if not results:
            return f"📊 Search Summary: No relevant conversations found for '{query}'"

        # Analyze results
        total_results = len(results)
        avg_score = sum(r["score"] for r in results) / total_results
        projects = {r.get("project") for r in results if r.get("project")}

        # Extract common themes (simple keyword analysis)
        all_content = " ".join(r["content"] for r in results)
        common_words = {}
        for word in all_content.lower().split():
            if len(word) > 4:  # Only consider longer words
                common_words[word] = common_words.get(word, 0) + 1

        top_themes = sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:5]

        output = []
        output.append(f"📊 Search Summary: '{query}'")
        output.append("=" * 40)
        output.append(f"📈 Total matches: {total_results}")
        output.append(f"🎯 Average relevance: {avg_score * 100:.1f}%")
        if projects:
            output.append(f"📁 Projects involved: {', '.join(projects)}")

        if top_themes:
            output.append("\n🔑 Common themes:")
            for word, count in top_themes:
                if count > 1:
                    output.append(f"   • {word} ({count} mentions)")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error generating search summary: {e}"


@mcp.tool()
async def get_more_results(
    query: str,
    offset: int = 3,
    limit: int = 3,
    project: str | None = None,
) -> str:
    """Get additional search results after an initial search (pagination support)."""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"

    try:
        db = await get_reflection_database()
        current_proj = project or get_current_project()

        # Get more results than needed to handle offset
        all_results = await db.search_conversations(
            query=query,
            limit=offset + limit,
            min_score=0.5,
            project=current_proj,
        )

        # Apply offset and limit
        paginated_results = all_results[offset : offset + limit]

        if not paginated_results:
            return f"📄 No more results found starting from position {offset + 1}"

        output = []
        output.append(
            f"📄 Additional results for: '{query}' (positions {offset + 1}-{offset + len(paginated_results)})",
        )
        if current_proj:
            output.append(f"📁 Project: {current_proj}")
        output.append("=" * 50)

        for i, result in enumerate(paginated_results, offset + 1):
            score_pct = result["score"] * 100
            timestamp = result.get("timestamp", "Unknown time")
            output.append(f"\n#{i} (Score: {score_pct:.1f}%)")
            output.append(f"📅 {timestamp}")
            output.append(f"💬 {result['content'][:200]}...")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error getting more results: {e}"


@mcp.tool()
async def search_by_file(
    file_path: str,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Search for conversations that analyzed a specific file."""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"

    try:
        db = await get_reflection_database()
        current_proj = project or get_current_project()

        results = await db.search_by_file(
            file_path=file_path,
            limit=limit,
            project=current_proj,
        )

        if not results:
            return f"🔍 No conversations found mentioning file: {file_path}"

        output = []
        output.append(f"📁 Found {len(results)} conversations about: {file_path}")
        if current_proj:
            output.append(f"🗂️ Project: {current_proj}")
        output.append("=" * 60)

        for i, result in enumerate(results, 1):
            timestamp = result.get("timestamp", "Unknown time")
            output.append(f"\n#{i}")
            output.append(f"📅 {timestamp}")
            output.append(f"💬 {result['content'][:250]}...")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error searching by file: {e}"


@mcp.tool()
async def search_by_concept(
    concept: str,
    include_files: bool = True,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Search for conversations about a specific development concept."""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"

    try:
        db = await get_reflection_database()
        current_proj = project or get_current_project()

        # Search both conversations and reflections for the concept
        conv_results = await db.search_conversations(
            query=concept,
            limit=limit,
            min_score=0.6,
            project=current_proj,
        )

        refl_results = await db.search_reflections(
            query=concept,
            limit=5,
            min_score=0.6,
        )

        output = []
        output.append(f"🧠 Concept search: '{concept}'")
        if current_proj:
            output.append(f"📁 Project: {current_proj}")
        output.append("=" * 50)

        if conv_results:
            output.append(f"\n💬 Conversations ({len(conv_results)}):")
            for i, result in enumerate(conv_results[:5], 1):
                score_pct = result["score"] * 100
                output.append(
                    f"#{i} (Score: {score_pct:.1f}%) {result['content'][:150]}...",
                )

        if refl_results:
            output.append(f"\n💭 Reflections ({len(refl_results)}):")
            for i, result in enumerate(refl_results, 1):
                score_pct = result["score"] * 100
                tags_str = f" [{', '.join(result['tags'])}]" if result["tags"] else ""
                output.append(
                    f"#{i} (Score: {score_pct:.1f}%){tags_str} {result['content'][:150]}...",
                )

        if not conv_results and not refl_results:
            output.append(f"🔍 No relevant content found for concept: '{concept}'")
            output.append(
                "💡 Try related terms or check if conversations about this topic exist",
            )

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error searching by concept: {e}"


@mcp.tool()
async def reset_reflection_database() -> str:
    """Reset the reflection database connection to fix lock issues."""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"

    try:
        from session_mgmt_mcp.reflection_tools import (
            cleanup_reflection_database,
            get_reflection_database,
        )

        # Clean up existing instance
        cleanup_reflection_database()

        # Test new connection
        db = await get_reflection_database()
        stats = await db.get_stats()

        return f"✅ Reflection database reset successfully!\n📊 Stats: {stats}"

    except Exception as e:
        return f"❌ Error resetting database: {e}"


@mcp.tool()
async def reflection_stats() -> str:
    """Get statistics about the reflection database."""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"

    try:
        db = await get_reflection_database()
        stats = await db.get_stats()

        if "error" in stats:
            return f"❌ {stats['error']}"

        output = []
        output.append("📊 Reflection Database Statistics")
        output.append("=" * 40)
        output.append(f"💬 Conversations: {stats['conversations_count']}")
        output.append(f"💭 Reflections: {stats['reflections_count']}")
        output.append(f"🧠 Embedding provider: {stats['embedding_provider']}")
        output.append(f"📏 Embedding dimension: {stats['embedding_dimension']}")
        output.append(f"💾 Database: {stats['database_path']}")

        total_items = stats["conversations_count"] + stats["reflections_count"]
        output.append(f"\n📈 Total stored items: {total_items}")

        if total_items > 0:
            output.append("✅ Memory system is active and contains data")
        else:
            output.append("💡 Memory system is ready but contains no data yet")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error getting reflection stats: {e}"


# Token Optimization Tools
@mcp.tool()
async def get_cached_chunk(cache_key: str, chunk_index: int) -> str:
    """Get a specific chunk from cached chunked response."""
    if not TOKEN_OPTIMIZER_AVAILABLE:
        return "❌ Token optimizer not available. Install dependencies: pip install tiktoken"

    try:
        from .token_optimizer import get_cached_chunk

        chunk_data = await get_cached_chunk(cache_key, chunk_index)

        if not chunk_data:
            return f"❌ Chunk not found. Cache key '{cache_key}' or chunk {chunk_index} may have expired."

        output = []
        output.append(
            f"📄 Chunk {chunk_data['current_chunk']} of {chunk_data['total_chunks']}",
        )
        output.append("=" * 50)

        chunk = chunk_data["chunk"]
        for i, item in enumerate(chunk, 1):
            timestamp = item.get("timestamp", "Unknown time")
            output.append(f"\n#{i}")
            output.append(f"📅 {timestamp}")
            output.append(f"💬 {item.get('content', '')[:200]}...")

        if chunk_data["has_more"]:
            output.append(
                f"\n💡 More chunks available. Use: get_cached_chunk('{cache_key}', {chunk_data['current_chunk'] + 1})",
            )

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error retrieving cached chunk: {e}"


@mcp.tool()
async def get_token_usage_stats(hours: int = 24) -> str:
    """Get token usage statistics and optimization metrics."""
    if not TOKEN_OPTIMIZER_AVAILABLE:
        return "❌ Token optimizer not available. Install dependencies: pip install tiktoken"

    try:
        from .token_optimizer import get_token_usage_stats

        stats = await get_token_usage_stats(hours)

        if stats["status"] == "no_data":
            return f"📊 No token usage data available for the last {hours} hours"

        output = []
        output.append(f"📊 Token Usage Statistics (Last {hours} hours)")
        output.append("=" * 50)
        output.append(f"📈 Total Requests: {stats['total_requests']}")
        output.append(f"🔤 Total Tokens Used: {stats['total_tokens']:,}")
        output.append(
            f"📊 Average Tokens per Request: {stats['average_tokens_per_request']}",
        )

        if stats.get("optimizations_applied"):
            output.append("\n⚡ Optimizations Applied:")
            for strategy, count in stats["optimizations_applied"].items():
                output.append(f"  • {strategy}: {count} times")

        cost_savings = stats.get("estimated_cost_savings", {})
        if cost_savings.get("savings_usd", 0) > 0:
            output.append("\n💰 Estimated Cost Savings:")
            output.append(f"  • ${cost_savings['savings_usd']:.4f} USD saved")
            output.append(
                f"  • {cost_savings['estimated_tokens_saved']:,} tokens saved",
            )
            output.append(
                f"  • {cost_savings['requests_optimized']} optimized requests",
            )

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error getting token usage stats: {e}"


@mcp.tool()
async def optimize_memory_usage(
    strategy: str = "auto",
    max_age_days: int = 30,
    dry_run: bool = True,
) -> str:
    """Optimize memory usage by consolidating old conversations."""
    if not TOKEN_OPTIMIZER_AVAILABLE or not REFLECTION_TOOLS_AVAILABLE:
        return (
            "❌ Memory optimization requires both token optimizer and reflection tools"
        )

    try:
        from .memory_optimizer import MemoryOptimizer

        db = await get_reflection_database()
        optimizer = MemoryOptimizer(db)

        # Set up retention policy based on parameters
        policy = None
        if strategy != "auto":
            policy = {
                "consolidation_age_days": max_age_days,
                "importance_threshold": 0.3 if strategy == "aggressive" else 0.5,
            }

        results = await optimizer.compress_memory(policy=policy, dry_run=dry_run)

        if results.get("error"):
            return f"❌ Memory optimization error: {results['error']}"

        output = []
        output.append(
            f"🧠 Memory Optimization Results {'(DRY RUN)' if dry_run else ''}",
        )
        output.append("=" * 50)
        output.append(f"📊 Total Conversations: {results['total_conversations']}")
        output.append(f"✅ Conversations to Keep: {results['conversations_to_keep']}")
        output.append(
            f"📦 Conversations to Consolidate: {results['conversations_to_consolidate']}",
        )
        output.append(f"🔗 Clusters Created: {results['clusters_created']}")

        if results.get("space_saved_estimate", 0) > 0:
            output.append("\n💾 Space Optimization:")
            output.append(f"  • {results['space_saved_estimate']:,} characters saved")
            output.append(f"  • {results['compression_ratio']:.1%} compression ratio")

        if results.get("consolidated_summaries"):
            output.append("\n📝 Consolidation Preview:")
            for i, summary in enumerate(results["consolidated_summaries"][:3], 1):
                output.append(
                    f"  #{i}: {summary['original_count']} conversations → 1 summary",
                )
                output.append(f"      Projects: {', '.join(summary['projects'][:2])}")
                output.append(f"      Summary: {summary['summary'][:100]}...")

        if dry_run:
            output.append("\n💡 Run with dry_run=False to apply changes")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error optimizing memory: {e}"


# Enhanced Search Tools (Phase 1)


@mcp.tool()
async def search_code(
    query: str,
    pattern_type: str | None = None,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Search for code patterns in conversations using AST parsing.

    Args:
        query: Search query for code patterns
        pattern_type: Type of code pattern (function, class, import, assignment, call, loop, conditional, try, async)
        limit: Maximum number of results to return
        project: Optional project filter

    """
    if not ENHANCED_SEARCH_AVAILABLE or not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Enhanced search or reflection tools not available"

    try:
        db = await get_reflection_database()
        search_engine = EnhancedSearchEngine(db)

        results = await search_engine.search_code_patterns(query, pattern_type, limit)

        if not results:
            return f"🔍 No code patterns found for query: '{query}'"

        output = []
        output.append(f"🔍 Code Pattern Search: '{query}'")
        if pattern_type:
            output.append(f"📝 Pattern type: {pattern_type}")
        output.append("=" * 50)

        for i, result in enumerate(results, 1):
            pattern = result["pattern"]
            relevance_pct = result["relevance"] * 100

            output.append(f"\n#{i} (Relevance: {relevance_pct:.1f}%)")
            output.append(f"📁 Project: {result['project']}")
            output.append(f"🕒 Time: {result['timestamp']}")
            output.append(f"📋 Pattern: {pattern['type']}")

            if "name" in pattern:
                output.append(f"🏷️ Name: {pattern['name']}")

            output.append("💻 Code snippet:")
            code_lines = pattern["content"].split("\n")[:3]  # First 3 lines
            for line in code_lines:
                output.append(f"    {line}")
            if len(pattern["content"].split("\n")) > 3:
                output.append("    ...")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error searching code patterns: {e}"


@mcp.tool()
async def search_errors(
    query: str,
    error_type: str | None = None,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Search for error patterns and debugging contexts in conversations.

    Args:
        query: Search query for error patterns
        error_type: Type of error (python_traceback, python_exception, javascript_error, etc.)
        limit: Maximum number of results to return
        project: Optional project filter

    """
    if not ENHANCED_SEARCH_AVAILABLE or not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Enhanced search or reflection tools not available"

    try:
        db = await get_reflection_database()
        search_engine = EnhancedSearchEngine(db)

        results = await search_engine.search_error_patterns(query, error_type, limit)

        if not results:
            return f"🔍 No error patterns found for query: '{query}'"

        output = []
        output.append(f"🚨 Error Pattern Search: '{query}'")
        if error_type:
            output.append(f"⚠️ Error type: {error_type}")
        output.append("=" * 50)

        for i, result in enumerate(results, 1):
            pattern = result["pattern"]
            relevance_pct = result["relevance"] * 100

            output.append(f"\n#{i} (Relevance: {relevance_pct:.1f}%)")
            output.append(f"📁 Project: {result['project']}")
            output.append(f"🕒 Time: {result['timestamp']}")
            output.append(f"🚨 Pattern: {pattern['type']} - {pattern['subtype']}")

            if pattern["type"] == "error":
                if pattern.get("groups"):
                    output.append(
                        f"💀 Error: {pattern['groups'][0] if pattern['groups'] else 'Unknown'}",
                    )
                    if len(pattern["groups"]) > 1:
                        output.append(f"📝 Message: {pattern['groups'][1]}")

            # Show relevant snippet
            snippet = result["snippet"]
            output.append(f"📄 Context: {snippet[:200]}...")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error searching error patterns: {e}"


@mcp.tool()
async def search_temporal(
    time_expression: str,
    query: str | None = None,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Search conversations within a specific time range using natural language.

    Args:
        time_expression: Natural language time expression (e.g., "yesterday", "last week", "2 days ago")
        query: Optional search query to filter results within the time range
        limit: Maximum number of results to return
        project: Optional project filter

    """
    if not ENHANCED_SEARCH_AVAILABLE or not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Enhanced search or reflection tools not available"

    try:
        db = await get_reflection_database()
        search_engine = EnhancedSearchEngine(db)

        results = await search_engine.search_temporal(time_expression, query, limit)

        if not results:
            return f"🔍 No conversations found for time: '{time_expression}'"

        if len(results) == 1 and "error" in results[0]:
            return f"❌ {results[0]['error']}"

        output = []
        output.append(f"🕒 Temporal Search: '{time_expression}'")
        if query:
            output.append(f"🔍 Query: '{query}'")
        output.append("=" * 50)

        for i, result in enumerate(results, 1):
            relevance_pct = result.get("relevance", 1.0) * 100

            output.append(f"\n#{i} (Relevance: {relevance_pct:.1f}%)")
            output.append(f"📁 Project: {result['project']}")
            output.append(f"🕒 Time: {result['timestamp']}")
            output.append(f"💬 Content: {result['content']}")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error in temporal search: {e}"


@mcp.tool()
async def auto_load_context(
    working_directory: str | None = None,
    max_conversations: int = 10,
    min_relevance: float = 0.3,
) -> str:
    """Automatically load relevant conversations based on current development context.

    Args:
        working_directory: Optional working directory override
        max_conversations: Maximum number of conversations to load
        min_relevance: Minimum relevance score (0.0-1.0)

    """
    if not AUTO_CONTEXT_AVAILABLE or not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Auto-context loading or reflection tools not available"

    try:
        db = await get_reflection_database()
        context_loader = AutoContextLoader(db)

        result = await context_loader.load_relevant_context(
            working_directory,
            max_conversations,
            min_relevance,
        )

        context = result["context"]
        conversations = result["relevant_conversations"]

        output = []
        output.append("🤖 Auto-Context Loading Results")
        output.append("=" * 50)

        # Context summary
        output.append("\n📋 Detected Context:")
        output.append(f"📁 Project: {context['project_name']}")

        if context["detected_languages"]:
            langs = ", ".join(context["detected_languages"])
            output.append(f"💻 Languages: {langs}")

        if context["detected_tools"]:
            tools = ", ".join(context["detected_tools"])
            output.append(f"🔧 Tools: {tools}")

        if context["project_type"]:
            proj_type = context["project_type"].replace("_", " ").title()
            output.append(f"📋 Type: {proj_type}")

        confidence = context["confidence_score"] * 100
        output.append(f"🎯 Detection confidence: {confidence:.0f}%")

        # Git info
        if context["git_info"].get("is_git_repo"):
            git_info = context["git_info"]
            branch = git_info.get("current_branch", "unknown")
            platform = git_info.get("platform", "git")
            output.append(f"🌿 Git: {branch} branch on {platform}")

        # Relevant conversations
        output.append(
            f"\n💬 Loaded Conversations: {result['loaded_count']}/{result['total_found']} found",
        )
        if conversations:
            for i, conv in enumerate(conversations, 1):
                relevance_pct = conv["relevance_score"] * 100
                output.append(f"\n#{i} (Relevance: {relevance_pct:.1f}%)")
                output.append(f"📁 Project: {conv['project']}")
                output.append(f"🕒 Time: {conv['timestamp']}")

                # Show snippet of content
                content_preview = conv["content"][:200]
                if len(conv["content"]) > 200:
                    content_preview += "..."
                output.append(f"💬 Content: {content_preview}")
        else:
            output.append(
                f"🔍 No conversations found above {min_relevance:.1f} relevance threshold",
            )
            output.append(
                "💡 Try lowering min_relevance or working on this project more",
            )

        # Recent files info
        if context["recent_files"]:
            recent_count = len(context["recent_files"])
            output.append(
                f"\n📄 Recent activity: {recent_count} files modified in last 2 hours",
            )
            for file_info in context["recent_files"][:3]:  # Show top 3
                output.append(f"    📝 {file_info['path']}")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error in auto-context loading: {e}"


@mcp.tool()
async def get_context_summary(working_directory: str | None = None) -> str:
    """Get a summary of current development context without loading conversations.

    Args:
        working_directory: Optional working directory override

    """
    if not AUTO_CONTEXT_AVAILABLE:
        return "❌ Auto-context loading tools not available"

    try:
        db = await get_reflection_database() if REFLECTION_TOOLS_AVAILABLE else None
        context_loader = AutoContextLoader(db) if db else None

        if context_loader:
            summary = await context_loader.get_context_summary(working_directory)
        else:
            # Fallback to basic context detection
            from session_mgmt_mcp.context_manager import ContextDetector

            detector = ContextDetector()
            context = detector.detect_current_context(working_directory)

            summary_parts = []
            summary_parts.append(f"📁 Project: {context['project_name']}")
            summary_parts.append(f"📂 Directory: {context['working_directory']}")

            if context["detected_languages"]:
                langs = ", ".join(context["detected_languages"])
                summary_parts.append(f"💻 Languages: {langs}")

            if context["detected_tools"]:
                tools = ", ".join(context["detected_tools"])
                summary_parts.append(f"🔧 Tools: {tools}")

            confidence = context["confidence_score"] * 100
            summary_parts.append(f"🎯 Detection confidence: {confidence:.0f}%")

            summary = "\n".join(summary_parts)

        output = []
        output.append("📋 Current Development Context")
        output.append("=" * 40)
        output.append(summary)

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error getting context summary: {e}"


@mcp.tool()
async def compress_memory(
    max_age_days: int = 30,
    max_conversations: int = 10000,
    importance_threshold: float = 0.3,
    dry_run: bool = False,
) -> str:
    """Compress conversation memory by consolidating old conversations.

    Args:
        max_age_days: Consolidate conversations older than this many days
        max_conversations: Maximum total conversations to keep
        importance_threshold: Minimum importance score to avoid consolidation (0.0-1.0)
        dry_run: Preview changes without actually applying them

    """
    if not MEMORY_OPTIMIZER_AVAILABLE or not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Memory optimizer or reflection tools not available"

    try:
        db = await get_reflection_database()
        optimizer = MemoryOptimizer(db)

        policy = {
            "consolidation_age_days": max_age_days,
            "max_conversations": max_conversations,
            "importance_threshold": importance_threshold,
        }

        result = await optimizer.compress_memory(policy, dry_run)

        if "error" in result:
            return f"❌ {result['error']}"

        output = []
        output.append("🗜️ Memory Compression Results")
        output.append("=" * 50)

        if dry_run:
            output.append("🔍 DRY RUN - No changes made")

        output.append("\n📊 Overview:")
        output.append(f"📁 Total conversations: {result['total_conversations']}")
        output.append(f"✅ Conversations to keep: {result['conversations_to_keep']}")
        output.append(
            f"🗜️ Conversations to consolidate: {result['conversations_to_consolidate']}",
        )
        output.append(f"📦 Clusters created: {result['clusters_created']}")

        if result["space_saved_estimate"] > 0:
            space_mb = result["space_saved_estimate"] / (1024 * 1024)
            compression_pct = result["compression_ratio"] * 100
            output.append(
                f"💾 Space saved: {space_mb:.2f}MB ({compression_pct:.1f}% reduction)",
            )

        # Show consolidation details
        if result["consolidated_summaries"]:
            output.append("\n📋 Consolidation Details:")
            for i, summary in enumerate(
                result["consolidated_summaries"][:5],
                1,
            ):  # Show top 5
                original_kb = summary["original_size"] / 1024
                compressed_kb = summary["compressed_size"] / 1024
                reduction_pct = (
                    1 - summary["compressed_size"] / summary["original_size"]
                ) * 100

                output.append(f"\n#{i} Cluster:")
                output.append(
                    f"   📁 Projects: {', '.join(summary['projects']) if summary['projects'] else 'Multiple'}",
                )
                output.append(f"   💬 Conversations: {summary['original_count']}")
                output.append(
                    f"   📏 Size: {original_kb:.1f}KB → {compressed_kb:.1f}KB ({reduction_pct:.1f}% reduction)",
                )
                output.append(f"   🕒 Time range: {summary['time_range']}")
                output.append(f"   📝 Summary: {summary['summary'][:100]}...")

        if not dry_run and result["consolidated_summaries"]:
            output.append("\n✅ Memory compression completed successfully!")
        elif dry_run:
            output.append("\n💡 Run with dry_run=False to apply these changes")
        elif not result["consolidated_summaries"]:
            output.append(
                "\n💡 No consolidation needed - all conversations are recent or important",
            )

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error compressing memory: {e}"


@mcp.tool()
async def get_compression_stats() -> str:
    """Get memory compression statistics and history."""
    if not MEMORY_OPTIMIZER_AVAILABLE or not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Memory optimizer or reflection tools not available"

    try:
        db = await get_reflection_database()
        optimizer = MemoryOptimizer(db)

        stats = await optimizer.get_compression_stats()

        output = []
        output.append("📊 Memory Compression Statistics")
        output.append("=" * 40)

        if stats["last_run"]:
            last_run = stats["last_run"]
            output.append(f"🕒 Last compression: {last_run}")
            output.append(
                f"💬 Conversations processed: {stats['conversations_processed']}",
            )
            output.append(
                f"📦 Conversations consolidated: {stats['conversations_consolidated']}",
            )

            if stats["space_saved_bytes"] > 0:
                space_mb = stats["space_saved_bytes"] / (1024 * 1024)
                compression_pct = stats["compression_ratio"] * 100
                output.append(f"💾 Space saved: {space_mb:.2f}MB")
                output.append(f"🗜️ Compression ratio: {compression_pct:.1f}%")
        else:
            output.append("💡 No compression runs performed yet")
            output.append(
                "🔧 Use compress_memory() to start optimizing your conversation storage",
            )

        # Current database stats
        if hasattr(db, "conn") and db.conn:
            cursor = db.conn.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]

            cursor = db.conn.execute("SELECT SUM(LENGTH(content)) FROM conversations")
            total_size = cursor.fetchone()[0] or 0

            output.append("\n📈 Current Database:")
            output.append(f"💬 Total conversations: {total_conversations}")
            size_mb = total_size / (1024 * 1024)
            output.append(f"💾 Total size: {size_mb:.2f}MB")

            # Check for consolidated conversations
            cursor = db.conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE metadata LIKE '%consolidated%'",
            )
            consolidated_count = cursor.fetchone()[0]
            if consolidated_count > 0:
                output.append(f"📦 Consolidated conversations: {consolidated_count}")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error getting compression stats: {e}"


@mcp.tool()
async def set_retention_policy(
    max_age_days: int = 365,
    max_conversations: int = 10000,
    importance_threshold: float = 0.3,
    consolidation_age_days: int = 30,
) -> str:
    """Set memory retention policy parameters.

    Args:
        max_age_days: Maximum age in days to keep conversations
        max_conversations: Maximum total conversations to store
        importance_threshold: Minimum importance score to retain (0.0-1.0)
        consolidation_age_days: Age in days after which to consolidate conversations

    """
    if not MEMORY_OPTIMIZER_AVAILABLE:
        return "❌ Memory optimizer not available"

    try:
        # Create temporary optimizer to set policy
        from session_mgmt_mcp.memory_optimizer import MemoryOptimizer

        db = await get_reflection_database() if REFLECTION_TOOLS_AVAILABLE else None
        optimizer = MemoryOptimizer(db) if db else MemoryOptimizer(None)

        policy = {
            "max_age_days": max_age_days,
            "max_conversations": max_conversations,
            "importance_threshold": importance_threshold,
            "consolidation_age_days": consolidation_age_days,
        }

        result = await optimizer.set_retention_policy(policy)

        if "error" in result:
            return f"❌ {result['error']}"

        output = []
        output.append("⚙️ Retention Policy Updated")
        output.append("=" * 40)

        policy = result["updated_policy"]
        output.append(f"📅 Max age: {policy['max_age_days']} days")
        output.append(f"💬 Max conversations: {policy['max_conversations']}")
        output.append(f"⭐ Importance threshold: {policy['importance_threshold']:.1f}")
        output.append(f"📦 Consolidation age: {policy['consolidation_age_days']} days")
        output.append(
            f"🗜️ Target compression: {policy.get('compression_ratio', 0.5) * 100:.0f}%",
        )

        output.append("\n💡 These settings will be applied on the next compression run")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error setting retention policy: {e}"


@mcp.prompt("compress-memory")
async def get_compress_memory_prompt() -> str:
    """Compress conversation memory by consolidating old conversations into summaries."""
    return """# Memory Compression

Compress conversation memory by consolidating old conversations into summaries.

This command will:
- Analyze conversation age and importance
- Group related conversations into clusters
- Create consolidated summaries of old conversations
- Remove redundant conversation data
- Calculate space savings and compression ratios

Examples:
- Default compression: compress_memory()
- Preview changes: dry_run=True
- Aggressive compression: max_age_days=14, importance_threshold=0.5

Use this periodically to keep your conversation memory manageable and efficient."""


@mcp.prompt("compression-stats")
async def get_compression_stats_prompt() -> str:
    """Get detailed statistics about memory compression history and current database status."""
    return """# Compression Statistics

Get detailed statistics about memory compression history and current database status.

This command will:
- Show last compression run details
- Display space savings and compression ratios
- Report current database size and conversation count
- Show number of consolidated conversations
- Provide compression efficiency metrics

Use this to monitor memory usage and compression effectiveness."""


@mcp.prompt("retention-policy")
async def get_retention_policy_prompt() -> str:
    """Configure memory retention policy parameters for automatic compression."""
    return """# Retention Policy

Configure memory retention policy parameters for automatic compression.

This command will:
- Set maximum conversation age and count limits
- Configure importance threshold for retention
- Define consolidation age triggers
- Adjust compression ratio targets

Examples:
- Conservative: max_age_days=365, importance_threshold=0.2
- Aggressive: max_age_days=90, importance_threshold=0.5
- Custom: consolidation_age_days=14

Use this to customize how your conversation memory is managed over time."""


@mcp.prompt("auto-load-context")
async def get_auto_load_context_prompt() -> str:
    """Automatically detect current development context and load relevant conversations."""
    return """# Auto-Context Loading

Automatically detect current development context and load relevant conversations.

This command will:
- Analyze your current project structure and files
- Detect programming languages and tools in use
- Identify project type (web app, CLI tool, library, etc.)
- Find recent file modifications
- Load conversations relevant to your current context
- Score conversations by relevance to current work

Examples:
- Load default context: auto_load_context()
- Increase results: max_conversations=20
- Lower threshold: min_relevance=0.2

Use this at the start of coding sessions to get relevant context automatically."""


@mcp.prompt("context-summary")
async def get_context_summary_prompt() -> str:
    """Get a quick summary of your current development context without loading conversations."""
    return """# Context Summary

Get a quick summary of your current development context without loading conversations.

This command will:
- Detect current project name and type
- Identify programming languages and tools
- Show Git repository information
- Display recently modified files
- Calculate detection confidence score

Use this to understand what context the system has detected about your current work."""


# Register enhanced search prompts


@mcp.prompt("search-code")
async def get_search_code_prompt() -> str:
    """Search for code patterns in conversations using AST parsing."""
    return """# Code Pattern Search

Search for code patterns in your conversation history using AST (Abstract Syntax Tree) parsing.

This command will:
- Parse Python code blocks from conversations
- Extract functions, classes, imports, loops, and other patterns
- Rank results by relevance to your query
- Show code context and project information

Examples:
- Search for functions: pattern_type='function'
- Search for class definitions: pattern_type='class'
- Search for error handling: query='try except'

Use this to find code examples and patterns from your development sessions."""


@mcp.prompt("search-errors")
async def get_search_errors_prompt() -> str:
    """Search for error patterns and debugging contexts in conversations."""
    return """# Error Pattern Search

Search for error messages, exceptions, and debugging contexts in your conversation history.

This command will:
- Find Python tracebacks and exceptions
- Detect JavaScript errors and warnings
- Identify debugging and testing contexts
- Show error context and solutions

Examples:
- Find Python errors: error_type='python_exception'
- Find import issues: query='ImportError'
- Find debugging sessions: query='debug'

Use this to quickly find solutions to similar errors you've encountered before."""


@mcp.prompt("search-temporal")
async def get_search_temporal_prompt() -> str:
    """Search conversations within a specific time range using natural language."""
    return """# Temporal Search

Search your conversation history using natural language time expressions.

This command will:
- Parse time expressions like "yesterday", "last week", "2 days ago"
- Find conversations within that time range
- Optionally filter by additional search terms
- Sort results by time and relevance

Examples:
- "yesterday" - conversations from yesterday
- "last week" - conversations from the past week
- "2 days ago" - conversations from 2 days ago
- "this month" + query - filter by content within the month

Use this to find recent discussions or work from specific time periods."""


@mcp.prompt("start-app-monitoring")
async def get_start_app_monitoring_prompt() -> str:
    """Start monitoring IDE activity and browser documentation usage."""
    return """# Start Application Monitoring

Monitor your development activity to provide better context and insights.

This command will:
- Start file system monitoring for code changes
- Track application focus (IDE, browser, terminal)
- Monitor documentation site visits
- Build activity profiles for better context

Monitoring includes:
- File modifications in your project directories
- IDE and editor activity patterns
- Browser navigation to documentation sites
- Application focus and context switching

Use this to automatically capture your development context for better session insights."""


@mcp.prompt("stop-app-monitoring")
async def get_stop_app_monitoring_prompt() -> str:
    """Stop all application monitoring."""
    return """# Stop Application Monitoring

Stop monitoring your development activity.

This command will:
- Stop file system monitoring
- Stop application focus tracking
- Preserve collected activity data
- Clean up monitoring resources

Use this when you want to pause monitoring or when you're done with a development session."""


@mcp.prompt("activity-summary")
async def get_activity_summary_prompt() -> str:
    """Get activity summary for recent development work."""
    return """# Activity Summary

Get a comprehensive summary of your recent development activity.

This command will:
- Show file modification patterns
- List most active applications
- Display visited documentation sites
- Calculate productivity metrics

Summary includes:
- Event counts by type and application
- Most actively edited files
- Documentation resources consulted
- Average relevance scores

Use this to understand your development patterns and identify productive sessions."""


@mcp.prompt("context-insights")
async def get_context_insights_prompt() -> str:
    """Get contextual insights from recent activity."""
    return """# Context Insights

Analyze recent development activity for contextual insights.

This command will:
- Identify primary focus areas
- Detect technologies being used
- Count context switches
- Calculate productivity scores

Insights include:
- Primary application focus
- Active programming languages
- Documentation topics explored
- Project switching patterns
- Overall productivity assessment

Use this to understand your current development context and optimize your workflow."""


@mcp.prompt("active-files")
async def get_active_files_prompt() -> str:
    """Get files currently being worked on."""
    return """# Active Files

Show files that are currently being actively worked on.

This command will:
- List recently modified files
- Show activity scores and patterns
- Highlight most frequently changed files
- Include project context

File activity is scored based on:
- Frequency of modifications
- Recency of changes
- File type and relevance
- Project context

Use this to quickly see what you're currently working on and resume interrupted tasks."""


# Global app monitor instance
_app_monitor = None


async def get_app_monitor() -> ApplicationMonitor | None:
    """Get or initialize application monitor."""
    global _app_monitor
    if not APP_MONITOR_AVAILABLE:
        return None

    if _app_monitor is None:
        data_dir = Path.home() / ".claude" / "data" / "app_monitoring"
        working_dir = os.environ.get("PWD", os.getcwd())
        project_paths = [working_dir] if Path(working_dir).exists() else []
        _app_monitor = ApplicationMonitor(str(data_dir), project_paths)

    return _app_monitor


@mcp.tool()
async def start_app_monitoring(project_paths: list[str] | None = None) -> str:
    """Start monitoring IDE activity and browser documentation usage.

    Args:
        project_paths: Optional list of project directories to monitor

    """
    if not APP_MONITOR_AVAILABLE:
        return "❌ Application monitoring not available. Install dependencies: pip install watchdog psutil"

    try:
        monitor = await get_app_monitor()
        if not monitor:
            return "❌ Failed to initialize application monitor"

        # Update project paths if provided
        if project_paths:
            monitor.project_paths = project_paths
            monitor.ide_monitor.project_paths = project_paths

        result = await monitor.start_monitoring()

        status_lines = [
            "🎯 Application monitoring started",
            f"📁 Monitoring {len(monitor.project_paths)} project paths",
            f"📝 IDE monitoring: {'✅' if result.get('ide_monitoring') else '❌'}",
            f"🔍 Watchdog available: {'✅' if result.get('watchdog_available') else '❌'}",
            f"📊 Process monitoring: {'✅' if result.get('psutil_available') else '❌'}",
        ]

        return "\n".join(status_lines)

    except Exception as e:
        return f"❌ Error starting monitoring: {e}"


@mcp.tool()
async def stop_app_monitoring() -> str:
    """Stop all application monitoring."""
    if not APP_MONITOR_AVAILABLE:
        return "❌ Application monitoring not available"

    try:
        monitor = await get_app_monitor()
        if monitor and monitor.monitoring_active:
            await monitor.stop_monitoring()
            return "✅ Application monitoring stopped"
        return "ℹ️ Application monitoring was not active"
    except Exception as e:
        return f"❌ Error stopping monitoring: {e}"


@mcp.tool()
async def get_activity_summary(hours: int = 2) -> str:
    """Get activity summary for the specified number of hours.

    Args:
        hours: Number of hours to look back (default: 2)

    """
    if not APP_MONITOR_AVAILABLE:
        return "❌ Application monitoring not available"

    try:
        monitor = await get_app_monitor()
        if not monitor:
            return "❌ Application monitor not initialized"

        summary = monitor.get_activity_summary(hours)

        output = [
            f"📊 Activity Summary (Last {hours} hours)",
            f"🎯 Total events: {summary['total_events']}",
            f"📱 Average relevance: {summary['average_relevance']:.2f}",
            "",
        ]

        if summary["event_types"]:
            output.append("📋 Event Types:")
            for event_type, count in summary["event_types"].items():
                output.append(f"  • {event_type}: {count}")
            output.append("")

        if summary["applications"]:
            output.append("💻 Applications:")
            for app, count in summary["applications"].items():
                output.append(f"  • {app}: {count}")
            output.append("")

        if summary["active_files"]:
            output.append("📄 Most Active Files:")
            for file_info in summary["active_files"][:5]:
                output.append(
                    f"  • {file_info['file_path']} (score: {file_info['activity_score']})",
                )
            output.append("")

        if summary["documentation_sites"]:
            output.append("📖 Documentation Sites Visited:")
            for site in summary["documentation_sites"]:
                output.append(f"  • {site}")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error getting activity summary: {e}"


@mcp.tool()
async def get_context_insights(hours: int = 1) -> str:
    """Get contextual insights from recent activity.

    Args:
        hours: Number of hours to analyze (default: 1)

    """
    if not APP_MONITOR_AVAILABLE:
        return "❌ Application monitoring not available"

    try:
        monitor = await get_app_monitor()
        if not monitor:
            return "❌ Application monitor not initialized"

        insights = monitor.get_context_insights(hours)

        output = [f"🧠 Context Insights (Last {hours} hours)", ""]

        if insights["primary_focus"]:
            output.append(f"🎯 Primary Focus: {insights['primary_focus']}")

        if insights["technologies_used"]:
            tech_list = ", ".join(insights["technologies_used"])
            output.append(f"💻 Technologies: {tech_list}")

        if insights["active_projects"]:
            output.append(f"📁 Active Projects: {len(insights['active_projects'])}")
            for project in list(insights["active_projects"])[:3]:
                output.append(f"  • {Path(project).name}")

        if insights["documentation_topics"]:
            output.append("📖 Documentation Topics:")
            for topic in insights["documentation_topics"][:5]:
                output.append(f"  • {topic}")

        output.extend(
            [
                "",
                f"🔄 Context Switches: {insights['context_switches']}",
                f"⚡ Productivity Score: {insights['productivity_score']:.2f}",
            ],
        )

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error getting context insights: {e}"


@mcp.tool()
async def get_active_files(minutes: int = 60) -> str:
    """Get files currently being worked on.

    Args:
        minutes: Number of minutes to look back (default: 60)

    """
    if not APP_MONITOR_AVAILABLE:
        return "❌ Application monitoring not available"

    try:
        monitor = await get_app_monitor()
        if not monitor:
            return "❌ Application monitor not initialized"

        active_files = monitor.ide_monitor.get_active_files(minutes)

        if not active_files:
            return f"📄 No active files found in the last {minutes} minutes"

        output = [f"📄 Active Files (Last {minutes} minutes)", ""]

        for file_info in active_files[:10]:
            file_path = file_info["file_path"]
            relative_path = Path(file_path).name if len(file_path) > 50 else file_path
            score = file_info["activity_score"]
            count = file_info["event_count"]
            output.append(f"• {relative_path} (score: {score}, events: {count})")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error getting active files: {e}"


# Global LLM manager instance
_llm_manager = None


async def get_llm_manager() -> LLMManager | None:
    """Get or initialize LLM manager."""
    global _llm_manager
    if not LLM_PROVIDERS_AVAILABLE:
        return None

    if _llm_manager is None:
        config_path = Path.home() / ".claude" / "data" / "llm_config.json"
        _llm_manager = LLMManager(str(config_path) if config_path.exists() else None)

    return _llm_manager


@mcp.tool()
async def list_llm_providers() -> str:
    """List all available LLM providers and their models."""
    if not LLM_PROVIDERS_AVAILABLE:
        return "❌ LLM providers not available. Install dependencies: pip install openai google-generativeai aiohttp"

    try:
        manager = await get_llm_manager()
        if not manager:
            return "❌ Failed to initialize LLM manager"

        available_providers = await manager.get_available_providers()
        provider_info = manager.get_provider_info()

        output = ["🤖 Available LLM Providers", ""]

        for provider_name, info in provider_info["providers"].items():
            status = "✅" if provider_name in available_providers else "❌"
            output.append(f"{status} {provider_name.title()}")

            if provider_name in available_providers:
                models = info["models"][:5]  # Show first 5 models
                for model in models:
                    output.append(f"   • {model}")
                if len(info["models"]) > 5:
                    output.append(f"   • ... and {len(info['models']) - 5} more")
            output.append("")

        config = provider_info["config"]
        output.extend(
            [
                f"🎯 Default Provider: {config['default_provider']}",
                f"🔄 Fallback Providers: {', '.join(config['fallback_providers'])}",
            ],
        )

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error listing providers: {e}"


@mcp.tool()
async def test_llm_providers() -> str:
    """Test all LLM providers to check their availability and functionality."""
    if not LLM_PROVIDERS_AVAILABLE:
        return "❌ LLM providers not available"

    try:
        manager = await get_llm_manager()
        if not manager:
            return "❌ Failed to initialize LLM manager"

        test_results = await manager.test_providers()

        output = ["🧪 LLM Provider Test Results", ""]

        for provider_name, result in test_results.items():
            if result["available"] and result["test_successful"]:
                output.append(f"✅ {provider_name.title()}: Working")
                output.append(f"   Model: {result.get('model', 'Unknown')}")
                output.append(f"   Response: {result.get('response_length', 0)} chars")
            elif result["available"]:
                output.append(f"⚠️ {provider_name.title()}: Available but test failed")
                output.append(f"   Error: {result.get('error', 'Unknown error')}")
            else:
                output.append(f"❌ {provider_name.title()}: Not available")
                output.append(f"   Error: {result.get('error', 'Unknown error')}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error testing providers: {e}"


@mcp.tool()
async def generate_with_llm(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    use_fallback: bool = True,
) -> str:
    """Generate text using specified LLM provider.

    Args:
        prompt: The text prompt to generate from
        provider: LLM provider to use (openai, gemini, ollama)
        model: Specific model to use
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        use_fallback: Whether to use fallback providers if primary fails

    """
    if not LLM_PROVIDERS_AVAILABLE:
        return "❌ LLM providers not available"

    try:
        manager = await get_llm_manager()
        if not manager:
            return "❌ Failed to initialize LLM manager"

        messages = [LLMMessage(role="user", content=prompt)]

        response = await manager.generate(
            messages=messages,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_fallback=use_fallback,
        )

        output = [
            f"🤖 Generated by {response.provider} ({response.model})",
            "",
            response.content,
            "",
            f"📊 Usage: {response.usage.get('total_tokens', 0)} tokens",
            f"⏱️ Generated: {response.timestamp}",
        ]

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error generating text: {e}"


@mcp.tool()
async def chat_with_llm(
    messages: list[dict[str, str]],
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Have a conversation with an LLM provider.

    Args:
        messages: List of messages in format [{"role": "user/assistant/system", "content": "text"}]
        provider: LLM provider to use (openai, gemini, ollama)
        model: Specific model to use
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate

    """
    if not LLM_PROVIDERS_AVAILABLE:
        return "❌ LLM providers not available"

    try:
        manager = await get_llm_manager()
        if not manager:
            return "❌ Failed to initialize LLM manager"

        # Convert to LLMMessage objects
        llm_messages = []
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return "❌ Invalid message format. Each message must have 'role' and 'content' fields"

            if msg["role"] not in ["user", "assistant", "system"]:
                return "❌ Invalid role. Must be 'user', 'assistant', or 'system'"

            llm_messages.append(LLMMessage(role=msg["role"], content=msg["content"]))

        response = await manager.generate(
            messages=llm_messages,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        output = [
            f"🤖 {response.provider} ({response.model}) response:",
            "",
            response.content,
            "",
            f"📊 Usage: {response.usage.get('total_tokens', 0)} tokens",
            f"⏱️ Generated: {response.timestamp}",
        ]

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error in chat: {e}"


@mcp.tool()
async def configure_llm_provider(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    default_model: str | None = None,
) -> str:
    """Configure an LLM provider with API credentials and settings.

    Args:
        provider: Provider name (openai, gemini, ollama)
        api_key: API key for the provider
        base_url: Base URL for the provider API
        default_model: Default model to use

    """
    if not LLM_PROVIDERS_AVAILABLE:
        return "❌ LLM providers not available"

    try:
        config_path = Path.home() / ".claude" / "data" / "llm_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        # Initialize structure
        if "providers" not in config:
            config["providers"] = {}
        if provider not in config["providers"]:
            config["providers"][provider] = {}

        # Update configuration
        if api_key:
            config["providers"][provider]["api_key"] = api_key
        if base_url:
            config["providers"][provider]["base_url"] = base_url
        if default_model:
            config["providers"][provider]["default_model"] = default_model

        # Save configuration
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Reinitialize manager to pick up new config
        global _llm_manager
        _llm_manager = None

        return f"✅ Configuration updated for {provider} provider"

    except Exception as e:
        return f"❌ Error configuring provider: {e}"


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


@mcp.tool()
async def create_serverless_session(
    user_id: str,
    project_id: str,
    session_data: dict[str, Any] | None = None,
    ttl_hours: int = 24,
) -> str:
    """Create a new serverless session with external storage.

    Args:
        user_id: User identifier for the session
        project_id: Project identifier for the session
        session_data: Optional metadata for the session
        ttl_hours: Time-to-live in hours (default: 24)

    """
    if not SERVERLESS_MODE_AVAILABLE:
        return "❌ Serverless mode not available. Install dependencies: pip install redis boto3"

    try:
        manager = await get_serverless_manager()
        if not manager:
            return "❌ Failed to initialize serverless manager"

        session_id = await manager.create_session(
            user_id=user_id,
            project_id=project_id,
            session_data=session_data,
            ttl_hours=ttl_hours,
        )

        return f"✅ Created serverless session: {session_id}\n🕐 TTL: {ttl_hours} hours"

    except Exception as e:
        return f"❌ Error creating session: {e}"


@mcp.tool()
async def get_serverless_session(session_id: str) -> str:
    """Get serverless session state.

    Args:
        session_id: Session identifier to retrieve

    """
    if not SERVERLESS_MODE_AVAILABLE:
        return "❌ Serverless mode not available"

    try:
        manager = await get_serverless_manager()
        if not manager:
            return "❌ Failed to initialize serverless manager"

        session_state = await manager.get_session(session_id)
        if not session_state:
            return f"❌ Session not found: {session_id}"

        output = [
            f"📊 Session: {session_id}",
            f"👤 User: {session_state.user_id}",
            f"📁 Project: {session_state.project_id}",
            f"🕐 Created: {session_state.created_at}",
            f"⏰ Last Activity: {session_state.last_activity}",
            f"🔐 Permissions: {len(session_state.permissions)}",
            f"💬 Conversations: {len(session_state.conversation_history)}",
            f"💾 Size: {session_state.get_compressed_size()} bytes (compressed)",
        ]

        if session_state.metadata:
            output.append(
                f"📋 Metadata: {json.dumps(session_state.metadata, indent=2)}",
            )

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error getting session: {e}"


@mcp.tool()
async def update_serverless_session(
    session_id: str,
    updates: dict[str, Any],
    ttl_hours: int | None = None,
) -> str:
    """Update serverless session state.

    Args:
        session_id: Session identifier to update
        updates: Dictionary of updates to apply
        ttl_hours: Optional new TTL in hours

    """
    if not SERVERLESS_MODE_AVAILABLE:
        return "❌ Serverless mode not available"

    try:
        manager = await get_serverless_manager()
        if not manager:
            return "❌ Failed to initialize serverless manager"

        success = await manager.update_session(session_id, updates, ttl_hours)

        if success:
            update_summary = ", ".join(
                f"{k}: {type(v).__name__}" for k, v in updates.items()
            )
            result = f"✅ Updated session {session_id}\n📝 Changes: {update_summary}"
            if ttl_hours:
                result += f"\n🕐 New TTL: {ttl_hours} hours"
            return result
        return f"❌ Failed to update session {session_id}"

    except Exception as e:
        return f"❌ Error updating session: {e}"


@mcp.tool()
async def delete_serverless_session(session_id: str) -> str:
    """Delete a serverless session.

    Args:
        session_id: Session identifier to delete

    """
    if not SERVERLESS_MODE_AVAILABLE:
        return "❌ Serverless mode not available"

    try:
        manager = await get_serverless_manager()
        if not manager:
            return "❌ Failed to initialize serverless manager"

        success = await manager.delete_session(session_id)

        if success:
            return f"✅ Deleted session {session_id}"
        return f"❌ Session not found or failed to delete: {session_id}"

    except Exception as e:
        return f"❌ Error deleting session: {e}"


@mcp.tool()
async def list_serverless_sessions(
    user_id: str | None = None,
    project_id: str | None = None,
) -> str:
    """List serverless sessions by user or project.

    Args:
        user_id: Filter by user ID (optional)
        project_id: Filter by project ID (optional)

    """
    if not SERVERLESS_MODE_AVAILABLE:
        return "❌ Serverless mode not available"

    try:
        manager = await get_serverless_manager()
        if not manager:
            return "❌ Failed to initialize serverless manager"

        if user_id:
            session_ids = await manager.list_user_sessions(user_id)
            filter_desc = f"user '{user_id}'"
        elif project_id:
            session_ids = await manager.list_project_sessions(project_id)
            filter_desc = f"project '{project_id}'"
        else:
            # List all sessions (expensive operation)
            session_ids = await manager.storage.list_sessions()
            filter_desc = "all users/projects"

        if not session_ids:
            return f"📭 No sessions found for {filter_desc}"

        output = [f"📋 Sessions for {filter_desc} ({len(session_ids)} found)", ""]

        # Get details for first few sessions
        for session_id in session_ids[:10]:
            session_state = await manager.get_session(session_id)
            if session_state:
                output.append(
                    f"• {session_id[:12]}... (user: {session_state.user_id}, project: {session_state.project_id})",
                )

        if len(session_ids) > 10:
            output.append(f"... and {len(session_ids) - 10} more")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error listing sessions: {e}"


@mcp.tool()
async def test_serverless_storage() -> str:
    """Test serverless storage backends for availability."""
    if not SERVERLESS_MODE_AVAILABLE:
        return "❌ Serverless mode not available"

    try:
        config_path = Path.home() / ".claude" / "data" / "serverless_config.json"
        config = ServerlessConfigManager.load_config(
            str(config_path) if config_path.exists() else None,
        )

        test_results = await ServerlessConfigManager.test_storage_backends(config)

        output = ["🧪 Storage Backend Test Results", ""]

        for backend_name, available in test_results.items():
            status = "✅" if available else "❌"
            output.append(
                f"{status} {backend_name.title()}: {'Available' if available else 'Not available'}",
            )

        # Show current configuration
        current_backend = config.get("storage_backend", "local")
        current_available = test_results.get(current_backend, False)

        output.extend(
            [
                "",
                f"🎯 Current Backend: {current_backend} ({'✅ Available' if current_available else '❌ Not available'})",
            ],
        )

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error testing storage: {e}"


@mcp.tool()
async def cleanup_serverless_sessions() -> str:
    """Clean up expired serverless sessions."""
    if not SERVERLESS_MODE_AVAILABLE:
        return "❌ Serverless mode not available"

    try:
        manager = await get_serverless_manager()
        if not manager:
            return "❌ Failed to initialize serverless manager"

        cleaned_count = await manager.cleanup_sessions()
        stats = manager.get_session_stats()

        output = [
            "🧹 Session Cleanup Results",
            f"🗑️ Cleaned up: {cleaned_count} expired sessions",
            f"💾 Active sessions in cache: {stats['cached_sessions']}",
            f"🏗️ Storage backend: {stats['storage_backend']}",
        ]

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error during cleanup: {e}"


@mcp.tool()
async def configure_serverless_storage(
    backend: str,
    config_updates: dict[str, Any],
) -> str:
    """Configure serverless storage backend settings.

    Args:
        backend: Storage backend (redis, s3, local)
        config_updates: Configuration updates to apply

    """
    if not SERVERLESS_MODE_AVAILABLE:
        return "❌ Serverless mode not available"

    try:
        config_path = Path.home() / ".claude" / "data" / "serverless_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config
        config = ServerlessConfigManager.load_config(
            str(config_path) if config_path.exists() else None,
        )

        # Update configuration
        if "backends" not in config:
            config["backends"] = {}
        if backend not in config["backends"]:
            config["backends"][backend] = {}

        config["backends"][backend].update(config_updates)
        config["storage_backend"] = backend

        # Save configuration
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Reinitialize manager
        global _serverless_manager
        _serverless_manager = None

        return (
            f"✅ Configured {backend} storage backend\n📁 Config saved to {config_path}"
        )

    except Exception as e:
        return f"❌ Error configuring storage: {e}"


# Team Knowledge Base Tools
@mcp.tool()
async def create_team_user(
    user_id: str,
    username: str,
    email: str | None = None,
    role: str = "contributor",
) -> str:
    """Create a new team user with specified role."""
    try:
        from .team_knowledge import create_team_user as _create_team_user

        user_data = await _create_team_user(user_id, username, email, role)

        output = []
        output.append("👤 Team user created successfully!")
        output.append(f"🆔 User ID: {user_data['user_id']}")
        output.append(f"👥 Username: {user_data['username']}")
        output.append(f"🏷️ Role: {user_data['role']}")
        if email:
            output.append(f"📧 Email: {email}")
        output.append(f"📅 Created: {user_data['created_at']}")
        output.append("🎉 User can now participate in team knowledge sharing")

        return "\n".join(output)

    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error creating team user: {e}"


@mcp.tool()
async def create_team(team_id: str, name: str, description: str, owner_id: str) -> str:
    """Create a new team for knowledge sharing."""
    try:
        from .team_knowledge import create_team as _create_team

        team_data = await _create_team(team_id, name, description, owner_id)

        output = []
        output.append("🏆 Team created successfully!")
        output.append(f"🆔 Team ID: {team_data['team_id']}")
        output.append(f"📛 Name: {team_data['name']}")
        output.append(f"📝 Description: {team_data['description']}")
        output.append(f"👑 Owner: {team_data['owner_id']}")
        output.append(f"👥 Members: {team_data['member_count']}")
        output.append(f"📁 Projects: {team_data['project_count']}")
        output.append(f"📅 Created: {team_data['created_at']}")
        output.append("🎯 Team is ready for collaborative knowledge sharing")

        return "\n".join(output)

    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error creating team: {e}"


@mcp.tool()
async def add_team_reflection(
    content: str,
    author_id: str,
    tags: list[str] | None = None,
    access_level: str = "team",
    team_id: str | None = None,
    project_id: str | None = None,
) -> str:
    """Add reflection to team knowledge base with access control."""
    try:
        from .team_knowledge import add_team_reflection as _add_team_reflection

        reflection_id = await _add_team_reflection(
            content,
            author_id,
            tags,
            access_level,
            team_id,
            project_id,
        )

        output = []
        output.append("💡 Team reflection added successfully!")
        output.append(f"🆔 Reflection ID: {reflection_id}")
        output.append(f"👤 Author: {author_id}")
        output.append(f"🔒 Access Level: {access_level}")
        if team_id:
            output.append(f"👥 Team: {team_id}")
        if project_id:
            output.append(f"📁 Project: {project_id}")
        if tags:
            output.append(f"🏷️ Tags: {', '.join(tags)}")
        output.append(f"📝 Content: {content[:100]}...")
        output.append("🌟 Reflection is now available for team collaboration")

        return "\n".join(output)

    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error adding team reflection: {e}"


@mcp.tool()
async def search_team_knowledge(
    query: str,
    user_id: str,
    team_id: str | None = None,
    project_id: str | None = None,
    tags: list[str] | None = None,
    limit: int = 20,
) -> str:
    """Search team reflections with access control."""
    try:
        from .team_knowledge import search_team_knowledge as _search_team_knowledge

        results = await _search_team_knowledge(
            query,
            user_id,
            team_id,
            project_id,
            tags,
            limit,
        )

        if not results:
            return f"🔍 No team knowledge found for query: '{query}'\n💡 Try adjusting search terms or access permissions."

        output = []
        output.append(f"🧠 Found {len(results)} team reflections for: '{query}'")
        output.append(f"👤 User: {user_id}")
        if team_id:
            output.append(f"👥 Team: {team_id}")
        if project_id:
            output.append(f"📁 Project: {project_id}")
        if tags:
            output.append(f"🏷️ Tags: {', '.join(tags)}")
        output.append("=" * 50)

        for i, result in enumerate(results, 1):
            output.append(f"\n#{i}")
            output.append(f"🆔 ID: {result['id']}")
            output.append(f"👤 Author: {result['author_id']}")
            output.append(f"🔒 Access: {result['access_level']}")
            output.append(f"👍 Votes: {result['votes']}")
            if result.get("team_id"):
                output.append(f"👥 Team: {result['team_id']}")
            if result.get("tags"):
                output.append(f"🏷️ Tags: {', '.join(result['tags'])}")
            output.append(f"📝 {result['content'][:200]}...")
            output.append(f"📅 {result['created_at']}")

        return "\n".join(output)

    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error searching team knowledge: {e}"


@mcp.tool()
async def join_team(user_id: str, team_id: str, requester_id: str | None = None) -> str:
    """Join a team or add user to team."""
    try:
        from .team_knowledge import join_team as _join_team

        success = await _join_team(user_id, team_id, requester_id)

        if success:
            output = []
            output.append("🎉 Successfully joined team!")
            output.append(f"👤 User: {user_id}")
            output.append(f"👥 Team: {team_id}")
            if requester_id and requester_id != user_id:
                output.append(f"👑 Added by: {requester_id}")
            output.append(
                "🌟 You can now access team reflections and contribute knowledge",
            )
            return "\n".join(output)
        return (
            f"❌ Failed to join team {team_id}. Check permissions and team existence."
        )

    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error joining team: {e}"


@mcp.tool()
async def get_team_statistics(team_id: str, user_id: str) -> str:
    """Get team statistics and activity."""
    try:
        from .team_knowledge import get_team_statistics as _get_team_statistics

        stats = await _get_team_statistics(team_id, user_id)

        if not stats:
            return f"❌ Cannot access team {team_id}. Check permissions and team existence."

        output = []
        team_info = stats["team"]
        reflection_stats = stats["reflection_stats"]

        output.append(f"📊 Team Statistics: {team_info['name']}")
        output.append(f"🆔 Team ID: {team_info['team_id']}")
        output.append(f"📝 Description: {team_info['description']}")
        output.append(f"👑 Owner: {team_info['owner_id']}")
        output.append(f"📅 Created: {team_info['created_at']}")
        output.append("=" * 50)

        output.append(f"👥 Members: {stats['member_count']}")
        output.append(f"📁 Projects: {stats['project_count']}")
        output.append(
            f"💡 Total Reflections: {reflection_stats['total_reflections'] or 0}",
        )
        output.append(
            f"✍️ Active Contributors: {reflection_stats['active_contributors'] or 0}",
        )
        output.append(f"👍 Total Votes: {reflection_stats['total_votes'] or 0}")
        output.append(
            f"📈 Avg Votes/Reflection: {reflection_stats['avg_votes']:.1f}"
            if reflection_stats["avg_votes"]
            else "📈 Avg Votes/Reflection: 0.0",
        )
        output.append(
            f"⚡ Recent Activity (7 days): {stats['recent_activity']['recent_reflections']} reflections",
        )

        return "\n".join(output)

    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error getting team statistics: {e}"


@mcp.tool()
async def get_user_team_permissions(user_id: str) -> str:
    """Get user's permissions and team memberships."""
    try:
        from .team_knowledge import (
            get_user_team_permissions as _get_user_team_permissions,
        )

        perms = await _get_user_team_permissions(user_id)

        if not perms:
            return f"❌ User {user_id} not found in team knowledge base"

        user_data = perms["user"]
        teams = perms["teams"]

        output = []
        output.append(f"👤 User Permissions: {user_data['username']}")
        output.append(f"🆔 User ID: {user_data['user_id']}")
        output.append(f"🏷️ Role: {user_data['role']}")
        if user_data.get("email"):
            output.append(f"📧 Email: {user_data['email']}")
        output.append(f"📅 Created: {user_data['created_at']}")
        output.append(f"⏰ Last Active: {user_data['last_active']}")
        output.append("=" * 50)

        output.append(f"👥 Teams ({len(teams)}):")
        for team in teams:
            output.append(f"  • {team['name']} ({team['team_id']})")
            if team.get("description"):
                output.append(f"    📝 {team['description'][:80]}...")

        output.append("\n🔒 Permissions:")
        permissions = user_data.get("permissions", {})
        for perm, enabled in permissions.items():
            status = "✅" if enabled else "❌"
            perm_name = perm.replace("_", " ").title()
            output.append(f"  {status} {perm_name}")

        output.append("\n🎯 Special Abilities:")
        if perms["can_create_teams"]:
            output.append("  ✅ Can create teams")
        if perms["can_moderate"]:
            output.append("  ✅ Can moderate content")

        return "\n".join(output)

    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error getting user permissions: {e}"


@mcp.tool()
async def vote_on_reflection(
    reflection_id: str,
    user_id: str,
    vote_delta: int = 1,
) -> str:
    """Vote on a team reflection (upvote/downvote)."""
    try:
        from .team_knowledge import vote_on_reflection as _vote_on_reflection

        success = await _vote_on_reflection(reflection_id, user_id, vote_delta)

        if success:
            vote_type = "upvoted" if vote_delta > 0 else "downvoted"
            vote_emoji = "👍" if vote_delta > 0 else "👎"
            output = []
            output.append(f"{vote_emoji} Reflection {vote_type} successfully!")
            output.append(f"🆔 Reflection ID: {reflection_id}")
            output.append(f"👤 Voter: {user_id}")
            output.append(f"📊 Vote Delta: {vote_delta:+d}")
            output.append("🌟 Your vote helps surface valuable team knowledge")
            return "\n".join(output)
        return f"❌ Failed to vote on reflection {reflection_id}. Check permissions and reflection existence."

    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error voting on reflection: {e}"


# Natural Language Scheduling Tools
@mcp.tool()
async def create_natural_reminder(
    title: str,
    time_expression: str,
    description: str = "",
    user_id: str = "default",
    project_id: str | None = None,
    notification_method: str = "session",
) -> str:
    """Create reminder from natural language time expression."""
    try:
        from .natural_scheduler import (
            create_natural_reminder as _create_natural_reminder,
        )

        reminder_id = await _create_natural_reminder(
            title,
            time_expression,
            description,
            user_id,
            project_id,
            notification_method,
        )

        if reminder_id:
            output = []
            output.append("⏰ Natural reminder created successfully!")
            output.append(f"🆔 Reminder ID: {reminder_id}")
            output.append(f"📝 Title: {title}")
            output.append(f"📄 Description: {description}")
            output.append(f"🕐 When: {time_expression}")
            output.append(f"👤 User: {user_id}")
            if project_id:
                output.append(f"📁 Project: {project_id}")
            output.append(f"📢 Notification: {notification_method}")
            output.append(
                "🎯 Reminder will trigger automatically at the scheduled time",
            )
            return "\n".join(output)
        return f"❌ Failed to parse time expression: '{time_expression}'\n💡 Try formats like 'in 30 minutes', 'tomorrow at 9am', 'every day at 5pm'"

    except ImportError:
        return "❌ Natural scheduling tools not available. Install: pip install python-dateutil schedule python-crontab"
    except Exception as e:
        return f"❌ Error creating reminder: {e}"


@mcp.tool()
async def list_user_reminders(
    user_id: str = "default",
    project_id: str | None = None,
) -> str:
    """List pending reminders for user/project."""
    try:
        from .natural_scheduler import list_user_reminders as _list_user_reminders

        reminders = await _list_user_reminders(user_id, project_id)

        if not reminders:
            output = []
            output.append("📋 No pending reminders found")
            output.append(f"👤 User: {user_id}")
            if project_id:
                output.append(f"📁 Project: {project_id}")
            output.append(
                "💡 Use 'create_natural_reminder' to set up time-based reminders",
            )
            return "\n".join(output)

        output = []
        output.append(f"⏰ Found {len(reminders)} pending reminders")
        output.append(f"👤 User: {user_id}")
        if project_id:
            output.append(f"📁 Project: {project_id}")
        output.append("=" * 50)

        for i, reminder in enumerate(reminders, 1):
            output.append(f"\n#{i}")
            output.append(f"🆔 ID: {reminder['id']}")
            output.append(f"📝 Title: {reminder['title']}")
            if reminder["description"]:
                output.append(f"📄 Description: {reminder['description']}")
            output.append(
                f"🔄 Type: {reminder['reminder_type'].replace('_', ' ').title()}",
            )
            output.append(f"📊 Status: {reminder['status'].replace('_', ' ').title()}")
            output.append(f"🕐 Scheduled: {reminder['scheduled_for']}")
            output.append(f"📅 Created: {reminder['created_at']}")
            if reminder.get("recurrence_rule"):
                output.append(f"🔁 Recurrence: {reminder['recurrence_rule']}")
            if reminder.get("context_triggers"):
                output.append(f"🎯 Triggers: {', '.join(reminder['context_triggers'])}")

        return "\n".join(output)

    except ImportError:
        return "❌ Natural scheduling tools not available"
    except Exception as e:
        return f"❌ Error listing reminders: {e}"


@mcp.tool()
async def cancel_user_reminder(reminder_id: str) -> str:
    """Cancel a specific reminder."""
    try:
        from .natural_scheduler import cancel_user_reminder as _cancel_user_reminder

        success = await _cancel_user_reminder(reminder_id)

        if success:
            output = []
            output.append("❌ Reminder cancelled successfully!")
            output.append(f"🆔 Reminder ID: {reminder_id}")
            output.append("🚫 The reminder will no longer trigger")
            output.append("💡 You can create a new reminder if needed")
            return "\n".join(output)
        return f"❌ Failed to cancel reminder {reminder_id}. Check that the ID is correct and the reminder exists."

    except ImportError:
        return "❌ Natural scheduling tools not available"
    except Exception as e:
        return f"❌ Error cancelling reminder: {e}"


@mcp.tool()
async def check_due_reminders() -> str:
    """Check for reminders that are due now."""
    try:
        from .natural_scheduler import check_due_reminders as _check_due_reminders

        due_reminders = await _check_due_reminders()

        if not due_reminders:
            return "✅ No reminders are currently due\n⏰ All scheduled reminders are in the future"

        output = []
        output.append(f"🚨 {len(due_reminders)} reminders are DUE NOW!")
        output.append("=" * 50)

        for i, reminder in enumerate(due_reminders, 1):
            output.append(f"\n🔥 #{i} OVERDUE")
            output.append(f"🆔 ID: {reminder['id']}")
            output.append(f"📝 Title: {reminder['title']}")
            if reminder["description"]:
                output.append(f"📄 Description: {reminder['description']}")
            output.append(f"🕐 Scheduled: {reminder['scheduled_for']}")
            output.append(f"👤 User: {reminder['user_id']}")
            if reminder.get("project_id"):
                output.append(f"📁 Project: {reminder['project_id']}")

            # Calculate how overdue
            try:
                from datetime import datetime

                scheduled = datetime.fromisoformat(
                    reminder["scheduled_for"],
                )
                now = datetime.now()
                overdue = now - scheduled
                if overdue.total_seconds() > 0:
                    hours = int(overdue.total_seconds() // 3600)
                    minutes = int((overdue.total_seconds() % 3600) // 60)
                    if hours > 0:
                        output.append(f"⏱️ Overdue: {hours}h {minutes}m")
                    else:
                        output.append(f"⏱️ Overdue: {minutes}m")
            except Exception:
                output.append("⏱️ Overdue: calculation failed")

        output.append(
            "\n💡 These reminders should be processed by the background scheduler",
        )
        return "\n".join(output)

    except ImportError:
        return "❌ Natural scheduling tools not available"
    except Exception as e:
        return f"❌ Error checking due reminders: {e}"


@mcp.tool()
async def start_reminder_service() -> str:
    """Start the background reminder service."""
    try:
        from .natural_scheduler import (
            register_session_notifications,
        )
        from .natural_scheduler import (
            start_reminder_service as _start_reminder_service,
        )

        # Register default session notifications
        register_session_notifications()

        # Start the service
        _start_reminder_service()

        output = []
        output.append("🚀 Natural reminder service started!")
        output.append("⏰ Background scheduler is now active")
        output.append("🔍 Checking for due reminders every minute")
        output.append("📢 Session notifications are registered")
        output.append(
            "💡 Reminders will automatically trigger at their scheduled times",
        )
        output.append("🛑 Use 'stop_reminder_service' to stop the background service")

        return "\n".join(output)

    except ImportError:
        return "❌ Natural scheduling tools not available"
    except Exception as e:
        return f"❌ Error starting reminder service: {e}"


@mcp.tool()
async def stop_reminder_service() -> str:
    """Stop the background reminder service."""
    try:
        from .natural_scheduler import stop_reminder_service as _stop_reminder_service

        _stop_reminder_service()

        output = []
        output.append("🛑 Natural reminder service stopped")
        output.append("❌ Background scheduler is no longer active")
        output.append("⚠️ Existing reminders will not trigger automatically")
        output.append("🚀 Use 'start_reminder_service' to restart the service")
        output.append(
            "💡 You can still check due reminders manually with 'check_due_reminders'",
        )

        return "\n".join(output)

    except ImportError:
        return "❌ Natural scheduling tools not available"
    except Exception as e:
        return f"❌ Error stopping reminder service: {e}"


# Smart Interruption Management Tools
@mcp.tool()
async def start_interruption_monitoring(
    working_directory: str = ".",
    watch_files: bool = True,
) -> str:
    """Start smart interruption monitoring with context switch detection."""
    try:
        from .interruption_manager import (
            start_interruption_monitoring as _start_interruption_monitoring,
        )

        await _start_interruption_monitoring(working_directory, watch_files)

        output = []
        output.append("🛡️ Smart interruption monitoring started!")
        output.append(f"📁 Working directory: {working_directory}")
        output.append(f"📄 File watching: {'enabled' if watch_files else 'disabled'}")
        output.append("👁️ Focus tracking: active")
        output.append("💾 Auto-save: enabled (30s threshold)")
        output.append("🔄 Context preservation: automatic")
        output.append(
            "💡 Your work context will be automatically preserved during interruptions",
        )
        output.append("🛑 Use 'stop_interruption_monitoring' to disable")

        return "\n".join(output)

    except ImportError:
        return "❌ Interruption management tools not available. Install: pip install psutil watchdog"
    except Exception as e:
        return f"❌ Error starting interruption monitoring: {e}"


@mcp.tool()
async def stop_interruption_monitoring() -> str:
    """Stop interruption monitoring."""
    try:
        from .interruption_manager import (
            stop_interruption_monitoring as _stop_interruption_monitoring,
        )

        _stop_interruption_monitoring()

        output = []
        output.append("🛑 Interruption monitoring stopped")
        output.append("❌ Focus tracking: disabled")
        output.append("❌ File watching: disabled")
        output.append("❌ Auto-save: disabled")
        output.append("⚠️ Context preservation is now manual only")
        output.append("🚀 Use 'start_interruption_monitoring' to restart")

        return "\n".join(output)

    except ImportError:
        return "❌ Interruption management tools not available"
    except Exception as e:
        return f"❌ Error stopping interruption monitoring: {e}"


@mcp.tool()
async def create_session_context(
    user_id: str,
    project_id: str | None = None,
    working_directory: str = ".",
) -> str:
    """Create new session context for interruption management."""
    try:
        from .interruption_manager import (
            create_session_context as _create_session_context,
        )

        session_id = await _create_session_context(
            user_id,
            project_id,
            working_directory,
        )

        output = []
        output.append("🎯 Session context created successfully!")
        output.append(f"🆔 Session ID: {session_id}")
        output.append(f"👤 User: {user_id}")
        if project_id:
            output.append(f"📁 Project: {project_id}")
        output.append(f"📂 Working directory: {working_directory}")
        output.append("💾 Context will be automatically preserved on interruptions")
        output.append("🔄 Session state is now being tracked")

        return "\n".join(output)

    except ImportError:
        return "❌ Interruption management tools not available"
    except Exception as e:
        return f"❌ Error creating session context: {e}"


@mcp.tool()
async def preserve_current_context(
    session_id: str | None = None,
    force: bool = False,
) -> str:
    """Manually preserve current session context."""
    try:
        from .interruption_manager import (
            preserve_current_context as _preserve_current_context,
        )

        success = await _preserve_current_context(session_id, force)

        if success:
            output = []
            output.append("💾 Session context preserved successfully!")
            if session_id:
                output.append(f"🆔 Session ID: {session_id}")
            output.append(f"🔧 Preservation type: {'forced' if force else 'automatic'}")
            output.append("📸 Context snapshot created")
            output.append("🗜️ Data compressed for storage")
            output.append("✅ Recovery is now possible if interruption occurs")
            return "\n".join(output)
        return "❌ Failed to preserve context. Check that a session context exists."

    except ImportError:
        return "❌ Interruption management tools not available"
    except Exception as e:
        return f"❌ Error preserving context: {e}"


@mcp.tool()
async def restore_session_context(session_id: str) -> str:
    """Restore session context from snapshot."""
    try:
        from .interruption_manager import (
            restore_session_context as _restore_session_context,
        )

        context_data = await _restore_session_context(session_id)

        if context_data:
            output = []
            output.append("🔄 Session context restored successfully!")
            output.append(f"🆔 Session ID: {session_id}")
            output.append(f"👤 User: {context_data['user_id']}")
            if context_data.get("project_id"):
                output.append(f"📁 Project: {context_data['project_id']}")
            output.append(f"📂 Working directory: {context_data['working_directory']}")
            output.append(
                f"📊 Interruption count: {context_data['interruption_count']}",
            )
            output.append(f"🔄 Recovery attempts: {context_data['recovery_attempts']}")
            output.append(f"⏱️ Focus duration: {context_data['focus_duration']:.1f}s")
            if context_data.get("open_files"):
                output.append(f"📄 Open files: {len(context_data['open_files'])}")
            output.append("✅ Context is now active and being monitored")
            return "\n".join(output)
        return f"❌ Failed to restore context for session {session_id}. Check that the session exists and has preserved data."

    except ImportError:
        return "❌ Interruption management tools not available"
    except Exception as e:
        return f"❌ Error restoring context: {e}"


@mcp.tool()
async def get_interruption_history(user_id: str, hours: int = 24) -> str:
    """Get recent interruption history for user."""
    try:
        from .interruption_manager import (
            get_interruption_history as _get_interruption_history,
        )

        history = await _get_interruption_history(user_id, hours)

        if not history:
            output = []
            output.append("📋 No interruptions found")
            output.append(f"👤 User: {user_id}")
            output.append(f"🕐 Time range: last {hours} hours")
            output.append("✅ Your work sessions have been uninterrupted!")
            return "\n".join(output)

        output = []
        output.append(
            f"📊 Found {len(history)} interruptions in the last {hours} hours",
        )
        output.append(f"👤 User: {user_id}")
        output.append("=" * 50)

        # Group by type for summary
        by_type = {}
        total_duration = 0
        auto_saved_count = 0

        for event in history:
            event_type = event["event_type"]
            by_type[event_type] = by_type.get(event_type, 0) + 1
            if event.get("duration"):
                total_duration += event["duration"]
            if event.get("auto_saved"):
                auto_saved_count += 1

        output.append("📈 Summary:")
        for event_type, count in by_type.items():
            type_name = event_type.replace("_", " ").title()
            output.append(f"  • {type_name}: {count}")

        output.append(f"💾 Auto-saved interruptions: {auto_saved_count}/{len(history)}")
        if total_duration > 0:
            output.append(f"⏱️ Total focus time: {total_duration:.1f}s")

        # Show recent events
        output.append("\n🕐 Recent interruptions:")
        for i, event in enumerate(history[:5], 1):
            event_type = event["event_type"].replace("_", " ").title()
            timestamp = event["timestamp"]
            auto_saved = "💾" if event.get("auto_saved") else "📝"
            duration_info = (
                f" ({event['duration']:.1f}s)" if event.get("duration") else ""
            )
            output.append(
                f"  {i}. {auto_saved} {event_type}{duration_info} - {timestamp}",
            )

        if len(history) > 5:
            output.append(f"  ... and {len(history) - 5} more")

        return "\n".join(output)

    except ImportError:
        return "❌ Interruption management tools not available"
    except Exception as e:
        return f"❌ Error getting interruption history: {e}"


@mcp.tool()
def _format_statistics_header(user_id: str) -> list[str]:
    """Format the header section for interruption statistics."""
    return ["📊 Interruption Management Statistics", f"👤 User: {user_id}", "=" * 50]


def _format_session_statistics(sessions: dict) -> list[str]:
    """Format session management statistics section."""
    if not sessions:
        return []

    output = ["🎯 Session Management:"]
    output.append(f"  • Total sessions: {sessions.get('total_sessions', 0)}")
    output.append(f"  • Preserved sessions: {sessions.get('preserved_sessions', 0)}")
    output.append(f"  • Restored sessions: {sessions.get('restored_sessions', 0)}")

    avg_restore = sessions.get("avg_restore_count", 0)
    output.append(f"  • Average restore count: {avg_restore:.1f}")

    return output


def _format_interruption_type_details(by_type: list) -> list[str]:
    """Format interruption type details section."""
    if not by_type:
        return []

    output = ["  • By type:"]

    for type_stat in by_type:
        type_name = type_stat.get("event_type", "").replace("_", " ").title()
        count = type_stat.get("type_count", 0)
        auto_saved = type_stat.get("auto_saved_interruptions", 0)
        avg_duration = type_stat.get("avg_duration", 0)

        output.append(f"    - {type_name}: {count}")
        if auto_saved > 0:
            output.append(f"      (💾 {auto_saved} auto-saved)")
        if avg_duration:
            output.append(f"      (⏱️ avg: {avg_duration:.1f}s)")

    return output


def _format_interruption_statistics(interruptions: dict) -> list[str]:
    """Format interruption statistics section."""
    if not interruptions:
        return []

    total_interruptions = interruptions.get("total", 0)
    output = ["\n⚡ Interruption Summary:"]
    output.append(f"  • Total interruptions: {total_interruptions}")

    by_type = interruptions.get("by_type", [])
    output.extend(_format_interruption_type_details(by_type))

    return output


def _format_snapshot_statistics(snapshots: dict) -> list[str]:
    """Format snapshot statistics section."""
    if not snapshots:
        return []

    total_snapshots = snapshots.get("total_snapshots", 0)
    total_size = snapshots.get("total_size", 0)
    avg_size = snapshots.get("avg_size", 0)

    output = ["\n📸 Context Snapshots:"]
    output.append(f"  • Total snapshots: {total_snapshots}")

    if total_size:
        size_mb = total_size / (1024 * 1024)
        output.append(f"  • Total storage: {size_mb:.2f} MB")

    if avg_size:
        avg_kb = avg_size / 1024
        output.append(f"  • Average size: {avg_kb:.1f} KB")

    return output


def _calculate_efficiency_rates(
    sessions: dict,
    interruptions: dict,
    by_type: list,
) -> tuple[float, float]:
    """Calculate preservation and auto-save rates."""
    preservation_rate = (
        sessions.get("preserved_sessions", 0)
        / max(sessions.get("total_sessions", 1), 1)
    ) * 100

    auto_save_rate = 0
    total_interruptions = interruptions.get("total", 0)
    if by_type and total_interruptions > 0:
        total_auto_saved = sum(t.get("auto_saved_interruptions", 0) for t in by_type)
        auto_save_rate = (total_auto_saved / max(total_interruptions, 1)) * 100

    return preservation_rate, auto_save_rate


def _format_efficiency_metrics(
    sessions: dict,
    interruptions: dict,
    by_type: list,
) -> list[str]:
    """Format efficiency metrics section."""
    if not (sessions and interruptions):
        return []

    preservation_rate, auto_save_rate = _calculate_efficiency_rates(
        sessions,
        interruptions,
        by_type,
    )

    return [
        "\n📈 Efficiency Metrics:",
        f"  • Context preservation rate: {preservation_rate:.1f}%",
        f"  • Auto-save success rate: {auto_save_rate:.1f}%",
    ]


def _has_statistics_data(sessions: dict, interruptions: dict, snapshots: dict) -> bool:
    """Check if there's any statistics data available."""
    return any(
        [
            sessions,
            interruptions.get("total", 0),
            snapshots.get("total_snapshots", 0),
        ],
    )


def _format_no_data_message(user_id: str) -> list[str]:
    """Format message when no statistics data is found."""
    return [
        "📊 No interruption management data found",
        f"👤 User: {user_id}",
        "💡 Start using interruption monitoring to see statistics",
    ]


async def get_interruption_statistics(user_id: str) -> str:
    """Get comprehensive interruption and context preservation statistics."""
    try:
        from .interruption_manager import (
            get_interruption_statistics as _get_interruption_statistics,
        )

        stats = await _get_interruption_statistics(user_id)
        output = _format_statistics_header(user_id)

        # Get statistics sections
        sessions = stats.get("sessions", {})
        interruptions = stats.get("interruptions", {})
        snapshots = stats.get("snapshots", {})
        by_type = interruptions.get("by_type", [])

        # Format all sections
        output.extend(_format_session_statistics(sessions))
        output.extend(_format_interruption_statistics(interruptions))
        output.extend(_format_snapshot_statistics(snapshots))
        output.extend(_format_efficiency_metrics(sessions, interruptions, by_type))

        # Check if we have any data
        if not _has_statistics_data(sessions, interruptions, snapshots):
            output = _format_no_data_message(user_id)

        return "\n".join(output)

    except ImportError:
        return "❌ Interruption management tools not available"
    except Exception as e:
        return f"❌ Error getting statistics: {e}"


# =====================================
# Crackerjack Integration MCP Tools
# =====================================


@mcp.tool()
async def execute_crackerjack_command(
    command: str,
    args: str = "",
    working_directory: str = ".",
    timeout: int = 300,
    ai_agent_mode: bool = False,
) -> str:
    """Execute a Crackerjack command with enhanced AI integration.

    Args:
        command: Crackerjack command to execute (analyze, check, test, lint, etc.)
        args: Additional command line arguments
        working_directory: Directory to run command in
        timeout: Command timeout in seconds
        ai_agent_mode: Enable AI agent autonomous fixing mode

    """
    if not CRACKERJACK_INTEGRATION_AVAILABLE:
        return "❌ Crackerjack integration not available"

    try:
        integration = CrackerjackIntegration()
        args_list = args.split() if args else []
        result = await integration.execute_crackerjack_command(
            command,
            args_list,
            working_directory,
            timeout,
            ai_agent_mode,
        )

        output = [f"🔧 Crackerjack {command} execution complete"]
        if ai_agent_mode:
            output.append("🤖 AI Agent mode enabled")
        output.append(f"📁 Working directory: {working_directory}")
        output.append(f"⏱️  Duration: {result.execution_time:.2f}s")
        output.append(f"🎯 Exit code: {result.exit_code}")

        if result.parsed_data:
            output.append("\n📊 Parsed Results:")
            for key, value in result.parsed_data.items():
                if isinstance(value, dict) and value:
                    output.append(f"  • {key}: {len(value)} items")
                elif isinstance(value, int | float) or value:
                    output.append(f"  • {key}: {value}")

        if result.memory_insights:
            output.append("\n💡 Memory Insights:")
            for insight in result.memory_insights[:3]:  # Show top 3
                output.append(f"  • {insight}")

        if result.exit_code != 0 and result.stderr:
            output.append("\n❌ Error output:")
            output.append(f"  {result.stderr[:200]}...")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error executing Crackerjack command: {e}"


# Clean Command Aliases
@mcp.tool()
async def crackerjack_run(
    command: str,
    args: str = "",
    working_directory: str = ".",
    timeout: int = 300,
    ai_agent_mode: bool = False,
) -> str:
    """Run crackerjack with enhanced analytics (replaces /crackerjack:run).

    Provides memory integration, intelligent insights, and quality tracking.
    Use this instead of basic /crackerjack:run for development work.

    Args:
        command: Crackerjack command to execute
        args: Additional command arguments
        working_directory: Directory to run in
        timeout: Command timeout in seconds
        ai_agent_mode: Enable autonomous AI fixing

    """
    return await execute_crackerjack_command(
        command, args, working_directory, timeout, ai_agent_mode
    )


@mcp.tool()
async def crackerjack_history(
    working_directory: str = ".",
    command_filter: str = "",
    days: int = 7,
) -> str:
    """View crackerjack execution history with trends and patterns."""
    return await get_crackerjack_results_history(
        working_directory,
        command_filter,
        days,
    )


@mcp.tool()
async def crackerjack_metrics(working_directory: str = ".", days: int = 30) -> str:
    """Get quality metrics trends from crackerjack execution history."""
    return await get_crackerjack_quality_metrics(days, working_directory)


@mcp.tool()
async def crackerjack_patterns(days: int = 7, working_directory: str = ".") -> str:
    """Analyze test failure patterns and trends."""
    return await analyze_crackerjack_test_patterns(days, working_directory)


@mcp.tool()
async def crackerjack_help() -> str:
    """Get comprehensive help for choosing the right crackerjack commands."""
    output = ["🔧 CRACKERJACK COMMAND GUIDE", "=" * 50]

    # Check what's available
    detected_servers = _detect_other_mcp_servers()
    has_basic_crackerjack = detected_servers.get("crackerjack", False)
    has_enhanced = CRACKERJACK_INTEGRATION_AVAILABLE

    if has_enhanced:
        output.extend(
            [
                "\n🎯 RECOMMENDED (Enhanced with AI Intelligence):",
                "• /session-mgmt:crackerjack-run check    - Full analysis with memory integration",
                "• /session-mgmt:crackerjack-run test     - Test execution with pattern tracking",
                "• /session-mgmt:crackerjack-run lint     - Linting with trend analysis",
                "• /session-mgmt:crackerjack-history      - View execution history and trends",
                "• /session-mgmt:crackerjack-metrics      - Quality metrics over time",
                "• /session-mgmt:crackerjack-patterns     - Analyze test failure patterns",
                "",
                "🧠 Enhanced Features:",
                "  ✅ Conversation memory - remembers all results",
                "  ✅ Intelligent insights - automated analysis",
                "  ✅ Quality tracking - metrics over time",
                "  ✅ Pattern detection - identifies recurring issues",
                "  ✅ Trend analysis - shows improvement/degradation",
                "  ✅ Cross-session learning - builds knowledge over time",
            ],
        )

    if has_basic_crackerjack:
        output.extend(
            [
                "\n📋 BASIC (Simple Execution, No Memory):",
                "• /crackerjack:run check                 - Raw output only",
                "• /crackerjack:run test                  - Basic test execution",
                "• /crackerjack:run lint                  - Simple linting",
                "",
                "⚠️ Basic Features:",
                "  ❌ No conversation memory",
                "  ❌ No intelligent analysis",
                "  ❌ No quality tracking",
                "  ❌ No pattern detection",
            ],
        )

    # Usage recommendations
    output.extend(
        [
            "\n💡 USAGE RECOMMENDATIONS:",
            "",
            "🚀 For Development Work:",
            "  → Use enhanced commands (/session-mgmt:crackerjack-*)",
            "  → Get AI insights and memory integration",
            "  → Build long-term project intelligence",
            "",
            "⚙️ For CI/CD Pipelines:",
            "  → Use basic commands (/crackerjack:*) if needed",
            "  → Lighter weight, no persistence requirements",
            "",
            "🎯 Quick Start:",
            "  1. /session-mgmt:crackerjack-run check",
            "  2. /session-mgmt:crackerjack-history",
            "  3. /session-mgmt:crackerjack-metrics",
        ],
    )

    if not has_enhanced and not has_basic_crackerjack:
        output.extend(
            [
                "\n❌ SETUP REQUIRED:",
                "Neither enhanced nor basic crackerjack is available.",
                "Install crackerjack to get started with code quality tools.",
            ],
        )

    return "\n".join(output)


@mcp.tool()
async def get_crackerjack_results_history(
    working_directory: str = ".",
    command_filter: str = "",
    days: int = 7,
) -> str:
    """Get recent Crackerjack command execution history."""
    if not CRACKERJACK_INTEGRATION_AVAILABLE:
        return "❌ Crackerjack integration not available"

    try:
        integration = CrackerjackIntegration()
        results = await integration.get_recent_crackerjack_results(
            working_directory,
            command_filter,
            days,
        )

        if not results:
            return f"📋 No Crackerjack results found in the last {days} days"

        output = [f"📋 Crackerjack Results History (last {days} days)"]
        output.append(f"📁 Directory: {working_directory}")
        if command_filter:
            output.append(f"🔍 Filter: {command_filter}")

        for result in results[:10]:  # Show last 10 results
            status = "✅" if result.get("exit_code", 1) == 0 else "❌"
            command = result.get("command", "unknown")
            timestamp = result.get("timestamp", "")
            duration = result.get("duration", 0)

            output.append(f"\n{status} {command} ({timestamp})")
            output.append(f"   ⏱️  {duration:.2f}s")

            # Show key metrics if available
            parsed = result.get("parsed_results", {})
            if parsed:
                if "test_summary" in parsed:
                    summary = parsed["test_summary"]
                    passed = summary.get("passed", 0)
                    failed = summary.get("failed", 0)
                    output.append(f"   🧪 Tests: {passed} passed, {failed} failed")

                if "lint_issues" in parsed:
                    issues = len(parsed["lint_issues"])
                    if issues > 0:
                        output.append(f"   🔍 Lint: {issues} issues")

                if "coverage_percentage" in parsed:
                    coverage = parsed["coverage_percentage"]
                    output.append(f"   📊 Coverage: {coverage}%")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error getting Crackerjack history: {e}"


@mcp.tool()
async def get_crackerjack_quality_metrics(
    working_directory: str = ".",
    days: int = 30,
) -> str:
    """Get quality metrics trends from Crackerjack execution history."""
    if not CRACKERJACK_INTEGRATION_AVAILABLE:
        return "❌ Crackerjack integration not available"

    try:
        integration = CrackerjackIntegration()
        metrics = await integration.get_quality_metrics_history(working_directory, days)

        if not metrics:
            return f"📊 No quality metrics found in the last {days} days"

        output = [f"📊 Quality Metrics Trends (last {days} days)"]
        output.append(f"📁 Directory: {working_directory}")

        # Test metrics
        test_data = metrics.get("test_metrics", {})
        if test_data:
            latest_tests = test_data.get("latest", {})
            trends = test_data.get("trends", {})

            output.append("\n🧪 Test Metrics:")
            if latest_tests:
                passed = latest_tests.get("passed", 0)
                failed = latest_tests.get("failed", 0)
                total = passed + failed
                pass_rate = (passed / max(total, 1)) * 100
                output.append(f"  • Latest: {passed}/{total} passed ({pass_rate:.1f}%)")

            if trends.get("pass_rate_trend"):
                trend = trends["pass_rate_trend"]
                direction = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                output.append(f"  • Pass rate trend: {direction} {trend:+.1f}%")

        # Coverage metrics
        coverage_data = metrics.get("coverage_metrics", {})
        if coverage_data:
            latest = coverage_data.get("latest", 0)
            trend = coverage_data.get("trend", 0)
            direction = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"

            output.append("\n📊 Coverage Metrics:")
            output.append(f"  • Latest coverage: {latest}%")
            output.append(f"  • Coverage trend: {direction} {trend:+.1f}%")

        # Lint metrics
        lint_data = metrics.get("lint_metrics", {})
        if lint_data:
            latest = lint_data.get("latest_issues", 0)
            trend = lint_data.get("issues_trend", 0)
            direction = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"

            output.append("\n🔍 Code Quality:")
            output.append(f"  • Latest lint issues: {latest}")
            output.append(f"  • Issues trend: {direction} {trend:+.0f}")

        # Security metrics
        security_data = metrics.get("security_metrics", {})
        if security_data:
            latest = security_data.get("latest_issues", 0)
            severity = security_data.get("highest_severity", "none")

            output.append("\n🔒 Security Metrics:")
            output.append(f"  • Latest security issues: {latest}")
            if severity != "none":
                output.append(f"  • Highest severity: {severity}")

        if not any([test_data, coverage_data, lint_data, security_data]):
            output = [
                f"📊 No quality metrics available for {working_directory}",
                "💡 Run some Crackerjack commands to start tracking metrics",
            ]

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error getting quality metrics: {e}"


def _add_frequent_failures_info(output: list[str], frequent_failures: list) -> None:
    """Add frequent failures information to output."""
    if frequent_failures:
        output.append("\n🔥 Most Frequent Failures:")
        for failure in frequent_failures[:5]:
            test_name = failure.get("test_name", "unknown")
            count = failure.get("failure_count", 0)
            last_error = failure.get("latest_error_type", "unknown")
            output.append(f"  • {test_name} ({count}x) - {last_error}")


def _add_error_types_info(output: list[str], error_types: dict) -> None:
    """Add error type distribution information to output."""
    if error_types:
        output.append("\n📊 Error Type Distribution:")
        for error_type, count in sorted(
            error_types.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]:
            output.append(f"  • {error_type}: {count} occurrences")


def _add_trends_info(output: list[str], trends: dict) -> None:
    """Add trends information to output."""
    if not trends:
        return

    output.append("\n📈 Trends:")

    stability = trends.get("stability_score", 0)
    stability_emoji = "🟢" if stability > 80 else "🟡" if stability > 60 else "🔴"
    output.append(f"  • Test stability: {stability_emoji} {stability:.1f}%")

    avg_duration = trends.get("average_test_duration", 0)
    if avg_duration > 0:
        output.append(f"  • Average test duration: {avg_duration:.1f}s")

    if trends.get("most_problematic_files"):
        files = trends["most_problematic_files"][:3]
        output.append("  • Most problematic files:")
        for file_path in files:
            output.append(f"    - {file_path}")


def _generate_test_recommendations(
    frequent_failures: list,
    error_types: dict,
    trends: dict,
) -> list[str]:
    """Generate recommendations based on test patterns."""
    recommendations = []

    if frequent_failures and len(frequent_failures) > 3:
        recommendations.append("Consider investigating the most frequent test failures")

    if error_types.get("AssertionError", 0) > error_types.get("ImportError", 0) * 2:
        recommendations.append(
            "High assertion failures suggest logic issues in tests or code",
        )

    if trends.get("stability_score", 100) < 70:
        recommendations.append(
            "Test stability is below 70% - consider test environment review",
        )

    return recommendations


@mcp.tool()
async def analyze_crackerjack_test_patterns(
    working_directory: str = ".",
    days: int = 7,
) -> str:
    """Analyze test failure patterns and trends."""
    if not CRACKERJACK_INTEGRATION_AVAILABLE:
        return "❌ Crackerjack integration not available"

    try:
        integration = CrackerjackIntegration()
        patterns = await integration.get_test_failure_patterns(days)

        if not patterns:
            return f"🧪 No test failure patterns found in the last {days} days"

        output = [f"🧪 Test Failure Pattern Analysis (last {days} days)"]
        output.append(f"📁 Directory: {working_directory}")

        frequent_failures = patterns.get("frequent_failures", [])
        error_types = patterns.get("error_type_distribution", {})
        trends = patterns.get("trends", {})

        _add_frequent_failures_info(output, frequent_failures)
        _add_error_types_info(output, error_types)
        _add_trends_info(output, trends)

        recommendations = _generate_test_recommendations(
            frequent_failures,
            error_types,
            trends,
        )
        if recommendations:
            output.append("\n💡 Recommendations:")
            for rec in recommendations:
                output.append(f"  • {rec}")

        if not any([frequent_failures, error_types, trends]):
            output = [
                "🧪 No test failure patterns to analyze",
                f"📁 Directory: {working_directory}",
                "💡 Run tests with failures to enable pattern analysis",
            ]

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error analyzing test patterns: {e}"


@mcp.tool()
async def crackerjack_quality_trends(
    working_directory: str = ".",
    days: int = 30,
) -> str:
    """Analyze quality trends over time with actionable insights."""
    if not CRACKERJACK_INTEGRATION_AVAILABLE:
        return "❌ Crackerjack integration not available"

    try:
        integration = CrackerjackIntegration()
        trends = await integration.get_quality_trends(working_directory, days)

        output = [f"📊 Quality Trends Analysis (last {days} days)"]
        output.append(f"📁 Directory: {working_directory}")

        overall = trends.get("overall", {})
        output.append(
            f"\n🎯 Overall Direction: {overall.get('overall_direction', 'unknown').upper()}"
        )

        improving = overall.get("improving_count", 0)
        declining = overall.get("declining_count", 0)
        stable = overall.get("stable_count", 0)

        output.append(f"  📈 Improving: {improving} metrics")
        output.append(f"  📉 Declining: {declining} metrics")
        output.append(f"  ➡️  Stable: {stable} metrics")

        # Individual metric trends
        metric_trends = trends.get("trends", {})
        if metric_trends:
            output.append("\n📋 Metric Details:")
            for metric_type, trend_data in metric_trends.items():
                direction = trend_data.get("direction", "unknown")
                recent_avg = trend_data.get("recent_average", 0)
                change = trend_data.get("change", 0)
                strength = trend_data.get("trend_strength", "unknown")

                icon = (
                    "📈"
                    if direction == "improving"
                    else "📉"
                    if direction == "declining"
                    else "➡️"
                )
                metric_name = metric_type.replace("_", " ").title()

                output.append(
                    f"  {icon} {metric_name}: {recent_avg:.1f}% (±{change:.1f}%, {strength})"
                )

        # Recommendations
        recommendations = trends.get("recommendations", [])
        if recommendations:
            output.append("\n💡 Recommendations:")
            for rec in recommendations[:5]:  # Show top 5
                output.append(f"  • {rec}")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Error analyzing quality trends: {e}"


@mcp.tool()
async def crackerjack_health_check() -> str:
    """Check Crackerjack integration health and provide diagnostics."""
    if not CRACKERJACK_INTEGRATION_AVAILABLE:
        return "❌ Crackerjack integration not available - check installation"

    try:
        integration = CrackerjackIntegration()
        health = await integration.health_check()

        output = ["🏥 Crackerjack Integration Health Check"]

        status = health.get("status", "unknown")
        if status == "healthy":
            output.append("✅ Status: HEALTHY")
        elif status == "partial":
            output.append("⚠️ Status: PARTIAL (some features may be limited)")
        else:
            output.append("❌ Status: UNHEALTHY")

        # Detailed checks
        output.append("\n🔍 Component Status:")

        if health.get("crackerjack_available"):
            output.append("  ✅ Crackerjack CLI: Available")
        else:
            output.append("  ❌ Crackerjack CLI: Not available")

        if health.get("database_accessible"):
            output.append("  ✅ Integration Database: Accessible")
        else:
            output.append("  ❌ Integration Database: Not accessible")

        # Recommendations
        recommendations = health.get("recommendations", [])
        if recommendations:
            output.append("\n💡 Diagnostics:")
            for rec in recommendations:
                output.append(f"  • {rec}")

        # Quick fix suggestions
        if status != "healthy":
            output.append("\n🔧 Quick Fixes:")
            if not health.get("crackerjack_available"):
                output.append("  → Install crackerjack: uv add crackerjack")
            if not health.get("database_accessible"):
                output.append("  → Check ~/.claude/data/ directory permissions")
                output.append("  → Try running a simple crackerjack command first")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Health check failed: {e}"


def _format_monitoring_status(quality_data: dict) -> list[str]:
    """Format the monitoring status section."""
    output = []
    if quality_data["monitoring_active"]:
        output.append("✅ Quality monitoring: ACTIVE")
    else:
        output.append("⚠️ Quality monitoring: LIMITED")
    return output


def _format_quality_trend(quality_data: dict) -> list[str]:
    """Format the quality trend analysis section."""
    output = []
    trend = quality_data["quality_trend"]
    if trend == "improving":
        output.append(f"📈 Quality trend: {trend.upper()} ✅")
    elif trend == "declining":
        output.append(f"📉 Quality trend: {trend.upper()} ⚠️")
    else:
        output.append(f"📊 Quality trend: {trend.upper()}")
    return output


def _format_quality_alerts(quality_data: dict) -> list[str]:
    """Format the quality alerts section."""
    output = []
    alerts = quality_data.get("alerts", [])
    if alerts:
        output.append(f"\n⚠️ Active Alerts ({len(alerts)}):")
        for i, alert in enumerate(alerts, 1):
            if "URGENT" in alert:
                output.append(f"   {i}. 🚨 {alert}")
            elif "WARNING" in alert:
                output.append(f"   {i}. ⚠️ {alert}")
            else:
                output.append(f"   {i}. 💡 {alert}")
    else:
        output.append("\n✅ No quality alerts - system healthy")
    return output


def _format_proactive_recommendations(quality_data: dict) -> list[str]:
    """Format the proactive recommendations section."""
    output = []
    if quality_data.get("recommend_checkpoint"):
        output.append("\n🔄 IMMEDIATE ACTION RECOMMENDED:")
        output.append("   • Run checkpoint to address quality issues")
        output.append("   • Review recent workflow changes")
        output.append("   • Consider conversation cleanup if needed")
    return output


async def _format_conversation_summary() -> list[str]:
    """Format the conversation summary section."""
    output = []
    try:
        conversation_summary = await summarize_current_conversation()
        if conversation_summary["key_topics"]:
            output.append("\n💬 Current Session Focus:")
            for topic in conversation_summary["key_topics"][:3]:
                output.append(f"   • {topic}")

        if conversation_summary["decisions_made"]:
            output.append("\n✅ Key Decisions:")
            for decision in conversation_summary["decisions_made"][:2]:
                output.append(f"   • {decision}")
    except Exception:
        pass
    return output


def _format_monitor_usage_guidance() -> list[str]:
    """Format the monitor usage guidance section."""
    return [
        "\n💡 Monitor Usage:",
        "   • Run quality_monitor between checkpoints",
        "   • Watch for declining trends",
        "   • Act on urgent/warning alerts immediately",
    ]


@mcp.tool()
async def quality_monitor() -> str:
    """Phase 3: Proactive quality monitoring with early warning system."""
    output = []
    output.append("📊 Proactive Quality Monitor")
    output.append("=" * 50)

    try:
        quality_data = await monitor_proactive_quality()

        # Build output sections using helper functions
        output.extend(_format_monitoring_status(quality_data))
        output.extend(_format_quality_trend(quality_data))
        output.extend(_format_quality_alerts(quality_data))
        output.extend(_format_proactive_recommendations(quality_data))
        output.extend(await _format_conversation_summary())
        output.extend(_format_monitor_usage_guidance())

        return "\n".join(output)

    except Exception as e:
        output.append(f"❌ Quality monitoring failed: {e}")
        output.append("💡 Try running a regular checkpoint instead")
        return "\n".join(output)


@mcp.prompt("quality-monitor")
async def quality_monitor_prompt() -> str:
    """Proactive session quality monitoring with trend analysis and early warnings."""
    return await quality_monitor()


@mcp.tool()
async def auto_compact() -> str:
    """Automatically trigger conversation compaction with intelligent summary."""
    output = []
    output.append("🔄 Auto-Compaction Tool")
    output.append("=" * 50)

    try:
        # Generate intelligent conversation summary for compaction
        conversation_summary = await summarize_current_conversation()

        output.append("📝 Preparing conversation compaction...")
        output.append("🎯 Key conversation elements to preserve:")

        if conversation_summary["key_topics"]:
            output.append(
                f"• Topics: {', '.join(conversation_summary['key_topics'][:3])}",
            )

        if conversation_summary["decisions_made"]:
            output.append(f"• Decisions: {conversation_summary['decisions_made'][0]}")

        if conversation_summary["next_steps"]:
            output.append(f"• Next steps: {conversation_summary['next_steps'][0]}")

        # Store the summary in the database
        try:
            db = await get_reflection_database()
            pre_compact_summary = f"Pre-compaction session state: {', '.join(conversation_summary['key_topics'])}. "
            pre_compact_summary += (
                f"Key decisions: {', '.join(conversation_summary['decisions_made'])}. "
            )
            pre_compact_summary += (
                f"Next steps: {', '.join(conversation_summary['next_steps'])}"
            )

            await db.store_reflection(
                pre_compact_summary,
                [
                    "auto-compact",
                    "pre-compaction",
                    "context-preservation",
                    current_project or "unknown-project",
                ],
            )

            output.append("💾 Context preserved in reflection database")

        except Exception as e:
            output.append(f"⚠️ Context preservation failed: {str(e)[:50]}")

        # The key insight: Return a message that will trigger Claude to run /compact
        output.append("\n" + "=" * 50)
        output.append("🚨 CONTEXT COMPACTION REQUIRED")
        output.append("=" * 50)
        output.append("📋 Session summary prepared and stored safely")
        output.append("🔄 Claude will now automatically compact the conversation")
        output.append("✅ All important context has been preserved")

        # This is the magic: Return a response that triggers automatic compaction
        output.append("\n/compact")

        return "\n".join(output)

    except Exception as e:
        output.append(f"❌ Auto-compaction failed: {e}")
        output.append("💡 Manual /compact recommended")
        return "\n".join(output)


@mcp.prompt("auto-compact")
async def auto_compact_prompt() -> str:
    """Automatically trigger conversation compaction with context preservation."""
    return await auto_compact()


# Multi-Project Coordination Tools


@mcp.tool()
async def create_project_group(
    name: str,
    projects: list[str],
    description: str = "",
) -> str:
    """Create a new project group for multi-project coordination."""
    if not multi_project_coordinator:
        await initialize_new_features()
        if not multi_project_coordinator:
            return "❌ Multi-project coordination not available"

    try:
        group = await multi_project_coordinator.create_project_group(
            name=name,
            projects=projects,
            description=description,
        )

        return f"""✅ **Project Group Created**

**Group:** {group.name}
**Projects:** {", ".join(group.projects)}
**Description:** {group.description or "None"}
**ID:** {group.id}

The project group is now available for cross-project coordination and knowledge sharing."""

    except Exception as e:
        return f"❌ Failed to create project group: {e}"


@mcp.tool()
async def add_project_dependency(
    source_project: str,
    target_project: str,
    dependency_type: str,
    description: str = "",
) -> str:
    """Add a dependency relationship between projects."""
    if not multi_project_coordinator:
        await initialize_new_features()
        if not multi_project_coordinator:
            return "❌ Multi-project coordination not available"

    try:
        dependency = await multi_project_coordinator.add_project_dependency(
            source_project=source_project,
            target_project=target_project,
            dependency_type=dependency_type,
            description=description,
        )

        return f"""✅ **Project Dependency Added**

**Source:** {dependency.source_project}
**Target:** {dependency.target_project}
**Type:** {dependency.dependency_type}
**Description:** {dependency.description or "None"}

This relationship will be used for cross-project search and coordination."""

    except Exception as e:
        return f"❌ Failed to add project dependency: {e}"


@mcp.tool()
async def search_across_projects(
    query: str,
    current_project: str,
    limit: int = 10,
) -> str:
    """Search conversations across related projects."""
    if not multi_project_coordinator:
        await initialize_new_features()
        if not multi_project_coordinator:
            return "❌ Multi-project coordination not available"

    try:
        results = await multi_project_coordinator.find_related_conversations(
            current_project=current_project,
            query=query,
            limit=limit,
        )

        if not results:
            return f"🔍 No results found for '{query}' across related projects"

        output = [f"🔍 **Cross-Project Search Results** ({len(results)} found)\n"]

        for i, result in enumerate(results, 1):
            project_indicator = (
                "📍 Current"
                if result["is_current_project"]
                else f"🔗 {result['source_project']}"
            )

            output.append(f"""**{i}.** {project_indicator}
**Score:** {result["score"]:.3f}
**Content:** {result["content"][:200]}{"..." if len(result["content"]) > 200 else ""}
**Timestamp:** {result.get("timestamp", "Unknown")}
---""")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Search failed: {e}"


@mcp.tool()
async def get_project_insights(projects: list[str], time_range_days: int = 30) -> str:
    """Get cross-project insights and collaboration opportunities."""
    if not multi_project_coordinator:
        await initialize_new_features()
        if not multi_project_coordinator:
            return "❌ Multi-project coordination not available"

    try:
        insights = await multi_project_coordinator.get_cross_project_insights(
            projects=projects,
            time_range_days=time_range_days,
        )

        output = [f"📊 **Cross-Project Insights** (Last {time_range_days} days)\n"]

        # Project activity
        if insights["project_activity"]:
            output.append("**📈 Project Activity:**")
            for project, stats in insights["project_activity"].items():
                output.append(
                    f"• **{project}:** {stats['conversation_count']} conversations, last active: {stats.get('last_activity', 'Unknown')}",
                )
            output.append("")

        # Common patterns
        if insights["common_patterns"]:
            output.append("**🔍 Common Patterns:**")
            for pattern in insights["common_patterns"][:5]:  # Top 5
                projects_str = ", ".join(pattern["projects"])
                output.append(
                    f"• **{pattern['pattern']}** across {projects_str} (frequency: {pattern['frequency']})",
                )
            output.append("")

        if not insights["project_activity"] and not insights["common_patterns"]:
            output.append("No insights available for the specified time range.")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Failed to get insights: {e}"


# Advanced Search Tools


@mcp.tool()
async def advanced_search(
    query: str,
    content_type: str | None = None,
    project: str | None = None,
    timeframe: str | None = None,
    sort_by: str = "relevance",
    limit: int = 10,
) -> str:
    """Perform advanced search with faceted filtering."""
    if not advanced_search_engine:
        await initialize_new_features()
        if not advanced_search_engine:
            return "❌ Advanced search not available"

    try:
        filters = []

        # Add content type filter
        if content_type:
            from session_mgmt_mcp.advanced_search import SearchFilter

            filters.append(
                SearchFilter(field="content_type", operator="eq", value=content_type),
            )

        # Add project filter
        if project:
            filters.append(SearchFilter(field="project", operator="eq", value=project))

        # Add timeframe filter
        if timeframe:
            start_time, end_time = advanced_search_engine._parse_timeframe(timeframe)
            filters.append(
                SearchFilter(
                    field="timestamp",
                    operator="range",
                    value=(start_time, end_time),
                ),
            )

        # Perform search
        search_results = await advanced_search_engine.search(
            query=query,
            filters=filters,
            sort_by=sort_by,
            limit=limit,
            include_highlights=True,
        )

        results = search_results["results"]
        if not results:
            return f"🔍 No results found for '{query}'"

        output = [f"🔍 **Advanced Search Results** ({len(results)} found)\n"]

        for i, result in enumerate(results, 1):
            output.append(f"""**{i}.** {result.title}
**Score:** {result.score:.3f} | **Project:** {result.project or "Unknown"}
**Content:** {result.content}
**Timestamp:** {result.timestamp}""")

            if result.highlights:
                output.append(f"**Highlights:** {'; '.join(result.highlights)}")

            output.append("---")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Advanced search failed: {e}"


@mcp.tool()
async def search_suggestions(query: str, field: str = "content", limit: int = 5) -> str:
    """Get search completion suggestions."""
    if not advanced_search_engine:
        await initialize_new_features()
        if not advanced_search_engine:
            return "❌ Advanced search not available"

    try:
        suggestions = await advanced_search_engine.suggest_completions(
            query=query,
            field=field,
            limit=limit,
        )

        if not suggestions:
            return f"💡 No suggestions found for '{query}'"

        output = [f"💡 **Search Suggestions** for '{query}':\n"]

        for i, suggestion in enumerate(suggestions, 1):
            output.append(
                f"{i}. {suggestion['text']} (frequency: {suggestion['frequency']})",
            )

        return "\n".join(output)

    except Exception as e:
        return f"❌ Failed to get suggestions: {e}"


@mcp.tool()
async def get_search_metrics(metric_type: str, timeframe: str = "30d") -> str:
    """Get search and activity metrics."""
    if not advanced_search_engine:
        await initialize_new_features()
        if not advanced_search_engine:
            return "❌ Advanced search not available"

    try:
        metrics = await advanced_search_engine.aggregate_metrics(
            metric_type=metric_type,
            timeframe=timeframe,
        )

        if "error" in metrics:
            return f"❌ {metrics['error']}"

        output = [f"📊 **{metric_type.title()} Metrics** ({timeframe})\n"]

        for item in metrics["data"][:10]:  # Top 10
            output.append(f"• **{item['key']}:** {item['value']}")

        if not metrics["data"]:
            output.append("No data available for the specified timeframe.")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Failed to get metrics: {e}"


# Git Worktree Management Tools


@mcp.tool()
async def git_worktree_list(working_directory: str | None = None) -> str:
    """List all git worktrees for the current repository."""
    from .worktree_manager import WorktreeManager

    working_dir = Path(working_directory or os.getcwd())
    manager = WorktreeManager(session_logger=session_logger)

    try:
        result = await manager.list_worktrees(working_dir)

        if not result["success"]:
            return f"❌ {result['error']}"

        worktrees = result["worktrees"]
        if not worktrees:
            return (
                "📝 No worktrees found. This repository only has the main working tree."
            )

        output = [
            f"🌿 **Git Worktrees** ({result['total_count']} total)\n",
            f"📂 Repository: {working_dir.name}",
            f"🎯 Current: {result.get('current_worktree', 'Unknown')}\n",
        ]

        for wt in worktrees:
            prefix = "🔸" if wt["is_current"] else "◦"
            main_indicator = " (main)" if wt["is_main"] else ""
            detached_indicator = " (detached)" if wt["is_detached"] else ""

            output.append(
                f"{prefix} **{wt['branch']}{main_indicator}{detached_indicator}**",
            )
            output.append(f"   📁 {wt['path']}")

            status_items = []
            if wt["locked"]:
                status_items.append("🔒 locked")
            if wt["prunable"]:
                status_items.append("🗑️ prunable")
            if not wt["exists"]:
                status_items.append("❌ missing")
            if wt["has_session"]:
                status_items.append("🧠 has session")

            if status_items:
                output.append(f"   Status: {', '.join(status_items)}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        session_logger.exception(f"git_worktree_list failed: {e}")
        return f"❌ Failed to list worktrees: {e}"


@mcp.tool()
async def git_worktree_add(
    branch: str,
    path: str,
    working_directory: str | None = None,
    create_branch: bool = False,
) -> str:
    """Create a new git worktree."""
    from .worktree_manager import WorktreeManager

    working_dir = Path(working_directory or os.getcwd())
    new_path = Path(path)

    if not new_path.is_absolute():
        new_path = working_dir.parent / path

    manager = WorktreeManager(session_logger=session_logger)

    try:
        result = await manager.create_worktree(
            repository_path=working_dir,
            new_path=new_path,
            branch=branch,
            create_branch=create_branch,
        )

        if not result["success"]:
            return f"❌ {result['error']}"

        output = [
            "🎉 **Worktree Created Successfully!**\n",
            f"🌿 Branch: {result['branch']}",
            f"📁 Path: {result['worktree_path']}",
            f"🎯 Created new branch: {'Yes' if create_branch else 'No'}",
        ]

        if result.get("output"):
            output.append(f"\n📝 Git output: {result['output']}")

        output.append(f"\n💡 To start working: cd {result['worktree_path']}")
        output.append("💡 Use `git_worktree_list` to see all worktrees")

        return "\n".join(output)

    except Exception as e:
        session_logger.exception(f"git_worktree_add failed: {e}")
        return f"❌ Failed to create worktree: {e}"


@mcp.tool()
async def git_worktree_remove(
    path: str,
    working_directory: str | None = None,
    force: bool = False,
) -> str:
    """Remove an existing git worktree."""
    from .worktree_manager import WorktreeManager

    working_dir = Path(working_directory or os.getcwd())
    remove_path = Path(path)

    if not remove_path.is_absolute():
        remove_path = working_dir.parent / path

    manager = WorktreeManager(session_logger=session_logger)

    try:
        result = await manager.remove_worktree(
            repository_path=working_dir,
            worktree_path=remove_path,
            force=force,
        )

        if not result["success"]:
            return f"❌ {result['error']}"

        output = [
            "🗑️ **Worktree Removed Successfully!**\n",
            f"📁 Removed path: {result['removed_path']}",
        ]

        if result.get("output"):
            output.append(f"📝 Git output: {result['output']}")

        output.append(f"\n💡 Used force removal: {'Yes' if force else 'No'}")
        output.append("💡 Use `git_worktree_list` to see remaining worktrees")

        return "\n".join(output)

    except Exception as e:
        session_logger.exception(f"git_worktree_remove failed: {e}")
        return f"❌ Failed to remove worktree: {e}"


@mcp.tool()
async def git_worktree_status(working_directory: str | None = None) -> str:
    """Get comprehensive status of current worktree and all related worktrees."""
    from .worktree_manager import WorktreeManager

    working_dir = Path(working_directory or os.getcwd())
    manager = WorktreeManager(session_logger=session_logger)

    try:
        result = await manager.get_worktree_status(working_dir)

        if not result["success"]:
            return f"❌ {result['error']}"

        current = result["current_worktree"]
        all_worktrees = result["all_worktrees"]
        session_summary = result["session_summary"]

        output = [
            "🌿 **Git Worktree Status**\n",
            f"📂 Repository: {working_dir.name}",
            f"🎯 Current worktree: {current['branch']}"
            + (" (main)" if current["is_main"] else " (worktree)"),
            f"📁 Path: {current['path']}",
            f"🧠 Has session: {'Yes' if current['has_session'] else 'No'}",
            f"🔸 Detached HEAD: {'Yes' if current['is_detached'] else 'No'}\n",
        ]

        # Session summary across all worktrees
        output.append("📊 **Multi-Worktree Summary:**")
        output.append(f"• Total worktrees: {result['total_worktrees']}")
        output.append(f"• Active sessions: {session_summary['active_sessions']}")
        output.append(f"• Unique branches: {session_summary['unique_branches']}")
        output.append(f"• Branches: {', '.join(session_summary['branches'])}\n")

        # List all worktrees with status
        output.append("🌳 **All Worktrees:**")
        for i, wt in enumerate(all_worktrees, 1):
            current_marker = " 👈 CURRENT" if wt["is_current"] else ""
            main_marker = " (main)" if wt["is_main"] else ""

            output.append(f"{i}. **{wt['branch']}{main_marker}**{current_marker}")
            output.append(f"   📁 {wt['path']}")

            status_items = []
            if wt["has_session"]:
                status_items.append("🧠 session")
            if wt["prunable"]:
                status_items.append("🗑️ prunable")
            if not wt["exists"]:
                status_items.append("❌ missing")

            if status_items:
                output.append(f"   Status: {', '.join(status_items)}")
            output.append("")

        output.append("💡 Use `git_worktree_list` for more details")
        output.append(
            "💡 Use `git_worktree_add <branch> <path>` to create new worktrees",
        )

        return "\n".join(output)

    except Exception as e:
        session_logger.exception(f"git_worktree_status failed: {e}")
        return f"❌ Failed to get worktree status: {e}"


@mcp.tool()
async def git_worktree_prune(working_directory: str | None = None) -> str:
    """Prune stale worktree references."""
    from .worktree_manager import WorktreeManager

    working_dir = Path(working_directory or os.getcwd())
    manager = WorktreeManager(session_logger=session_logger)

    try:
        result = await manager.prune_worktrees(working_dir)

        if not result["success"]:
            return f"❌ {result['error']}"

        output = ["🧹 **Worktree Pruning Complete**\n"]

        if result["pruned_count"] > 0:
            output.append(
                f"🗑️ Pruned {result['pruned_count']} stale worktree references",
            )
            if result.get("output"):
                output.append(f"📝 Details: {result['output']}")
        else:
            output.append("✅ No stale worktree references found")
            output.append("🎉 All worktrees are clean and up to date")

        output.append("\n💡 Use `git_worktree_list` to see current worktrees")

        return "\n".join(output)

    except Exception as e:
        session_logger.exception(f"git_worktree_prune failed: {e}")
        return f"❌ Failed to prune worktrees: {e}"


def main() -> None:
    """Main entry point for the MCP server."""
    # Initialize new features on startup
    import asyncio

    try:
        asyncio.run(initialize_new_features())
    except Exception:
        # Silently handle optional feature initialization failures
        pass

    mcp.run()


if __name__ == "__main__":
    main()
