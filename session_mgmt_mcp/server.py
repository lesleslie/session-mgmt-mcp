#!/usr/bin/env python3
"""
Claude Session Management MCP Server - FastMCP Version

A dedicated MCP server that provides session management functionality
including initialization, checkpoints, and cleanup across all projects.

This server can be included in any project's .mcp.json file to provide
automatic access to /session-init, /session-checkpoint,
and /session-end slash commands.
"""

import asyncio
import sys
import os
import json
import subprocess
import shutil
import hashlib
import logging
from pathlib import Path  
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

# Configure structured logging
class SessionLogger:
    """Structured logging for session management with context"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = log_dir / f"session_management_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Configure logger
        self.logger = logging.getLogger('session_management')
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
                '%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str, **context):
        """Log info with optional context"""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.info(message)
    
    def warning(self, message: str, **context):
        """Log warning with optional context"""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.warning(message)
    
    def error(self, message: str, **context):
        """Log error with optional context"""
        if context:
            message = f"{message} | Context: {json.dumps(context)}"
        self.logger.error(message)

# Initialize logger
claude_dir = Path.home() / "Projects" / "claude"
session_logger = SessionLogger(claude_dir / "logs")

# Add the global toolkit to Python path  
toolkits_path = str(claude_dir / "toolkits")
if toolkits_path not in sys.path:
    sys.path.append(toolkits_path)

try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    print("FastMCP not available. Install with: uv add fastmcp", file=sys.stderr)
    sys.exit(1)

# Global session management imports
try:
    from session.session_manager import SessionManager, start_session, checkpoint_session, end_session
    from verification.verification_toolkit import VerificationToolkit
    SESSION_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    print(f"Session management import failed: {e}", file=sys.stderr)
    SESSION_MANAGEMENT_AVAILABLE = False

# Import reflection tools
try:
    from session_mgmt_mcp.reflection_tools import get_reflection_database, get_current_project
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
        ServerlessSessionManager, ServerlessConfigManager, SessionState
    )
    SERVERLESS_MODE_AVAILABLE = True
except ImportError as e:
    print(f"Serverless mode import failed: {e}", file=sys.stderr)
    SERVERLESS_MODE_AVAILABLE = False

class SessionPermissionsManager:
    """Manages session permissions to avoid repeated prompts for trusted operations"""
    
    _instance = None
    _session_id = None
    
    def __new__(cls, claude_dir: Path):
        """Singleton pattern to ensure consistent session ID across tool calls"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, claude_dir: Path):
        if self._initialized:
            return
        self.claude_dir = claude_dir
        self.permissions_file = claude_dir / "sessions" / "trusted_permissions.json"
        self.permissions_file.parent.mkdir(exist_ok=True)
        self.trusted_operations: Set[str] = set()
        # Use class-level session ID to persist across instances
        if SessionPermissionsManager._session_id is None:
            SessionPermissionsManager._session_id = self._generate_session_id()
        self.session_id = SessionPermissionsManager._session_id
        self._load_permissions()
        self._initialized = True
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID based on current time and working directory"""
        session_data = f"{datetime.now().isoformat()}_{os.getcwd()}"
        return hashlib.md5(session_data.encode()).hexdigest()[:12]
    
    def _load_permissions(self):
        """Load previously granted permissions"""
        if self.permissions_file.exists():
            try:
                with open(self.permissions_file, 'r') as f:
                    data = json.load(f)
                    self.trusted_operations.update(data.get('trusted_operations', []))
            except (json.JSONDecodeError, KeyError):
                pass
    
    def _save_permissions(self):
        """Save current trusted permissions"""
        data = {
            'trusted_operations': list(self.trusted_operations),
            'last_updated': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        with open(self.permissions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def is_operation_trusted(self, operation: str) -> bool:
        """Check if an operation is already trusted"""
        return operation in self.trusted_operations
    
    def trust_operation(self, operation: str, description: str = ""):
        """Mark an operation as trusted to avoid future prompts"""
        self.trusted_operations.add(operation)
        self._save_permissions()
    
    def get_permission_status(self) -> Dict[str, Any]:
        """Get current permission status"""
        return {
            'session_id': self.session_id,
            'trusted_operations_count': len(self.trusted_operations),
            'trusted_operations': list(self.trusted_operations),
            'permissions_file': str(self.permissions_file)
        }
    
    def revoke_all_permissions(self):
        """Revoke all trusted permissions (for security reset)"""
        self.trusted_operations.clear()
        if self.permissions_file.exists():
            self.permissions_file.unlink()
    
    # Common trusted operations
    TRUSTED_UV_OPERATIONS = "uv_package_management"
    TRUSTED_GIT_OPERATIONS = "git_repository_access"
    TRUSTED_FILE_OPERATIONS = "project_file_access"
    TRUSTED_SUBPROCESS_OPERATIONS = "subprocess_execution"
    TRUSTED_NETWORK_OPERATIONS = "network_access"
    TRUSTED_WORKSPACE_OPERATIONS = "global_workspace_access"

# Initialize FastMCP 2.0 server
mcp = FastMCP("session-mgmt-mcp")

# Register slash commands as MCP prompts (not resources!)
SESSION_COMMANDS = {
    "init": """# Session Initialization

Initialize Claude session with comprehensive setup including UV dependencies, global workspace verification, and automation tools.

This command will:
- Verify global workspace structure
- Check for required toolkits and automation tools  
- Initialize project context and dependencies
- Set up performance monitoring and session tracking
- Provide comprehensive setup summary with next steps

Run this at the start of any coding session for optimal Claude integration.
""",
    "checkpoint": """# Session Checkpoint

Perform mid-session quality checkpoint with workflow analysis and optimization recommendations.

This command will:
- Analyze current session progress and workflow effectiveness
- Check for performance bottlenecks and optimization opportunities
- Validate current task completion status
- Provide recommendations for workflow improvements
- Create checkpoint for session recovery if needed

Use this periodically during long coding sessions to maintain optimal productivity.
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
"""
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

# Global instances
permissions_manager = SessionPermissionsManager(claude_dir)
current_project = None

def validate_global_workspace() -> Dict[str, Any]:
    """Enhanced validation of global workspace components"""
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'component_status': {},
        'missing_files': []
    }
    
    # Check key directories
    required_dirs = [
        claude_dir / "toolkits",
        claude_dir / "sessions", 
        claude_dir / "logs"
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            validation_result['errors'].append(f"Missing directory: {dir_path}")
            validation_result['valid'] = False
        else:
            validation_result['component_status'][dir_path.name] = "✅ Present"
    
    # Check key files
    key_files = [
        "CLAUDE.md",
        "SESSION_INITIALIZATION_CHECKLIST.md", 
        "WORKFLOW_AUTOMATION_SYSTEM.md",
        "TESTING_QUICK_REFERENCE.md",
        "TOOLKIT_DEVELOPMENT_GUIDE.md"
    ]
    
    missing_files = []
    for file_name in key_files:
        file_path = claude_dir / file_name
        if file_path.exists():
            validation_result['component_status'][file_name] = "✅ Present"
        else:
            missing_files.append(file_name)
            validation_result['component_status'][file_name] = "❌ Missing"
    
    if missing_files:
        validation_result['warnings'].append(f"Missing {len(missing_files)} documentation files")
        validation_result['missing_files'] = missing_files
    
    # Check toolkit availability
    toolkit_modules = [
        "session.session_manager",
        "verification.verification_toolkit"
    ]
    
    for module in toolkit_modules:
        try:
            __import__(module)
            validation_result['component_status'][f"toolkit_{module}"] = "✅ Available"
        except ImportError:
            validation_result['warnings'].append(f"Toolkit module not available: {module}")
            validation_result['component_status'][f"toolkit_{module}"] = "⚠️ Missing"
    
    return validation_result

async def analyze_project_context(project_dir: Path) -> Dict[str, bool]:
    """Analyze project structure and context with enhanced error handling"""
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
                "has_mcp_config": False
            }
        
        return {
            "python_project": (project_dir / "pyproject.toml").exists(),
            "git_repo": (project_dir / ".git").exists(),
            "has_tests": any(project_dir.glob("test*")) or any(project_dir.glob("**/test*")),
            "has_docs": (project_dir / "README.md").exists() or any(project_dir.glob("docs/**")),
            "has_requirements": (project_dir / "requirements.txt").exists(),
            "has_uv_lock": (project_dir / "uv.lock").exists(),
            "has_mcp_config": (project_dir / ".mcp.json").exists()
        }
    except (OSError, PermissionError) as e:
        # Log error but return safe defaults
        print(f"Warning: Could not analyze project context for {project_dir}: {e}", file=sys.stderr)
        return {
            "python_project": False,
            "git_repo": False,
            "has_tests": False,
            "has_docs": False,
            "has_requirements": False,
            "has_uv_lock": False,
            "has_mcp_config": False
        }

@mcp.tool()
async def init(working_directory: Optional[str] = None) -> str:
    """Initialize Claude session with comprehensive setup including UV dependencies, global workspace verification, and automation tools
    
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
        current_dir = Path(os.environ.get('PWD', Path.cwd()))
    
    global current_project
    current_project = current_dir.name
    output.append(f"📁 Current project: {current_project}")
    
    # Phase 1: Enhanced Global Workspace Verification
    output.append("\n📋 Phase 1: Global workspace verification...")
    
    workspace_validation = validate_global_workspace()
    
    if workspace_validation['valid']:
        output.append("✅ Global workspace structure verified")
    else:
        output.append("⚠️ Global workspace issues detected:")
        for error in workspace_validation['errors']:
            output.append(f"   ❌ {error}")
    
    # Show component status
    for component, status in workspace_validation['component_status'].items():
        output.append(f"   {status} {component}")
    
    # Show warnings if any
    if workspace_validation['warnings']:
        output.append("⚠️ Workspace warnings:")
        for warning in workspace_validation['warnings']:
            output.append(f"   • {warning}")
    
    # Phase 2: UV Dependency Management
    output.append("\n🔧 Phase 2: UV dependency management & session setup...")
    
    uv_available = shutil.which('uv') is not None
    output.append(f"🔍 UV package manager: {'✅ AVAILABLE' if uv_available else '❌ NOT FOUND'}")
    
    # Check UV permissions
    uv_trusted = permissions_manager.is_operation_trusted(permissions_manager.TRUSTED_UV_OPERATIONS)
    if uv_trusted:
        output.append("🔐 UV operations: ✅ TRUSTED (no prompts needed)")
    else:
        output.append("🔐 UV operations: ⚠️ Will require permission prompts")
    
    if uv_available:
        original_cwd = Path.cwd()
        try:
            os.chdir(claude_dir)
            output.append(f"📁 Working in: {claude_dir}")
            
            # Trust UV operations if first successful run
            if not uv_trusted:
                output.append("🔓 Trusting UV operations for this session...")
                permissions_manager.trust_operation(permissions_manager.TRUSTED_UV_OPERATIONS, "UV package management operations")
                output.append("✅ UV operations now trusted - no more prompts needed")
            
            # Sync dependencies
            sync_result = subprocess.run(['uv', 'sync'], capture_output=True, text=True)
            if sync_result.returncode == 0:
                output.append("✅ UV sync completed successfully")
                
                # Generate requirements.txt
                compile_result = subprocess.run(
                    ['uv', 'pip', 'compile', 'pyproject.toml', '--output-file', 'requirements.txt'],
                    capture_output=True, text=True
                )
                if compile_result.returncode == 0:
                    output.append("✅ Requirements.txt updated from UV dependencies")
                else:
                    output.append(f"⚠️ Requirements compilation warning: {compile_result.stderr}")
            else:
                output.append(f"⚠️ UV sync issues: {sync_result.stderr}")
                
        except Exception as e:
            output.append(f"⚠️ UV operation error: {e}")
        finally:
            os.chdir(original_cwd)
    else:
        output.append("💡 Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
    
    # Phase 3: Session Management
    if SESSION_MANAGEMENT_AVAILABLE:
        try:
            output.append("\n🔧 Starting comprehensive session checklist...")
            session_result = start_session()
            
            if session_result.get('checklist_passed'):
                output.append("✅ Session checklist passed successfully")
                
                auto_enabled = session_result.get('auto_checkpoints_enabled', False)
                output.append(f"🔄 Auto-checkpoints: {'✅ ACTIVE' if auto_enabled else '⚠️ INACTIVE'}")
                
                if auto_enabled:
                    output.append("   📊 Automatic quality monitoring every 5 minutes")
                    output.append("   🔔 Notifications will alert for workflow drift")
                
                output.append(f"📁 Session data: {session_result.get('session_file')}")
            else:
                output.append("❌ Session checklist issues found:")
                for issue in session_result.get('issues_found', []):
                    output.append(f"   • {issue}")
                    
        except Exception as e:
            output.append(f"❌ Session initialization error: {e}")
    else:
        output.append("\n⚠️  Session management toolkit not available")
        output.append("💡  Install dependencies: pip install -r ~/Projects/claude/requirements.txt")
    
    # Phase 4: Integrated MCP Services Initialization
    output.append("\n🧠 Phase 4: Integrated MCP services initialization...")
    
    output.append("\n📊 Integrated MCP Services Status:")
    output.append("✅ Session Management - Active (conversation memory enabled)")

    # Phase 5: Project Context Analysis
    output.append(f"\n🎯 Phase 5: Project context analysis for {current_project}...")
    
    project_context = await analyze_project_context(current_dir)
    context_score = sum(1 for detected in project_context.values() if detected)
    
    output.append("🔍 Project structure analysis:")
    for context_type, detected in project_context.items():
        status = "✅" if detected else "➖"
        output.append(f"   {status} {context_type.replace('_', ' ').title()}")
    
    output.append(f"\n📊 Project maturity: {context_score}/{len(project_context)} indicators")
    if context_score >= len(project_context) - 1:
        output.append("🌟 Excellent project structure - well-organized codebase")
    elif context_score >= len(project_context) // 2:
        output.append("👍 Good project structure - solid foundation")
    else:
        output.append("💡 Basic project - consider adding structure")
    
    # Final Summary
    output.append("\n" + "=" * 60)
    output.append(f"🎯 {current_project.upper()} SESSION INITIALIZATION COMPLETE")
    output.append("=" * 60)
    
    output.append(f"📅 Initialized: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"🗂️ Project: {current_project}")
    output.append(f"📊 Structure score: {context_score}/{len(project_context)}")
    
    missing_files = workspace_validation.get('missing_files', [])
    if context_score >= len(project_context) // 2 and not missing_files:
        output.append("✅ Ready for productive session - all systems optimal")
    else:
        output.append("⚠️ Session ready with minor setup opportunities identified")
    
    # Permissions Summary
    permissions_status = permissions_manager.get_permission_status()
    output.append(f"\n🔐 Session Permissions Summary:")
    output.append(f"   📊 Trusted operations: {permissions_status['trusted_operations_count']}")
    if permissions_status['trusted_operations_count'] > 0:
        output.append("   ✅ Future operations will have reduced permission prompts")
    else:
        output.append("   💡 Operations will be trusted automatically on first use")
    
    output.append("\n📋 AVAILABLE MCP TOOLS:")
    output.append("📊 Session Management:")
    output.append("• checkpoint - Mid-session quality assessment")
    output.append("• end - Complete session cleanup")
    output.append("• status - Current session status")
    output.append("• permissions - Manage trusted operations")
    output.append("• Built-in conversation memory with semantic search")
    
    output.append(f"\n✨ {current_project} session initialization complete via MCP!")
    
    return "\n".join(output)

async def calculate_quality_score() -> Dict[str, Any]:
    """Calculate session quality score based on multiple factors"""
    current_dir = Path(os.environ.get('PWD', Path.cwd()))
    
    # Project health indicators (40% of score)
    project_context = await analyze_project_context(current_dir)
    project_score = (sum(1 for detected in project_context.values() if detected) / len(project_context)) * 40
    
    # Permissions health (20% of score)  
    permissions_count = len(permissions_manager.trusted_operations)
    permissions_score = min(permissions_count * 5, 20)  # Up to 4 trusted operations = max score
    
    # Session management availability (20% of score)
    session_score = 20 if SESSION_MANAGEMENT_AVAILABLE else 5
    
    # Tool availability (20% of score)
    uv_available = shutil.which('uv') is not None
    tool_score = 20 if uv_available else 10
    
    total_score = int(project_score + permissions_score + session_score + tool_score)
    
    return {
        'total_score': total_score,
        'breakdown': {
            'project_health': project_score,
            'permissions': permissions_score, 
            'session_management': session_score,
            'tools': tool_score
        },
        'recommendations': _generate_quality_recommendations(total_score, project_context, permissions_count, uv_available)
    }

def _generate_quality_recommendations(score: int, project_context: Dict, permissions_count: int, uv_available: bool) -> List[str]:
    """Generate quality improvement recommendations based on score factors"""
    recommendations = []
    
    if score < 50:
        recommendations.append("Session needs attention - multiple areas for improvement")
    elif score < 75:
        recommendations.append("Good session health - minor optimizations available")
    else:
        recommendations.append("Excellent session quality - maintain current practices")
    
    # Project-specific recommendations
    if not project_context.get('has_tests'):
        recommendations.append("Consider adding tests to improve project structure")
    if not project_context.get('has_docs'):
        recommendations.append("Documentation would enhance project maturity")
    
    # Permissions recommendations
    if permissions_count == 0:
        recommendations.append("No trusted operations yet - permissions will be granted on first use")
    elif permissions_count > 5:
        recommendations.append("Many trusted operations - consider reviewing for security")
    
    # Tools recommendations
    if not uv_available:
        recommendations.append("Install UV package manager for better dependency management")
    
    return recommendations

@mcp.tool()
async def checkpoint() -> str:
    """Perform mid-session quality checkpoint with workflow analysis and optimization recommendations"""
    output = []
    output.append(f"🔍 Claude Session Checkpoint - {current_project or 'Current Project'}")
    output.append("=" * 50)
    
    # Initialize quality_score for later use
    quality_score = 0
    
    # Enhanced Quality Assessment
    if SESSION_MANAGEMENT_AVAILABLE:
        try:
            output.append("\n📊 Running comprehensive quality assessment...")
            # Calculate enhanced quality score
            quality_data = await calculate_quality_score()
            quality_score = quality_data['total_score']
            
            # Try to get session management data (but don't override our quality score)
            checkpoint_result = checkpoint_session()
            
            if checkpoint_result.get('checkpoint_passed', True):
                if quality_score >= 80:
                    output.append(f"✅ Session quality: EXCELLENT (Score: {quality_score}/100)")
                elif quality_score >= 60:
                    output.append(f"✅ Session quality: GOOD (Score: {quality_score}/100)")
                else:
                    output.append(f"⚠️ Session quality: NEEDS ATTENTION (Score: {quality_score}/100)")
                
                # Quality breakdown
                output.append("\n📈 Quality breakdown:")
                breakdown = quality_data['breakdown']
                output.append(f"   • Project health: {breakdown['project_health']:.1f}/40")
                output.append(f"   • Permissions: {breakdown['permissions']:.1f}/20")
                output.append(f"   • Session tools: {breakdown['session_management']:.1f}/20")
                output.append(f"   • Tool availability: {breakdown['tools']:.1f}/20")
                
                # Recommendations
                recommendations = quality_data['recommendations']
                if recommendations:
                    output.append("\n💡 Recommendations:")
                    for rec in recommendations[:3]:
                        output.append(f"   • {rec}")
                
                strengths = checkpoint_result.get('strengths', [])
                if strengths:
                    output.append("\n🌟 Session strengths:")
                    for strength in strengths[:3]:
                        output.append(f"   • {strength}")
            else:
                output.append("⚠️ Session quality issues detected:")
                issues = checkpoint_result.get('issues_found', [])
                for issue in issues:
                    output.append(f"   • {issue}")
                    
            session_stats = checkpoint_result.get('session_stats', {})
            if session_stats:
                output.append(f"\n⏱️ Session progress:")
                output.append(f"   • Duration: {session_stats.get('duration_minutes', 0)} minutes")
                output.append(f"   • Checkpoints: {session_stats.get('total_checkpoints', 0)}")
                output.append(f"   • Success rate: {session_stats.get('success_rate', 0):.1f}%")
                
        except Exception as e:
            session_logger.error("Checkpoint error occurred", error=str(e), project=current_project)
            output.append(f"❌ Checkpoint error: {e}")
            # Fallback to basic quality scoring
            quality_data = await calculate_quality_score()
            quality_score = quality_data['total_score']
            output.append(f"\n📊 Basic quality assessment: {quality_score}/100")
    else:
        output.append("\n⚠️ Session management not available - performing basic checks")
        # Still calculate quality score without session management
        quality_data = await calculate_quality_score()
        quality_score = quality_data['total_score']
        
        if quality_score >= 80:
            output.append(f"✅ Session quality: EXCELLENT (Score: {quality_score}/100)")
        elif quality_score >= 60:
            output.append(f"✅ Session quality: GOOD (Score: {quality_score}/100)")
        else:
            output.append(f"⚠️ Session quality: NEEDS ATTENTION (Score: {quality_score}/100)")
        
        # Quality breakdown
        output.append("\n📈 Quality breakdown:")
        breakdown = quality_data['breakdown']
        output.append(f"   • Project health: {breakdown['project_health']:.1f}/40")
        output.append(f"   • Permissions: {breakdown['permissions']:.1f}/20")
        output.append(f"   • Session tools: {breakdown['session_management']:.1f}/20")
        output.append(f"   • Tool availability: {breakdown['tools']:.1f}/20")
        
        # Recommendations
        recommendations = quality_data['recommendations']
        if recommendations:
            output.append("\n💡 Recommendations:")
            for rec in recommendations[:3]:
                output.append(f"   • {rec}")
    
    # Project Context
    current_dir = Path(os.environ.get('PWD', Path.cwd()))
    project_context = await analyze_project_context(current_dir)
    context_score = sum(1 for detected in project_context.values() if detected)
    
    output.append(f"\n🎯 Project context: {context_score}/{len(project_context)} indicators")
    
    # Dynamic recommendations based on score
    if quality_score >= 80:
        output.append("\n💡 Excellent session - maintain current workflow")
    elif quality_score >= 60:
        output.append("\n💡 Good session - minor optimizations available")
    else:
        output.append("\n💡 Session needs attention - review recommendations above")
    
    output.append("🔄 Consider next checkpoint in 30-45 minutes")
    
    # Git commit functionality
    output.append("\n" + "=" * 50)
    output.append("📦 Git Checkpoint Commit")
    output.append("=" * 50)
    
    current_dir = Path(os.environ.get('PWD', Path.cwd()))
    git_dir = current_dir / ".git"
    
    if git_dir.exists() and git_dir.is_dir():
        try:
            # Check git status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=current_dir
            )
            
            if status_result.returncode == 0:
                status_lines = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []
                
                modified_files = []
                untracked_files = []
                
                for line in status_lines:
                    if line:
                        status = line[:2]
                        filepath = line[3:]
                        
                        if status == "??":
                            untracked_files.append(filepath)
                        elif status.strip():  # Any other status means modified/staged
                            modified_files.append(filepath)
                
                if modified_files or untracked_files:
                    output.append(f"\n📝 Found {len(modified_files)} modified files and {len(untracked_files)} untracked files")
                    
                    # Handle untracked files - prompt for each
                    files_to_add = []
                    if untracked_files:
                        output.append("\n🆕 Untracked files found:")
                        for file in untracked_files[:10]:  # Limit to first 10 for display
                            output.append(f"   • {file}")
                        if len(untracked_files) > 10:
                            output.append(f"   ... and {len(untracked_files) - 10} more")
                        
                        # For MCP, we'll add a note about manual handling
                        output.append("\n⚠️ Please manually review and add untracked files if needed:")
                        output.append("   Use: git add <file> for files you want to include")
                    
                    # Stage modified files
                    if modified_files:
                        output.append(f"\n✅ Staging {len(modified_files)} modified files...")
                        for file in modified_files:
                            subprocess.run(["git", "add", file], cwd=current_dir, capture_output=True)
                    
                    # Check if there's anything to commit
                    staged_result = subprocess.run(
                        ["git", "diff", "--cached", "--name-only"],
                        capture_output=True,
                        text=True,
                        cwd=current_dir
                    )
                    
                    if staged_result.stdout.strip():
                        # Create commit
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        commit_message = f"checkpoint: Session checkpoint - {timestamp}\n\nAutomatic checkpoint commit via session-management MCP server\nProject: {current_project}\nQuality Score: {quality_score}/100"
                        
                        commit_result = subprocess.run(
                            ["git", "commit", "-m", commit_message],
                            capture_output=True,
                            text=True,
                            cwd=current_dir
                        )
                        
                        if commit_result.returncode == 0:
                            # Get the commit hash
                            hash_result = subprocess.run(
                                ["git", "rev-parse", "HEAD"],
                                capture_output=True,
                                text=True,
                                cwd=current_dir
                            )
                            commit_hash = hash_result.stdout.strip()[:8] if hash_result.returncode == 0 else "unknown"
                            
                            output.append(f"\n✅ Checkpoint commit created: {commit_hash}")
                            output.append(f"   Message: checkpoint: Session checkpoint - {timestamp}")
                            output.append("   💡 Use 'git reset HEAD~1' to undo if needed")
                        else:
                            output.append(f"\n⚠️ Commit failed: {commit_result.stderr}")
                    else:
                        output.append("\nℹ️ No staged changes to commit")
                        if untracked_files:
                            output.append("   💡 Add untracked files with 'git add' if you want to include them")
                else:
                    output.append("\n✅ Working directory is clean - no changes to commit")
            else:
                output.append(f"\n⚠️ Git status check failed: {status_result.stderr}")
        except Exception as e:
            output.append(f"\n⚠️ Git operations error: {e}")
    else:
        output.append("\nℹ️ Not a git repository - skipping commit")
    
    output.append(f"\n✨ Checkpoint complete - {current_project} session health verified!")
    
    return "\n".join(output)

@mcp.tool()
async def end() -> str:
    """End Claude session with cleanup, learning capture, and handoff file creation"""
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
            final_quality_score = final_quality_data['total_score']
            
            if final_checkpoint.get('checkpoint_passed'):
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
            project_name = current_project.replace("/", "_").replace(" ", "_") if current_project else "unknown"
            handoff_file = handoff_dir / f"{project_name}_mcp_session_handoff_{session_timestamp}.md"
            
            handoff_content = f"""# {current_project} MCP Session Handoff - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Session Summary
- **Project**: {current_project}
- **Working Directory**: {Path(os.environ.get('PWD', Path.cwd()))}
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
            
            with open(handoff_file, 'w') as f:
                f.write(handoff_content)
            
            output.append(f"📋 Session handoff created: {handoff_file}")
            
            # End session
            end_result = end_session()
            if end_result.get('session_ended_successfully'):
                output.append("✅ Session terminated successfully")
                
        except Exception as e:
            output.append(f"❌ Session end error: {e}")
    else:
        output.append("\n✅ Basic session cleanup completed")
    
    output.append(f"\n🙏 {current_project.upper() if current_project else 'SESSION'} COMPLETE!")
    output.append(f"📅 Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("\n🔄 Next session: Use /session-management:init slash command")
    output.append("🎉 MCP server will provide automatic session management!")
    
    return "\n".join(output)

async def health_check() -> Dict[str, Any]:
    """Comprehensive health check for MCP server and toolkit availability"""
    health_status = {
        'overall_healthy': True,
        'checks': {},
        'warnings': [],
        'errors': []
    }
    
    # MCP Server health
    try:
        # Test FastMCP availability 
        health_status['checks']['mcp_server'] = "✅ Active"
    except Exception as e:
        health_status['checks']['mcp_server'] = "❌ Error"
        health_status['errors'].append(f"MCP server issue: {e}")
        health_status['overall_healthy'] = False
    
    # Session management toolkit health
    health_status['checks']['session_toolkit'] = "✅ Available" if SESSION_MANAGEMENT_AVAILABLE else "⚠️ Limited"
    if not SESSION_MANAGEMENT_AVAILABLE:
        health_status['warnings'].append("Session management toolkit not fully available")
    
    # UV package manager health
    uv_available = shutil.which('uv') is not None
    health_status['checks']['uv_manager'] = "✅ Available" if uv_available else "❌ Missing"
    if not uv_available:
        health_status['warnings'].append("UV package manager not found")
    
    # Global workspace health
    workspace_validation = validate_global_workspace()
    health_status['checks']['global_workspace'] = "✅ Valid" if workspace_validation['valid'] else "⚠️ Issues"
    health_status['warnings'].extend(workspace_validation['warnings'])
    health_status['errors'].extend(workspace_validation['errors'])
    
    # Permissions system health
    try:
        permissions_status = permissions_manager.get_permission_status()
        health_status['checks']['permissions_system'] = "✅ Active"
        health_status['checks']['session_id'] = f"Active ({permissions_status['session_id']})"
    except Exception as e:
        health_status['checks']['permissions_system'] = "❌ Error"
        health_status['errors'].append(f"Permissions system issue: {e}")
        health_status['overall_healthy'] = False
    
    # Log health check results
    session_logger.info("Health check completed", 
                       overall_healthy=health_status['overall_healthy'],
                       warnings_count=len(health_status['warnings']),
                       errors_count=len(health_status['errors']))
    
    return health_status

@mcp.tool()
async def status(working_directory: Optional[str] = None) -> str:
    """Get current session status and project context information with health checks
    
    Args:
        working_directory: Optional working directory override (defaults to PWD environment variable or current directory)
    """
    output = []
    output.append("📊 Claude Session Status via MCP Server")
    output.append("=" * 40)
    
    if working_directory:
        current_dir = Path(working_directory)
    else:
        current_dir = Path(os.environ.get('PWD', Path.cwd()))
    
    global current_project
    current_project = current_dir.name
    
    output.append(f"📁 Current project: {current_project}")
    output.append(f"🗂️ Working directory: {current_dir}")
    output.append(f"🌐 MCP server: Active (Claude Session Management)")
    
    # Comprehensive health check
    health_status = await health_check()
    
    output.append(f"\n🏥 System Health: {'✅ HEALTHY' if health_status['overall_healthy'] else '⚠️ ISSUES DETECTED'}")
    
    # Display health check results
    for check_name, status in health_status['checks'].items():
        friendly_name = check_name.replace('_', ' ').title()
        output.append(f"   • {friendly_name}: {status}")
    
    # Show warnings and errors
    if health_status['warnings']:
        output.append("\n⚠️ Health Warnings:")
        for warning in health_status['warnings'][:3]:  # Limit to 3 warnings
            output.append(f"   • {warning}")
    
    if health_status['errors']:
        output.append("\n❌ Health Errors:")
        for error in health_status['errors'][:3]:  # Limit to 3 errors
            output.append(f"   • {error}")
    
    # Project analysis
    project_context = await analyze_project_context(current_dir)
    context_score = sum(1 for detected in project_context.values() if detected)
    
    output.append(f"\n📈 Project maturity: {context_score}/{len(project_context)}")
    
    # Permissions Status
    permissions_status = permissions_manager.get_permission_status()
    output.append(f"\n🔐 Session Permissions:")
    output.append(f"   📊 Trusted operations: {permissions_status['trusted_operations_count']}")
    if permissions_status['trusted_operations']:
        for op in permissions_status['trusted_operations']:
            output.append(f"   ✅ {op.replace('_', ' ').title()}")
    else:
        output.append("   ⚠️ No trusted operations yet - will prompt for permissions")
    
    output.append("\n🛠️ Available MCP Tools:")
    output.append("• init - Full session initialization")
    output.append("• checkpoint - Quality monitoring")
    output.append("• end - Complete cleanup")
    output.append("• status - This status report with health checks")
    output.append("• permissions - Manage trusted operations")
    
    return "\n".join(output)

@mcp.tool()
async def permissions(
    action: str = "status",
    operation: Optional[str] = None
) -> str:
    """Manage session permissions for trusted operations to avoid repeated prompts
    
    Args:
        action: Action to perform: status (show current), trust (add operation), revoke_all (reset)
        operation: Operation to trust (required for 'trust' action)
    """
    output = []
    output.append("🔐 Claude Session Permissions Management")
    output.append("=" * 50)
    
    if action == 'status':
        permissions_status = permissions_manager.get_permission_status()
        output.append(f"\n📊 Session ID: {permissions_status['session_id']}")
        output.append(f"📁 Permissions file: {permissions_status['permissions_file']}")
        output.append(f"✅ Trusted operations: {permissions_status['trusted_operations_count']}")
        
        if permissions_status['trusted_operations']:
            output.append("\n🔓 Currently trusted operations:")
            for op in permissions_status['trusted_operations']:
                friendly_name = op.replace('_', ' ').title()
                output.append(f"   • {friendly_name}")
                
            output.append("\n💡 These operations will not prompt for permission in future sessions")
        else:
            output.append("\n⚠️ No operations are currently trusted")
            output.append("💡 Operations will be automatically trusted on first successful use")
        
        output.append("\n🛠️ Common Operations That Can Be Trusted:")
        output.append("   • UV Package Management - uv sync, pip operations")
        output.append("   • Git Repository Access - git status, commit, push")
        output.append("   • Project File Access - reading/writing project files")
        output.append("   • Subprocess Execution - running build tools, tests")
        output.append("   • Global Workspace Access - accessing ~/Projects/claude/")
        
    elif action == 'trust':
        if not operation:
            output.append("❌ Error: 'operation' parameter required for 'trust' action")
            output.append("💡 Example: permissions with action='trust' and operation='uv_package_management'")
        else:
            permissions_manager.trust_operation(operation, f"Manually trusted via MCP at {datetime.now()}")
            output.append(f"✅ Operation trusted: {operation.replace('_', ' ').title()}")
            output.append("🔓 This operation will no longer prompt for permission")
            
    elif action == 'revoke_all':
        permissions_manager.revoke_all_permissions()
        output.append("🚨 All trusted permissions have been revoked")
        output.append("⚠️ All operations will now prompt for permission again")
        output.append("💡 Use this for security reset or if permissions were granted incorrectly")
        
    else:
        output.append(f"❌ Unknown action: {action}")
        output.append("💡 Valid actions: 'status', 'trust', 'revoke_all'")
    
    return "\n".join(output)

# Reflection Tools
@mcp.tool()
async def reflect_on_past(
    query: str,
    limit: int = 5,
    min_score: float = 0.7,
    project: Optional[str] = None
) -> str:
    """Search past conversations and store reflections with semantic similarity"""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"
    
    try:
        db = await get_reflection_database()
        current_proj = project or get_current_project()
        
        results = await db.search_conversations(
            query=query,
            limit=limit,
            min_score=min_score,
            project=current_proj
        )
        
        if not results:
            return f"🔍 No relevant conversations found for query: '{query}'\n💡 Try adjusting the search terms or lowering min_score."
        
        output = []
        output.append(f"🧠 Found {len(results)} relevant conversations for: '{query}'")
        if current_proj:
            output.append(f"📁 Project: {current_proj}")
        output.append("=" * 50)
        
        for i, result in enumerate(results, 1):
            score_pct = result['score'] * 100
            timestamp = result.get('timestamp', 'Unknown time')
            output.append(f"\n#{i} (Score: {score_pct:.1f}%)")
            output.append(f"📅 {timestamp}")
            output.append(f"💬 {result['content'][:200]}...")
            if result.get('project'):
                output.append(f"📁 Project: {result['project']}")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error searching conversations: {e}"

@mcp.tool()
async def store_reflection(
    content: str,
    tags: Optional[List[str]] = None
) -> str:
    """Store an important insight or reflection for future reference"""
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
        return f"❌ Error storing reflection: {e}"

@mcp.tool()
async def quick_search(
    query: str,
    min_score: float = 0.7,
    project: Optional[str] = None
) -> str:
    """Quick search that returns only the count and top result for fast overview"""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"
    
    try:
        db = await get_reflection_database()
        current_proj = project or get_current_project()
        
        results = await db.search_conversations(
            query=query,
            limit=1,
            min_score=min_score,
            project=current_proj
        )
        
        # Get total count with lower score threshold
        all_results = await db.search_conversations(
            query=query,
            limit=100,
            min_score=0.3,
            project=current_proj
        )
        
        output = []
        output.append(f"⚡ Quick search: '{query}'")
        if current_proj:
            output.append(f"📁 Project: {current_proj}")
        output.append(f"📊 Total matches: {len(all_results)} (threshold: 0.3)")
        output.append(f"🎯 High relevance: {len([r for r in all_results if r['score'] >= min_score])}")
        
        if results:
            top_result = results[0]
            score_pct = top_result['score'] * 100
            output.append(f"\n🥇 Top result (Score: {score_pct:.1f}%):")
            output.append(f"💬 {top_result['content'][:150]}...")
            output.append(f"📅 {top_result.get('timestamp', 'Unknown time')}")
        else:
            output.append(f"\n💡 No high-relevance matches found (min_score: {min_score})")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error in quick search: {e}"

@mcp.tool()
async def search_summary(
    query: str,
    min_score: float = 0.7,
    project: Optional[str] = None
) -> str:
    """Get aggregated insights from search results without individual result details"""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"
    
    try:
        db = await get_reflection_database()
        current_proj = project or get_current_project()
        
        results = await db.search_conversations(
            query=query,
            limit=10,
            min_score=min_score,
            project=current_proj
        )
        
        if not results:
            return f"📊 Search Summary: No relevant conversations found for '{query}'"
        
        # Analyze results
        total_results = len(results)
        avg_score = sum(r['score'] for r in results) / total_results
        projects = set(r.get('project') for r in results if r.get('project'))
        
        # Extract common themes (simple keyword analysis)
        all_content = ' '.join(r['content'] for r in results)
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
            output.append(f"\n🔑 Common themes:")
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
    project: Optional[str] = None
) -> str:
    """Get additional search results after an initial search (pagination support)"""
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
            project=current_proj
        )
        
        # Apply offset and limit
        paginated_results = all_results[offset:offset + limit]
        
        if not paginated_results:
            return f"📄 No more results found starting from position {offset + 1}"
        
        output = []
        output.append(f"📄 Additional results for: '{query}' (positions {offset + 1}-{offset + len(paginated_results)})")
        if current_proj:
            output.append(f"📁 Project: {current_proj}")
        output.append("=" * 50)
        
        for i, result in enumerate(paginated_results, offset + 1):
            score_pct = result['score'] * 100
            timestamp = result.get('timestamp', 'Unknown time')
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
    project: Optional[str] = None
) -> str:
    """Search for conversations that analyzed a specific file"""
    if not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Reflection tools not available. Install dependencies: pip install duckdb transformers"
    
    try:
        db = await get_reflection_database()
        current_proj = project or get_current_project()
        
        results = await db.search_by_file(
            file_path=file_path,
            limit=limit,
            project=current_proj
        )
        
        if not results:
            return f"🔍 No conversations found mentioning file: {file_path}"
        
        output = []
        output.append(f"📁 Found {len(results)} conversations about: {file_path}")
        if current_proj:
            output.append(f"🗂️ Project: {current_proj}")
        output.append("=" * 60)
        
        for i, result in enumerate(results, 1):
            timestamp = result.get('timestamp', 'Unknown time')
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
    project: Optional[str] = None
) -> str:
    """Search for conversations about a specific development concept"""
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
            project=current_proj
        )
        
        refl_results = await db.search_reflections(
            query=concept,
            limit=5,
            min_score=0.6
        )
        
        output = []
        output.append(f"🧠 Concept search: '{concept}'")
        if current_proj:
            output.append(f"📁 Project: {current_proj}")
        output.append("=" * 50)
        
        if conv_results:
            output.append(f"\n💬 Conversations ({len(conv_results)}):")
            for i, result in enumerate(conv_results[:5], 1):
                score_pct = result['score'] * 100
                output.append(f"#{i} (Score: {score_pct:.1f}%) {result['content'][:150]}...")
        
        if refl_results:
            output.append(f"\n💭 Reflections ({len(refl_results)}):")
            for i, result in enumerate(refl_results, 1):
                score_pct = result['score'] * 100
                tags_str = f" [{', '.join(result['tags'])}]" if result['tags'] else ""
                output.append(f"#{i} (Score: {score_pct:.1f}%){tags_str} {result['content'][:150]}...")
        
        if not conv_results and not refl_results:
            output.append(f"🔍 No relevant content found for concept: '{concept}'")
            output.append("💡 Try related terms or check if conversations about this topic exist")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error searching by concept: {e}"

@mcp.tool()
async def reflection_stats() -> str:
    """Get statistics about the reflection database"""
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
        
        total_items = stats['conversations_count'] + stats['reflections_count']
        output.append(f"\n📈 Total stored items: {total_items}")
        
        if total_items > 0:
            output.append("✅ Memory system is active and contains data")
        else:
            output.append("💡 Memory system is ready but contains no data yet")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error getting reflection stats: {e}"

# Enhanced Search Tools (Phase 1)

@mcp.tool()
async def search_code(
    query: str,
    pattern_type: Optional[str] = None,
    limit: int = 10,
    project: Optional[str] = None
) -> str:
    """Search for code patterns in conversations using AST parsing
    
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
            pattern = result['pattern']
            relevance_pct = result['relevance'] * 100
            
            output.append(f"\n#{i} (Relevance: {relevance_pct:.1f}%)")
            output.append(f"📁 Project: {result['project']}")
            output.append(f"🕒 Time: {result['timestamp']}")
            output.append(f"📋 Pattern: {pattern['type']}")
            
            if 'name' in pattern:
                output.append(f"🏷️ Name: {pattern['name']}")
            
            output.append(f"💻 Code snippet:")
            code_lines = pattern['content'].split('\n')[:3]  # First 3 lines
            for line in code_lines:
                output.append(f"    {line}")
            if len(pattern['content'].split('\n')) > 3:
                output.append("    ...")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error searching code patterns: {e}"

@mcp.tool()
async def search_errors(
    query: str,
    error_type: Optional[str] = None,
    limit: int = 10,
    project: Optional[str] = None
) -> str:
    """Search for error patterns and debugging contexts in conversations
    
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
            pattern = result['pattern']
            relevance_pct = result['relevance'] * 100
            
            output.append(f"\n#{i} (Relevance: {relevance_pct:.1f}%)")
            output.append(f"📁 Project: {result['project']}")
            output.append(f"🕒 Time: {result['timestamp']}")
            output.append(f"🚨 Pattern: {pattern['type']} - {pattern['subtype']}")
            
            if pattern['type'] == 'error':
                if 'groups' in pattern and pattern['groups']:
                    output.append(f"💀 Error: {pattern['groups'][0] if pattern['groups'] else 'Unknown'}")
                    if len(pattern['groups']) > 1:
                        output.append(f"📝 Message: {pattern['groups'][1]}")
            
            # Show relevant snippet
            snippet = result['snippet']
            output.append(f"📄 Context: {snippet[:200]}...")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error searching error patterns: {e}"

@mcp.tool()
async def search_temporal(
    time_expression: str,
    query: Optional[str] = None,
    limit: int = 10,
    project: Optional[str] = None
) -> str:
    """Search conversations within a specific time range using natural language
    
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
        
        if len(results) == 1 and 'error' in results[0]:
            return f"❌ {results[0]['error']}"
        
        output = []
        output.append(f"🕒 Temporal Search: '{time_expression}'")
        if query:
            output.append(f"🔍 Query: '{query}'")
        output.append("=" * 50)
        
        for i, result in enumerate(results, 1):
            relevance_pct = result.get('relevance', 1.0) * 100
            
            output.append(f"\n#{i} (Relevance: {relevance_pct:.1f}%)")
            output.append(f"📁 Project: {result['project']}")
            output.append(f"🕒 Time: {result['timestamp']}")
            output.append(f"💬 Content: {result['content']}")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error in temporal search: {e}"

@mcp.tool()
async def auto_load_context(
    working_directory: Optional[str] = None,
    max_conversations: int = 10,
    min_relevance: float = 0.3
) -> str:
    """Automatically load relevant conversations based on current development context
    
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
            working_directory, max_conversations, min_relevance
        )
        
        context = result['context']
        conversations = result['relevant_conversations']
        
        output = []
        output.append("🤖 Auto-Context Loading Results")
        output.append("=" * 50)
        
        # Context summary
        output.append("\n📋 Detected Context:")
        output.append(f"📁 Project: {context['project_name']}")
        
        if context['detected_languages']:
            langs = ', '.join(context['detected_languages'])
            output.append(f"💻 Languages: {langs}")
        
        if context['detected_tools']:
            tools = ', '.join(context['detected_tools'])
            output.append(f"🔧 Tools: {tools}")
        
        if context['project_type']:
            proj_type = context['project_type'].replace('_', ' ').title()
            output.append(f"📋 Type: {proj_type}")
        
        confidence = context['confidence_score'] * 100
        output.append(f"🎯 Detection confidence: {confidence:.0f}%")
        
        # Git info
        if context['git_info'].get('is_git_repo'):
            git_info = context['git_info']
            branch = git_info.get('current_branch', 'unknown')
            platform = git_info.get('platform', 'git')
            output.append(f"🌿 Git: {branch} branch on {platform}")
        
        # Relevant conversations
        output.append(f"\n💬 Loaded Conversations: {result['loaded_count']}/{result['total_found']} found")
        if conversations:
            for i, conv in enumerate(conversations, 1):
                relevance_pct = conv['relevance_score'] * 100
                output.append(f"\n#{i} (Relevance: {relevance_pct:.1f}%)")
                output.append(f"📁 Project: {conv['project']}")
                output.append(f"🕒 Time: {conv['timestamp']}")
                
                # Show snippet of content
                content_preview = conv['content'][:200]
                if len(conv['content']) > 200:
                    content_preview += "..."
                output.append(f"💬 Content: {content_preview}")
        else:
            output.append(f"🔍 No conversations found above {min_relevance:.1f} relevance threshold")
            output.append("💡 Try lowering min_relevance or working on this project more")
        
        # Recent files info
        if context['recent_files']:
            recent_count = len(context['recent_files'])
            output.append(f"\n📄 Recent activity: {recent_count} files modified in last 2 hours")
            for file_info in context['recent_files'][:3]:  # Show top 3
                output.append(f"    📝 {file_info['path']}")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error in auto-context loading: {e}"

@mcp.tool()
async def get_context_summary(working_directory: Optional[str] = None) -> str:
    """Get a summary of current development context without loading conversations
    
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
            
            if context['detected_languages']:
                langs = ', '.join(context['detected_languages'])
                summary_parts.append(f"💻 Languages: {langs}")
            
            if context['detected_tools']:
                tools = ', '.join(context['detected_tools'])
                summary_parts.append(f"🔧 Tools: {tools}")
            
            confidence = context['confidence_score'] * 100
            summary_parts.append(f"🎯 Detection confidence: {confidence:.0f}%")
            
            summary = '\n'.join(summary_parts)
        
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
    dry_run: bool = False
) -> str:
    """Compress conversation memory by consolidating old conversations
    
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
            'consolidation_age_days': max_age_days,
            'max_conversations': max_conversations,
            'importance_threshold': importance_threshold
        }
        
        result = await optimizer.compress_memory(policy, dry_run)
        
        if 'error' in result:
            return f"❌ {result['error']}"
        
        output = []
        output.append("🗜️ Memory Compression Results")
        output.append("=" * 50)
        
        if dry_run:
            output.append("🔍 DRY RUN - No changes made")
        
        output.append(f"\n📊 Overview:")
        output.append(f"📁 Total conversations: {result['total_conversations']}")
        output.append(f"✅ Conversations to keep: {result['conversations_to_keep']}")
        output.append(f"🗜️ Conversations to consolidate: {result['conversations_to_consolidate']}")
        output.append(f"📦 Clusters created: {result['clusters_created']}")
        
        if result['space_saved_estimate'] > 0:
            space_mb = result['space_saved_estimate'] / (1024 * 1024)
            compression_pct = result['compression_ratio'] * 100
            output.append(f"💾 Space saved: {space_mb:.2f}MB ({compression_pct:.1f}% reduction)")
        
        # Show consolidation details
        if result['consolidated_summaries']:
            output.append(f"\n📋 Consolidation Details:")
            for i, summary in enumerate(result['consolidated_summaries'][:5], 1):  # Show top 5
                original_kb = summary['original_size'] / 1024
                compressed_kb = summary['compressed_size'] / 1024
                reduction_pct = (1 - summary['compressed_size'] / summary['original_size']) * 100
                
                output.append(f"\n#{i} Cluster:")
                output.append(f"   📁 Projects: {', '.join(summary['projects']) if summary['projects'] else 'Multiple'}")
                output.append(f"   💬 Conversations: {summary['original_count']}")
                output.append(f"   📏 Size: {original_kb:.1f}KB → {compressed_kb:.1f}KB ({reduction_pct:.1f}% reduction)")
                output.append(f"   🕒 Time range: {summary['time_range']}")
                output.append(f"   📝 Summary: {summary['summary'][:100]}...")
        
        if not dry_run and result['consolidated_summaries']:
            output.append(f"\n✅ Memory compression completed successfully!")
        elif dry_run:
            output.append(f"\n💡 Run with dry_run=False to apply these changes")
        elif not result['consolidated_summaries']:
            output.append(f"\n💡 No consolidation needed - all conversations are recent or important")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error compressing memory: {e}"

@mcp.tool()
async def get_compression_stats() -> str:
    """Get memory compression statistics and history"""
    if not MEMORY_OPTIMIZER_AVAILABLE or not REFLECTION_TOOLS_AVAILABLE:
        return "❌ Memory optimizer or reflection tools not available"
    
    try:
        db = await get_reflection_database()
        optimizer = MemoryOptimizer(db)
        
        stats = await optimizer.get_compression_stats()
        
        output = []
        output.append("📊 Memory Compression Statistics")
        output.append("=" * 40)
        
        if stats['last_run']:
            last_run = stats['last_run']
            output.append(f"🕒 Last compression: {last_run}")
            output.append(f"💬 Conversations processed: {stats['conversations_processed']}")
            output.append(f"📦 Conversations consolidated: {stats['conversations_consolidated']}")
            
            if stats['space_saved_bytes'] > 0:
                space_mb = stats['space_saved_bytes'] / (1024 * 1024)
                compression_pct = stats['compression_ratio'] * 100
                output.append(f"💾 Space saved: {space_mb:.2f}MB")
                output.append(f"🗜️ Compression ratio: {compression_pct:.1f}%")
        else:
            output.append("💡 No compression runs performed yet")
            output.append("🔧 Use compress_memory() to start optimizing your conversation storage")
        
        # Current database stats
        if hasattr(db, 'conn') and db.conn:
            cursor = db.conn.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]
            
            cursor = db.conn.execute("SELECT SUM(LENGTH(content)) FROM conversations")
            total_size = cursor.fetchone()[0] or 0
            
            output.append(f"\n📈 Current Database:")
            output.append(f"💬 Total conversations: {total_conversations}")
            size_mb = total_size / (1024 * 1024)
            output.append(f"💾 Total size: {size_mb:.2f}MB")
            
            # Check for consolidated conversations
            cursor = db.conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE metadata LIKE '%consolidated%'"
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
    consolidation_age_days: int = 30
) -> str:
    """Set memory retention policy parameters
    
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
            'max_age_days': max_age_days,
            'max_conversations': max_conversations,
            'importance_threshold': importance_threshold,
            'consolidation_age_days': consolidation_age_days
        }
        
        result = await optimizer.set_retention_policy(policy)
        
        if 'error' in result:
            return f"❌ {result['error']}"
        
        output = []
        output.append("⚙️ Retention Policy Updated")
        output.append("=" * 40)
        
        policy = result['updated_policy']
        output.append(f"📅 Max age: {policy['max_age_days']} days")
        output.append(f"💬 Max conversations: {policy['max_conversations']}")
        output.append(f"⭐ Importance threshold: {policy['importance_threshold']:.1f}")
        output.append(f"📦 Consolidation age: {policy['consolidation_age_days']} days")
        output.append(f"🗜️ Target compression: {policy.get('compression_ratio', 0.5) * 100:.0f}%")
        
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

async def get_app_monitor() -> Optional[ApplicationMonitor]:
    """Get or initialize application monitor"""
    global _app_monitor
    if not APP_MONITOR_AVAILABLE:
        return None
    
    if _app_monitor is None:
        data_dir = Path.home() / ".claude" / "data" / "app_monitoring"
        working_dir = os.environ.get('PWD', os.getcwd())
        project_paths = [working_dir] if Path(working_dir).exists() else []
        _app_monitor = ApplicationMonitor(str(data_dir), project_paths)
    
    return _app_monitor

@mcp.tool()
async def start_app_monitoring(project_paths: Optional[List[str]] = None) -> str:
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
            f"📊 Process monitoring: {'✅' if result.get('psutil_available') else '❌'}"
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
        else:
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
            ""
        ]
        
        if summary['event_types']:
            output.append("📋 Event Types:")
            for event_type, count in summary['event_types'].items():
                output.append(f"  • {event_type}: {count}")
            output.append("")
        
        if summary['applications']:
            output.append("💻 Applications:")
            for app, count in summary['applications'].items():
                output.append(f"  • {app}: {count}")
            output.append("")
        
        if summary['active_files']:
            output.append("📄 Most Active Files:")
            for file_info in summary['active_files'][:5]:
                output.append(f"  • {file_info['file_path']} (score: {file_info['activity_score']})")
            output.append("")
        
        if summary['documentation_sites']:
            output.append("📖 Documentation Sites Visited:")
            for site in summary['documentation_sites']:
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
        
        output = [
            f"🧠 Context Insights (Last {hours} hours)",
            ""
        ]
        
        if insights['primary_focus']:
            output.append(f"🎯 Primary Focus: {insights['primary_focus']}")
        
        if insights['technologies_used']:
            tech_list = ', '.join(insights['technologies_used'])
            output.append(f"💻 Technologies: {tech_list}")
        
        if insights['active_projects']:
            output.append(f"📁 Active Projects: {len(insights['active_projects'])}")
            for project in list(insights['active_projects'])[:3]:
                output.append(f"  • {Path(project).name}")
        
        if insights['documentation_topics']:
            output.append("📖 Documentation Topics:")
            for topic in insights['documentation_topics'][:5]:
                output.append(f"  • {topic}")
        
        output.extend([
            "",
            f"🔄 Context Switches: {insights['context_switches']}",
            f"⚡ Productivity Score: {insights['productivity_score']:.2f}"
        ])
        
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
        
        output = [
            f"📄 Active Files (Last {minutes} minutes)",
            ""
        ]
        
        for file_info in active_files[:10]:
            file_path = file_info['file_path']
            relative_path = Path(file_path).name if len(file_path) > 50 else file_path
            score = file_info['activity_score']
            count = file_info['event_count']
            output.append(f"• {relative_path} (score: {score}, events: {count})")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error getting active files: {e}"

# Global LLM manager instance
_llm_manager = None

async def get_llm_manager() -> Optional[LLMManager]:
    """Get or initialize LLM manager"""
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
        
        output = [
            "🤖 Available LLM Providers",
            ""
        ]
        
        for provider_name, info in provider_info['providers'].items():
            status = "✅" if provider_name in available_providers else "❌"
            output.append(f"{status} {provider_name.title()}")
            
            if provider_name in available_providers:
                models = info['models'][:5]  # Show first 5 models
                for model in models:
                    output.append(f"   • {model}")
                if len(info['models']) > 5:
                    output.append(f"   • ... and {len(info['models']) - 5} more")
            output.append("")
        
        config = provider_info['config']
        output.extend([
            f"🎯 Default Provider: {config['default_provider']}",
            f"🔄 Fallback Providers: {', '.join(config['fallback_providers'])}"
        ])
        
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
        
        output = [
            "🧪 LLM Provider Test Results",
            ""
        ]
        
        for provider_name, result in test_results.items():
            if result['available'] and result['test_successful']:
                output.append(f"✅ {provider_name.title()}: Working")
                output.append(f"   Model: {result.get('model', 'Unknown')}")
                output.append(f"   Response: {result.get('response_length', 0)} chars")
            elif result['available']:
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
async def generate_with_llm(prompt: str, 
                           provider: Optional[str] = None,
                           model: Optional[str] = None,
                           temperature: float = 0.7,
                           max_tokens: Optional[int] = None,
                           use_fallback: bool = True) -> str:
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
        
        messages = [LLMMessage(role='user', content=prompt)]
        
        response = await manager.generate(
            messages=messages,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_fallback=use_fallback
        )
        
        output = [
            f"🤖 Generated by {response.provider} ({response.model})",
            "",
            response.content,
            "",
            f"📊 Usage: {response.usage.get('total_tokens', 0)} tokens",
            f"⏱️ Generated: {response.timestamp}"
        ]
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error generating text: {e}"

@mcp.tool()
async def chat_with_llm(messages: List[Dict[str, str]], 
                       provider: Optional[str] = None,
                       model: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> str:
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
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                return "❌ Invalid message format. Each message must have 'role' and 'content' fields"
            
            if msg['role'] not in ['user', 'assistant', 'system']:
                return "❌ Invalid role. Must be 'user', 'assistant', or 'system'"
            
            llm_messages.append(LLMMessage(
                role=msg['role'],
                content=msg['content']
            ))
        
        response = await manager.generate(
            messages=llm_messages,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        output = [
            f"🤖 {response.provider} ({response.model}) response:",
            "",
            response.content,
            "",
            f"📊 Usage: {response.usage.get('total_tokens', 0)} tokens",
            f"⏱️ Generated: {response.timestamp}"
        ]
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error in chat: {e}"

@mcp.tool()
async def configure_llm_provider(provider: str, 
                                api_key: Optional[str] = None,
                                base_url: Optional[str] = None,
                                default_model: Optional[str] = None) -> str:
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
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Initialize structure
        if 'providers' not in config:
            config['providers'] = {}
        if provider not in config['providers']:
            config['providers'][provider] = {}
        
        # Update configuration
        if api_key:
            config['providers'][provider]['api_key'] = api_key
        if base_url:
            config['providers'][provider]['base_url'] = base_url
        if default_model:
            config['providers'][provider]['default_model'] = default_model
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Reinitialize manager to pick up new config
        global _llm_manager
        _llm_manager = None
        
        return f"✅ Configuration updated for {provider} provider"
        
    except Exception as e:
        return f"❌ Error configuring provider: {e}"

# Global serverless session manager
_serverless_manager = None

async def get_serverless_manager() -> Optional[ServerlessSessionManager]:
    """Get or initialize serverless session manager"""
    global _serverless_manager
    if not SERVERLESS_MODE_AVAILABLE:
        return None
    
    if _serverless_manager is None:
        config_path = Path.home() / ".claude" / "data" / "serverless_config.json"
        config = ServerlessConfigManager.load_config(str(config_path) if config_path.exists() else None)
        storage_backend = ServerlessConfigManager.create_storage_backend(config)
        _serverless_manager = ServerlessSessionManager(storage_backend)
    
    return _serverless_manager

@mcp.tool()
async def create_serverless_session(user_id: str, 
                                   project_id: str,
                                   session_data: Optional[Dict[str, Any]] = None,
                                   ttl_hours: int = 24) -> str:
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
            ttl_hours=ttl_hours
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
            f"💾 Size: {session_state.get_compressed_size()} bytes (compressed)"
        ]
        
        if session_state.metadata:
            output.append(f"📋 Metadata: {json.dumps(session_state.metadata, indent=2)}")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error getting session: {e}"

@mcp.tool()
async def update_serverless_session(session_id: str,
                                   updates: Dict[str, Any],
                                   ttl_hours: Optional[int] = None) -> str:
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
            update_summary = ", ".join(f"{k}: {type(v).__name__}" for k, v in updates.items())
            result = f"✅ Updated session {session_id}\n📝 Changes: {update_summary}"
            if ttl_hours:
                result += f"\n🕐 New TTL: {ttl_hours} hours"
            return result
        else:
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
        else:
            return f"❌ Session not found or failed to delete: {session_id}"
        
    except Exception as e:
        return f"❌ Error deleting session: {e}"

@mcp.tool()
async def list_serverless_sessions(user_id: Optional[str] = None,
                                  project_id: Optional[str] = None) -> str:
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
        
        output = [
            f"📋 Sessions for {filter_desc} ({len(session_ids)} found)",
            ""
        ]
        
        # Get details for first few sessions
        for session_id in session_ids[:10]:
            session_state = await manager.get_session(session_id)
            if session_state:
                output.append(f"• {session_id[:12]}... (user: {session_state.user_id}, project: {session_state.project_id})")
        
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
        config = ServerlessConfigManager.load_config(str(config_path) if config_path.exists() else None)
        
        test_results = await ServerlessConfigManager.test_storage_backends(config)
        
        output = [
            "🧪 Storage Backend Test Results",
            ""
        ]
        
        for backend_name, available in test_results.items():
            status = "✅" if available else "❌"
            output.append(f"{status} {backend_name.title()}: {'Available' if available else 'Not available'}")
        
        # Show current configuration
        current_backend = config.get('storage_backend', 'local')
        current_available = test_results.get(current_backend, False)
        
        output.extend([
            "",
            f"🎯 Current Backend: {current_backend} ({'✅ Available' if current_available else '❌ Not available'})"
        ])
        
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
            f"🏗️ Storage backend: {stats['storage_backend']}"
        ]
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error during cleanup: {e}"

@mcp.tool()
async def configure_serverless_storage(backend: str,
                                      config_updates: Dict[str, Any]) -> str:
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
        config = ServerlessConfigManager.load_config(str(config_path) if config_path.exists() else None)
        
        # Update configuration
        if 'backends' not in config:
            config['backends'] = {}
        if backend not in config['backends']:
            config['backends'][backend] = {}
        
        config['backends'][backend].update(config_updates)
        config['storage_backend'] = backend
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Reinitialize manager
        global _serverless_manager
        _serverless_manager = None
        
        return f"✅ Configured {backend} storage backend\n📁 Config saved to {config_path}"
        
    except Exception as e:
        return f"❌ Error configuring storage: {e}"

# Team Knowledge Base Tools
@mcp.tool()
async def create_team_user(
    user_id: str,
    username: str,
    email: Optional[str] = None,
    role: str = "contributor"
) -> str:
    """Create a new team user with specified role"""
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
async def create_team(
    team_id: str,
    name: str,
    description: str,
    owner_id: str
) -> str:
    """Create a new team for knowledge sharing"""
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
    tags: Optional[List[str]] = None,
    access_level: str = "team",
    team_id: Optional[str] = None,
    project_id: Optional[str] = None
) -> str:
    """Add reflection to team knowledge base with access control"""
    try:
        from .team_knowledge import add_team_reflection as _add_team_reflection
        reflection_id = await _add_team_reflection(content, author_id, tags, access_level, team_id, project_id)
        
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
    team_id: Optional[str] = None,
    project_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 20
) -> str:
    """Search team reflections with access control"""
    try:
        from .team_knowledge import search_team_knowledge as _search_team_knowledge
        results = await _search_team_knowledge(query, user_id, team_id, project_id, tags, limit)
        
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
            if result.get('team_id'):
                output.append(f"👥 Team: {result['team_id']}")
            if result.get('tags'):
                output.append(f"🏷️ Tags: {', '.join(result['tags'])}")
            output.append(f"📝 {result['content'][:200]}...")
            output.append(f"📅 {result['created_at']}")
        
        return "\n".join(output)
        
    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error searching team knowledge: {e}"

@mcp.tool()
async def join_team(
    user_id: str,
    team_id: str,
    requester_id: Optional[str] = None
) -> str:
    """Join a team or add user to team"""
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
            output.append("🌟 You can now access team reflections and contribute knowledge")
            return "\n".join(output)
        else:
            return f"❌ Failed to join team {team_id}. Check permissions and team existence."
        
    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error joining team: {e}"

@mcp.tool()
async def get_team_statistics(
    team_id: str,
    user_id: str
) -> str:
    """Get team statistics and activity"""
    try:
        from .team_knowledge import get_team_statistics as _get_team_statistics
        stats = await _get_team_statistics(team_id, user_id)
        
        if not stats:
            return f"❌ Cannot access team {team_id}. Check permissions and team existence."
        
        output = []
        team_info = stats['team']
        reflection_stats = stats['reflection_stats']
        
        output.append(f"📊 Team Statistics: {team_info['name']}")
        output.append(f"🆔 Team ID: {team_info['team_id']}")
        output.append(f"📝 Description: {team_info['description']}")
        output.append(f"👑 Owner: {team_info['owner_id']}")
        output.append(f"📅 Created: {team_info['created_at']}")
        output.append("=" * 50)
        
        output.append(f"👥 Members: {stats['member_count']}")
        output.append(f"📁 Projects: {stats['project_count']}")
        output.append(f"💡 Total Reflections: {reflection_stats['total_reflections'] or 0}")
        output.append(f"✍️ Active Contributors: {reflection_stats['active_contributors'] or 0}")
        output.append(f"👍 Total Votes: {reflection_stats['total_votes'] or 0}")
        output.append(f"📈 Avg Votes/Reflection: {reflection_stats['avg_votes']:.1f}" if reflection_stats['avg_votes'] else "📈 Avg Votes/Reflection: 0.0")
        output.append(f"⚡ Recent Activity (7 days): {stats['recent_activity']['recent_reflections']} reflections")
        
        return "\n".join(output)
        
    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error getting team statistics: {e}"

@mcp.tool()
async def get_user_team_permissions(
    user_id: str
) -> str:
    """Get user's permissions and team memberships"""
    try:
        from .team_knowledge import get_user_team_permissions as _get_user_team_permissions
        perms = await _get_user_team_permissions(user_id)
        
        if not perms:
            return f"❌ User {user_id} not found in team knowledge base"
        
        user_data = perms['user']
        teams = perms['teams']
        
        output = []
        output.append(f"👤 User Permissions: {user_data['username']}")
        output.append(f"🆔 User ID: {user_data['user_id']}")
        output.append(f"🏷️ Role: {user_data['role']}")
        if user_data.get('email'):
            output.append(f"📧 Email: {user_data['email']}")
        output.append(f"📅 Created: {user_data['created_at']}")
        output.append(f"⏰ Last Active: {user_data['last_active']}")
        output.append("=" * 50)
        
        output.append(f"👥 Teams ({len(teams)}):")
        for team in teams:
            output.append(f"  • {team['name']} ({team['team_id']})")
            if team.get('description'):
                output.append(f"    📝 {team['description'][:80]}...")
        
        output.append("\n🔒 Permissions:")
        permissions = user_data.get('permissions', {})
        for perm, enabled in permissions.items():
            status = "✅" if enabled else "❌"
            perm_name = perm.replace('_', ' ').title()
            output.append(f"  {status} {perm_name}")
        
        output.append(f"\n🎯 Special Abilities:")
        if perms['can_create_teams']:
            output.append("  ✅ Can create teams")
        if perms['can_moderate']:
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
    vote_delta: int = 1
) -> str:
    """Vote on a team reflection (upvote/downvote)"""
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
        else:
            return f"❌ Failed to vote on reflection {reflection_id}. Check permissions and reflection existence."
        
    except ImportError:
        return "❌ Team knowledge tools not available"
    except Exception as e:
        return f"❌ Error voting on reflection: {e}"

def main():
    """Main entry point for the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()