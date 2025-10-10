# Session Management MCP Server

[![Code style: crackerjack](https://img.shields.io/badge/code%20style-crackerjack-000042)](https://github.com/lesleslie/crackerjack)
[![Python: 3.13+](https://img.shields.io/badge/python-3.13%2B-green)](https://www.python.org/downloads/)
![Coverage](https://img.shields.io/badge/coverage-34.6%25-red)

A dedicated MCP server that provides comprehensive session management functionality for Claude Code sessions across any project.

## Features

- **🚀 Session Initialization**: Complete setup with UV dependency management, project analysis, and automation tools
- **🔍 Quality Checkpoints**: Mid-session quality monitoring with workflow analysis and optimization recommendations
- **🏁 Session Cleanup**: Comprehensive cleanup with learning capture and handoff file creation
- **📊 Status Monitoring**: Real-time session status and project context analysis
- **⚡ Auto-Generated Shortcuts**: Automatically creates `/start`, `/checkpoint`, and `/end` Claude Code slash commands

## 🚀 Automatic Session Management (NEW!)

**For Git Repositories:**

- ✅ **Automatic initialization** when Claude Code connects
- ✅ **Automatic cleanup** when session ends (quit, crash, or network failure)
- ✅ **Intelligent auto-compaction** during checkpoints
- ✅ **Zero manual intervention** required

**For Non-Git Projects:**

- 📝 Use `/start` for manual initialization
- 📝 Use `/end` for manual cleanup
- 📝 Full session management features available on-demand

The server automatically detects git repositories and provides seamless session lifecycle management with crash resilience and network failure recovery. Non-git projects retain manual control for flexible workflow management.

## Available MCP Tools

**Total: 70+ specialized tools** organized into 10 functional categories:

### 🎯 Core Session Management (8 tools)

- **`start`** - Comprehensive session initialization with project analysis, UV sync, and memory setup
- **`checkpoint`** - Mid-session quality assessment with V2 scoring system and workflow analysis
- **`end`** - Complete session cleanup with learning capture and handoff documentation
- **`status`** - Current session overview with health checks and diagnostics
- **`permissions`** - Manage trusted operations to reduce permission prompts
- **`auto_compact`** - Automatic context window compaction when needed
- **`quality_monitor`** - Real-time quality monitoring and tracking
- **`session_welcome`** - Session connection information and continuity

### 🧠 Memory & Conversation Search (14 tools)

**Semantic Search & Retrieval**:

- **`reflect_on_past`** / **`search_reflections`** - Semantic search through past conversations using local AI embeddings (all-MiniLM-L6-v2)
- **`quick_search`** - Fast overview search with count and top results
- **`search_summary`** - Aggregated insights without individual results
- **`get_more_results`** - Pagination support for large result sets

**Targeted Search**:

- **`search_by_file`** - Find conversations about specific files
- **`search_by_concept`** - Search for development concepts and patterns
- **`search_code`** - Code-specific search with pattern matching
- **`search_errors`** - Search error patterns and resolutions
- **`search_temporal`** - Time-based search queries

**Storage & Management**:

- **`store_reflection`** - Store insights with tagging and embeddings
- **`reflection_stats`** - Memory system statistics and health
- **`reset_reflection_database`** - Reset/rebuild memory database

**Advanced**:

- **`_optimize_search_results`** - Token-aware result optimization

### 📊 Crackerjack Quality Integration (11 tools)

**Command Execution**:

- **`execute_crackerjack_command`** / **`crackerjack_run`** - Execute crackerjack with analytics
- **`crackerjack_help`** - Comprehensive help for choosing commands

**Metrics & Analysis**:

- **`crackerjack_metrics`** - Quality metrics trends over time
- **`crackerjack_quality_trends`** - Trend analysis with actionable insights
- **`get_crackerjack_quality_metrics`** - Detailed quality metric extraction
- **`get_crackerjack_results_history`** - Command execution history

**Pattern Detection**:

- **`crackerjack_patterns`** - Test failure pattern analysis
- **`analyze_crackerjack_test_patterns`** - Deep test pattern analysis
- **`crackerjack_history`** - Execution history with trends

**Health & Status**:

- **`crackerjack_health_check`** - Integration health diagnostics

### 🤖 LLM Provider Management (5 tools)

- **`list_llm_providers`** - List available LLM providers and models
- **`test_llm_providers`** - Test provider availability and functionality
- **`generate_with_llm`** - Generate text using specified provider
- **`chat_with_llm`** - Have conversations with LLM providers
- **`configure_llm_provider`** - Configure provider credentials and settings

### ☁️ Serverless Session Management (8 tools)

- **`create_serverless_session`** - Create session with external storage
- **`get_serverless_session`** - Retrieve session state
- **`update_serverless_session`** - Update session state
- **`delete_serverless_session`** - Delete session
- **`list_serverless_sessions`** - List sessions by user/project
- **`test_serverless_storage`** - Test storage backend availability
- **`cleanup_serverless_sessions`** - Clean up expired sessions
- **`configure_serverless_storage`** - Configure storage backends (Redis, S3, local)

### 👥 Team Collaboration & Knowledge Sharing (4 tools)

- **`create_team`** - Create team for knowledge sharing
- **`search_team_knowledge`** - Search team reflections with access control
- **`get_team_statistics`** - Team activity and statistics
- **`vote_on_reflection`** - Vote on team reflections (upvote/downvote)

### 🔗 Multi-Project Coordination (4 tools)

- **`create_project_group`** - Create project groups for coordination
- **`add_project_dependency`** - Add dependency relationships between projects
- **`search_across_projects`** - Search conversations across related projects
- **`get_project_insights`** - Cross-project insights and collaboration opportunities

### 📱 Application & Activity Monitoring (5 tools)

- **`start_app_monitoring`** - Start IDE and browser activity monitoring
- **`stop_app_monitoring`** - Stop activity monitoring
- **`get_activity_summary`** - Activity summary over time period
- **`get_context_insights`** - Generate insights from development behavior
- **`get_active_files`** - Get recently active files

### 🔄 Interruption & Context Management (7 tools)

- **`start_interruption_monitoring`** - Smart detection and context preservation
- **`stop_interruption_monitoring`** - Stop interruption monitoring
- **`create_session_context`** - Create session context snapshot
- **`preserve_current_context`** - Preserve context during interruptions
- **`restore_session_context`** - Restore preserved session context
- **`get_interruption_history`** - Interruption history and statistics
- **`get_interruption_statistics`** - Comprehensive interruption stats

### ⏰ Natural Language Scheduling (5 tools)

- **`create_natural_reminder`** - Create reminder from natural language
- **`list_user_reminders`** - List pending reminders
- **`cancel_user_reminder`** - Cancel specific reminder
- **`start_reminder_service`** - Start background reminder service
- **`stop_reminder_service`** - Stop reminder service

### 🌳 Git Worktree Management (3 tools)

- **`git_worktree_add`** - Create new git worktree
- **`git_worktree_remove`** - Remove existing worktree
- **`git_worktree_switch`** - Switch context between worktrees with session preservation

### 🔍 Advanced Search Features (3 tools)

- **`advanced_search`** - Faceted search with filtering
- **`search_suggestions`** - Search completion suggestions
- **`get_search_metrics`** - Search and activity metrics

All tools use **local processing** for privacy, with **DuckDB vector storage** (FLOAT[384] embeddings) and **ONNX-based semantic search** requiring no external API calls.

## 🚀 Integration with Crackerjack

Session-mgmt includes deep integration with [Crackerjack](https://github.com/lesleslie/crackerjack), the AI-driven Python development platform:

**Integrated Features:**

- **📊 Quality Metrics Tracking**: Automatically captures and tracks Crackerjack quality scores over time
- **🧪 Test Result Monitoring**: Learns from test patterns, failures, and successful fixes
- **🔍 Error Pattern Recognition**: Remembers how specific errors were resolved and suggests solutions
- **📝 Command History Analysis**: Tracks which Crackerjack commands are most effective for different scenarios
- **🎯 Progress Intelligence**: Predicts completion times based on historical data

**Why Use Both Together:**

- **Crackerjack**: Enforces code quality, runs tests, manages releases, and provides AI auto-fixing
- **Session-mgmt**: Remembers what worked, tracks progress evolution, and maintains context
- **Synergy**: Creates an intelligent development environment that learns from every interaction

**Example Integrated Workflow:**

1. 🚀 **Session-mgmt `start`** - Sets up your session with accumulated context from previous work
1. 🔧 **Crackerjack runs** quality checks and applies AI agent fixes to resolve issues
1. 💾 **Session-mgmt captures** successful patterns, quality improvements, and error resolutions
1. 🧠 **Next session starts** with all accumulated knowledge and learned patterns
1. 📈 **Continuous improvement** as both systems get smarter with each interaction

**Technical Integration:**
The `crackerjack_integration.py` module (50KB+) provides:

- Real-time progress tracking during Crackerjack operations
- Quality metric extraction and trend analysis
- Test result pattern detection and storage
- Error resolution pattern matching for faster fixes
- Command effectiveness scoring for workflow optimization

**Configuration Example:**

```json
{
  "mcpServers": {
    "crackerjack": {
      "command": "python",
      "args": ["-m", "crackerjack", "--start-mcp-server"]
    },
    "session-mgmt": {
      "command": "python",
      "args": ["-m", "session_mgmt_mcp.server"]
    }
  }
}
```

The integration is automatic once both servers are configured - they coordinate through the MCP protocol without requiring additional setup.

### Crackerjack MCP Tool Usage

When using Crackerjack through MCP tools, follow these patterns for correct usage:

#### ✅ Correct Usage

```python
# Run tests with AI auto-fix
await crackerjack_run(command="test", ai_agent_mode=True)

# Run all checks with verbose output
await crackerjack_run(
    command="check",
    args="--verbose",
    ai_agent_mode=True,
    timeout=600,  # 10 minutes for complex fixes
)

# Dry-run to preview fixes
await crackerjack_run(command="test", args="--dry-run", ai_agent_mode=True)

# Run security checks
await execute_crackerjack_command(command="security")

# Run with custom iteration limit
await crackerjack_run(command="test", args="--max-iterations 15", ai_agent_mode=True)
```

#### ❌ Common Mistakes

```python
# WRONG - Don't put flags in command parameter
await crackerjack_run(command="--ai-fix -t")

# WRONG - Don't put --ai-fix in args
await crackerjack_run(command="test", args="--ai-fix")

# WRONG - Don't use CLI flag syntax
await execute_crackerjack_command(command="-t --verbose")

# CORRECT
await crackerjack_run(command="test", ai_agent_mode=True)
```

#### Parameters

- **`command`** (required): Semantic command name

  - Valid: `test`, `lint`, `check`, `format`, `security`, `complexity`, `all`
  - Invalid: `--ai-fix`, `-t`, any CLI flags

- **`ai_agent_mode`** (optional, default False): Enable AI-powered auto-fix

  - Replaces the `--ai-fix` CLI flag
  - Requires Anthropic API key configured in crackerjack
  - Max 10 iterations by default (configurable via `--max-iterations` in args)

- **`args`** (optional): Additional arguments

  - Examples: `--verbose`, `--dry-run`, `--max-iterations 5`
  - Do NOT include `--ai-fix` here - use `ai_agent_mode=True` instead

- **`working_directory`** (optional, default "."): Working directory for command execution

- **`timeout`** (optional, default 300): Timeout in seconds

  - Increase for complex auto-fix operations (e.g., 600-1200 seconds)

#### Auto-Fix Workflow

When `ai_agent_mode=True`, Crackerjack will:

1. Run pre-commit hooks and detect issues
1. Apply AI-powered fixes using Claude AI
1. Re-run hooks to verify fixes
1. Iterate up to 10 times (or custom `--max-iterations`) until convergence
1. Stop when all hooks pass or no progress can be made

**Configuration Requirements:**

```bash
# 1. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Configure adapter in settings/adapters.yml
ai: claude
```

See [Crackerjack AUTO_FIX_GUIDE.md](https://github.com/lesleslie/crackerjack/blob/main/docs/AUTO_FIX_GUIDE.md) for detailed auto-fix documentation.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/lesleslie/session-mgmt-mcp.git
cd session-mgmt-mcp

# Install with all dependencies (development + testing)
uv sync --group dev

# Or install minimal production dependencies only
uv sync

# Or use pip (for production only)
pip install session-mgmt-mcp
```

### MCP Configuration

Add to your project's `.mcp.json` file:

```json
{
  "mcpServers": {
    "session-mgmt": {
      "command": "python",
      "args": ["-m", "session_mgmt_mcp.server"],
      "cwd": "/path/to/session-mgmt-mcp",
      "env": {
        "PYTHONPATH": "/path/to/session-mgmt-mcp"
      }
    }
  }
}
```

### Alternative: Use Script Entry Point

If installed with pip/uv, you can use the script entry point:

```json
{
  "mcpServers": {
    "session-mgmt": {
      "command": "session-mgmt-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

### Dependencies

**Core Requirements** (from pyproject.toml):

- Python 3.13+
- `fastmcp>=2` - MCP server framework
- `duckdb>=0.9` - Conversation storage with vector support
- `numpy>=1.24` - Numerical operations for embeddings
- `pydantic>=2.0` - Data validation and settings management
- `tiktoken>=0.5` - Token counting and optimization
- `crackerjack` - Code quality and testing integration
- `onnxruntime>=1.15` - Local ONNX model inference
- `transformers>=4.21` - Tokenizer for embedding models
- `psutil>=7.0.0` - System and process utilities
- `rich>=14.1.0` - Terminal formatting and output
- `structlog>=25.4` - Structured logging
- `pydantic-settings>=2.0` - Settings management
- `tomli>=2.2.1` - TOML parsing
- `typer>=0.17.4` - CLI interface

**Development Dependencies** (install with `--group dev`):

- `pytest>=7` + `pytest-asyncio>=0.21` - Testing framework
- `pytest-cov>=4`, `pytest-benchmark>=4` - Coverage and benchmarking
- `pytest-xdist>=3`, `pytest-timeout>=2.1` - Parallel execution and timeouts
- `hypothesis>=6.70` - Property-based testing
- `coverage>=7` - Code coverage analysis
- `pytest-mock>=3.10` - Mocking utilities
- `psutil>=5.9` - Process monitoring

Install all dependencies:

```bash
# Full installation with development + testing tools
uv sync --group dev

# Minimal installation (production only)
uv sync

# Install from PyPI with pip
pip install session-mgmt-mcp

# Add to existing UV project
uv add session-mgmt-mcp

# Add with development dependencies
uv add session-mgmt-mcp --group dev
```

## Usage

Once configured, the following slash commands become available in Claude Code:

### Primary Session Commands

- `/session-mgmt:start` - Full session initialization with workspace verification
- `/session-mgmt:checkpoint` - Quality monitoring checkpoint with scoring
- `/session-mgmt:end` - Complete session cleanup with learning capture
- `/session-mgmt:status` - Current status overview with health checks

### Auto-Generated Shortcuts

The first time you run `/session-mgmt:start`, convenient shortcuts are automatically created:

- **`/start`** → `/session-mgmt:start` - Quick session initialization
- **`/checkpoint [name]`** → `/session-mgmt:checkpoint` - Create named checkpoints
- **`/end`** → `/session-mgmt:end` - Quick session cleanup

> These shortcuts are created in `~/.claude/commands/` and work across all projects

### Memory & Search Commands

- `/session-mgmt:reflect_on_past` - Search past conversations with semantic similarity
- `/session-mgmt:store_reflection` - Store important insights with tagging
- `/session-mgmt:quick_search` - Fast search with overview results
- `/session-mgmt:permissions` - Manage trusted operations

### Advanced Usage

**Running Server Directly** (for development):

```bash
python -m session_mgmt_mcp.server
# or
session-mgmt-mcp
```

**Testing Memory Features**:

```bash
# The memory system automatically stores conversations and provides:
# - Semantic search across all past conversations
# - Local embedding generation (no external API needed)
# - Cross-project conversation history
# - Time-decay prioritization for recent content
```

## Memory System Architecture

### Built-in Conversation Memory

- **Local Storage**: DuckDB database at `~/.claude/data/reflection.duckdb`
- **Embeddings**: Local ONNX models (all-MiniLM-L6-v2) for semantic search
- **Vector Storage**: FLOAT[384] arrays for similarity matching
- **No External Dependencies**: Everything runs locally for privacy
- **Cross-Project History**: Conversations tagged by project context

### Search Capabilities

- **Semantic Search**: Vector similarity with customizable thresholds
- **Text Fallback**: Standard text search when embeddings unavailable
- **Time Decay**: Recent conversations prioritized in results
- **Project Context**: Filter searches by project or search across all
- **Batch Operations**: Efficient bulk storage and retrieval

## Data Storage

This server manages its data locally in the user's home directory:

- **Memory Storage**: `~/.claude/data/reflection.duckdb`
- **Session Logs**: `~/.claude/logs/`
- **Configuration**: Uses pyproject.toml and environment variables

## Recommended Session Workflow

1. **Initialize Session**: `/session-mgmt:start`

   - UV dependency synchronization
   - Project context analysis and health monitoring
   - Session quality tracking setup
   - Memory system initialization
   - Permission system setup

1. **Monitor Progress**: `/session-mgmt:checkpoint` (every 30-45 minutes)

   - Real-time quality scoring
   - Workflow optimization recommendations
   - Progress tracking and goal alignment
   - Automatic Git checkpoint commits

1. **Search Past Work**: `/session-mgmt:reflect_on_past`

   - Semantic search through project history
   - Find relevant past conversations and solutions
   - Build on previous insights

1. **Store Important Insights**: `/session-mgmt:store_reflection`

   - Capture key learnings and solutions
   - Tag insights for easy retrieval
   - Build project knowledge base

1. **End Session**: `/session-mgmt:end`

   - Final quality assessment
   - Learning capture across categories
   - Session handoff file creation
   - Memory persistence and cleanup

## Benefits

### Comprehensive Coverage

- **Session Quality**: Real-time monitoring and optimization
- **Memory Persistence**: Cross-session conversation retention
- **Project Structure**: Context-aware development workflows

### Reduced Friction

- **Single Command Setup**: One `/session-mgmt:start` sets up everything
- **Local Dependencies**: No external API calls or services required
- **Intelligent Permissions**: Reduces repeated permission prompts
- **Automated Workflows**: Structured processes for common tasks

### Enhanced Productivity

- **Quality Scoring**: Guides session effectiveness
- **Built-in Memory**: Enables building on past work automatically
- **Project Templates**: Accelerates development setup
- **Knowledge Persistence**: Maintains context across sessions

## Documentation

The project documentation is organized into the following categories:

### For Developers

- **[Testing Guide](docs/TESTING.md)** - Comprehensive testing strategy, status, and standards
- **[Parameter Validation](docs/developer/PARAMETER_VALIDATION.md)** - Pydantic parameter validation guide
- **[Architecture](docs/developer/ARCHITECTURE.md)** - System architecture and design patterns
- **[Integration](docs/developer/INTEGRATION.md)** - Integration patterns and best practices

### For Users

- **[Quick Start](docs/user/QUICK_START.md)** - Getting started guide
- **[Configuration](docs/user/CONFIGURATION.md)** - Setup and configuration options
- **[Deployment](docs/user/DEPLOYMENT.md)** - Deployment and production setup
- **[MCP Tools Reference](docs/user/MCP_TOOLS_REFERENCE.md)** - Complete tool documentation

### Features

- **[AI Integration](docs/features/AI_INTEGRATION.md)** - AI integration strategies and patterns
- **[Token Optimization](docs/features/TOKEN_OPTIMIZATION.md)** - Token management and chunking features
- **[Auto Lifecycle](docs/features/AUTO_LIFECYCLE.md)** - Automatic session management
- **[Crackerjack Integration](docs/CRACKERJACK.md)** - Comprehensive code quality integration
- **[Selective Auto-Store](docs/features/SELECTIVE_AUTO_STORE.md)** - Smart reflection storage

### Development & Migration

- **[ACB Config Migration Summary](docs/ACB_CONFIG_MIGRATION_SUMMARY.md)** - Phase 1 ACB integration (complete)
- **[Comprehensive Improvement Plan](docs/SESSION-MGMT-MCP-COMPREHENSIVE-IMPROVEMENT-PLAN.md)** - 16-week ACB transformation roadmap
- **[Config Migration Strategy](docs/config-migration-strategy.md)** - Detailed migration approach analysis

### Reference

- **[MCP Schema Reference](docs/reference/MCP_SCHEMA_REFERENCE.md)** - MCP protocol schemas
- **[Slash Command Shortcuts](docs/reference/slash-command-shortcuts.md)** - Command reference

## Troubleshooting

### Common Issues

- **Memory/embedding issues**: Ensure all dependencies are installed with `uv sync` (embeddings are now included by default)
- **Path errors**: Ensure `cwd` and `PYTHONPATH` are set correctly in `.mcp.json`
- **Permission issues**: Use `/session-mgmt:permissions` to trust operations
- **Project context**: Analyze current project health and structure

### Debug Mode

```bash
# Run with verbose logging
PYTHONPATH=/path/to/session-mgmt-mcp python -m session_mgmt_mcp.server --debug
```
