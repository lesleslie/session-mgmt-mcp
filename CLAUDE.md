# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Claude Session Management MCP (Model Context Protocol) server that provides comprehensive session management functionality for Claude Code across any project. It operates as a standalone MCP server with its own isolated environment to avoid dependency conflicts.

## Development Commands

### Installation & Setup
```bash
# Install development dependencies
uv sync --group dev

# Run directly as a module
python -m session_mgmt_mcp.server
```

### Testing & Development
```bash
# Run comprehensive test suite with coverage
python tests/scripts/run_tests.py

# Quick smoke tests for development
python tests/scripts/run_tests.py --quick

# Run specific test categories
python tests/scripts/run_tests.py --unit           # Unit tests only
python tests/scripts/run_tests.py --integration    # Integration tests only
python tests/scripts/run_tests.py --performance    # Performance tests only
python tests/scripts/run_tests.py --security       # Security tests only

# Run single test file
pytest tests/unit/test_session_permissions.py -v

# Run with parallel execution
python tests/scripts/run_tests.py --parallel

# Coverage report only (no test execution)
python tests/scripts/run_tests.py --coverage-only

# Development with debugging
python tests/scripts/run_tests.py --verbose --no-cleanup

# Check MCP server functionality
python -c "from session_mgmt_mcp.server import mcp; print('MCP server loads successfully')"

# Test memory system
python -c "from session_mgmt_mcp.reflection_tools import ReflectionDatabase; print('Memory system available')"
```

## Architecture Overview

### Core Components

1. **server.py**: Main MCP server implementation
   - FastMCP server setup and tool registration
   - Session lifecycle management (init, checkpoint, end, status)
   - Permissions management system with trusted operations
   - Global workspace validation and project analysis
   - Git integration for automatic checkpoint commits

2. **reflection_tools.py**: Memory & conversation search system
   - DuckDB-based conversation storage with embeddings
   - Local ONNX semantic search (all-MiniLM-L6-v2 model)
   - Reflection storage and retrieval
   - Fallback text search when embeddings unavailable

3. **crackerjack_integration.py**: Integration layer with Crackerjack code quality tools

4. **context_manager.py**: Context management and session tracking

5. **memory_optimizer.py**: Memory usage optimization for long-running sessions

6. **team_knowledge.py**: Shared knowledge management across team members

### Additional Components

- **app_monitor.py**: Application monitoring and health checks
- **interruption_manager.py**: Graceful handling of session interruptions
- **llm_providers.py**: Integration with various LLM providers
- **natural_scheduler.py**: Natural language-based task scheduling
- **search_enhanced.py**: Enhanced search capabilities
- **serverless_mode.py**: Serverless execution mode support

### Key Design Patterns

- **Database Layer**: ReflectionDatabase manages DuckDB operations with async/await
- **Graceful Degradation**: System works with reduced functionality if dependencies missing
- **MCP Tool Registration**: All functions exposed via FastMCP decorators (@mcp.tool(), @mcp.prompt())

### Session Management Workflow

1. **Initialization** (`init` tool):
   - Sets up ~/.claude directory structure
   - Syncs UV dependencies and generates requirements.txt
   - Analyzes project context and calculates maturity score
   - Sets up session permissions and auto-checkpoints

2. **Quality Monitoring** (`checkpoint` tool):
   - Calculates multi-factor quality score (project health, permissions, tools)
   - Creates automatic Git commits with checkpoint metadata
   - Provides workflow optimization recommendations

3. **Session Cleanup** (`end` tool):
   - Generates session handoff documentation
   - Performs final quality assessment
   - Cleans up session artifacts

### Memory System Architecture

- **Embedding Storage**: Uses DuckDB with FLOAT[384] arrays for vector similarity
- **Dual Search**: Semantic search with ONNX + fallback text search
- **Async Operations**: All database operations use executor threads
- **Project Context**: Conversations tagged with project metadata

## Configuration & Integration

### MCP Configuration (.mcp.json)
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

### Directory Structure
The server uses the ~/.claude directory for data storage:
- **~/.claude/logs/**: Session management logging
- **~/.claude/data/**: Reflection database storage

### Environment Variables
- `PWD`: Used to detect current working directory

## Development Notes

### Dependencies & Isolation
- Uses isolated virtual environment to prevent conflicts
- Required: `fastmcp>=2.0.0`, `duckdb>=0.9.0`
- Optional: `onnxruntime`, `transformers` (for semantic search)
- Falls back gracefully when optional dependencies unavailable

### Testing Architecture
The project uses a comprehensive pytest-based testing framework with four main test categories:

**Test Structure:**
- **Unit Tests** (`tests/unit/`): Core functionality testing
  - Session permissions and lifecycle management
  - Reflection database operations with async/await patterns
  - Mock fixtures for isolated component testing

- **Integration Tests** (`tests/integration/`): Complete MCP workflow validation
  - End-to-end session management workflows
  - MCP tool registration and execution
  - Database integrity with concurrent operations

- **Performance Tests** (`tests/performance/`): Database and system performance
  - Bulk operation benchmarking with memory profiling
  - Concurrent access patterns under load
  - Performance regression detection with baselines

- **Security Tests** (`tests/security/`): Permission system validation
  - SQL injection prevention for DuckDB operations
  - Input sanitization across MCP tool parameters
  - Rate limiting and permission boundary testing

**Key Testing Features:**
- Async/await support for MCP server testing
- Temporary database fixtures with automatic cleanup
- Data factories for realistic test data generation
- Performance metrics collection and baseline comparison
- Mock MCP server creation for isolated testing

## Available MCP Tools

### Session Management Tools
- **`init`** (`mcp__session-mgmt__init`) - Complete session initialization with workspace verification
- **`checkpoint`** (`mcp__session-mgmt__checkpoint`) - Mid-session quality assessment and optimization
- **`end`** (`mcp__session-mgmt__end`) - Complete session cleanup with learning capture
- **`status`** (`mcp__session-mgmt__status`) - Current session status with health checks
- **`permissions`** (`mcp__session-mgmt__permissions`) - Manage trusted operations

### Memory & Reflection Tools
- **`reflect_on_past`** (`mcp__session-mgmt__reflect_on_past`) - Search past conversations with semantic similarity
- **`store_reflection`** (`mcp__session-mgmt__store_reflection`) - Store important insights with tagging
- **`search_nodes`** (`mcp__session-mgmt__search_nodes`) - Advanced search through stored knowledge
- **`quick_search`** (`mcp__session-mgmt__quick_search`) - Fast overview search with count and top result
- **`search_summary`** (`mcp__session-mgmt__search_summary`) - Get aggregated insights without individual results
- **`get_more_results`** (`mcp__session-mgmt__get_more_results`) - Pagination support for large result sets
- **`search_by_file`** (`mcp__session-mgmt__search_by_file`) - Find conversations about specific files
- **`search_by_concept`** (`mcp__session-mgmt__search_by_concept`) - Search for development concepts
- **`reflection_stats`** (`mcp__session-mgmt__reflection_stats`) - Get statistics about stored knowledge

## Server Architecture Notes

⚠️ **Current Issue**: server.py is 4756 lines - violates clean code principles

### Recommended Modularization
The server should be refactored into focused modules:

```
session_mgmt_mcp/
├── server.py (main FastMCP setup only)
├── core/
│   ├── mcp_server.py (FastMCP configuration)
│   └── session_manager.py (session state)
├── tools/
│   ├── session_tools.py (init, checkpoint, end, status)
│   ├── memory_tools.py (reflect_on_past, store_reflection, etc.)
│   └── permission_tools.py (permissions management)
├── utils/
│   ├── logging.py (SessionLogger)
│   └── git_operations.py (Git commit functions)
└── reflection_tools.py (memory database - keep as-is)
```

## Development Guidelines

### Adding New MCP Tools
1. Define function with `@mcp.tool()` decorator in appropriate tools/ module
2. Add corresponding prompt with `@mcp.prompt()` for slash command support
3. Import and register in main server.py
4. Update status() tool to report new functionality
5. Add tests in appropriate test category

### Extending Memory System
1. Add new table schemas in reflection_tools.py:_ensure_tables()
2. Implement storage/retrieval methods in ReflectionDatabase class
3. Add corresponding MCP tools in tools/memory_tools.py
4. Update reflection_stats() to include new metrics
5. Add performance tests for new operations

### Testing New Features
1. Add unit tests for individual functions
2. Add integration tests for MCP tool workflows
3. Add performance tests for database operations
4. Add security tests for input validation
5. Ensure 85% minimum coverage is maintained

