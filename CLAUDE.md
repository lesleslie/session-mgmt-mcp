# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Claude Session Management MCP (Model Context Protocol) server that provides comprehensive session management functionality for Claude Code across any project. It operates as a standalone MCP server with its own isolated environment to avoid dependency conflicts.

## Development Commands

### Installation & Setup
```bash
# Install dependencies using UV (recommended)
uv sync

# Alternative: Install using pip
pip install -e .

# Install development dependencies
uv sync --group dev
```

### Running the Server
```bash
# Run directly as a module
python -m session_mgmt_mcp.server

# Or use the console script
session-mgmt-mcp --start-server

# Or use UV to run
uvx session-mgmt-mcp --start-server
```

### Testing & Development
```bash
# Run tests using crackerjack (if available)
crackerjack

# Check MCP server functionality
python -c "from session_mgmt_mcp.server import mcp; print('MCP server loads successfully')"

# Validate pyproject.toml
uv build --check
```

## Architecture Overview

### Core Components

1. **server.py** (Lines 1-1565): Main MCP server implementation
   - FastMCP 2.0 server setup and tool registration
   - Session lifecycle management (init, checkpoint, end, status)
   - Permissions management system with trusted operations
   - Global workspace validation and project analysis
   - Git integration for automatic checkpoint commits

2. **reflection_tools.py** (Lines 1-411): Memory & conversation search system
   - DuckDB-based conversation storage with embeddings
   - Local ONNX semantic search (all-MiniLM-L6-v2 model)
   - Reflection storage and retrieval
   - Fallback text search when embeddings unavailable

3. **__init__.py**: Package initialization and version information

### Key Design Patterns

- **Singleton Pattern**: SessionPermissionsManager ensures consistent session state
- **Database Layer**: ReflectionDatabase manages DuckDB operations with async/await
- **Graceful Degradation**: System works with reduced functionality if dependencies missing
- **MCP Tool Registration**: All functions exposed via FastMCP decorators (@mcp.tool(), @mcp.prompt())

### Session Management Workflow

1. **Initialization** (`init` tool):
   - Validates global workspace at ~/Projects/claude/
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
    "session-management": {
      "command": "uvx",
      "args": ["session-mgmt-mcp", "--start-server"],
      "env": {}
    }
  }
}
```

### Global Workspace Dependencies
The server integrates with the global Claude workspace system:
- **~/Projects/claude/toolkits/**: Session management and verification toolkits
- **~/Projects/claude/logs/**: Structured logging output
- **~/.claude/data/**: Reflection database storage

### Environment Variables
- `PWD`: Used to detect current working directory
- `PYTHONPATH`: Extended to include global toolkits when available

## Development Notes

### Dependencies & Isolation
- Uses isolated virtual environment to prevent conflicts
- Required: `fastmcp>=2.0.0`, `duckdb>=0.9.0`
- Optional: `onnxruntime`, `transformers` (for semantic search)
- Falls back gracefully when optional dependencies unavailable

### Error Handling Strategy
- Structured logging with context preservation (server.py:26-78)
- Health checks validate all system components (server.py:1020-1071)
- Graceful degradation when external services unavailable
- Permission errors handled with trusted operations system

### Testing Strategy
- Health check system provides runtime validation
- MCP tool registration can be tested via direct imports
- Database operations include error recovery mechanisms
- Integration testing via example.mcp.json configuration

### Code Quality Metrics
The system includes built-in quality scoring based on:
- Project health indicators (40%): tests, docs, git, dependencies
- Permissions management (20%): trusted operations count
- Session management availability (20%): toolkit integration
- Tool availability (20%): UV package manager, etc.

## Common Development Tasks

### Adding New MCP Tools
1. Define function with `@mcp.tool()` decorator in server.py
2. Add corresponding prompt with `@mcp.prompt()` for slash command support
3. Include tool in SESSION_COMMANDS dictionary for documentation
4. Update status() tool to report new functionality

### Extending Memory System
1. Add new table schemas in reflection_tools.py:_ensure_tables()
2. Implement storage/retrieval methods in ReflectionDatabase class
3. Add corresponding MCP tools in server.py
4. Update reflection_stats() to include new metrics

### Permission System Extensions
1. Add new operation constants to SessionPermissionsManager
2. Update trust_operation() calls in relevant tools
3. Modify permissions() tool to display new operation types
4. Update health checks to validate new permissions