# AGENTS.md

This file documents AI agent interaction patterns and tooling for the Session Management MCP Server.

## Overview

This project provides a comprehensive MCP (Model Context Protocol) server that enables sophisticated AI session management across any development project. It offers 40+ specialized tools for memory management, context preservation, project coordination, and development workflow optimization.

## MCP Server Integration

### Server Configuration

Add to your `.mcp.json`:

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

### Quick Start for AI Agents

```bash
# Initialize session with comprehensive setup
uvx session-mgmt-mcp init

# Mid-session quality checkpoint
uvx session-mgmt-mcp checkpoint

# End session with cleanup and handoff
uvx session-mgmt-mcp end
```

## Available MCP Tools

### 🚀 Session Management Tools

| Tool | Purpose | AI Usage Pattern |
|------|---------|------------------|
| `mcp__session-mgmt__init` | Complete session initialization | Start of any new project work |
| `mcp__session-mgmt__checkpoint` | Mid-session quality assessment | During development for progress check |
| `mcp__session-mgmt__end` | Session cleanup with learning capture | End of work session |
| `mcp__session-mgmt__status` | Current session status with health checks | When assessing project state |
| `mcp__session-mgmt__permissions` | Manage trusted operations | Configure agent permissions |

### 🧠 Memory & Knowledge Tools

| Tool | Purpose | AI Usage Pattern |
|------|---------|------------------|
| `mcp__session-mgmt__reflect_on_past` | Search past conversations with semantic similarity | Find previous solutions to similar problems |
| `mcp__session-mgmt__store_reflection` | Store important insights with tagging | Capture key learnings and decisions |
| `mcp__session-mgmt__quick_search` | Fast overview search with count and top result | Quick context gathering |
| `mcp__session-mgmt__search_summary` | Get aggregated insights without individual results | High-level pattern analysis |
| `mcp__session-mgmt__search_by_file` | Find conversations about specific files | File-focused development context |
| `mcp__session-mgmt__search_by_concept` | Search for development concepts | Conceptual knowledge retrieval |

### 🔍 Advanced Search Tools

| Tool | Purpose | AI Usage Pattern |
|------|---------|------------------|
| `mcp__session-mgmt__search_code` | AST-based code pattern search | Find code patterns and examples |
| `mcp__session-mgmt__search_errors` | Error pattern and debugging context search | Troubleshooting and debugging |
| `mcp__session-mgmt__search_temporal` | Time-based conversation search | Context from specific time periods |
| `mcp__session-mgmt__advanced_search` | Faceted search with multiple filters | Complex knowledge retrieval |

### 🎯 Context & Optimization Tools

| Tool | Purpose | AI Usage Pattern |
|------|---------|------------------|
| `mcp__session-mgmt__auto_load_context` | Automatically load relevant conversations | Smart context preparation |
| `mcp__session-mgmt__get_context_summary` | Summary of current development context | Project situational awareness |
| `mcp__session-mgmt__compress_memory` | Optimize memory usage | Long-running session maintenance |
| `mcp__session-mgmt__auto_compact` | Automated conversation compaction | Memory optimization |

### 🔧 Integration Tools

| Tool | Purpose | AI Usage Pattern |
|------|---------|------------------|
| `mcp__session-mgmt__execute_crackerjack_command` | Execute code quality commands | Quality analysis and improvement |
| `mcp__session-mgmt__get_crackerjack_results_history` | Historical quality metrics | Trend analysis |
| `mcp__session-mgmt__analyze_crackerjack_test_patterns` | Test failure pattern analysis | Testing and debugging |

### 👥 Collaboration Tools

| Tool | Purpose | AI Usage Pattern |
|------|---------|------------------|
| `mcp__session-mgmt__create_team_user` | Create team user with role | Team setup |
| `mcp__session-mgmt__search_team_knowledge` | Search team knowledge base | Collaborative knowledge access |
| `mcp__session-mgmt__add_team_reflection` | Add reflection to team knowledge | Knowledge sharing |

## AI-Assisted Development Workflows

### 1. Project Onboarding Workflow

```
AI Agent Process:
1. Use `init` tool → Comprehensive project setup
2. Use `auto_load_context` → Load relevant past conversations
3. Use `get_context_summary` → Understand current state
4. Use `reflect_on_past` → Find similar project patterns
```

### 2. Development Session Workflow

```
AI Agent Process:
1. Use `status` → Assess current project health
2. Use `search_by_file` → Get context for specific files
3. [Perform development work]
4. Use `checkpoint` → Mid-session quality assessment
5. Use `store_reflection` → Capture key decisions
```

### 3. Problem Solving Workflow

```
AI Agent Process:
1. Use `search_errors` → Find similar error patterns
2. Use `search_code` → Locate relevant code examples
3. Use `reflect_on_past` → Access previous solutions
4. [Implement solution]
5. Use `execute_crackerjack_command` → Validate quality
```

### 4. Session Closure Workflow

```
AI Agent Process:
1. Use `compress_memory` → Optimize long conversation
2. Use `store_reflection` → Capture session learnings
3. Use `end` → Complete cleanup with handoff documentation
```

## Integration Patterns

### Multi-MCP Server Coordination

This server works synergistically with other MCP servers:

```json
// Recommended MCP server stack
{
  "session-mgmt": "Session memory and context management",
  "crackerjack": "Code quality and development tools",
  "ast-grep": "Code analysis and pattern matching",
  "filesystem": "File operations with safety",
  "git": "Version control operations"
}
```

### Development Tool Chain

```
Session-Mgmt-MCP → Memory & Context
        ↓
   Crackerjack → Code Quality & Testing
        ↓
    AST-Grep → Code Analysis
        ↓
   Filesystem → File Operations
        ↓
       Git → Version Control
```

## Agent Behavior Patterns

### Memory Management

**Best Practice**: Use semantic search before creating new solutions

```python
# AI should first search for existing solutions
results = await search_code("authentication middleware")
if results:
    # Build upon existing patterns
else:
    # Create new implementation
```

### Context Preservation

**Best Practice**: Store important decisions and learnings

```python
# After making significant changes
await store_reflection(
    "Refactored authentication to use JWT tokens for better scalability",
    tags=["architecture", "auth", "performance"],
)
```

### Quality Integration

**Best Practice**: Use checkpoint for mid-session quality assessment

```python
# Regular quality checkpoints during development
checkpoint_result = await checkpoint()
if checkpoint_result.quality_score < 80:
    # Address quality issues before continuing
```

## Common AI Tasks

### 1. **Code Refactoring**

- Use `search_code` to find patterns
- Use `execute_crackerjack_command` for complexity analysis
- Use `checkpoint` to validate improvements

### 2. **Debugging**

- Use `search_errors` to find similar issues
- Use `reflect_on_past` for previous solutions
- Use `store_reflection` to capture fix details

### 3. **Architecture Decisions**

- Use `search_by_concept` for architectural patterns
- Use `reflect_on_past` for similar decisions
- Use `store_reflection` to document reasoning

### 4. **Testing Strategy**

- Use `analyze_crackerjack_test_patterns` for test insights
- Use `search_code` for test examples
- Use `execute_crackerjack_command` for test execution

## Advanced Features

### Token Optimization

The server automatically manages large responses:

```python
# Responses >4000 tokens are automatically chunked
result = await large_operation()
if result.get("chunked"):
    # Use get_cached_chunk for additional chunks
    next_chunk = await get_cached_chunk(result.cache_key, 2)
```

### Multi-Project Coordination

```python
# Coordinate related projects
await create_project_group(
    "microservices", ["user-service", "order-service", "payment-service"]
)

# Search across related projects
results = await search_across_projects("authentication", "user-service")
```

### Interruption Recovery

```python
# Context preservation during interruptions
await start_interruption_monitoring()
# ... work gets interrupted ...
await restore_session_context(session_id)
```

## Performance Characteristics

### Database Performance

- **Vector Search**: 384-dimensional embeddings with cosine similarity
- **Concurrent Access**: Connection pooling with async/await patterns
- **Memory Efficient**: Automatic conversation compaction for long sessions

### Response Handling

- **Chunked Responses**: Automatic splitting for large results
- **Caching**: Smart caching with 5-minute TTL for context queries
- **Fallback Strategy**: Graceful degradation when embeddings unavailable

## Troubleshooting for AI Agents

### Common Issues

1. **MCP Server Connection Failed**

   ```bash
   # Verify server setup
   python -c "from session_mgmt_mcp.server import mcp; print('✅ MCP ready')"
   ```

1. **Memory/Embedding System Issues**

   ```bash
   # Install embedding dependencies
   uv sync --extra embeddings
   ```

1. **Database Connection Problems**

   ```bash
   # Check DuckDB setup
   python -c "import duckdb; print('✅ DuckDB ready')"
   ```

### Performance Optimization

- Use `quick_search` for overview rather than full search
- Leverage `search_summary` for pattern analysis
- Use `compress_memory` for long-running sessions
- Batch operations where possible with MCP tools

## Best Practices for AI Agents

### 1. **Context-First Development**

- Always use `auto_load_context` at session start
- Search before creating (`reflect_on_past`, `search_code`)
- Store learnings (`store_reflection`)

### 2. **Quality-Driven Workflow**

- Use `checkpoint` regularly during development
- Integrate `execute_crackerjack_command` for quality validation
- Monitor trends with `get_crackerjack_results_history`

### 3. **Memory Management**

- Use semantic search to avoid duplicating work
- Store important decisions and architectural choices
- Compress memory for long sessions

### 4. **Collaborative Development**

- Use team knowledge tools for shared learning
- Document context for future team members
- Maintain searchable project knowledge base

## Contributing

When adding new MCP tools:

1. Define with `@mcp.tool()` decorator
1. Add corresponding `@mcp.prompt()` for slash commands
1. Update this AGENTS.md with usage patterns
1. Add integration tests for AI workflow validation

## Security Considerations

- All data stored locally (no external API calls)
- DuckDB file-based storage in `~/.claude/` directory
- Permission system for trusted operations
- Input validation on all MCP tool parameters

______________________________________________________________________

*This AGENTS.md file serves as both documentation and an example of emerging standards for AI-integrated development projects.*
