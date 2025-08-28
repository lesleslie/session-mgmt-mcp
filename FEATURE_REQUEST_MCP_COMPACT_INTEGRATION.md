# Feature Request: MCP Tool Integration with /compact Command

**Repository**: https://github.com/anthropics/claude-code/issues
**Type**: Enhancement
**Priority**: Medium-High

## Summary

Allow MCP (Model Context Protocol) tools to trigger Claude's built-in `/compact` command programmatically, eliminating the need for manual compaction steps in automated workflows.

## Problem Statement

Currently, MCP tools cannot trigger Claude's built-in `/compact` command, which creates workflow inefficiencies. When MCP tools like session checkpoint functions detect that context compaction would be beneficial (e.g., after large development sessions, complex multi-file operations, or natural workflow breakpoints), users must manually remember to run `/compact` as a separate step.

### Current Workflow Problem

1. User runs MCP tool (e.g., `/session-mgmt:checkpoint`)
1. Tool analyzes context and determines compaction would be beneficial
1. Tool can only **suggest** that user run `/compact` manually
1. User must remember to run `/compact` in separate command
1. "Context low" warnings persist until manual intervention

## Proposed Solutions

### Option 1: Direct MCP API Integration

Allow MCP tools to trigger compaction via API call:

```python
@mcp.tool()
async def checkpoint():
    # ... perform checkpoint logic ...

    if should_suggest_compact():
        await claude.compact()  # Direct integration

    return results
```

### Option 2: Return Value Metadata

Allow MCP tools to return metadata indicating compaction should occur:

```python
@mcp.tool()
async def checkpoint():
    # ... checkpoint logic ...
    return {
        "content": results,
        "metadata": {
            "trigger_compact": True,
            "reason": "Large project detected - compaction recommended",
        },
    }
```

### Option 3: Configuration-Based Auto-Compaction

Provide user configuration for automatic compaction:

```json
{
  "mcp_settings": {
    "auto_compact_after": [
      "session-mgmt:checkpoint",
      "custom-cleanup-tool"
    ],
    "compact_threshold": "auto"
  }
}
```

### Option 4: MCP Hook System

Implement pre/post hooks for MCP tools:

```json
{
  "mcp_hooks": {
    "post_execution": {
      "session-mgmt:checkpoint": ["compact_if_recommended"]
    }
  }
}
```

## Use Cases

### Session Management Tools

- **Checkpoint Tools**: Create Git commits and analyze session progress
- **Context Analyzers**: Detect complex multi-file operations
- **Workflow Optimizers**: Identify natural break points in development

### Development Workflow Integration

- **Large Codebase Work**: Automatically compact after analyzing 50+ files
- **Multi-Project Sessions**: Compact when switching between project contexts
- **Long Development Sessions**: Periodic compaction during extended work

### Quality Assurance Tools

- **Code Review Tools**: Compact after comprehensive analysis
- **Testing Frameworks**: Compact after running extensive test suites
- **Documentation Generators**: Compact after processing large documentation sets

## Benefits

### User Experience

- **Seamless Workflow**: No manual intervention required
- **Reduced Cognitive Load**: Users don't need to remember compact commands
- **Consistent Optimization**: More predictable context management
- **Better Performance**: Optimal compaction timing based on actual usage

### MCP Ecosystem Enhancement

- **Sophisticated Tools**: Enable more advanced MCP capabilities
- **Workflow Integration**: Better alignment with development processes
- **Tool Chaining**: Enable complex multi-step automated workflows
- **Professional Usage**: Support enterprise and power-user scenarios

### Technical Advantages

- **Context Efficiency**: Compaction at optimal moments
- **Memory Management**: Better resource utilization
- **Session Quality**: Improved conversation flow and focus

## Implementation Considerations

### Security

- Compaction should be optional and user-controlled
- Tools should clearly indicate when they will trigger compaction
- User consent/configuration should be required

### Performance

- Compaction should be asynchronous to avoid blocking tools
- Provide feedback during compaction process
- Handle compaction failures gracefully

### Compatibility

- Maintain backward compatibility with existing MCP tools
- Provide clear migration path for tools wanting to use feature
- Ensure feature works across different Claude Code versions

## Current Workaround

We've implemented intelligent context analysis in our session-management MCP server:

```python
def should_suggest_compact() -> tuple[bool, str]:
    """Detect when compaction would be beneficial"""
    # Analyze project size, git activity, file complexity
    # Return recommendation with reasoning


async def perform_strategic_compaction():
    """Provide detailed compaction recommendations"""
    should_compact, reason = should_suggest_compact()

    if should_compact:
        return [
            "🔄 RECOMMENDATION: Run /compact to optimize context",
            "💡 WORKFLOW: After checkpoint completes, run: /compact",
            f"📊 Reason: {reason}",
        ]
```

This provides helpful guidance but requires manual intervention, creating workflow friction.

## Urgency and Impact

### High Value Use Cases

- **Development Teams**: Using MCP tools for project management
- **Long Sessions**: Extended coding sessions with complex contexts
- **Enterprise Users**: Sophisticated workflow automation needs
- **Tool Developers**: Building advanced MCP integrations

### Ecosystem Impact

This feature would enable a new class of sophisticated MCP tools and significantly improve the professional development experience with Claude Code.

## References

- **MCP Documentation**: https://docs.anthropic.com/en/docs/claude-code/mcp
- **Context Management**: Existing `/compact` command functionality
- **Session Management Example**: https://github.com/user/session-mgmt-mcp (example implementation)

## Submission Information

**Submitted by**: session-mgmt-mcp MCP Server Development Team
**Date**: August 27, 2025
**Contact**: Submit via GitHub Issues for follow-up questions

______________________________________________________________________

*This feature request was generated automatically by the session-mgmt-mcp server during workflow optimization analysis.*
