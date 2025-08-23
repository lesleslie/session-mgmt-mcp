# Session Management MCP Server

A dedicated MCP server that provides comprehensive session management functionality for Claude Code sessions across any project.

## Features

- **🚀 Session Initialization**: Complete setup with UV dependency management, global workspace verification, and automation tools
- **🔍 Quality Checkpoints**: Mid-session quality monitoring with workflow analysis and optimization recommendations  
- **🏁 Session Cleanup**: Comprehensive cleanup with learning capture and handoff file creation
- **📊 Status Monitoring**: Real-time session status and project context analysis

## Available Tools

### `init`
Comprehensive session initialization including:
- Global workspace verification 
- UV dependency synchronization
- Session management setup with auto-checkpoints
- Project context analysis
- Automation tools verification

### `checkpoint`  
Mid-session quality assessment with:
- Real-time quality scoring
- Workflow drift detection
- Progress tracking and goal alignment
- Performance optimization recommendations
- Automatic git checkpoint commits (if in git repo)

### `end`
Complete session cleanup featuring:
- Final quality checkpoint
- Learning capture across key categories
- Workspace cleanup and optimization
- Session handoff file creation for continuity

### `status`
Current session status including:
- Project context analysis
- Tool availability verification
- Session management status
- Available MCP tools listing

## Installation

Add to your project's `.mcp.json` file:

```json
{
  "mcpServers": {
    "session-mgmt": {
      "command": "/Users/les/Projects/claude/mcp-servers/session-mgmt/.venv/bin/python",
      "args": ["/Users/les/Projects/claude/mcp-servers/session-mgmt/session_mgmt/server.py"]
    }
  }
}
```

**Note**: The MCP server runs in its own virtual environment with isolated dependencies to avoid conflicts with your project environment.

## Usage

Once configured, the following slash commands become available in Claude Code:

- `/claude-session-init` - Full session initialization
- `/claude-session-checkpoint` - Quality monitoring checkpoint  
- `/claude-session-end` - Complete session cleanup
- `/claude-session-status` - Current status overview

## Integration Benefits

- **🔄 Automatic Availability**: Tools available immediately when Claude starts
- **📁 Project-Agnostic**: Works consistently across all project types  
- **🎯 Context-Aware**: Adapts behavior based on current project structure
- **📋 Standardized Workflow**: Consistent session management across all projects
- **💾 Session Continuity**: Handoff files maintain context between sessions

## Requirements

- Python 3.13+
- MCP library (`pip install mcp`)
- Global Claude workspace at `~/Projects/claude/`
- UV package manager (recommended)

## Global Workspace Integration

This server integrates with the global Claude workspace automation system:
- Session management toolkit (`~/Projects/claude/toolkits/session/`)
- Verification tools (`~/Projects/claude/toolkits/verification/`)
- Analytics tracking (`~/Projects/claude/toolkits/analytics/`)
- Research automation (`~/Projects/claude/toolkits/research/`)

## Session Workflow

1. **Start Session**: `/claude-session-init`
   - UV dependency sync
   - Global tools verification
   - Auto-checkpoint setup
   - Project analysis

2. **Monitor Progress**: `/claude-session-checkpoint` (every 30-45 minutes)
   - Quality assessment
   - Workflow optimization
   - Progress tracking

3. **End Session**: `/claude-session-end`
   - Final quality check
   - Learning capture
   - Cleanup and handoff

This creates a complete, automated session management workflow that maintains high productivity standards and preserves knowledge across sessions.