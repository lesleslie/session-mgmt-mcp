# Crackerjack Integration Guide

This document describes how session-mgmt-mcp integrates with crackerjack for AI-powered code quality automation.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [MCP Tool Interface](#mcp-tool-interface)
4. [Hook Output Parsing](#hook-output-parsing)
5. [Workflow Examples](#workflow-examples)
6. [Error Handling](#error-handling)
7. [Configuration](#configuration)

---

## Overview

Session-mgmt-mcp provides an MCP (Model Context Protocol) server that wraps crackerjack functionality, enabling remote execution and AI-assisted code fixing through Claude Code.

### Key Features

- **Remote Execution**: Run crackerjack from Claude Code via MCP
- **Input Validation**: Prevent common usage mistakes
- **Result Parsing**: Extract structured data from hook output
- **AI Integration**: Enable auto-fix workflows with `ai_agent_mode`

---

## Architecture

### Component Diagram

```
┌─────────────────┐
│   Claude Code   │
│   (User Agent)  │
└────────┬────────┘
         │ MCP Protocol
         ▼
┌─────────────────────────────────┐
│  session-mgmt-mcp MCP Server    │
│                                 │
│  ┌───────────────────────────┐ │
│  │  crackerjack_tools.py     │ │
│  │  - Input Validation       │ │
│  │  - Command Building       │ │
│  │  - Result Formatting      │ │
│  └───────────┬───────────────┘ │
│              │                  │
│  ┌───────────▼───────────────┐ │
│  │  hook_parser.py           │ │
│  │  - Reverse Parsing        │ │
│  │  - Status Extraction      │ │
│  └───────────────────────────┘ │
└────────────┬────────────────────┘
             │ subprocess
             ▼
     ┌────────────────┐
     │  Crackerjack   │
     │  CLI Process   │
     └────────────────┘
```

### Data Flow

```
User → MCP Server → validate_command() → build command →
execute subprocess → parse output → format results → return to user
```

---

## MCP Tool Interface

### Tool Definition

```python
@mcp.tool()
async def crackerjack_run(
    command: str,
    args: str = "",
    working_directory: str = ".",
    timeout: int = 300,
    ai_agent_mode: bool = False,
) -> str:
    """Run crackerjack with enhanced analytics."""
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `command` | str | Yes | Semantic command (test, lint, check, etc.) |
| `args` | str | No | Additional arguments (NOT --ai-fix) |
| `working_directory` | str | No | Working directory (default: ".") |
| `timeout` | int | No | Timeout in seconds (default: 300) |
| `ai_agent_mode` | bool | No | Enable AI auto-fix (default: False) |

### Correct Usage

```python
# ✅ CORRECT - Semantic command with ai_agent_mode
await crackerjack_run(command="test", ai_agent_mode=True)

# ✅ CORRECT - With additional args
await crackerjack_run(
    command="check",
    args="--verbose",
    ai_agent_mode=True,
    timeout=600
)

# ✅ CORRECT - Dry-run mode
await crackerjack_run(
    command="test",
    args="--dry-run",
    ai_agent_mode=True
)
```

### Incorrect Usage (Will Error)

```python
# ❌ WRONG - Flags in command parameter
await crackerjack_run(command="--ai-fix -t")
# Error: "Invalid Command: '--ai-fix'"

# ❌ WRONG - --ai-fix in args
await crackerjack_run(command="test", args="--ai-fix")
# Error: "Invalid Args: Found '--ai-fix' in args parameter"

# ❌ WRONG - Unknown command
await crackerjack_run(command="invalid")
# Error: "Unknown Command: 'invalid'"
```

---

## Hook Output Parsing

### Algorithm: Reverse Parsing

The hook parser uses a **reverse parsing** algorithm to reliably extract hook names and status markers:

```python
def parse_hook_line(line: str) -> HookResult:
    """Parse hook output line.

    Format: hook_name + padding_dots + space + status_marker
    Example: "refurb...................... ❌"
    """
    # Step 1: Split from right on whitespace
    left_part, status_marker = line.rsplit(maxsplit=1)

    # Step 2: Validate status marker
    if status_marker not in ["✅", "Passed", "❌", "Failed"]:
        raise ParseError(f"Unknown status marker: {status_marker}")

    # Step 3: Extract hook name (strip padding dots)
    hook_name = left_part.rstrip(".")

    return HookResult(hook_name=hook_name, passed=(status_marker in ["✅", "Passed"]))
```

### Why Reverse Parsing?

**Problem**: Hook names can contain dots (e.g., `test.integration.api`)

**Solution**: Parse from right to left:
1. Split on last whitespace: `hook_name....... ❌` → `[hook_name......., ❌]`
2. Strip dots from right: `hook_name.......` → `hook_name`
3. Handles any hook name pattern correctly

### Example Parsing

```
Input Line:
"test.integration.api...................... ✅"

Step 1 - rsplit(maxsplit=1):
["test.integration.api......................", "✅"]

Step 2 - Validate marker:
"✅" in PASS_MARKERS → True

Step 3 - rstrip("."):
"test.integration.api......................" → "test.integration.api"

Result:
HookResult(hook_name="test.integration.api", passed=True)
```

### Supported Status Markers

**Pass Markers**:
- `✅` (green checkmark)
- `Passed` (text)

**Fail Markers**:
- `❌` (red X)
- `Failed` (text)

---

## Workflow Examples

### Basic Mode (No AI)

```python
# Run all hooks without auto-fix
result = await crackerjack_run(command="check")

# Output:
"""
✅ **Status**: Success

All hooks passed successfully!
"""
```

### AI Mode (Auto-Fix Enabled)

```python
# Run with AI-powered auto-fix
result = await crackerjack_run(
    command="test",
    ai_agent_mode=True,
    timeout=600  # Allow time for AI fixes
)

# Output (if fixes applied):
"""
✅ **Status**: Success after 3 iterations

**Fixes Applied**: 5
- refurb: Fixed unnecessary comprehension
- complexity: Simplified nested conditions
- ruff: Fixed line length issues

**Convergence**: All hooks now passing
"""
```

### Handling Failures

```python
# When hooks fail (basic mode)
result = await crackerjack_run(command="lint")

# Output:
"""
❌ **Status**: Failed (exit code: 1)
**Failed Hooks**: refurb, complexipy

refurb................................................................ ❌
complexipy............................................................ ❌
bandit................................................................ ✅
ruff.................................................................. ✅
"""
```

---

## Error Handling

### Input Validation Errors

**Error**: Command with flags
```python
await crackerjack_run(command="--ai-fix")

# Returns:
"""
❌ **Invalid Command**: '--ai-fix'

**Error**: Commands should be semantic names, not flags.

**Valid commands**: all, check, complexity, format, lint, security, test

**Correct usage**:
```python
crackerjack_run(command='test', ai_agent_mode=True)
```
"""
```

**Error**: --ai-fix in args
```python
await crackerjack_run(command="test", args="--ai-fix")

# Returns:
"""
❌ **Invalid Args**: Found '--ai-fix' in args parameter

**Use instead**: Set `ai_agent_mode=True` parameter

**Correct**:
```python
crackerjack_run(command='test', ai_agent_mode=True)
```
"""
```

### Execution Errors

**Timeout**:
```python
await crackerjack_run(command="all", timeout=10)

# Returns:
"""
❌ **Execution Error**: Command timed out after 10 seconds

**Suggestion**: Increase timeout parameter or run fewer hooks
"""
```

**Parsing Errors**:
```python
# If hook output format is unexpected
"""
⚠️ **Parsing Warning**: Could not parse some hook output

**Raw Output**:
[original output shown here]

**Suggestion**: Check hook configuration or contact support
"""
```

---

## Configuration

### MCP Server Settings

**Location**: `session_mgmt_mcp/config.py`

```python
# Crackerjack integration settings
CRACKERJACK_TIMEOUT_DEFAULT = 300  # 5 minutes
CRACKERJACK_TIMEOUT_MAX = 1800     # 30 minutes
CRACKERJACK_MAX_RETRIES = 3
```

### Hook Parser Settings

**Location**: `session_mgmt_mcp/tools/hook_parser.py`

```python
# Status markers (frozen sets for performance)
_PASS_MARKERS = frozenset(["✅", "Passed"])
_FAIL_MARKERS = frozenset(["❌", "Failed"])
```

### Environment Variables

```bash
# Optional: Override crackerjack binary location
export CRACKERJACK_BIN=/path/to/crackerjack

# Optional: Default working directory
export CRACKERJACK_WORKDIR=/path/to/project
```

---

## Integration Flow

### Sequence Diagram: Basic Execution

```
┌─────┐         ┌─────────┐         ┌──────────┐         ┌────────────┐
│User │         │   MCP   │         │Validator │         │Crackerjack │
└──┬──┘         └────┬────┘         └────┬─────┘         └─────┬──────┘
   │                 │                   │                     │
   │ crackerjack_run │                   │                     │
   ├────────────────>│                   │                     │
   │                 │                   │                     │
   │                 │ validate_command  │                     │
   │                 ├──────────────────>│                     │
   │                 │                   │                     │
   │                 │     valid/error   │                     │
   │                 │<──────────────────┤                     │
   │                 │                   │                     │
   │                 │         Execute subprocess              │
   │                 ├────────────────────────────────────────>│
   │                 │                                         │
   │                 │          stdout, stderr, exit_code      │
   │                 │<────────────────────────────────────────┤
   │                 │                                         │
   │                 │ parse_hook_output                       │
   │                 ├─────────┐                               │
   │                 │         │                               │
   │                 │<────────┘                               │
   │                 │                                         │
   │   Formatted     │                                         │
   │     Results     │                                         │
   │<────────────────┤                                         │
   │                 │                                         │
```

### Sequence Diagram: AI-Powered Execution

See [WORKFLOW-ARCHITECTURE.md](../../crackerjack/docs/WORKFLOW-ARCHITECTURE.md) for detailed AI workflow diagrams.

---

## Implementation Details

### File Structure

```
session_mgmt_mcp/
├── tools/
│   ├── crackerjack_tools.py    # MCP tool implementation
│   └── hook_parser.py          # Hook output parsing
├── config.py                    # Configuration settings
└── docs/
    └── CRACKERJACK-INTEGRATION.md  # This file
```

### Key Functions

**`crackerjack_tools.py`**:
```python
def _parse_hook_results(stdout: str) -> tuple[list[str], list[str]]:
    """Parse hook results to find failures.

    Uses production-ready hook_parser module.
    """
    results = parse_hook_output(stdout)
    passed = [r.hook_name for r in results if r.passed]
    failed = [r.hook_name for r in results if not r.passed]
    return passed, failed

def _format_execution_status(result: CrackerjackResult) -> str:
    """Format execution status for output.

    Validates parsing worked and determines true status.
    """
    passed_hooks, failed_hooks = _parse_hook_results(result.stdout)
    has_failures = result.exit_code != 0 or len(failed_hooks) > 0

    if has_failures:
        return f"❌ **Status**: Failed\n**Failed Hooks**: {', '.join(failed_hooks)}\n"
    return "✅ **Status**: Success\n"
```

**`hook_parser.py`**:
```python
def parse_hook_output(output: str) -> list[HookResult]:
    """Parse multiple lines of hook output.

    Skips empty lines, validates each non-empty line.
    """
    results = []
    for line_num, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            result = parse_hook_line(line)
            results.append(result)
        except ParseError as e:
            raise ParseError(f"Line {line_num}: {e}") from e
    return results
```

---

## Testing

### Unit Tests

**Location**: `tests/tools/test_hook_parser.py`

```python
def test_parse_simple_hook():
    line = "refurb............ ❌"
    result = parse_hook_line(line)
    assert result.hook_name == "refurb"
    assert result.passed is False

def test_parse_hook_with_dots():
    line = "test.integration.api......... ✅"
    result = parse_hook_line(line)
    assert result.hook_name == "test.integration.api"
    assert result.passed is True
```

### Integration Tests

```python
async def test_crackerjack_run_basic():
    """Test basic crackerjack execution."""
    result = await crackerjack_run(command="test")
    assert "Status" in result
    assert "Failed" in result or "Success" in result

async def test_crackerjack_run_ai_mode():
    """Test AI-powered auto-fix."""
    result = await crackerjack_run(
        command="test",
        ai_agent_mode=True
    )
    assert "Status" in result
```

---

## Troubleshooting

### Issue: MCP Server Not Responding

**Symptoms**: Timeout or no response from MCP server

**Solution**:
1. Check MCP server is running: `pgrep -fl session-mgmt-mcp`
2. Restart MCP server
3. Check logs: `tail -f ~/.mcp/logs/session-mgmt-mcp.log`

### Issue: Hook Parsing Fails

**Symptoms**: "ParseError" in output

**Causes**:
- Unexpected hook output format
- Hook output contains extra characters
- Status markers changed

**Solution**:
1. Check raw hook output
2. Verify hook configuration
3. Update hook_parser.py if format changed

### Issue: Command Validation Errors

**Symptoms**: "Invalid Command" or "Invalid Args" errors

**Solution**:
- Use semantic commands: `test`, `lint`, `check`
- Don't put flags in `command` parameter
- Use `ai_agent_mode=True` instead of `--ai-fix`

---

## Performance Optimization

### Caching

Future enhancement: Cache hook results for unchanged files

```python
# Planned feature
cache_key = hash(file_content)
if cache_key in results_cache:
    return results_cache[cache_key]
```

### Parallel Execution

Future enhancement: Run hooks in parallel

```python
# Planned feature
async with asyncio.TaskGroup() as group:
    for hook in hooks:
        group.create_task(run_hook(hook))
```

---

## References

- [Crackerjack Workflow Architecture](../../crackerjack/docs/WORKFLOW-ARCHITECTURE.md)
- [Implementation Plan](/Users/les/Projects/acb/CRACKERJACK-FIX-IMPLEMENTATION-PLAN.md)
- [MCP Protocol Specification](https://modelcontextprotocol.io)

---

**Last Updated**: 2025-01-03
**Version**: 2.0
**Status**: Production (Hook parsing fixed, AI integration pending)
