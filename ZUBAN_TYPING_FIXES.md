# Zuban Typing Failures: Systematic Implementation Guide

This document provides a comprehensive approach to systematically fixing zuban type checking failures in the session-mgmt-mcp project.

## Overview

Zuban is a fast Rust-based type checker for Python that enforces strict typing rules. The current codebase has several categories of typing issues that need systematic resolution.

## Categories of Issues Found

### 1. Import and Definition Conflicts (no-redef)

**Error**: `Name "ReflectionDatabase" already defined (possibly by an import)`
**Location**: `session_mgmt_mcp/tools/memory_tools.py:19:5`

**Root Cause**: Using conditional imports with fallback type definitions creates redefinition conflicts.

**Current Pattern**:

```python
try:
    from ..reflection_tools import ReflectionDatabase
except ImportError:
    # For type checking when imports fail
    ReflectionDatabase = Any  # type: ignore[misc,assignment]
```

**Solution Strategy**:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..reflection_tools import ReflectionDatabase
else:
    try:
        from ..reflection_tools import ReflectionDatabase
    except ImportError:
        ReflectionDatabase = Any  # type: ignore[import-untyped]
```

### 2. Method Signature Mismatches (call-arg)

**Error**: `Unexpected keyword argument "project" for "search_reflections"`
**Location**: Multiple locations in `memory_tools.py`

**Root Cause**: Method signatures have evolved but call sites haven't been updated.

**Investigation Required**:

1. Check actual `ReflectionDatabase.search_reflections` method signature
1. Update all call sites to match current signature
1. Ensure backward compatibility if needed

### 3. Async/Await Issues (return-value)

**Error**: `Incompatible return value type (got "Coroutine[Any, Any, list[dict[str, Any]]]", expected "list[dict[str, Any]]")`

**Root Cause**: Missing `await` keywords for async methods.

**Solution Pattern**:

```python
# Wrong:
def sync_method() -> list[dict[str, Any]]:
    return db.search_reflections(...)  # Returns coroutine


# Correct:
async def async_method() -> list[dict[str, Any]]:
    return await db.search_reflections(...)
```

## Systematic Fix Implementation Plan

### Phase 1: Import Resolution (Priority: High)

**Files to Fix**: All files with conditional imports

**Implementation Steps**:

1. **Identify Import Patterns**:

   ```bash
   grep -r "except ImportError:" session_mgmt_mcp/
   grep -r "type: ignore.*assignment" session_mgmt_mcp/
   ```

1. **Standard Import Pattern**:

   ```python
   from __future__ import annotations

   from typing import TYPE_CHECKING, Any

   if TYPE_CHECKING:
       from ..reflection_tools import ReflectionDatabase
   else:
       try:
           from ..reflection_tools import ReflectionDatabase
       except ImportError:
           ReflectionDatabase = Any  # type: ignore[import-untyped]
   ```

1. **Update type ignore comments** to use specific error codes instead of generic ones.

### Phase 2: Method Signature Verification (Priority: High)

**Files to Fix**: `session_mgmt_mcp/tools/memory_tools.py`

**Implementation Steps**:

1. **Audit Method Signatures**:

   ```bash
   # Find the actual ReflectionDatabase class definition
   grep -n "class ReflectionDatabase" session_mgmt_mcp/
   grep -n "def search_reflections" session_mgmt_mcp/
   ```

1. **Verify Call Sites**:

   ```bash
   # Find all calls to search_reflections
   grep -rn "search_reflections" session_mgmt_mcp/
   ```

1. **Update Method Calls**:

   - Remove unsupported `project` parameter if it doesn't exist
   - Add missing required parameters
   - Ensure parameter types match

### Phase 3: Async/Await Consistency (Priority: Medium)

**Files to Fix**: All files with async methods

**Implementation Steps**:

1. **Identify Async Methods**:

   ```bash
   grep -rn "async def" session_mgmt_mcp/ | grep -v "__"
   ```

1. **Check Return Types**:

   - Functions returning `Coroutine` should be `async def`
   - Functions calling async methods should `await` them
   - Update return type annotations accordingly

1. **Fix Pattern**:

   ```python
   # Before:
   def method() -> list[dict[str, Any]]:
       return async_method()  # Wrong - returns coroutine


   # After:
   async def method() -> list[dict[str, Any]]:
       return await async_method()  # Correct
   ```

### Phase 4: Comprehensive Type Annotation (Priority: Low)

**Files to Fix**: All Python files

**Implementation Steps**:

1. **Add Missing Return Types**:

   ```bash
   # Find functions without return type annotations
   grep -rn "def.*)" session_mgmt_mcp/ | grep -v " -> "
   ```

1. **Add Missing Parameter Types**:

   - Use zuban's `--disallow-untyped-defs` flag to identify
   - Add proper type annotations for all parameters

1. **Use Modern Type Hints**:

   ```python
   # Old style:
   from typing import List, Dict, Optional
   def method(items: List[Dict[str, Optional[str]]]) -> None:

   # New style (Python 3.9+):
   def method(items: list[dict[str, str | None]]) -> None:
   ```

## Automation Strategy

### Script-Based Fixes

Create automation scripts for common patterns:

```python
#!/usr/bin/env python3
"""Automated zuban typing fixes."""

import re
import subprocess
from pathlib import Path


def fix_import_patterns(file_path: Path) -> None:
    """Fix conditional import patterns."""
    content = file_path.read_text()

    # Pattern to match conditional imports
    pattern = r"try:\s*\n\s*from (.*) import (.*)\nexcept ImportError:\s*\n\s*# For type checking.*\n\s*(.*) = Any.*"

    def replacement(match):
        module, imported, name = match.groups()
        return f"""from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from {module} import {imported}
else:
    try:
        from {module} import {imported}
    except ImportError:
        {name} = Any  # type: ignore[import-untyped]"""

    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    file_path.write_text(new_content)


def fix_await_missing(file_path: Path) -> None:
    """Add missing await keywords."""
    content = file_path.read_text()

    # Look for common async method calls without await
    async_methods = ["search_reflections", "store_conversation", "get_conversations"]

    for method in async_methods:
        # Pattern: return db.method(...)
        pattern = rf"return (\w+\.{method}\([^)]*\))"
        content = re.sub(pattern, r"return await \1", content)

        # Pattern: result = db.method(...)
        pattern = rf"(\w+) = (\w+\.{method}\([^)]*\))"
        content = re.sub(pattern, r"\1 = await \2", content)

    file_path.write_text(content)
```

### Verification Strategy

```bash
#!/bin/bash
# verify_typing_fixes.sh

echo "Running zuban type checking..."
zuban check session_mgmt_mcp/ --show-error-codes --show-column-numbers --pretty > zuban_results.txt 2>&1

echo "Analyzing results..."
if [ $? -eq 0 ]; then
    echo "✅ All type checking passed!"
else
    echo "❌ Type checking failures found:"
    echo "Top issues:"
    grep -E "error:|note:" zuban_results.txt | head -20

    echo -e "\nError code summary:"
    grep "error:" zuban_results.txt | sed 's/.*\[\(.*\)\]$/\1/' | sort | uniq -c | sort -nr
fi
```

## File-Specific Implementation

### session_mgmt_mcp/tools/memory_tools.py

**Current Issues**:

1. Line 19: Import redefinition
1. Line 137: Invalid `project` parameter
1. Line 274: Missing `await` keyword

**Implementation**:

```python
# Fix 1: Import pattern
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..reflection_tools import ReflectionDatabase
else:
    try:
        from ..reflection_tools import ReflectionDatabase
    except ImportError:
        ReflectionDatabase = Any  # type: ignore[import-untyped]

# Fix 2: Check actual method signature and remove unsupported parameters
# Before:
results = await db.search_reflections(
    query=query,
    project=project,  # Remove if not supported
    limit=1,
    min_score=min_score,
)

# After (assuming project parameter doesn't exist):
results = await db.search_reflections(
    query=query,
    limit=1,
    min_score=min_score,
)


# Fix 3: Add missing await
async def _get_search_results(
    db: ReflectionDatabase,
    query: str,
    project: str | None,
    min_score: float,
) -> list[dict[str, Any]]:
    """Get search results from the database."""
    return await db.search_reflections(  # Add await
        query=query,
        limit=20,
        min_score=min_score,
    )
```

## Testing Strategy

### 1. Incremental Validation

```bash
# Test each file as you fix it
zuban check session_mgmt_mcp/tools/memory_tools.py --show-error-codes

# Run subset tests
zuban check session_mgmt_mcp/tools/ --show-error-codes
```

### 2. Regression Testing

```bash
# Ensure fixes don't break functionality
pytest session_mgmt_mcp/tests/ -v

# Run specific test modules
pytest session_mgmt_mcp/tests/test_memory_tools.py -v
```

### 3. Performance Verification

```bash
# Ensure type checking doesn't significantly slow down
time zuban check session_mgmt_mcp/ > /dev/null
```

## Configuration Recommendations

### pyproject.toml

```toml
[tool.mypy]
python_version = "3.13"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_unreachable = true
strict_equality = true
show_error_codes = true

[tool.zuban]
strict = true
show_error_codes = true
show_column_numbers = true
pretty = true
```

## Success Metrics

### Quantitative Goals

- **Zero type checking errors** with `zuban check session_mgmt_mcp/`
- **\<2 seconds** for full type checking run
- **100% test pass rate** after typing fixes

### Qualitative Goals

- **Consistent type annotation patterns** across codebase
- **Clear error messages** with specific error codes
- **Maintainable type hints** that aid development

## Common Pitfalls and Solutions

### 1. Circular Imports

**Problem**: Type hints create circular import dependencies
**Solution**: Use `from __future__ import annotations` and string literals for forward references

### 2. Any Type Overuse

**Problem**: Using `Any` to silence type checker
**Solution**: Create specific type aliases and protocols

### 3. Runtime Import Checks

**Problem**: `TYPE_CHECKING` imports not available at runtime
**Solution**: Use proper conditional import patterns with fallbacks

## Maintenance Plan

### 1. Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
- repo: local
  hooks:
  - id: zuban-check
    name: zuban type checking
    entry: zuban check
    language: system
    types: [python]
    pass_filenames: false
    args: [session_mgmt_mcp/, --show-error-codes]
```

### 2. CI Integration

```bash
# In CI pipeline
zuban check session_mgmt_mcp/ --show-error-codes
if [ $? -ne 0 ]; then
    echo "Type checking failed. Please fix typing issues before merging."
    exit 1
fi
```

### 3. Regular Audits

- **Weekly**: Run full type checking on main branch
- **Monthly**: Review and update type annotations
- **Per PR**: Require clean type checking for all new code

______________________________________________________________________

This guide provides a systematic approach to resolving zuban typing failures. Follow the phases in order, test incrementally, and maintain good typing practices for long-term code quality.
