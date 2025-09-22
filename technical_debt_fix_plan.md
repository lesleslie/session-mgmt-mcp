# Technical Debt Fix Implementation Plan

## Overview

This document outlines a systematic approach to fix the substantial technical debt in the session-mgmt-mcp project, addressing 174 type errors, 20 security warnings, 47 complexity violations, and 1 test failure.

## Current Issues Summary

### 1. Type Errors (174 from zuban/mypy)

- Missing type annotations across multiple files
- `Any` returns without proper typing
- Untyped decorators
- Incompatible type assignments
- Undefined function references

### 2. Security Warnings (20 from bandit)

- 6 Medium severity SQL injection warnings in `advanced_search.py`
- All are false positives with proper parameterization but need annotation fixes

### 3. Complexity Violations (47 from complexipy)

- Functions with cognitive complexity 16-24 (limit is 15)
- Largest offenders: complexity 24 in `llm_providers.py` and `natural_scheduler.py`

### 4. Test Failures

- 1 failure in session status checking functionality

## Priority-Based Fix Strategy

### Phase 1: Critical Function Fixes (High Priority)

**Estimated Time: 2-4 hours**

#### 1.1 Fix Missing Function (BLOCKING)

- **File**: `session_mgmt_mcp/server.py:2022`
- **Issue**: `_extract_quality_scores_from_reflections` function is undefined but called
- **Action**: Implement the missing function or remove the call

#### 1.2 Fix Core Type Annotation Issues

- **Files**: `server.py`, `llm_providers.py`, `memory_optimizer.py`
- **Issues**: Functions returning `Any`, missing parameter types
- **Action**: Add comprehensive type hints

### Phase 2: Security Warning Resolution (Medium Priority)

**Estimated Time: 1-2 hours**

#### 2.1 SQL Injection Warning Fixes

- **File**: `session_mgmt_mcp/advanced_search.py`
- **Lines**: 144, 302, 321, 337, 354, 554
- **Issue**: String concatenation in SQL queries flagged by bandit
- **Action**: Refactor to use proper parameterized queries or add more specific `# nosec` comments

### Phase 3: Complexity Reduction (Medium Priority)

**Estimated Time: 4-6 hours**

#### 3.1 High-Complexity Functions (24+ complexity)

1. `OllamaProvider::stream_generate` (complexity 24)
1. `NaturalLanguageParser::parse_time_expression` (complexity 24)

#### 3.2 Medium-Complexity Functions (20-23 complexity)

3. `AdvancedSearchEngine::search` (complexity 22)
1. `SessionLifecycleManager::checkpoint` (complexity 22)
1. `create_checkpoint_commit` (complexity 22)

### Phase 4: Comprehensive Type Coverage (Low Priority)

**Estimated Time: 6-8 hours**

#### 4.1 Systematic Type Annotation Addition

- Add missing return type annotations
- Fix untyped decorator parameters
- Resolve incompatible type assignments

## Detailed Implementation Steps

### Step 1: Fix Critical Missing Function

```python
# In session_mgmt_mcp/server.py, add missing function:
def _extract_quality_scores_from_reflections(
    reflections: list[dict[str, Any]],
) -> list[float]:
    """Extract quality scores from reflection data."""
    scores = []
    for reflection in reflections:
        # Extract quality score from reflection content or metadata
        if "quality_score" in reflection.get("metadata", {}):
            scores.append(float(reflection["metadata"]["quality_score"]))
        elif "score" in reflection:
            scores.append(float(reflection["score"]))
    return scores
```

### Step 2: Key Type Annotation Patterns

```python
# Pattern 1: Function return types
def analyze_project_context(project_dir: Path) -> dict[str, bool]:
    # Implementation
    pass


# Pattern 2: Async function types
async def calculate_quality_score() -> dict[str, Any]:
    # Implementation
    pass


# Pattern 3: Decorator type hints
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def decorator(func: Callable[P, T]) -> Callable[P, T]:
    # Implementation
    pass
```

### Step 3: SQL Security Fix Patterns

```python
# Current (flagged by bandit):
sql = "SELECT * FROM table " + where_clause

# Fixed approach 1 - Use f-strings with validation:
sql = f"SELECT * FROM table {validated_where_clause}"  # nosec B608 - validated input


# Fixed approach 2 - Use proper parameterization:
def build_query(conditions: dict[str, Any]) -> tuple[str, list[Any]]:
    placeholders = []
    params = []
    for key, value in conditions.items():
        if key in ALLOWED_FIELDS:  # Whitelist validation
            placeholders.append(f"{key} = ?")
            params.append(value)
    where_clause = " AND ".join(placeholders)
    return f"SELECT * FROM table WHERE {where_clause}", params
```

### Step 4: Complexity Reduction Strategies

#### Strategy 1: Extract Helper Functions

```python
# Before (complexity 24):
def complex_function(data):
    # 50+ lines of logic
    pass


# After (complexity reduced):
def complex_function(data):
    validated_data = _validate_input(data)
    processed_data = _process_data(validated_data)
    return _format_output(processed_data)


def _validate_input(data):
    # 10-15 lines
    pass


def _process_data(data):
    # 10-15 lines
    pass


def _format_output(data):
    # 10-15 lines
    pass
```

#### Strategy 2: Early Returns

```python
# Before:
def function(param):
    if condition1:
        if condition2:
            if condition3:
                # deep nesting
                pass


# After:
def function(param):
    if not condition1:
        return default_value
    if not condition2:
        return alternative_value
    if not condition3:
        return fallback_value
    # main logic
```

## File-Specific Fix Priorities

### 1. `session_mgmt_mcp/server.py`

- **Priority**: CRITICAL
- **Issues**: Missing function, 20+ type errors
- **Estimated Time**: 2-3 hours

### 2. `session_mgmt_mcp/llm_providers.py`

- **Priority**: HIGH
- **Issues**: Untyped decorators, complexity 24 function
- **Estimated Time**: 2-3 hours

### 3. `session_mgmt_mcp/memory_optimizer.py`

- **Priority**: HIGH
- **Issues**: Type mismatches, Any returns
- **Estimated Time**: 1-2 hours

### 4. `session_mgmt_mcp/advanced_search.py`

- **Priority**: MEDIUM
- **Issues**: 6 SQL injection warnings, complexity 22
- **Estimated Time**: 1-2 hours

### 5. `session_mgmt_mcp/tools/memory_tools.py`

- **Priority**: MEDIUM
- **Issues**: Any returns, untyped parameters
- **Estimated Time**: 1-2 hours

## Testing Strategy

### 1. Incremental Testing

- Run `pytest -m "not slow"` after each file fix
- Ensure no regressions introduced

### 2. Type Checking Validation

- Run `zubanls` on individual files as they're fixed
- Target zero type errors per file

### 3. Complexity Validation

- Use `complexipy --max-complexity-allowed 15` to verify fixes
- Ensure all functions stay under complexity 15

### 4. Security Validation

- Run `bandit` on modified files
- Verify all security warnings are addressed

## Success Metrics

### Target Goals

- **Type Errors**: 174 → 0
- **Security Warnings**: 20 → 0 (or all properly annotated)
- **Complexity Violations**: 47 → 0
- **Test Failures**: 1 → 0

### Quality Gates

- All tests pass: `pytest`
- No type errors: `zubanls`
- No security issues: `bandit -ll`
- No complexity violations: `complexipy --max-complexity-allowed 15`
- Code coverage maintained: `pytest --cov=session_mgmt_mcp --cov-fail-under=85`

## Risk Mitigation

### 1. Backup Strategy

- Create feature branch before starting fixes
- Commit frequently with descriptive messages
- Use `git stash` for experimental changes

### 2. Rollback Plan

- Test each phase independently
- Keep working main branch unchanged until complete
- Use `git revert` for problematic commits

### 3. Regression Prevention

- Run full test suite after each major change
- Validate MCP server functionality manually
- Check that existing tools/prompts still work

## Timeline Estimate

- **Phase 1 (Critical)**: 2-4 hours
- **Phase 2 (Security)**: 1-2 hours
- **Phase 3 (Complexity)**: 4-6 hours
- **Phase 4 (Type Coverage)**: 6-8 hours
- **Testing & Validation**: 2-3 hours

**Total Estimated Time**: 15-23 hours

## Next Steps

1. **Start with Phase 1**: Fix the missing function and critical type errors
1. **Validate incrementally**: Test after each file/function fix
1. **Monitor progress**: Use quality tools to track improvement
1. **Document changes**: Update type hints and docstrings as you go
1. **Final validation**: Run complete quality check before declaring complete

This plan provides a systematic approach to eliminating the technical debt while minimizing risk and maintaining code quality throughout the process.
