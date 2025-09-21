# Quality Fix Plan for session-mgmt-mcp

## Current Status

- **ZUBAN (Type Errors)**: 412 errors - Missing annotations, undefined names, type mismatches
- **BANDIT (Security)**: 93 issues - SQL injection vectors, subprocess usage
- **REFURB (Modernization)**: 6 issues - Replace try/except with contextlib.suppress
- **COMPLEXIPY**: 42 functions >15 complexity - Need refactoring

## Priority Order & Approach

### Phase 1: REFURB (Quick Modernization Wins)

**Issues**: 6 instances of try/except pass that should use contextlib.suppress
**Files affected**:

- session_mgmt_mcp/llm_providers.py
- session_mgmt_mcp/search_enhanced.py
- session_mgmt_mcp/tools/crackerjack_tools.py
- session_mgmt_mcp/tools/memory_tools.py
- session_mgmt_mcp/tools/monitoring_tools.py
- session_mgmt_mcp/tools/serverless_tools.py

**Fix**: Replace `try: ... except: pass` with `with contextlib.suppress(Exception): ...`

### Phase 2: BANDIT (Security Critical)

**Key Issues**:

1. **SQL Injection (B608)**: 86 instances - Use parameterized queries
1. **Subprocess (B602/B607)**: 7 instances - Add shell=False, validate inputs

**Strategy**:

- All DuckDB queries already use parameterized queries ($1, $2, etc.)
- The B608 warnings are false positives for DuckDB's SQL construction
- Focus on subprocess calls that need shell=False

### Phase 3: ZUBAN (Type Annotations)

**Major Categories**:

1. **Missing return type annotations**: ~200 functions
1. **Undefined names**: Import issues and missing dependencies
1. **Type mismatches**: Incorrect type hints

**Strategy**:

- Add comprehensive type hints using Python 3.13+ syntax
- Fix import issues (missing imports, circular dependencies)
- Use `typing as t` for complex types
- Ensure all async functions have proper return types

### Phase 4: COMPLEXIPY (Refactoring)

**Top Complex Functions** (complexity >20):

1. `server.py::checkpoint` (38)
1. `server.py::init` (36)
1. `server.py::end` (35)
1. `quality_utils.py::_analyze_with_confidence` (29)
1. `server.py::status` (28)

**Strategy**:

- Extract helper functions for distinct logic blocks
- Use early returns to reduce nesting
- Split large functions into smaller, focused functions
- Apply DRY principle to repeated patterns

## Execution Plan

### Step 1: REFURB Fixes (5 minutes)

```bash
# Apply contextlib.suppress pattern
python -m crackerjack --ai-agent -t --focus refurb
```

### Step 2: BANDIT Security (15 minutes)

```bash
# Fix subprocess and validate SQL patterns
python -m crackerjack --ai-agent -t --focus bandit
```

### Step 3: ZUBAN Type Annotations (30 minutes)

```bash
# Add comprehensive type hints
python -m crackerjack --ai-agent -t --focus zuban
```

### Step 4: COMPLEXIPY Refactoring (45 minutes)

```bash
# Refactor high-complexity functions
python -m crackerjack --ai-agent -t --focus complexipy
```

### Step 5: Final Validation

```bash
# Run full quality check
python -m crackerjack -t
```

## Success Criteria

- All REFURB issues resolved
- BANDIT issues reduced to acceptable false positives
- ZUBAN type errors \<50 (from 412)
- COMPLEXIPY functions all ≤15 complexity
- Test coverage maintained or improved
- All tests passing

## Clean Code Principles Applied

- **EVERY LINE IS A LIABILITY**: Remove unnecessary code
- **DRY**: Extract common patterns into reusable functions
- **YAGNI**: Don't add unnecessary abstraction
- **KISS**: Keep refactoring simple and readable
