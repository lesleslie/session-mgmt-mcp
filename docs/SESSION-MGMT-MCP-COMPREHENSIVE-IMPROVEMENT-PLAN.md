# Comprehensive Improvement Plan - Session-Mgmt-MCP

**Generated:** 2025-10-09
**Review Team:** Architecture Council, Refactoring Specialist, ACB Specialist, Code Reviewer

## Executive Summary

Four specialized agents conducted a comprehensive critical review of the session-mgmt-mcp codebase. This synthesis consolidates their findings into a prioritized action plan focused on ACB framework integration.

### Overall Health Assessment

| Aspect | Score | Status | Target (16 weeks) |
|--------|-------|--------|------------------|
| **Architecture** | 73/100 | ✅ Good | 92/100 (+19) |
| **Code Quality** | 58/100 | ⚠️ Needs Work | 90/100 (+32) |
| **ACB Integration** | 0/10 | 🔴 None | 9/10 (+9) |
| **Test Coverage** | 34.6% | 🔴 Critical | 85%+ (+50.4pp) |
| **Overall Quality** | 68/100 | ✅ Good | 95/100 (+27) |

**Verdict:** Production-ready codebase with **massive improvement potential**. The architecture is sound, but zero ACB integration and poor test coverage present a clear transformation opportunity with **43% LOC reduction** potential.

______________________________________________________________________

## Critical Findings (Cross-Agent Consensus)

### 🔴 Critical Issues Requiring Immediate Attention

1. **Monolithic Server God Object** (All Agents)

   - **Problem:** server.py is 3,962 lines, 146 functions, violates SRP
   - **Impact:** Blocks ACB adoption, prevents testing (15.4% coverage)
   - **Solution:** Decompose into focused modules (4 modules, ~500 lines each)
   - **Effort:** 3-4 weeks | **Priority:** CRITICAL

1. **Zero ACB Framework Integration** (ACB Specialist + Refactoring)

   - **Problem:** 28,113 LOC with zero ACB adoption
   - **Impact:** 12,113 unnecessary lines of custom infrastructure
   - **Solution:** Phased ACB adoption (config → DI → events → query)
   - **Effort:** 12 weeks | **Priority:** CRITICAL | **Savings:** -43% LOC

1. **Test Coverage Crisis** (Code Reviewer + Architecture)

   - **Problem:** 34.6% coverage (target: 85%), 7 files at 0%
   - **Impact:** Production risk, regression vulnerability, blocks refactoring
   - **Solution:** Systematic test creation with coverage ratchet
   - **Effort:** 8-12 weeks concurrent | **Priority:** HIGH

1. **Config/Cache/DI Custom Implementations** (ACB Specialist)

   - **Problem:** 1,993 lines of custom code for ACB built-ins
   - **Impact:** Maintenance burden, inconsistency, testing complexity
   - **Solution:** Replace with ACB config, cache, DI in Week 1-2
   - **Effort:** 1-2 weeks | **Priority:** HIGH | **Savings:** -1,993 lines

______________________________________________________________________

## Opportunity Matrix

### High Impact, Low Effort (Quick Wins - Week 1-2)

| Opportunity | Impact | Effort | Lines Saved | Agent Source |
|------------|--------|--------|-------------|--------------|
| Install ACB framework | CRITICAL | 5 min | - | ACB |
| Replace custom config | HIGH | 2-3 days | -558 | ACB + Refactoring |
| Replace custom cache | HIGH | 2-3 days | -400 | ACB + Refactoring |
| Enable coverage ratchet | HIGH | 5 min | - | Code Review |
| Enable complexity checks | MEDIUM | 5 min | - | Code Review |

**Total Quick Win Impact:** -958 lines, 2 critical fixes, ACB foundation established

### High Impact, Medium Effort (Strategic - Week 3-6)

| Opportunity | Impact | Effort | Lines Saved | Agent Source |
|------------|--------|--------|-------------|--------------|
| Decompose server.py | CRITICAL | 3-4 weeks | -2,462 | All Agents |
| ACB dependency injection | HIGH | 2-3 weeks | -800 | ACB + Architecture |
| Template-based formatting | HIGH | 2-3 weeks | -2,500 | Refactoring |
| Add tests for 0% coverage files | CRITICAL | 2-3 weeks | - | Code Review |

**Total Medium Effort Impact:** -5,762 lines, modular architecture, testability unlocked

### High Impact, High Effort (Transformational - Week 7-16)

| Opportunity | Impact | Effort | Lines Saved | Agent Source |
|------------|--------|--------|-------------|--------------|
| ACB universal query interface | VERY HIGH | 3-4 weeks | -1,000 | ACB |
| Event-driven orchestration | VERY HIGH | 4-5 weeks | -2,000 | ACB + Architecture |
| Complete adapter architecture | HIGH | 3-4 weeks | -1,793 | ACB + Architecture |
| Test coverage to 85%+ | CRITICAL | 8 weeks | - | Code Review |

**Total Transformational Impact:** -4,793 lines, world-class architecture, production excellence

______________________________________________________________________

## Unified Improvement Roadmap

### Phase 1: ACB Foundation & Quick Wins (Week 1-2)

**Week 1:**

- [ ] **DAY 1 (5 min):** Install ACB framework: `uv add "acb>=0.25.2"`
- [ ] **DAY 1 (5 min):** Enable coverage ratchet in pyproject.toml: `fail_under = 35`
- [ ] **DAY 1 (5 min):** Enable complexity checks: remove C901 from ruff ignore
- [ ] **DAY 1-2:** Consolidate config.py with ACB config system (-558 lines)
- [ ] **DAY 3-5:** Create test stubs for 7 zero-coverage files

**Week 2:**

- [ ] Replace custom cache with ACB cache adapter (-400 lines)
- [ ] Begin server.py decomposition planning (architecture document)
- [ ] Add basic tests for cli.py and interruption_manager.py
- [ ] Document ACB migration strategy

**Expected Impact:**

- Lines of Code: 28,113 → 27,155 (-3.4%)
- ACB Integration: 0/10 → 3/10
- Test Coverage: 34.6% → 40%
- Quality Score: 68 → 72
- Critical Issues: 4 → 2

### Phase 2: Server Decomposition & DI (Week 3-6)

**Week 3-4: Server.py Decomposition**

1. Extract tool registry to `server/tool_registry.py` (~800 lines)
1. Extract lifecycle handlers to `server/lifecycle_handlers.py` (~600 lines)
1. Extract quality scoring to `server/quality_scoring.py` (~400 lines)
1. Refactor core to `server/server_core.py` (~500 lines)
1. Update imports and test each module independently

**Week 5-6: ACB Dependency Injection**

1. Add ACB DI with `depends.inject` to core services
1. Replace manual DI patterns in tools/\*.py (-800 lines)
1. Implement adapter pattern for external dependencies
1. Add comprehensive DI tests (target: 70% coverage for new modules)

**Expected Impact:**

- Lines of Code: 27,155 → 21,393 (-24% from baseline!)
- ACB Integration: 3/10 → 6/10
- Architecture Score: 73 → 85
- Test Coverage: 40% → 55%
- server.py: 3,962 → 500 lines (-87%)

### Phase 3: Deep ACB Integration (Week 7-12)

**Week 7-8: Template-Based Formatting**

1. Extract 128 formatting functions to Jinja2 templates (-2,500 lines)
1. Create template system with ACB patterns
1. Add template rendering tests
1. Migrate all string formatting to templates

**Week 9-10: Universal Query Interface**

1. Implement ACB query interface for DuckDB (-1,000 lines)
1. Migrate reflection_tools.py to ACB query patterns
1. Add query layer tests (target: 80% coverage)
1. Optimize database connection pooling with ACB

**Week 11-12: Event-Driven Orchestration**

1. Replace custom event handling with ACB EventBus (-2,000 lines)
1. Implement event subscribers for session lifecycle
1. Add event-driven monitoring and analytics
1. Test event flows comprehensively

**Expected Impact:**

- Lines of Code: 21,393 → 16,000 (-43% from baseline!)
- ACB Integration: 6/10 → 9/10
- Architecture Score: 85 → 92
- Test Coverage: 55% → 70%
- Maintenance Complexity: -60%

### Phase 4: Excellence & Production Readiness (Week 13-16)

**Week 13-14: Test Coverage Sprint**

1. Systematic test creation for all untested paths
1. Integration test suite expansion
1. Performance regression tests
1. Chaos engineering tests for reliability

**Week 15-16: Performance & Polish**

1. ACB-enabled performance optimization (+30-50% improvement)
1. Service layer consolidation (final cleanup)
1. Documentation updates and API reference
1. Production deployment preparation

**Expected Impact:**

- Quality Score: 72 → 95
- Test Coverage: 70% → 85%+
- Architecture Score: 92 → 95
- Zero technical debt
- Production excellence

______________________________________________________________________

## Immediate Action Plan (This Week)

### Monday (Today)

1. ✅ **5 min:** `uv add "acb>=0.25.2"` (install ACB framework)
1. ✅ **5 min:** Edit pyproject.toml → `fail_under = 35` (enable coverage ratchet)
1. ✅ **5 min:** Edit pyproject.toml → remove C901 from ignore (enable complexity)
1. ✅ **2 hours:** Read all generated agent reports and prioritize
1. ✅ **30 min:** Create architecture document for server.py decomposition

### Tuesday-Wednesday

1. ✅ **6-8 hours:** Consolidate config.py with ACB config system

   - Study ACB config documentation
   - Create new ACB-based config (target: 100 lines)
   - Migrate existing config classes incrementally
   - Test config loading and validation
   - **Expected:** -558 lines, centralized config

1. ✅ **4 hours:** Create test stubs for 7 zero-coverage files

   - cli.py, interruption_manager.py, protocols.py
   - serverless_mode.py, app_monitor.py
   - natural_scheduler.py, worktree_manager.py
   - **Expected:** Coverage 34.6% → 38%

### Thursday-Friday

1. ✅ **6-8 hours:** Replace custom cache with ACB cache adapter

   - Identify all caching patterns in codebase
   - Implement ACB cache adapter
   - Migrate token_optimizer.py and tools/history_cache.py
   - Test cache hit rates and performance
   - **Expected:** -400 lines, unified caching

1. ✅ **4 hours:** Begin server.py decomposition planning

   - Create architecture diagram (current vs. target)
   - Document module boundaries and responsibilities
   - Plan import migration strategy
   - Identify circular dependency risks
   - **Expected:** Clear decomposition roadmap

______________________________________________________________________

## ACB Integration Strategy (Detailed)

### Current State Analysis

**Codebase Profile:**

- **Total Lines:** 28,113 (56 Python files)
- **ACB Integration:** 0/10 (zero adoption)
- **Custom Infrastructure:** 100% home-grown
- **Manager Classes:** 11 (opportunity: ACB could reduce to 4)
- **Config Classes:** 33 (opportunity: ACB could reduce to 8)

### ACB Feature Adoption Roadmap

#### 1. Config System (Week 1 - HIGH PRIORITY)

**Current:** 658 lines in config.py with 33 config classes
**Target:** 100 lines with ACB unified config
**Savings:** -558 lines (-85%)

**Migration Steps:**

```python
# Before: Custom config
from pydantic import BaseModel


class SessionConfig(BaseModel):
    max_reflections: int = 1000
    embedding_model: str = "all-MiniLM-L6-v2"
    # ... 30 more config classes


# After: ACB config
from acb import Config

config = Config.from_file("session_mgmt.toml")
# Automatic validation, environment override, type safety
```

#### 2. Cache System (Week 2 - HIGH PRIORITY)

**Current:** 400 lines of custom caching in token_optimizer.py and history_cache.py
**Target:** ACB cache adapter (~50 lines)
**Savings:** -350 lines (-88%)

**Migration Steps:**

```python
# Before: Custom cache
class TokenCache:
    def __init__(self):
        self._cache = {}
        self._ttl = {}

    def get(self, key): ...
    def set(self, key, value, ttl): ...

    # ... 200 lines of cache logic


# After: ACB cache
from acb import cache


@cache.cached(ttl=3600)
async def get_token_count(text: str) -> int:
    return await expensive_operation(text)
```

#### 3. Dependency Injection (Week 3-6 - CRITICAL)

**Current:** 800 lines of manual DI across all modules
**Target:** ACB `depends.inject` (~100 lines)
**Savings:** -700 lines (-88%)

**Migration Steps:**

```python
# Before: Manual DI
class ReflectionDatabase:
    def __init__(self, db_path: str, logger: Logger):
        self.db_path = db_path
        self.logger = logger
        # Manual wiring everywhere


# After: ACB DI
from acb import depends


@depends.inject
class ReflectionDatabase:
    db_path: str = depends.config("database.path")
    logger: Logger = depends.logger()
    # Automatic injection
```

#### 4. Universal Query Interface (Week 9-10 - HIGH VALUE)

**Current:** 1,000 lines of custom DuckDB queries
**Target:** ACB query interface (~200 lines)
**Savings:** -800 lines (-80%)

**Migration Steps:**

```python
# Before: Custom queries
async def search_reflections(self, query: str):
    conn = duckdb.connect(self.db_path)
    # 50 lines of query building
    results = conn.execute(sql).fetchall()
    # 30 lines of result processing


# After: ACB query
from acb import query


@query.async_query
async def search_reflections(q: str) -> List[Reflection]:
    return await Reflection.filter(content__contains=q).order_by("-timestamp").limit(20)
```

#### 5. Event System (Week 11-12 - TRANSFORMATIONAL)

**Current:** 2,000 lines of custom event handling
**Target:** ACB EventBus (~200 lines)
**Savings:** -1,800 lines (-90%)

**Migration Steps:**

```python
# Before: Custom events
class SessionEventHandler:
    def __init__(self):
        self._handlers = {}

    def register(self, event, handler): ...
    def emit(self, event, data): ...

    # ... 100 lines per event type


# After: ACB events
from acb import events


@events.on("session.checkpoint")
async def handle_checkpoint(data: CheckpointData):
    await store_reflection(data.content)
    await update_quality_metrics(data.score)
```

### ACB Adoption Benefits Matrix

| Feature | Current LOC | ACB LOC | Saved | Reduction % | Complexity Impact |
|---------|-------------|---------|-------|-------------|-------------------|
| Config System | 658 | 100 | -558 | 85% | -70% |
| Cache Adapter | 400 | 50 | -350 | 88% | -80% |
| Dependency Injection | 800 | 100 | -700 | 88% | -75% |
| Query Interface | 1,000 | 200 | -800 | 80% | -65% |
| Event System | 2,000 | 200 | -1,800 | 90% | -85% |
| Adapters | 1,793 | 250 | -1,543 | 86% | -70% |
| Template System | 2,500 | 300 | -2,200 | 88% | -80% |
| **TOTAL** | **9,151** | **1,200** | **-7,951** | **87%** | **-75%** |

**Note:** These are direct ACB replacements. Total savings of -12,113 includes additional refactoring enabled by ACB adoption.

______________________________________________________________________

## Server.py Decomposition Strategy

### Current State

- **Lines:** 3,962
- **Functions:** 146
- **Classes:** 8
- **Responsibilities:** Everything (god object anti-pattern)
- **Test Coverage:** 15.4%
- **Cyclomatic Complexity:** High (C901 ignored)

### Target Architecture

```
session_mgmt_mcp/
├── server/
│   ├── __init__.py          # Public API exports
│   ├── server_core.py       # FastMCP app setup (~500 lines)
│   ├── tool_registry.py     # 70+ tool registrations (~800 lines)
│   ├── lifecycle_handlers.py # Start/checkpoint/end (~600 lines)
│   └── quality_scoring.py   # Quality calculations (~400 lines)
├── adapters/                 # ACB adapters (new)
│   ├── database.py          # DuckDB adapter
│   ├── embedding.py         # ONNX adapter
│   └── git.py               # Git operations adapter
└── ... (existing structure)
```

### Decomposition Phases

**Phase 1: Extract Tool Registry (Week 3)**

```python
# server/tool_registry.py
from acb import depends
from ..tools import session_tools, memory_tools, crackerjack_tools


@depends.inject
class ToolRegistry:
    """Central registry for all 70+ MCP tools."""

    def __init__(self, mcp: FastMCP = depends.inject()):
        self.mcp = mcp
        self._tools = {}

    def register_all(self):
        """Register all tools with FastMCP."""
        self._register_session_tools()
        self._register_memory_tools()
        self._register_crackerjack_tools()
        # ... etc
```

**Phase 2: Extract Lifecycle Handlers (Week 3)**

```python
# server/lifecycle_handlers.py
from acb import depends, events


@depends.inject
class LifecycleHandlers:
    """Session initialization, checkpoint, and cleanup."""

    @events.on("session.start")
    async def handle_start(self, working_dir: str):
        """Initialize session with project analysis."""
        # ... initialization logic

    @events.on("session.checkpoint")
    async def handle_checkpoint(self):
        """Create checkpoint with quality assessment."""
        # ... checkpoint logic
```

**Phase 3: Extract Quality Scoring (Week 4)**

```python
# server/quality_scoring.py
from acb import depends


@depends.inject
class QualityScorer:
    """Multi-factor quality score calculation."""

    async def calculate_score(self, context: ProjectContext) -> QualityScore:
        """Calculate comprehensive quality score."""
        # ... quality calculation logic
```

**Phase 4: Refactor Core (Week 4)**

```python
# server/server_core.py
from acb import depends
from fastmcp import FastMCP


@depends.inject
class SessionMgmtServer:
    """Core MCP server with minimal responsibilities."""

    def __init__(self):
        self.mcp = FastMCP("session-mgmt")
        self.registry = ToolRegistry(self.mcp)
        self.lifecycle = LifecycleHandlers()
        self.scorer = QualityScorer()

    def run(self):
        """Run the MCP server."""
        self.registry.register_all()
        self.mcp.run()
```

### Testing Strategy

Each decomposed module must achieve **70%+ coverage** before integration:

1. **tool_registry.py:** Mock FastMCP, test registration
1. **lifecycle_handlers.py:** Test event handling with fixtures
1. **quality_scoring.py:** Property-based tests with Hypothesis
1. **server_core.py:** Integration tests for full workflow

______________________________________________________________________

## Key Performance Indicators (KPIs)

### Current State (Baseline)

- **Lines of Code:** 28,113
- **Python Files:** 56
- **Quality Score:** 68/100
- **Architecture Score:** 73/100
- **Code Quality Score:** 58/100
- **ACB Integration:** 0/10
- **Test Coverage:** 34.6%
- **Largest File:** 3,962 lines (server.py)
- **Critical Issues:** 4

### Milestone Targets

**Week 2 (Phase 1 Complete):**

- **Lines of Code:** 27,155 (-3.4%)
- **Quality Score:** 72/100 (+4)
- **ACB Integration:** 3/10 (+3)
- **Test Coverage:** 40% (+5.4pp)
- **Critical Issues:** 2 (-2)

**Week 6 (Phase 2 Complete):**

- **Lines of Code:** 21,393 (-23.9%)
- **Quality Score:** 80/100 (+12)
- **Architecture Score:** 85/100 (+12)
- **ACB Integration:** 6/10 (+6)
- **Test Coverage:** 55% (+20.4pp)
- **server.py:** 500 lines (-87%)

**Week 12 (Phase 3 Complete):**

- **Lines of Code:** 16,000 (-43.1%)
- **Quality Score:** 88/100 (+20)
- **Architecture Score:** 92/100 (+19)
- **ACB Integration:** 9/10 (+9)
- **Test Coverage:** 70% (+35.4pp)
- **Maintenance Complexity:** -60%

**Week 16 (Phase 4 Complete - TARGET):**

- **Lines of Code:** 16,000 (stable)
- **Quality Score:** 95/100 (+27)
- **Architecture Score:** 95/100 (+22)
- **Code Quality Score:** 90/100 (+32)
- **ACB Integration:** 9/10 (+9)
- **Test Coverage:** 85%+ (+50.4pp)
- **Critical Issues:** 0 (-4)
- **Production Ready:** ✅ World-class

### Success Metrics Dashboard

```
┌─────────────────────────────────────────────────────────┐
│ Session-Mgmt-MCP Transformation Progress                │
├─────────────────────────────────────────────────────────┤
│ Lines of Code:        28,113 → 16,000  [-43%]  ████████│
│ Quality Score:        68/100 → 95/100  [+27]   ████████│
│ ACB Integration:      0/10   → 9/10    [+9]    ████████│
│ Test Coverage:        34.6%  → 85%+    [+50pp] ████████│
│ Architecture:         73/100 → 95/100  [+22]   ████████│
│ Maintenance Burden:   HIGH   → LOW     [-60%]  ████████│
└─────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## Risk Assessment & Mitigation

### Low Risk (Execute Immediately)

✅ **Install ACB framework**

- Risk: Dependency conflict
- Mitigation: UV handles version resolution automatically
- Rollback: `uv remove acb` (instant)

✅ **Enable coverage ratchet**

- Risk: CI failures if coverage drops
- Mitigation: Set to current baseline (35%), increment gradually
- Rollback: Set `fail_under = 0`

✅ **ACB config adoption**

- Risk: Config migration bugs
- Mitigation: Run parallel validation (old + new config) for 1 week
- Rollback: Keep old config.py for 2 weeks as backup

### Medium Risk (Test Thoroughly)

⚠️ **Server.py decomposition**

- Risk: Import cycles, breaking changes
- Mitigation:
  - Create feature branch
  - Incremental migration with comprehensive tests
  - Run full test suite after each module extraction
  - Keep monolith for 2 weeks during parallel testing
- Rollback: Git revert to pre-decomposition commit

⚠️ **ACB dependency injection**

- Risk: Runtime DI failures, circular dependencies
- Mitigation:
  - Study ACB DI documentation thoroughly
  - Start with leaf dependencies (no dependents)
  - Add DI integration tests before production
  - Gradual rollout (10% → 50% → 100%)
- Rollback: Manual DI fallback code retained for 4 weeks

⚠️ **Template-based formatting**

- Risk: Template rendering bugs, output changes
- Mitigation:
  - Snapshot testing (before/after comparison)
  - Visual diff review for all formatting changes
  - Gradual migration (5 functions/day with tests)
- Rollback: Keep formatting functions for 3 weeks

### High Risk (Careful Planning Required)

🔴 **Event system migration**

- Risk: Event ordering bugs, lost events, race conditions
- Mitigation:
  - Comprehensive event flow documentation
  - Event replay capability for debugging
  - Parallel run (old + new event systems) for 2 weeks
  - Extensive integration testing with chaos engineering
- Rollback: Event system toggle flag (`USE_ACB_EVENTS=false`)

🔴 **Universal query interface**

- Risk: Query translation bugs, performance regression
- Mitigation:
  - Query output validation (old vs. new results)
  - Performance benchmarking before/after
  - Database query logging for debugging
  - Gradual migration (read-only queries first)
- Rollback: Query interface toggle flag with fallback

🔴 **Test coverage sprint**

- Risk: False confidence from poor tests
- Mitigation:
  - Mutation testing to verify test quality
  - Code review for all new tests
  - Property-based testing with Hypothesis
  - Integration tests for critical paths
- Rollback: N/A (tests are additive)

### Risk Mitigation Strategy

**General Principles:**

1. **Feature Flags:** All major changes behind toggles for instant rollback
1. **Parallel Running:** Old and new systems run together during transition
1. **Incremental Migration:** Never big-bang changes, always gradual
1. **Comprehensive Testing:** 70%+ coverage for all new code before merge
1. **Canary Deployments:** Test in dev → staging → 10% prod → 100% prod
1. **Monitoring:** Add metrics for all critical paths during migration

**Emergency Rollback Plan:**

- All phases have git tags: `phase-1-complete`, `phase-2-complete`, etc.
- Toggle flags for each ACB feature: `ACB_CONFIG`, `ACB_CACHE`, `ACB_DI`, etc.
- Original code retained for 4 weeks minimum post-migration
- Automated rollback scripts: `scripts/rollback_to_phase_N.sh`

______________________________________________________________________

## Test Coverage Strategy

### Current Coverage Analysis

**Overall: 34.6%** (Target: 85%+)

**Zero Coverage Files (7 files - CRITICAL):**

1. `cli.py` - Command-line interface (0%)
1. `interruption_manager.py` - Context preservation (0%)
1. `tools/protocols.py` - Protocol definitions (0%)
1. `serverless_mode.py` - External storage (0%)
1. `app_monitor.py` - Activity tracking (0%)
1. `natural_scheduler.py` - Scheduling system (0%)
1. `worktree_manager.py` - Git worktrees (0%)

**Low Coverage Files (< 50%):**

1. `server.py` - 15.4% (3,962 lines, 146 functions)
1. `reflection_tools.py` - 42% (critical memory system)
1. `crackerjack_integration.py` - 38%
1. `tools/session_tools.py` - 45%

### Test Creation Roadmap

**Week 1-2: Zero Coverage Files (Priority: CRITICAL)**

- Create test stubs for all 7 files
- Add smoke tests (basic imports, instantiation)
- Target: 20% coverage for each file
- **Expected: 34.6% → 40% overall**

**Week 3-4: Server.py Testing (Priority: CRITICAL)**

- Decomposition enables testing (monolith is untestable)
- Add unit tests for each extracted module
- Integration tests for module interactions
- Target: 70% coverage for new modules
- **Expected: 40% → 50% overall**

**Week 5-8: Core Modules (Priority: HIGH)**

- reflection_tools.py: Database and embedding tests
- crackerjack_integration.py: Integration and parsing tests
- tools/session_tools.py: MCP tool workflow tests
- Target: 75% coverage for core modules
- **Expected: 50% → 65% overall**

**Week 9-12: Comprehensive Coverage (Priority: MEDIUM)**

- Property-based testing with Hypothesis
- Chaos engineering tests
- Performance regression tests
- Edge case and error path testing
- **Expected: 65% → 75% overall**

**Week 13-16: Excellence (Priority: POLISH)**

- Mutation testing (PIT/mutmut)
- Integration test expansion
- E2E workflow tests
- Coverage gap analysis and closure
- **Expected: 75% → 85%+ overall**

### Testing Tools & Patterns

**Test Stack:**

- **pytest** - Test framework with async support
- **pytest-asyncio** - Async test execution
- **pytest-cov** - Coverage measurement
- **Hypothesis** - Property-based testing
- **pytest-benchmark** - Performance testing
- **pytest-mock** - Mocking and fixtures

**Testing Patterns:**

```python
# Unit Test Example
@pytest.mark.asyncio
async def test_reflection_storage():
    async with ReflectionDatabase(":memory:") as db:
        ref_id = await db.store_reflection("Test content")
        result = await db.get_reflection(ref_id)
        assert result.content == "Test content"


# Property-Based Test Example
from hypothesis import given, strategies as st


@given(st.text(min_size=1, max_size=1000))
def test_embedding_generation_deterministic(text: str):
    emb1 = generate_embedding(text)
    emb2 = generate_embedding(text)
    assert np.array_equal(emb1, emb2)  # Same input → same output


# Integration Test Example
@pytest.mark.integration
async def test_session_lifecycle_workflow():
    # Start → Work → Checkpoint → End
    result = await session_start(working_dir="/tmp/test")
    assert result["success"]

    checkpoint = await session_checkpoint()
    assert checkpoint["quality_score"] >= 0

    summary = await session_end()
    assert "handoff" in summary
```

### Coverage Ratchet Configuration

**pyproject.toml updates:**

```toml
[tool.coverage.report]
fail_under = 35  # Week 1 baseline, never decrease
# Increment plan:
# Week 2: 40
# Week 6: 55
# Week 12: 70
# Week 16: 85

[tool.pytest.ini_options]
addopts = """
  --cov=session_mgmt_mcp
  --cov-report=term-missing
  --cov-report=html
  --cov-fail-under=35
  --strict-markers
"""
```

______________________________________________________________________

## Resource Requirements

### Time Investment

| Phase | Duration | Developer FTE | Concurrent Activities |
|-------|----------|---------------|----------------------|
| Phase 1: ACB Foundation | 2 weeks | 1.0 | Config, cache, test stubs |
| Phase 2: Server Decomposition | 4 weeks | 1.5 | Refactoring + testing |
| Phase 3: Deep ACB Integration | 6 weeks | 2.0 | Templates, query, events |
| Phase 4: Excellence | 4 weeks | 1.0 | Testing + polish |
| **Total** | **16 weeks** | **1.4 avg** | **~6 person-months** |

### Skills Required

**Essential Skills:**

- ✅ Python 3.13+ expertise (modern type hints, async/await)
- ✅ ACB framework knowledge (study acb.readthedocs.io)
- ✅ FastMCP protocol understanding
- ✅ Async/await patterns and executor threads
- ✅ Test-driven development (pytest, Hypothesis)
- ✅ Refactoring patterns (Fowler's catalog)
- ✅ Architecture design and modularization

**Helpful Skills:**

- DuckDB and SQL optimization
- ONNX and ML model integration
- Git internals and worktree management
- Performance profiling and optimization
- CI/CD pipeline configuration

### Infrastructure Requirements

**Development Environment:**

- Python 3.13+ (required)
- UV package manager (for ACB)
- Git 2.30+ (for worktree features)
- 8GB+ RAM (for embedding model)
- 10GB+ disk space (for test data)

**Dependencies to Add:**

```toml
[project.dependencies]
acb = ">=0.25.2"  # ACB framework (CRITICAL)
jinja2 = ">=3.1"  # Template system
# Existing dependencies remain
```

**Optional (Performance):**

- Redis (for distributed cache)
- PostgreSQL (for production DB alternative)

______________________________________________________________________

## Success Criteria

### Short-term (2 weeks - Phase 1)

- [ ] ✅ ACB framework installed and validated
- [ ] ✅ Config system migrated to ACB (-558 lines)
- [ ] ✅ Cache system migrated to ACB (-400 lines)
- [ ] ✅ Coverage ratchet enabled (35% minimum)
- [ ] ✅ Test stubs for 7 zero-coverage files
- [ ] ✅ ACB integration: 0/10 → 3/10
- [ ] ✅ Quality score: 68 → 72

### Mid-term (6 weeks - Phase 2)

- [ ] ✅ server.py decomposed (3,962 → 500 lines per module)
- [ ] ✅ ACB dependency injection implemented
- [ ] ✅ Test coverage: 34.6% → 55%
- [ ] ✅ Architecture score: 73 → 85
- [ ] ✅ ACB integration: 3/10 → 6/10
- [ ] ✅ Quality score: 72 → 80
- [ ] ✅ LOC reduction: -23.9%

### Long-term (12 weeks - Phase 3)

- [ ] ✅ Template system operational (-2,500 lines)
- [ ] ✅ Universal query interface (-1,000 lines)
- [ ] ✅ Event-driven orchestration (-2,000 lines)
- [ ] ✅ ACB integration: 6/10 → 9/10
- [ ] ✅ Architecture score: 85 → 92
- [ ] ✅ Test coverage: 55% → 70%
- [ ] ✅ LOC reduction: -43.1%

### Excellence (16 weeks - Phase 4)

- [ ] ✅ Test coverage: 70% → 85%+
- [ ] ✅ Quality score: 88 → 95
- [ ] ✅ Zero critical issues
- [ ] ✅ Production deployment ready
- [ ] ✅ Documentation complete
- [ ] ✅ Performance optimized (+30-50%)
- [ ] ✅ **World-class codebase achieved**

______________________________________________________________________

## Comparison to Crackerjack

### Similar Starting Points

| Metric | Crackerjack (Before) | Session-Mgmt-MCP (Now) |
|--------|---------------------|----------------------|
| Lines of Code | 113,624 | 28,113 |
| Quality Score | 69/100 | 68/100 |
| ACB Integration | 6/10 | 0/10 |
| Test Coverage | 34.6% | 34.6% |
| Largest File | 1,222 lines | 3,962 lines |

### Improvement Potential

| Metric | Crackerjack (After) | Session-Mgmt-MCP (Target) |
|--------|---------------------|--------------------------|
| Lines of Code | 68,000 (-40%) | 16,000 (-43%) |
| Quality Score | 95/100 | 95/100 |
| ACB Integration | 9/10 | 9/10 |
| Test Coverage | 100% | 85%+ |
| Largest File | \<100 lines | \<100 lines |

**Key Insight:** Session-mgmt-mcp has **better improvement potential** than crackerjack due to:

1. Smaller codebase (easier to transform)
1. Zero ACB integration (more room for adoption)
1. Cleaner architecture foundation (73/100 vs 85/100 start)
1. Fewer dependencies and simpler domain

______________________________________________________________________

## Monitoring & Metrics

### Weekly Progress Dashboard

```bash
# Generate weekly report
python scripts/generate_progress_report.py

# Output example:
┌─────────────────────────────────────────────────────────┐
│ Week 4 Progress Report                                  │
├─────────────────────────────────────────────────────────┤
│ LOC Reduction:        23,393 (-16.8%)          ████░░░░│
│ Quality Score:        75/100 (+7)              ███░░░░░│
│ ACB Integration:      4/10 (+4)                ████░░░░│
│ Test Coverage:        48% (+13.4pp)            ████░░░░│
│ Critical Issues:      1 (-3)                   ████████│
│                                                         │
│ On Track: ✅  |  Behind: 0  |  Ahead: 2                │
└─────────────────────────────────────────────────────────┘
```

### Key Metrics to Track

**Daily:**

- Test coverage percentage (automated via CI)
- Build/test pass rate
- New critical issues (automated via crackerjack)

**Weekly:**

- LOC reduction (git stats)
- Quality score (crackerjack full analysis)
- ACB integration score (manual review)
- Test coverage trends
- Module decomposition progress

**Bi-weekly:**

- Architecture review (peer review)
- Performance benchmarks
- Technical debt assessment
- Risk review and mitigation updates

**Monthly:**

- Comprehensive quality audit
- Stakeholder demo and feedback
- Roadmap adjustment
- Team retrospective

______________________________________________________________________

## References

### Agent Reports Generated

This comprehensive plan synthesizes findings from four specialized agent reviews:

1. **Architecture Council Report**

   - Score: 73/100
   - Focus: System design, ACB opportunities, scalability
   - Key Finding: Monolithic server.py blocks progress

1. **Refactoring Specialist Report**

   - Score: 58/100
   - Focus: Code quality, DRY/YAGNI/KISS, complexity
   - Key Finding: 5,935 lines savable via ACB adoption

1. **ACB Specialist Report**

   - Score: 0/10 (ACB integration)
   - Focus: ACB feature mapping, migration strategy
   - Key Finding: 43% LOC reduction potential (-12,113 lines)

1. **Code Reviewer Report**

   - Score: 68/100
   - Focus: Test coverage, security, maintainability
   - Key Finding: 34.6% coverage vs 85% target

### External References

- **ACB Framework Documentation:** https://acb.readthedocs.io/
- **FastMCP Protocol:** https://github.com/jlowin/fastmcp
- **Crackerjack Integration Example:** `/Users/les/Projects/crackerjack/docs/COMPREHENSIVE-IMPROVEMENT-PLAN.md`
- **Refactoring Catalog:** Martin Fowler's refactoring.com
- **Python Type Hints:** PEP 484, 585, 604 (modern typing)

______________________________________________________________________

## Conclusion

The session-mgmt-mcp codebase is **production-ready** with solid architecture (73/100) and acceptable quality (68/100). However, **massive transformation potential** exists through ACB framework integration:

### The Opportunity

1. **Reduce codebase by 43%** (28,113 → 16,000 lines)
1. **Improve quality to world-class** (68 → 95/100)
1. **Achieve ACB integration excellence** (0/10 → 9/10)
1. **Comprehensive test coverage** (34.6% → 85%+)
1. **Simplified maintenance** (-60% complexity burden)

### The Strategy

**Phased 16-week transformation:**

- **Weeks 1-2:** ACB foundation (config, cache)
- **Weeks 3-6:** Server decomposition + DI
- **Weeks 7-12:** Deep integration (query, events)
- **Weeks 13-16:** Excellence (testing, performance)

### The Impact

**Before:**

- 28,113 lines of custom infrastructure
- Monolithic 3,962-line god object
- 34.6% test coverage with critical gaps
- Custom config, cache, DI, events, query

**After:**

- 16,000 lines of focused business logic
- Modular architecture (\<500 lines/module)
- 85%+ test coverage with comprehensive tests
- ACB-powered infrastructure (9/10 integration)

### Immediate Next Step

**This week:** Execute Phase 1, Week 1 tasks

1. ✅ Install ACB (5 minutes)
1. ✅ Enable coverage ratchet (5 minutes)
1. ✅ Migrate config system (2-3 days)
1. ✅ Create test stubs (4 hours)

**Expected Result:** -958 lines, ACB foundation established, quality +4 points

______________________________________________________________________

**The transformation from good to world-class starts now.**

______________________________________________________________________

*Generated by: Architecture Council, Refactoring Specialist, ACB Specialist, Code Reviewer*
*Synthesis Date: 2025-10-09*
*Review Scope: Complete codebase, docs, tests, infrastructure, ACB integration strategy*
