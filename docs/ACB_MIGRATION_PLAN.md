# ACB Alignment Migration Plan - Option 2

**Status**: In Progress
**Start Date**: 2025-01-16
**Estimated Timeline**: 14-16 days
**Success Probability**: 90%
**Expected Code Reduction**: 91% in storage layer (~880 lines → ~80 lines)

## Executive Summary

This document tracks the refactoring effort to align session-mgmt-mcp with ACB (Asynchronous Component Base) and crackerjack best practices. The primary goal is to replace custom backend implementations with native ACB adapters, following dependency injection patterns already established in the codebase.

### Key Objectives

- ✅ Replace custom storage backends with ACB storage adapters
- ✅ Migrate KnowledgeGraphDatabaseAdapter to use ACB Graph adapter
- ✅ Maintain ReflectionDatabaseAdapter (already ACB-compliant)
- ✅ Follow existing DI patterns (direct imports + depends.set)
- ✅ Achieve 91% code reduction in storage layer
- ✅ Zero breaking changes for users

## Pre-Migration Fixes

### ✅ Phase 0: Critical Bug Fixes (Completed)

- [x] **ACB_LIBRARY_MODE Environment Variable** (Commit: 36f6c96)
  - Fixed crackerjack hooks by setting `ACB_LIBRARY_MODE=true` in subprocess calls
  - Locations updated: crackerjack_integration.py (2 places), crackerjack_tools.py (1 place)

- [x] **Type Checking Errors** (Commit: 9a4f9f2)
  - Fixed missing imports in `utils/scheduler/time_parser.py`
  - Fixed SessionPermissionsManager export path in `__init__.py`
  - Reduced errors from 121 → 112 (remaining in optional features)

- [x] **Auto-Formatting** (Commit: 49d72bb)
  - Applied crackerjack auto-formatting fixes
  - Committed to resolve stop hook warnings

## Migration Phases

### Phase 1: Foundation & Setup (Days 1-3)

**Goal**: Establish infrastructure for ACB adapter integration without breaking existing functionality.

#### Day 1: Storage Adapter Registry

- [ ] **Task 1.1**: Create storage adapter registry module
  - File: `session_mgmt_mcp/adapters/storage_registry.py`
  - Register ACB storage adapters (S3, Azure, GCS, File, Memory)
  - Follow Vector/Graph adapter registration pattern

  ```python
  from acb.adapters.storage.s3 import S3Storage, S3StorageSettings
  from acb.adapters.storage.file import FileStorage, FileStorageSettings
  from acb.adapters.storage.memory import MemoryStorage
  from acb.config import Config
  from acb.depends import depends

  def configure_storage_adapters():
      """Register ACB storage adapters with DI container."""
      config = depends.get_sync(Config)
      config.ensure_initialized()

      # S3 Storage Adapter
      s3_settings = S3StorageSettings(
          bucket_name=config.storage.s3_bucket,
          endpoint_url=config.storage.s3_endpoint,
          access_key=config.storage.s3_access_key,
          secret_key=config.storage.s3_secret_key,
      )
      config.storage.s3 = s3_settings
      s3_adapter = S3Storage()
      s3_adapter.config = config
      depends.set(S3Storage, s3_adapter)

      # File Storage Adapter
      file_settings = FileStorageSettings(
          base_path=str(paths.data_dir / "sessions"),
      )
      config.storage.file = file_settings
      file_adapter = FileStorage()
      file_adapter.config = config
      depends.set(FileStorage, file_adapter)

      # Memory Storage Adapter (testing)
      memory_adapter = MemoryStorage()
      depends.set(MemoryStorage, memory_adapter)
  ```

- [ ] **Task 1.2**: Update DI configuration
  - File: `session_mgmt_mcp/di/__init__.py`
  - Import and call `configure_storage_adapters()` in `configure()`
  - Follow existing pattern used for Vector/Graph adapters

#### Day 2: Configuration Migration

- [ ] **Task 2.1**: Create storage configuration schema
  - File: `session_mgmt_mcp/config/storage.yaml`
  - Define S3, Azure, GCS, File, Memory settings
  - Support environment variable overrides

  ```yaml
  storage:
    default_backend: "file"  # file, s3, azure, gcs, memory

    s3:
      bucket_name: "${S3_BUCKET:session-mgmt}"
      endpoint_url: "${S3_ENDPOINT:}"
      access_key: "${S3_ACCESS_KEY:}"
      secret_key: "${S3_SECRET_KEY:}"
      region: "${S3_REGION:us-east-1}"

    azure:
      account_name: "${AZURE_ACCOUNT:}"
      account_key: "${AZURE_KEY:}"
      container: "${AZURE_CONTAINER:sessions}"

    gcs:
      bucket_name: "${GCS_BUCKET:}"
      credentials_path: "${GCS_CREDENTIALS:}"

    file:
      base_path: "${SESSION_DATA_DIR:~/.claude/sessions}"

    memory:
      max_size_mb: 100
  ```

- [ ] **Task 2.2**: Update Config class to load storage settings
  - File: `session_mgmt_mcp/config.py`
  - Add StorageConfig dataclass
  - Load from YAML with environment overrides

#### Day 3: Unified Storage Interface

- [ ] **Task 3.1**: Create SessionStorageAdapter facade
  - File: `session_mgmt_mcp/adapters/session_storage_adapter.py`
  - Unified interface wrapping ACB storage adapters
  - Runtime backend selection based on config

  ```python
  from acb.adapters.storage.s3 import S3Storage
  from acb.adapters.storage.file import FileStorage
  from acb.adapters.storage.memory import MemoryStorage
  from acb.depends import depends

  class SessionStorageAdapter:
      """Unified storage adapter for session state persistence."""

      def __init__(self, backend: str = "file"):
          self.backend = backend
          self._adapter = self._get_adapter()

      def _get_adapter(self):
          """Get ACB storage adapter based on backend config."""
          if self.backend == "s3":
              return depends.get_sync(S3Storage)
          elif self.backend == "file":
              return depends.get_sync(FileStorage)
          elif self.backend == "memory":
              return depends.get_sync(MemoryStorage)
          else:
              raise ValueError(f"Unknown backend: {self.backend}")

      async def store_session(self, session_id: str, state: dict) -> None:
          """Store session state using ACB adapter."""
          key = f"sessions/{session_id}/state.json"
          await self._adapter.put(key, json.dumps(state).encode())

      async def load_session(self, session_id: str) -> dict | None:
          """Load session state using ACB adapter."""
          key = f"sessions/{session_id}/state.json"
          data = await self._adapter.get(key)
          return json.loads(data) if data else None
  ```

- [ ] **Task 3.2**: Add tests for SessionStorageAdapter
  - File: `tests/unit/test_session_storage_adapter.py`
  - Test all backend types (S3, File, Memory)
  - Test error handling and fallbacks

**Phase 1 Success Criteria**:
- ✅ Storage adapters registered in DI container
- ✅ Configuration loads from YAML with env overrides
- ✅ SessionStorageAdapter works with all backends
- ✅ Tests passing with 85%+ coverage
- ✅ No breaking changes to existing functionality

---

### Phase 2: Backend Consolidation (Days 4-7)

**Goal**: Migrate serverless_mode.py to use SessionStorageAdapter, deprecate old backends.

#### Day 4: Serverless Mode Migration - Part 1

- [ ] **Task 4.1**: Update serverless_mode.py imports
  - Replace `from session_mgmt_mcp.backends` imports
  - Import `SessionStorageAdapter` instead
  - Update type hints to remove old backend references

- [ ] **Task 4.2**: Refactor RedisStorage usage
  - File: `session_mgmt_mcp/serverless_mode.py`
  - Replace RedisStorage with ACB cache adapter (if available)
  - Or use SessionStorageAdapter with S3/file backend

  ```python
  # OLD (backends/redis_backend.py - 200 lines):
  from session_mgmt_mcp.backends.redis_backend import RedisStorage

  storage = RedisStorage(
      host=config.redis_host,
      port=config.redis_port,
      password=config.redis_password,
  )

  # NEW (ACB storage adapter - ~10 lines):
  from session_mgmt_mcp.adapters import SessionStorageAdapter
  from acb.depends import depends

  storage = depends.get_sync(SessionStorageAdapter)
  ```

#### Day 5: Serverless Mode Migration - Part 2

- [ ] **Task 5.1**: Refactor S3Storage usage
  - Replace S3Storage with SessionStorageAdapter
  - Update configuration to use ACB S3 adapter
  - Remove custom boto3 implementation (~280 lines)

  ```python
  # OLD (backends/s3_backend.py - 280 lines):
  from session_mgmt_mcp.backends.s3_backend import S3Storage

  storage = S3Storage(
      bucket_name=config.s3_bucket,
      access_key=config.s3_access_key,
      secret_key=config.s3_secret_key,
      endpoint_url=config.s3_endpoint,
  )

  # NEW (ACB storage adapter - ~5 lines):
  from session_mgmt_mcp.adapters import SessionStorageAdapter

  storage = SessionStorageAdapter(backend="s3")
  ```

- [ ] **Task 5.2**: Refactor LocalFileStorage usage
  - Replace LocalFileStorage with SessionStorageAdapter
  - Update to use ACB file adapter

#### Day 6: Tool Updates & Testing

- [ ] **Task 6.1**: Update serverless tools
  - File: `session_mgmt_mcp/tools/serverless_tools.py`
  - Update all tools to use SessionStorageAdapter
  - Remove backend-specific code

- [ ] **Task 6.2**: Integration tests
  - File: `tests/integration/test_serverless_migration.py`
  - Test S3 backend with SessionStorageAdapter
  - Test File backend with SessionStorageAdapter
  - Test Memory backend for testing scenarios

#### Day 7: Backend Deprecation

- [ ] **Task 7.1**: Add deprecation warnings
  - Files: `backends/s3_backend.py`, `backends/redis_backend.py`, etc.
  - Add `@deprecated` decorators
  - Log warnings when old backends imported

  ```python
  import warnings

  warnings.warn(
      "backends.s3_backend is deprecated. Use adapters.SessionStorageAdapter instead.",
      DeprecationWarning,
      stacklevel=2,
  )
  ```

- [ ] **Task 7.2**: Update documentation
  - File: `CLAUDE.md`
  - Document new storage adapter usage
  - Add migration guide for users

**Phase 2 Success Criteria**:
- ✅ serverless_mode.py uses SessionStorageAdapter
- ✅ All serverless tools updated
- ✅ Integration tests passing
- ✅ Old backends deprecated with warnings
- ✅ Documentation updated

---

### Phase 2.5: Graph Adapter Migration (Days 8-9)

**Goal**: Migrate KnowledgeGraphDatabaseAdapter from raw DuckDB to ACB Graph adapter.

#### Day 8: Graph Adapter Investigation & Setup

- [ ] **Task 8.1**: Audit current KnowledgeGraphDatabaseAdapter
  - File: `session_mgmt_mcp/adapters/knowledge_graph_adapter.py` (current: ~700 lines)
  - Document all DuckDB SQL operations
  - Map to ACB Graph adapter methods
  - Identify any custom operations needing preservation

- [ ] **Task 8.2**: Study ACB Graph adapter API
  - Package: `acb.adapters.graph.duckdb_pgq`
  - Review available methods for entities, relationships, queries
  - Compare with current KnowledgeGraphDatabaseAdapter interface
  - Note: ACB Graph already registered in DI (lines 242+ in di/__init__.py)

  ```python
  # Current state (NOT using ACB):
  import duckdb  # ❌ Direct DuckDB

  class KnowledgeGraphDatabaseAdapter:
      def initialize(self):
          self.conn = duckdb.connect(db_path)  # ❌ Raw SQL
          self.conn.execute("CREATE TABLE kg_entities ...")  # ❌ Manual schema

  # Target state (using ACB Graph):
  from acb.adapters.graph.duckdb_pgq import Graph
  from acb.depends import depends

  class KnowledgeGraphDatabaseAdapter:
      def initialize(self):
          self.graph_adapter = depends.get_sync(Graph)  # ✅ Uses ACB!
          # ACB Graph handles schema automatically
  ```

#### Day 9: Graph Adapter Migration Implementation

- [ ] **Task 9.1**: Refactor entity operations
  - Replace raw SQL `CREATE TABLE kg_entities` with ACB Graph methods
  - Update `create_entity()` to use `graph_adapter.create_node()`
  - Update `get_entity()` to use `graph_adapter.get_node()`
  - Update `update_entity()` to use `graph_adapter.update_node()`

  ```python
  # OLD (~30 lines of SQL):
  async def create_entity(self, name: str, entity_type: str, metadata: dict) -> dict:
      self.conn.execute(
          "INSERT INTO kg_entities (name, type, metadata) VALUES (?, ?, ?)",
          (name, entity_type, json.dumps(metadata))
      )
      return {"id": str(uuid.uuid4()), "name": name}

  # NEW (~5 lines with ACB):
  async def create_entity(self, name: str, entity_type: str, metadata: dict) -> dict:
      node = await self.graph_adapter.create_node(
          label=entity_type,
          properties={"name": name, **metadata}
      )
      return {"id": node.id, "name": name}
  ```

- [ ] **Task 9.2**: Refactor relationship operations
  - Replace raw SQL `CREATE TABLE kg_relationships` with ACB Graph methods
  - Update `create_relationship()` to use `graph_adapter.create_edge()`
  - Update `get_relationships()` to use `graph_adapter.get_edges()`

  ```python
  # OLD (~40 lines of SQL):
  async def create_relationship(self, source_id: str, target_id: str, rel_type: str) -> dict:
      self.conn.execute(
          "INSERT INTO kg_relationships (source, target, type) VALUES (?, ?, ?)",
          (source_id, target_id, rel_type)
      )
      return {"source": source_id, "target": target_id, "type": rel_type}

  # NEW (~5 lines with ACB):
  async def create_relationship(self, source_id: str, target_id: str, rel_type: str) -> dict:
      edge = await self.graph_adapter.create_edge(
          from_id=source_id,
          to_id=target_id,
          label=rel_type
      )
      return {"source": source_id, "target": target_id, "type": rel_type}
  ```

- [ ] **Task 9.3**: Refactor query operations
  - Replace raw SQL queries with ACB Graph traversal methods
  - Update `search_entities()` to use graph_adapter query methods
  - Update `get_connected_entities()` to use graph traversal

  ```python
  # OLD (~50 lines of complex SQL):
  async def search_entities(self, query: str, limit: int = 10) -> list[dict]:
      cursor = self.conn.execute(
          """
          SELECT * FROM kg_entities
          WHERE name LIKE ? OR metadata LIKE ?
          LIMIT ?
          """,
          (f"%{query}%", f"%{query}%", limit)
      )
      return [dict(row) for row in cursor.fetchall()]

  # NEW (~10 lines with ACB):
  async def search_entities(self, query: str, limit: int = 10) -> list[dict]:
      nodes = await self.graph_adapter.query_nodes(
          where={"name__contains": query},
          limit=limit
      )
      return [{"id": node.id, **node.properties} for node in nodes]
  ```

- [ ] **Task 9.4**: Remove DuckDB dependencies
  - Remove `import duckdb` from knowledge_graph_adapter.py
  - Remove manual connection management code
  - Remove custom schema creation SQL
  - Expected reduction: ~700 lines → ~150 lines (78% reduction)

- [ ] **Task 9.5**: Update tests
  - File: `tests/unit/test_knowledge_graph_adapter.py`
  - Update mocks to use ACB Graph adapter
  - Test all entity and relationship operations
  - Verify query functionality

**Phase 2.5 Success Criteria**:
- ✅ KnowledgeGraphDatabaseAdapter uses ACB Graph adapter
- ✅ All entity/relationship operations working
- ✅ Query and traversal functionality maintained
- ✅ ~78% code reduction in knowledge_graph_adapter.py
- ✅ Tests passing with 85%+ coverage
- ✅ No raw DuckDB SQL remaining

**Phase 2.5 Benefits**:
- ✅ Consistent ACB pattern across all adapters
- ✅ Better connection pooling and resource management
- ✅ Improved testability through DI
- ✅ Reduced maintenance burden

---

### Phase 3: Testing & Validation (Days 10-12)

**Goal**: Comprehensive testing across all backends and migration scenarios.

#### Day 10: Integration Tests

- [ ] **Task 10.1**: S3 backend integration tests
  - File: `tests/integration/test_s3_storage.py`
  - Test SessionStorageAdapter with S3 backend
  - Test session persistence and retrieval
  - Test error handling (connection failures, etc.)

- [ ] **Task 10.2**: File backend integration tests
  - File: `tests/integration/test_file_storage.py`
  - Test SessionStorageAdapter with file backend
  - Test file creation, updates, deletions
  - Test concurrent access scenarios

- [ ] **Task 10.3**: Memory backend tests
  - File: `tests/integration/test_memory_storage.py`
  - Test in-memory storage for testing scenarios
  - Test session lifecycle operations

#### Day 11: Migration & Compatibility Tests

- [ ] **Task 11.1**: Backward compatibility tests
  - File: `tests/integration/test_migration_compatibility.py`
  - Test loading sessions created with old backends
  - Test data format compatibility
  - Test graceful degradation when ACB unavailable

- [ ] **Task 11.2**: Graph adapter migration tests
  - File: `tests/integration/test_graph_migration.py`
  - Test migrating existing graph data to ACB format
  - Test entity and relationship preservation
  - Test query result consistency

- [ ] **Task 11.3**: Cross-adapter tests
  - Test Vector + Graph + Storage adapters working together
  - Test DI container isolation and cleanup
  - Test concurrent adapter usage

#### Day 12: Performance & Load Tests

- [ ] **Task 12.1**: Performance benchmarks
  - File: `tests/performance/test_storage_benchmarks.py`
  - Benchmark S3 vs File vs Memory backends
  - Compare with old backend performance
  - Document any regressions

- [ ] **Task 12.2**: Load tests
  - Test 100+ concurrent sessions
  - Test large session state (>1MB)
  - Test rapid session creation/deletion

- [ ] **Task 12.3**: Memory profiling
  - Profile memory usage with ACB adapters
  - Compare with old backend memory footprint
  - Identify any memory leaks

**Phase 3 Success Criteria**:
- ✅ All integration tests passing
- ✅ Migration compatibility verified
- ✅ Performance meets or exceeds old backends
- ✅ No memory leaks detected
- ✅ 85%+ test coverage maintained

---

### Phase 4: Documentation & Cleanup (Days 13-16)

**Goal**: Complete migration with documentation, cleanup, and final validation.

#### Day 13: User Documentation

- [ ] **Task 13.1**: Update CLAUDE.md
  - Document new storage adapter usage
  - Update configuration examples
  - Add troubleshooting guide for ACB adapters

- [ ] **Task 13.2**: Create migration guide
  - File: `docs/MIGRATION_GUIDE_V2.md`
  - Step-by-step migration instructions
  - Configuration migration examples
  - Common issues and solutions

- [ ] **Task 13.3**: Update API documentation
  - Update docstrings for SessionStorageAdapter
  - Update KnowledgeGraphDatabaseAdapter docs
  - Document ACB adapter configuration

#### Day 14: Code Cleanup

- [ ] **Task 14.1**: Remove deprecated backends (after one release)
  - Delete `backends/s3_backend.py` (~280 lines)
  - Delete `backends/redis_backend.py` (~200 lines)
  - Delete `backends/local_backend.py` (~150 lines)
  - Delete `backends/acb_cache_backend.py` (~250 lines)
  - Keep `backends/base.py` (SessionState model only)

- [ ] **Task 14.2**: Update imports across codebase
  - Search for old backend imports
  - Replace with SessionStorageAdapter
  - Update type hints

- [ ] **Task 14.3**: Clean up DI configuration
  - Remove old backend registration code
  - Simplify configure() function
  - Add comments documenting ACB adapter setup

#### Day 15: Final Validation

- [ ] **Task 15.1**: Full test suite run
  - Run `pytest --cov=session_mgmt_mcp --cov-fail-under=85`
  - Verify all tests passing
  - Check coverage metrics

- [ ] **Task 15.2**: Quality checks
  - Run `python -m crackerjack -t`
  - Ensure all hooks passing
  - Verify type checking passes
  - Check code complexity ≤15

- [ ] **Task 15.3**: Manual testing
  - Test session management workflow end-to-end
  - Test with S3, File, and Memory backends
  - Test graph operations with new adapter
  - Verify all MCP tools working

#### Day 16: Release Preparation

- [ ] **Task 16.1**: Update changelog
  - File: `CHANGELOG.md`
  - Document all changes
  - Note breaking changes (if any)
  - List new features and improvements

- [ ] **Task 16.2**: Version bump
  - Update version in `pyproject.toml`
  - Update `__version__` in `__init__.py`
  - Follow semantic versioning

- [ ] **Task 16.3**: Final commit and PR
  - Commit all changes with descriptive message
  - Push to branch `claude/fix-crackerjack-hooks-01SNFACYTnFCvfcLMFBRuKLU`
  - Create PR with comprehensive description

**Phase 4 Success Criteria**:
- ✅ Documentation complete and accurate
- ✅ Deprecated code removed
- ✅ All tests passing
- ✅ Code quality checks passing
- ✅ Ready for release

---

## Success Metrics

### Code Reduction Targets

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| S3 Backend | 280 lines | ~20 lines | 93% |
| Redis Backend | 200 lines | Removed | 100% |
| Local Backend | 150 lines | ~10 lines | 93% |
| ACB Cache Backend | 250 lines | Removed | 100% |
| Knowledge Graph Adapter | 700 lines | ~150 lines | 78% |
| **Total Storage Layer** | **880 lines** | **~80 lines** | **91%** |
| **Total with Graph** | **1580 lines** | **~230 lines** | **85%** |

### Quality Metrics

- ✅ Test coverage: ≥85%
- ✅ Code complexity: ≤15 per function
- ✅ Type checking: 100% passing
- ✅ Security checks: No vulnerabilities
- ✅ Performance: No regressions vs old backends

### Migration Success Indicators

- ✅ Zero breaking changes for users
- ✅ All existing sessions load correctly
- ✅ All MCP tools functioning
- ✅ Configuration backward compatible
- ✅ Documentation complete

---

## Risk Mitigation

### Identified Risks

1. **ACB Adapter API Changes** (Probability: Low, Impact: High)
   - Mitigation: Pin ACB version, abstract with SessionStorageAdapter facade
   - Fallback: Keep old backends deprecated but functional for one release

2. **Performance Regression** (Probability: Medium, Impact: Medium)
   - Mitigation: Comprehensive benchmarking in Phase 3
   - Fallback: Add caching layer if needed

3. **Data Migration Issues** (Probability: Low, Impact: High)
   - Mitigation: Extensive migration compatibility tests
   - Fallback: Provide migration scripts for manual intervention

4. **Graph Adapter Limitations** (Probability: Medium, Impact: Medium)
   - Mitigation: Thoroughly test ACB Graph adapter capabilities first
   - Fallback: Keep hybrid approach (ACB for simple ops, raw DuckDB for complex)

### Rollback Plan

If critical issues discovered:
1. Revert commits on branch
2. Keep old backends active
3. Document issues and adjust plan
4. Re-attempt migration with fixes

---

## Dependencies

### Required ACB Components

- ✅ `acb.adapters.storage.s3` (S3/MinIO support)
- ✅ `acb.adapters.storage.file` (local file storage)
- ✅ `acb.adapters.storage.memory` (in-memory testing)
- ✅ `acb.adapters.graph.duckdb_pgq` (knowledge graph)
- ✅ `acb.adapters.vector.duckdb` (already used in ReflectionDatabaseAdapter)
- ✅ `acb.depends` (dependency injection)
- ✅ `acb.config` (configuration management)

### External Dependencies

No new external dependencies required - all ACB components already available.

---

## Progress Tracking

### Overall Progress: 3/58 tasks completed (5%)

- **Phase 0**: ✅ 3/3 completed (100%)
- **Phase 1**: ⬜ 0/8 completed (0%)
- **Phase 2**: ⬜ 0/10 completed (0%)
- **Phase 2.5**: ⬜ 0/9 completed (0%)
- **Phase 3**: ⬜ 0/10 completed (0%)
- **Phase 4**: ⬜ 0/18 completed (0%)

### Last Updated: 2025-01-16

**Current Status**: Phase 0 complete, ready to begin Phase 1.

---

## References

### Key Files

- **Storage Adapters**: `session_mgmt_mcp/adapters/storage_registry.py` (to be created)
- **Session Storage**: `session_mgmt_mcp/adapters/session_storage_adapter.py` (to be created)
- **Graph Adapter**: `session_mgmt_mcp/adapters/knowledge_graph_adapter.py` (existing, to be refactored)
- **Reflection Adapter**: `session_mgmt_mcp/adapters/reflection_adapter.py` (already ACB-compliant)
- **DI Config**: `session_mgmt_mcp/di/__init__.py` (existing, to be updated)
- **Serverless Mode**: `session_mgmt_mcp/serverless_mode.py` (existing, to be migrated)

### ACB Documentation

- Vector Adapter: `acb.adapters.vector.duckdb`
- Graph Adapter: `acb.adapters.graph.duckdb_pgq`
- Storage Adapters: `acb.adapters.storage.*`
- DI System: `acb.depends`

### Related Documents

- `CLAUDE.md` - Project development guidelines
- `docs/ACB_MIGRATION_COMPLETE.md` - Previous Vector/Graph migration docs
- `docs/refactoring/` - Historical refactoring documentation

---

## Notes

- This migration follows the same successful pattern used for Vector and Graph adapters
- ReflectionDatabaseAdapter already uses ACB Vector - serves as reference implementation
- Graph adapter is registered in DI but currently unused - opportunity identified
- 90% success probability based on proven ACB adapter track record
- Timeline assumes 1-2 developers working concurrently on different phases

---

**Document Version**: 1.0
**Last Modified**: 2025-01-16
**Author**: Claude Code Migration Team
