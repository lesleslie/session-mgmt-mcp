# Session-mgmt-mcp Complexity Reduction Strategy

## Executive Summary

This document provides a comprehensive strategy to reduce cognitive complexity across 31 functions that violate crackerjack's ≤15 complexity rule. The current complexity ranges from 16-21, with systematic patterns that can be addressed through strategic refactoring.

## Root Cause Analysis

### Primary Complexity Sources

1. **Single Responsibility Principle Violations**
   - Functions handling validation, processing, error handling, AND formatting
   - Mixed concerns in single functions (e.g., `_start_impl`, `_status_impl`)

2. **Complex Conditional Logic**
   - Deep nested if/else statements
   - Multiple code paths without clear separation
   - Configuration validation mixed with business logic

3. **Scattered Error Handling**
   - Try/catch blocks with complex recovery logic
   - Error formatting mixed with error handling
   - Repetitive error patterns not centralized

4. **Output Formatting Complexity**
   - String building mixed with core logic
   - Multiple formatting concerns in single functions
   - Repetitive output patterns

5. **Configuration Management Complexity**
   - Backend-specific logic not abstracted
   - Validation mixed with processing
   - Multiple storage types handled inline

## Strategic Refactoring Approach

### Phase 1: Extract Common Patterns (Foundation)

#### 1.1 Error Handling Patterns
```python
# Create centralized error handling utilities
class SessionError(Exception):
    """Base session management error."""
    pass

@dataclass
class ErrorResult:
    """Standardized error response."""
    success: bool = False
    error: str = ""
    context: dict[str, Any] = field(default_factory=dict)

def handle_session_error(operation: str) -> callable:
    """Decorator for consistent error handling."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"{operation} failed: {e}")
                return ErrorResult(error=str(e), context={"operation": operation})
        return wrapper
    return decorator
```

#### 1.2 Output Formatting Patterns
```python
@dataclass
class OutputBuilder:
    """Centralized output formatting with consistent styling."""
    sections: list[str] = field(default_factory=list)

    def add_header(self, title: str, level: int = 1) -> None:
        """Add formatted header."""
        separator = "=" * (60 if level == 1 else 50)
        self.sections.extend([title, separator])

    def add_section(self, title: str, items: list[str]) -> None:
        """Add formatted section with items."""
        self.sections.append(f"\n{title}:")
        self.sections.extend(f"   • {item}" for item in items)

    def add_status_item(self, name: str, status: bool, value: str = "") -> None:
        """Add status indicator item."""
        icon = "✅" if status else "❌"
        display = f"{name}: {icon}"
        if value:
            display += f" {value}"
        self.sections.append(f"   • {display}")

    def build(self) -> str:
        """Build final output string."""
        return "\n".join(self.sections)
```

#### 1.3 Configuration Validation Patterns
```python
@dataclass
class ConfigValidationResult:
    """Configuration validation result."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

class ConfigValidator:
    """Centralized configuration validation."""

    @staticmethod
    def validate_storage_config(backend: str, config: dict[str, Any]) -> ConfigValidationResult:
        """Validate storage backend configuration."""
        # Extract validation logic from complex functions
        pass

    @staticmethod
    def validate_session_config(config: dict[str, Any]) -> ConfigValidationResult:
        """Validate session configuration."""
        pass
```

### Phase 2: Priority Refactoring (High-Impact Functions)

#### Priority 1: Critical Functions (Complexity 20-21)

##### 2.1 WorktreeManager::create_worktree (Complexity: 21)
**Current Issues:**
- Mixed validation, git operations, error handling, and output formatting
- Complex branching logic for different scenarios
- Error recovery mixed with main logic

**Refactoring Strategy:**
```python
# Extract into smaller, focused functions
async def create_worktree(self, path: Path, branch: str, options: WorktreeOptions) -> WorktreeResult:
    """Main orchestration function - complexity target: ≤8"""
    validation_result = self._validate_worktree_request(path, branch, options)
    if not validation_result.valid:
        return WorktreeResult.from_validation_error(validation_result)

    git_result = await self._execute_worktree_creation(path, branch, options)
    if not git_result.success:
        return WorktreeResult.from_git_error(git_result)

    session_result = await self._setup_worktree_session(path, git_result)
    return WorktreeResult.from_session_setup(session_result)

def _validate_worktree_request(self, path: Path, branch: str, options: WorktreeOptions) -> ValidationResult:
    """Validate worktree creation request - complexity target: ≤5"""
    # Pure validation logic, no side effects

async def _execute_worktree_creation(self, path: Path, branch: str, options: WorktreeOptions) -> GitOperationResult:
    """Execute git worktree creation - complexity target: ≤8"""
    # Pure git operations with error handling

async def _setup_worktree_session(self, path: Path, git_result: GitOperationResult) -> SessionSetupResult:
    """Setup session for new worktree - complexity target: ≤5"""
    # Session initialization logic
```

##### 2.2 SessionLifecycleManager::_read_previous_session_info (Complexity: 20)
**Current Issues:**
- File reading, JSON parsing, validation, and error recovery in single function
- Complex fallback logic for different file formats
- Mixed I/O and business logic

**Refactoring Strategy:**
```python
async def _read_previous_session_info(self, working_dir: Path) -> SessionInfo:
    """Main function - complexity target: ≤8"""
    session_files = self._discover_session_files(working_dir)
    for file_path in session_files:
        session_info = await self._try_read_session_file(file_path)
        if session_info.is_valid():
            return session_info
    return SessionInfo.empty()

def _discover_session_files(self, working_dir: Path) -> list[Path]:
    """Find potential session files - complexity target: ≤3"""
    # File discovery logic only

async def _try_read_session_file(self, file_path: Path) -> SessionInfo:
    """Attempt to read and parse session file - complexity target: ≤8"""
    # File reading and parsing with error handling

@dataclass
class SessionInfo:
    """Immutable session information data."""
    session_id: str = ""
    last_activity: str = ""
    quality_score: int = 0
    context: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if session info is valid."""
        return bool(self.session_id and self.last_activity)

    @classmethod
    def empty(cls) -> SessionInfo:
        """Create empty session info."""
        return cls()
```

#### Priority 2: High-Impact Functions (Complexity 18-19)

##### 2.3 session_tools.py::_start_impl (Complexity: 17)
**Current Issues:**
- Session initialization, UV setup, output formatting, and error handling mixed
- Multiple unrelated concerns in single function
- Complex string building with business logic

**Refactoring Strategy:**
```python
async def _start_impl(working_directory: str | None = None) -> str:
    """Main implementation - complexity target: ≤8"""
    output_builder = OutputBuilder()
    output_builder.add_header("🚀 Claude Session Initialization via MCP Server")

    session_result = await self._initialize_session_core(working_directory)
    self._add_session_status_to_output(output_builder, session_result)

    if session_result.success:
        uv_result = self._setup_uv_environment(Path(session_result.working_directory))
        self._add_uv_status_to_output(output_builder, uv_result)

        shortcuts_result = self._create_session_shortcuts()
        self._add_shortcuts_status_to_output(output_builder, shortcuts_result)

    return output_builder.build()

async def _initialize_session_core(self, working_directory: str | None) -> SessionInitResult:
    """Core session initialization logic - complexity target: ≤8"""
    # Pure session initialization without output formatting

def _setup_uv_environment(self, working_dir: Path) -> UVSetupResult:
    """UV environment setup - complexity target: ≤8"""
    # UV-specific logic only

def _add_session_status_to_output(self, output_builder: OutputBuilder, result: SessionInitResult) -> None:
    """Add session status to output - complexity target: ≤5"""
    # Pure output formatting
```

##### 2.4 session_tools.py::_status_impl (Complexity: 17)
**Current Issues:**
- Status gathering, quality calculation, and output formatting mixed
- Multiple data sources queried in single function
- Complex conditional output logic

**Refactoring Strategy:**
```python
async def _status_impl(working_directory: str | None = None) -> str:
    """Main implementation - complexity target: ≤8"""
    output_builder = OutputBuilder()
    output_builder.add_header("📊 Claude Session Status Report")

    status_data = await self._gather_status_data(working_directory)
    if status_data.success:
        self._add_project_info_to_output(output_builder, status_data.project_info)
        self._add_quality_breakdown_to_output(output_builder, status_data.quality_data)
        self._add_system_health_to_output(output_builder, status_data.health_data)
        self._add_recommendations_to_output(output_builder, status_data.recommendations)
    else:
        output_builder.add_section("❌ Status Error", [status_data.error])

    return output_builder.build()

@dataclass
class StatusData:
    """Consolidated status information."""
    success: bool
    project_info: ProjectInfo
    quality_data: QualityData
    health_data: HealthData
    recommendations: list[str]
    error: str = ""

async def _gather_status_data(self, working_directory: str | None) -> StatusData:
    """Gather all status data - complexity target: ≤8"""
    # Data collection logic only

def _add_quality_breakdown_to_output(self, output_builder: OutputBuilder, quality_data: QualityData) -> None:
    """Add quality breakdown to output - complexity target: ≤5"""
    # Pure formatting logic
```

### Phase 3: Medium Priority Functions (Complexity 16-18)

#### 3.1 Common Patterns for Search Functions
```python
# Extract search result processing patterns
@dataclass
class SearchRequest:
    """Immutable search request."""
    query: str
    filters: list[SearchFilter]
    options: SearchOptions

@dataclass
class SearchResult:
    """Immutable search result."""
    results: list[Any]
    facets: dict[str, Any]
    metadata: SearchMetadata

class SearchProcessor:
    """Centralized search processing logic."""

    async def process_search(self, request: SearchRequest) -> SearchResult:
        """Main search processing - complexity target: ≤8"""
        query_builder = self._create_query_builder(request)
        raw_results = await self._execute_search_query(query_builder)
        processed_results = await self._process_search_results(raw_results, request.options)
        facets = await self._calculate_facets(request) if request.options.include_facets else {}

        return SearchResult(
            results=processed_results,
            facets=facets,
            metadata=SearchMetadata.from_request(request)
        )
```

#### 3.2 Serverless Storage Configuration Pattern
```python
# Extract configuration handling patterns
class StorageConfigurationHandler:
    """Centralized storage configuration management."""

    async def configure_storage(self, backend: str, config_updates: dict[str, Any]) -> ConfigurationResult:
        """Main configuration - complexity target: ≤8"""
        validator = self._get_validator_for_backend(backend)
        validation_result = validator.validate(config_updates)

        if not validation_result.valid:
            return ConfigurationResult.from_validation_error(validation_result)

        storage_result = await self._apply_configuration(backend, config_updates)
        return ConfigurationResult.from_storage_result(storage_result)

    def _get_validator_for_backend(self, backend: str) -> ConfigValidator:
        """Get appropriate validator - complexity target: ≤3"""
        # Simple factory method

    async def _apply_configuration(self, backend: str, config: dict[str, Any]) -> StorageResult:
        """Apply configuration to storage - complexity target: ≤8"""
        # Pure storage configuration logic
```

### Phase 4: Implementation Guidelines

#### 4.1 Refactoring Process

1. **Before Refactoring:**
   - Write comprehensive tests for existing functionality
   - Document current behavior and edge cases
   - Identify all callers of the function

2. **During Refactoring:**
   - Extract one concern at a time (validation → processing → formatting)
   - Maintain existing API contracts where possible
   - Use immutable data structures for data transfer
   - Apply single responsibility principle rigorously

3. **After Refactoring:**
   - Verify all tests pass
   - Check complexity with complexipy
   - Update documentation and type hints
   - Review error handling coverage

#### 4.2 Code Quality Rules

1. **Complexity Targets:**
   - Main orchestration functions: ≤8 complexity
   - Helper functions: ≤5 complexity
   - Pure validation/formatting functions: ≤3 complexity

2. **Function Responsibilities:**
   - **Orchestration:** Coordinate calls to helper functions
   - **Validation:** Pure validation logic, no side effects
   - **Processing:** Core business logic, minimal I/O
   - **Formatting:** Pure output formatting, no business logic

3. **Error Handling:**
   - Use consistent error types and patterns
   - Centralize error formatting
   - Separate error handling from business logic

4. **Data Flow:**
   - Use immutable data structures for inter-function communication
   - Avoid global state modifications in helper functions
   - Prefer explicit parameters over implicit dependencies

#### 4.3 Modern Python 3.13+ Patterns

```python
# Use pipe unions consistently
def process_result(data: dict[str, Any] | None) -> ProcessingResult | None:
    """Process result data."""
    pass

# Use dataclasses for structured data
@dataclass
class ProcessingContext:
    """Processing context with clear types."""
    working_dir: Path
    options: ProcessingOptions
    session_id: str | None = None

# Use proper async/await patterns
async def process_with_timeout(operation: Callable[[], Awaitable[T]], timeout_seconds: int = 30) -> T | None:
    """Process with timeout handling."""
    try:
        return await asyncio.wait_for(operation(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout_seconds}s")
        return None
```

## Implementation Priority Matrix

### Immediate Action Required (Week 1)

1. **WorktreeManager::create_worktree** (Complexity: 21)
   - **Impact:** High - Critical git operations
   - **Risk:** High - Complex error scenarios
   - **Effort:** Medium - Clear separation of concerns possible

2. **SessionLifecycleManager::_read_previous_session_info** (Complexity: 20)
   - **Impact:** High - Session continuity depends on this
   - **Risk:** Medium - Well-defined inputs/outputs
   - **Effort:** Low - Mostly I/O and parsing logic

3. **LLMManager::stream_generate** (Complexity: 20)
   - **Impact:** High - Core LLM functionality
   - **Risk:** High - Complex async operations
   - **Effort:** High - Multiple provider handling

### High Priority (Week 2)

4. **session_tools.py::_start_impl** (Complexity: 17)
5. **session_tools.py::_status_impl** (Complexity: 17)
6. **AdvancedSearchEngine::_update_search_facets** (Complexity: 17)

### Medium Priority (Week 3-4)

7. **serverless_tools.py::_configure_serverless_storage_impl** (Complexity: 16)
8. **search_tools.py::_search_by_concept_impl** (Complexity: 19)
9. **CrackerjackOutputParser::_parse_progress_output** (Complexity: 17)
10. **RedisStorage::cleanup_expired_sessions** (Complexity: 18)

## Success Metrics

### Quantitative Metrics
- **Primary Goal:** All functions ≤15 complexity (31 functions to fix)
- **Stretch Goal:** 80% of functions ≤10 complexity
- **Code Coverage:** Maintain >95% coverage throughout refactoring
- **Performance:** No degradation in critical path functions

### Qualitative Metrics
- **Maintainability:** Easier to understand and modify functions
- **Testability:** Each concern can be tested independently
- **Debuggability:** Clear separation makes issue isolation easier
- **Documentation:** Self-documenting code with clear responsibilities

## Conclusion

This systematic approach addresses complexity through architectural improvements rather than superficial changes. By extracting common patterns, separating concerns, and applying crackerjack's clean code philosophy consistently, we can reduce cognitive complexity while improving maintainability and testability.

The phased approach ensures critical functions are addressed first while building reusable patterns that benefit the entire codebase. Each refactoring maintains existing functionality while making the code more aligned with modern Python patterns and crackerjack standards.