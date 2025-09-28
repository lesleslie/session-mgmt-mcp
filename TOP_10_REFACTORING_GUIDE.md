# Top 10 Functions Refactoring Guide

This document provides specific, actionable refactoring guidance for the 10 most complex functions in session-mgmt-mcp, following crackerjack's clean code philosophy.

## Function Analysis & Refactoring Plans

### 1. WorktreeManager::create_worktree (Complexity: 21)
**Location:** `session_mgmt_mcp/worktree_manager.py`
**Current Issues:** Git operations, validation, session setup, and error recovery all mixed together

#### Refactoring Blueprint
```python
# BEFORE: Single complex function with multiple responsibilities
async def create_worktree(self, path: Path, branch: str, **options) -> str:
    # 100+ lines of mixed validation, git ops, session setup, error handling

# AFTER: Clean orchestration with focused helpers
async def create_worktree(self, path: Path, branch: str, options: WorktreeOptions) -> WorktreeResult:
    """Create new git worktree with session setup. Target complexity: ≤8"""
    # 1. Validate request (complexity ≤3)
    validation = self._validate_worktree_creation(path, branch, options)
    if not validation.is_valid:
        return WorktreeResult.validation_error(validation.errors)

    # 2. Execute git operations (complexity ≤8)
    git_result = await self._execute_git_worktree_add(path, branch, options)
    if not git_result.success:
        return WorktreeResult.git_error(git_result.error)

    # 3. Setup session if requested (complexity ≤5)
    if options.initialize_session:
        session_result = await self._initialize_worktree_session(path)
        return WorktreeResult.with_session(git_result, session_result)

    return WorktreeResult.success(git_result)

@dataclass
class WorktreeOptions:
    """Immutable worktree creation options."""
    initialize_session: bool = True
    checkout_branch: bool = True
    track_remote: bool = False
    force: bool = False

def _validate_worktree_creation(self, path: Path, branch: str, options: WorktreeOptions) -> ValidationResult:
    """Validate worktree creation parameters. Target complexity: ≤3"""
    errors = []

    if path.exists():
        errors.append(f"Path already exists: {path}")

    if not self._is_valid_branch_name(branch):
        errors.append(f"Invalid branch name: {branch}")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

**Key Improvements:**
- Separated validation, git operations, and session setup
- Used immutable data structures for options and results
- Clear error handling without deep nesting
- Each helper function has single responsibility

---

### 2. SessionLifecycleManager::_read_previous_session_info (Complexity: 20)
**Location:** `session_mgmt_mcp/core/session_manager.py`
**Current Issues:** File I/O, JSON parsing, validation, and fallback logic mixed

#### Refactoring Blueprint
```python
# BEFORE: Complex file reading with multiple formats and error recovery
async def _read_previous_session_info(self, working_dir: Path) -> dict[str, Any]:
    # Complex logic handling multiple file formats, parsing, validation

# AFTER: Clean separation of concerns
async def _read_previous_session_info(self, working_dir: Path) -> SessionInfo:
    """Read previous session information. Target complexity: ≤8"""
    session_files = self._discover_session_files(working_dir)

    for file_path in session_files:
        session_info = await self._parse_session_file(file_path)
        if session_info.is_complete():
            return session_info

    return SessionInfo.empty()

def _discover_session_files(self, working_dir: Path) -> list[Path]:
    """Find potential session files in priority order. Target complexity: ≤3"""
    candidates = [
        working_dir / ".claude" / "session.json",
        working_dir / ".claude" / "last_session.json",
        working_dir / "session_backup.json",
    ]
    return [path for path in candidates if path.exists()]

async def _parse_session_file(self, file_path: Path) -> SessionInfo:
    """Parse single session file with error handling. Target complexity: ≤8"""
    try:
        content = await self._read_file_safely(file_path)
        data = json.loads(content)
        return SessionInfo.from_dict(data)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Failed to parse session file {file_path}: {e}")
        return SessionInfo.empty()

@dataclass(frozen=True)
class SessionInfo:
    """Immutable session information."""
    session_id: str = ""
    last_activity: datetime | None = None
    quality_score: int = 0
    working_directory: str = ""
    context_data: dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if session info has required fields."""
        return bool(self.session_id and self.last_activity and self.working_directory)

    @classmethod
    def empty(cls) -> SessionInfo:
        """Create empty session info."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionInfo:
        """Create from dictionary with validation."""
        return cls(
            session_id=data.get("session_id", ""),
            last_activity=datetime.fromisoformat(data["last_activity"]) if data.get("last_activity") else None,
            quality_score=int(data.get("quality_score", 0)),
            working_directory=data.get("working_directory", ""),
            context_data=data.get("context_data", {})
        )
```

---

### 3. LLMManager::stream_generate (Complexity: 20)
**Location:** `session_mgmt_mcp/llm_providers.py`
**Current Issues:** Provider selection, streaming setup, error handling, and response processing mixed

#### Refactoring Blueprint
```python
# BEFORE: Complex streaming with provider management and error recovery
async def stream_generate(self, messages, **kwargs) -> AsyncGenerator[str, None]:
    # Complex provider selection, streaming, error handling logic

# AFTER: Clean separation with focused components
async def stream_generate(self, messages: list[Message], options: GenerationOptions) -> AsyncGenerator[StreamChunk, None]:
    """Generate streaming response. Target complexity: ≤8"""
    provider = await self._select_provider(options.preferred_provider)
    if not provider.is_available():
        yield StreamChunk.error("No available providers")
        return

    try:
        async for chunk in self._stream_from_provider(provider, messages, options):
            yield chunk
    except ProviderError as e:
        if provider.supports_fallback():
            fallback_provider = await self._get_fallback_provider(provider)
            async for chunk in self._stream_from_provider(fallback_provider, messages, options):
                yield chunk
        else:
            yield StreamChunk.error(f"Provider error: {e}")

@dataclass(frozen=True)
class GenerationOptions:
    """Immutable generation configuration."""
    preferred_provider: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = True
    use_fallback: bool = True

@dataclass(frozen=True)
class StreamChunk:
    """Immutable streaming response chunk."""
    content: str = ""
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def error(cls, message: str) -> StreamChunk:
        return cls(content="", is_error=True, metadata={"error": message})

async def _select_provider(self, preferred: str) -> LLMProvider:
    """Select appropriate provider. Target complexity: ≤5"""
    if preferred != "auto" and preferred in self.providers:
        return self.providers[preferred]

    for provider in self.providers.values():
        if provider.is_available():
            return provider

    raise ProviderError("No available providers")
```

---

### 4. server.py::advanced_search (Complexity: 20)
**Location:** `session_mgmt_mcp/server.py`
**Current Issues:** Query building, filtering, result processing, and formatting mixed

#### Refactoring Blueprint
```python
# BEFORE: Complex search with mixed concerns
async def advanced_search(query: str, **kwargs) -> str:
    # Complex query building, filtering, processing, formatting logic

# AFTER: Clean search pipeline
async def advanced_search(query: str, **search_params) -> str:
    """Execute advanced search. Target complexity: ≤8"""
    search_request = SearchRequest.from_params(query, search_params)

    search_engine = await self._get_search_engine()
    search_result = await search_engine.execute_search(search_request)

    formatter = SearchResultFormatter()
    return formatter.format_results(search_result)

@dataclass(frozen=True)
class SearchRequest:
    """Immutable search request."""
    query: str
    filters: list[SearchFilter] = field(default_factory=list)
    sort_by: str = "relevance"
    limit: int = 20
    include_facets: bool = False

    @classmethod
    def from_params(cls, query: str, params: dict[str, Any]) -> SearchRequest:
        """Create search request from parameters."""
        filters = []
        if content_type := params.get("content_type"):
            filters.append(SearchFilter.content_type(content_type))
        if timeframe := params.get("timeframe"):
            filters.append(SearchFilter.timeframe(timeframe))

        return cls(
            query=query,
            filters=filters,
            sort_by=params.get("sort_by", "relevance"),
            limit=int(params.get("limit", 20)),
            include_facets=params.get("facets", False)
        )

class SearchResultFormatter:
    """Format search results for display."""

    def format_results(self, result: SearchResult) -> str:
        """Format search results. Target complexity: ≤8"""
        output_builder = OutputBuilder()

        output_builder.add_header(f"🔍 Search Results ({len(result.results)} found)")

        if result.results:
            self._add_results_section(output_builder, result.results)

        if result.facets:
            self._add_facets_section(output_builder, result.facets)

        return output_builder.build()

    def _add_results_section(self, output_builder: OutputBuilder, results: list[SearchMatch]) -> None:
        """Add results to output. Target complexity: ≤5"""
        for i, match in enumerate(results, 1):
            output_builder.add_section(f"Result {i}", [
                f"Content: {match.content[:200]}...",
                f"Score: {match.score:.2f}",
                f"Source: {match.source}"
            ])
```

---

### 5. session_tools.py::_start_impl (Complexity: 17)
**Location:** `session_mgmt_mcp/tools/session_tools.py`
**Current Issues:** Session init, UV setup, shortcuts creation, and output formatting mixed

#### Refactoring Blueprint
```python
# BEFORE: Mixed initialization and formatting logic
async def _start_impl(working_directory: str | None = None) -> str:
    # Complex initialization with inline output formatting

# AFTER: Clean orchestration with formatting separation
async def _start_impl(working_directory: str | None = None) -> str:
    """Initialize session with comprehensive setup. Target complexity: ≤8"""
    formatter = SessionOutputFormatter()

    try:
        init_result = await session_manager.initialize_session(working_directory)

        if init_result.success:
            setup_results = await self._perform_environment_setup(init_result)
            return formatter.format_successful_start(init_result, setup_results)
        else:
            return formatter.format_start_error(init_result.error)

    except Exception as e:
        logger.exception("Session initialization error")
        return formatter.format_unexpected_error(str(e))

@dataclass
class EnvironmentSetupResults:
    """Results from environment setup operations."""
    uv_setup: UVSetupResult
    shortcuts_setup: ShortcutsResult

async def _perform_environment_setup(self, init_result: SessionInitResult) -> EnvironmentSetupResults:
    """Perform all environment setup tasks. Target complexity: ≤5"""
    working_dir = Path(init_result.working_directory)

    uv_result = self._setup_uv_dependencies(working_dir)
    shortcuts_result = self._create_session_shortcuts()

    return EnvironmentSetupResults(
        uv_setup=uv_result,
        shortcuts_setup=shortcuts_result
    )

class SessionOutputFormatter:
    """Format session management output."""

    def format_successful_start(self, init_result: SessionInitResult, setup_results: EnvironmentSetupResults) -> str:
        """Format successful start output. Target complexity: ≤8"""
        output_builder = OutputBuilder()
        output_builder.add_header("🚀 Claude Session Initialization via MCP Server")

        self._add_session_info(output_builder, init_result)
        self._add_environment_info(output_builder, setup_results)
        self._add_recommendations(output_builder, init_result.recommendations)

        output_builder.add_section("", ["✅ Session initialization completed successfully!"])
        return output_builder.build()
```

---

### 6. session_tools.py::_status_impl (Complexity: 17)
**Location:** `session_mgmt_mcp/tools/session_tools.py`
**Current Issues:** Status gathering, quality calculation, and output formatting mixed

#### Refactoring Blueprint
```python
# BEFORE: Mixed status gathering and formatting
async def _status_impl(working_directory: str | None = None) -> str:
    # Complex status collection with inline formatting

# AFTER: Clean separation of data and presentation
async def _status_impl(working_directory: str | None = None) -> str:
    """Get comprehensive session status. Target complexity: ≤8"""
    status_collector = SessionStatusCollector()
    formatter = SessionStatusFormatter()

    try:
        status_data = await status_collector.collect_all_status(working_directory)
        return formatter.format_status(status_data)
    except Exception as e:
        logger.exception("Status check error")
        return formatter.format_status_error(str(e))

@dataclass
class ComprehensiveStatus:
    """Complete session status information."""
    project_info: ProjectInfo
    quality_metrics: QualityMetrics
    system_health: SystemHealth
    recommendations: list[str]
    timestamp: datetime = field(default_factory=datetime.now)

class SessionStatusCollector:
    """Collect session status data."""

    async def collect_all_status(self, working_directory: str | None) -> ComprehensiveStatus:
        """Collect comprehensive status information. Target complexity: ≤8"""
        session_result = await session_manager.get_session_status(working_directory)

        if not session_result.success:
            raise SessionStatusError(session_result.error)

        return ComprehensiveStatus(
            project_info=self._extract_project_info(session_result),
            quality_metrics=self._extract_quality_metrics(session_result),
            system_health=self._extract_system_health(session_result),
            recommendations=session_result.get("recommendations", [])
        )

    def _extract_project_info(self, result: dict[str, Any]) -> ProjectInfo:
        """Extract project information. Target complexity: ≤3"""
        return ProjectInfo(
            name=result.get("project", "Unknown"),
            working_directory=result.get("working_directory", ""),
            quality_score=result.get("quality_score", 0)
        )

class SessionStatusFormatter:
    """Format session status for display."""

    def format_status(self, status: ComprehensiveStatus) -> str:
        """Format comprehensive status. Target complexity: ≤8"""
        output_builder = OutputBuilder()
        output_builder.add_header("📊 Claude Session Status Report")

        self._add_project_section(output_builder, status.project_info)
        self._add_quality_section(output_builder, status.quality_metrics)
        self._add_health_section(output_builder, status.system_health)
        self._add_recommendations_section(output_builder, status.recommendations)

        output_builder.add_section("", [f"⏰ Status generated: {status.timestamp}"])
        return output_builder.build()
```

---

### 7-10. Additional Priority Functions

#### 7. AdvancedSearchEngine::_update_search_facets (Complexity: 17)
**Key Refactoring:** Extract facet calculation, indexing, and storage into separate methods
```python
async def _update_search_facets(self, facets: list[str]) -> None:
    """Update search facets. Target complexity: ≤8"""
    for facet_type in facets:
        calculator = self._get_facet_calculator(facet_type)
        facet_data = await calculator.calculate_facet_data()
        await self._store_facet_data(facet_type, facet_data)
```

#### 8. LLMManager::generate (Complexity: 19)
**Key Refactoring:** Separate provider selection, generation, and response processing
```python
async def generate(self, messages: list[Message], options: GenerationOptions) -> GenerationResult:
    """Generate response. Target complexity: ≤8"""
    provider = await self._select_best_provider(options)
    raw_response = await provider.generate(messages, options)
    return self._process_generation_response(raw_response, options)
```

#### 9. CrackerjackOutputParser::_parse_progress_output (Complexity: 17)
**Key Refactoring:** Extract parsing rules, validation, and result building
```python
def _parse_progress_output(self, output: str) -> ProgressResult:
    """Parse crackerjack progress output. Target complexity: ≤8"""
    lines = self._normalize_output_lines(output)
    parsed_sections = self._extract_output_sections(lines)
    return self._build_progress_result(parsed_sections)
```

#### 10. search_tools.py::_search_by_concept_impl (Complexity: 19)
**Key Refactoring:** Separate concept matching, result ranking, and formatting
```python
async def _search_by_concept_impl(concept: str, **params) -> str:
    """Search by concept implementation. Target complexity: ≤8"""
    concept_matcher = ConceptMatcher()
    matches = await concept_matcher.find_concept_matches(concept, params)

    ranker = ConceptResultRanker()
    ranked_results = ranker.rank_by_relevance(matches, concept)

    formatter = ConceptSearchFormatter()
    return formatter.format_concept_results(concept, ranked_results)
```

## Common Refactoring Patterns

### Pattern 1: Orchestration + Helpers
```python
# Main function: Orchestrate calls, handle high-level errors
async def main_function(params) -> Result:
    """Target complexity: ≤8"""
    validation_result = validate_params(params)
    if not validation_result.is_valid:
        return Result.validation_error(validation_result)

    processing_result = await process_core_logic(params)
    return format_final_result(processing_result)

# Helper functions: Single responsibility, ≤5 complexity each
def validate_params(params) -> ValidationResult: ...
async def process_core_logic(params) -> ProcessingResult: ...
def format_final_result(result) -> Result: ...
```

### Pattern 2: Data-Driven Processing
```python
# Use immutable data structures to pass context
@dataclass(frozen=True)
class ProcessingContext:
    input_data: InputData
    options: ProcessingOptions
    metadata: dict[str, Any] = field(default_factory=dict)

# Process in pipeline stages
async def process_pipeline(context: ProcessingContext) -> ProcessingResult:
    stage1_result = await stage1_processor.process(context)
    stage2_result = await stage2_processor.process(stage1_result)
    return finalize_processing(stage2_result)
```

### Pattern 3: Error Handling Separation
```python
# Centralized error handling
@dataclass
class OperationResult:
    success: bool
    data: Any | None = None
    error: str = ""

    @classmethod
    def success_result(cls, data: Any) -> OperationResult:
        return cls(success=True, data=data)

    @classmethod
    def error_result(cls, error: str) -> OperationResult:
        return cls(success=False, error=error)

# Use consistent error patterns
async def operation_with_error_handling(params) -> OperationResult:
    try:
        result = await perform_operation(params)
        return OperationResult.success_result(result)
    except SpecificError as e:
        logger.error(f"Operation failed: {e}")
        return OperationResult.error_result(str(e))
```

## Implementation Checklist

For each function refactoring:

- [ ] **Pre-refactoring:**
  - [ ] Write comprehensive tests for existing behavior
  - [ ] Document current complexity score
  - [ ] Identify all function callers
  - [ ] Map out current responsibilities

- [ ] **During refactoring:**
  - [ ] Extract one responsibility at a time
  - [ ] Create immutable data structures for communication
  - [ ] Apply single responsibility principle
  - [ ] Use modern Python 3.13+ patterns
  - [ ] Maintain existing API contracts

- [ ] **Post-refactoring:**
  - [ ] Verify complexity ≤15 (target ≤8 for main functions)
  - [ ] Ensure all tests pass
  - [ ] Update type hints and documentation
  - [ ] Review error handling completeness
  - [ ] Check performance hasn't degraded

This guide provides concrete, actionable steps to systematically reduce complexity while maintaining functionality and improving code quality according to crackerjack standards.