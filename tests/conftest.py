#!/usr/bin/env python3
"""Global test configuration and fixtures for session-mgmt-mcp tests."""

import asyncio
import os
import tempfile
from collections.abc import AsyncGenerator, Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import duckdb
import numpy as np
import pytest
from fastmcp import FastMCP
from session_mgmt_mcp.reflection_tools import ReflectionDatabase

# Configure DI container BEFORE any other imports
# This ensures SessionLogger and other dependencies are available
from session_mgmt_mcp.di import configure as configure_di
from acb.depends import depends

# Initialize DI container for tests - force registration to bypass async checks
try:
    configure_di(force=True)
except Exception as e:
    # If DI configuration fails during import, we'll retry in the fixture
    import warnings
    warnings.warn(f"DI configuration failed during conftest import: {e}")


# =====================================
# MockFastMCP for Testing (Phase 2.6)
# =====================================


class MockFastMCP:
    """Minimal mock FastMCP for testing environments.

    Extracted from server.py Phase 2.6 to consolidate test infrastructure.
    """

    def __init__(self, name: str) -> None:
        """Initialize mock FastMCP server."""
        self.name = name
        self.tools: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Mock tool decorator."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def prompt(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Mock prompt decorator."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def run(self, *args: Any, **kwargs: Any) -> None:
        """Mock run method."""
        pass

# Import test factories for enhanced test data generation
# Note: After Phase 2.7 cleanup, only actively-used factories are imported
try:
    from tests.fixtures.data_factories import (
        LargeDatasetFactory,
        ReflectionDataFactory,
        SecurityTestDataFactory,
    )
except ImportError:
    # Create minimal mocks when factories aren't available
    class ReflectionDataFactory:
        @staticmethod
        def create():
            return {"content": "Test reflection", "tags": ["test"]}

    class LargeDatasetFactory:
        @staticmethod
        def generate_large_reflection_dataset(count: int = 1000):
            return [ReflectionDataFactory.create() for _ in range(count)]

    class SecurityTestDataFactory:
        @staticmethod
        def create():
            return {"valid_token": "test-token", "operation": "read"}


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop]:
    """Create session-scoped event loop for async tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    yield loop

    # Clean up pending tasks
    if not loop.is_closed():
        # Cancel all pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        # Wait for tasks to be cancelled
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        loop.close()


@pytest.fixture(autouse=True)
def reset_di_container():
    """Reset DI container between tests to ensure clean state.

    This fixture runs automatically for every test to prevent DI state
    leakage between tests. It cleans up all singleton instances from the
    bevy container, including:
    - SessionPaths, SessionLogger, SessionPermissionsManager, SessionLifecycleManager
    - ApplicationMonitor, LLMManager, ServerlessSessionManager
    - ReflectionDatabase, InterruptionManager

    Week 8 Day 1: Enhanced to fix test isolation issues by directly
    cleaning the bevy container instances dictionary. Reset happens both
    before and after test to ensure clean state for monkeypatching.
    """
    # Clean up BEFORE test to ensure monkeypatch can take effect
    try:
        from bevy import get_container
        from session_mgmt_mcp.di import SessionPaths

        container = get_container()

        # List of all singleton classes to clean up
        singleton_classes = [
            SessionPaths,
            # Core DI singletons
            "SessionLogger",
            "SessionPermissionsManager",
            "SessionLifecycleManager",
            # Instance manager singletons
            "ApplicationMonitor",
            "LLMManager",
            "ServerlessSessionManager",
            "ReflectionDatabase",
            "InterruptionManager",
        ]

        # Clean up each singleton from container
        for cls in singleton_classes:
            # Handle both direct class and string class names
            if isinstance(cls, str):
                # Import the class dynamically
                try:
                    if cls == "SessionLogger":
                        from session_mgmt_mcp.utils.logging import SessionLogger

                        cls = SessionLogger
                    elif cls == "SessionPermissionsManager":
                        from session_mgmt_mcp.server_core import (
                            SessionPermissionsManager,
                        )

                        cls = SessionPermissionsManager
                    elif cls == "SessionLifecycleManager":
                        from session_mgmt_mcp.core import SessionLifecycleManager

                        cls = SessionLifecycleManager
                    elif cls == "ApplicationMonitor":
                        from session_mgmt_mcp.app_monitor import ApplicationMonitor

                        cls = ApplicationMonitor
                    elif cls == "LLMManager":
                        from session_mgmt_mcp.llm_providers import LLMManager

                        cls = LLMManager
                    elif cls == "ServerlessSessionManager":
                        from session_mgmt_mcp.serverless_mode import (
                            ServerlessSessionManager,
                        )

                        cls = ServerlessSessionManager
                    elif cls == "ReflectionDatabase":
                        from session_mgmt_mcp.reflection_tools import ReflectionDatabase

                        cls = ReflectionDatabase
                    elif cls == "InterruptionManager":
                        from session_mgmt_mcp.interruption_manager import (
                            InterruptionManager,
                        )

                        cls = InterruptionManager
                except ImportError:
                    continue

            # Remove from container if present
            with suppress(KeyError, TypeError):
                container.instances.pop(cls, None)

        # Reset configuration flag BEFORE test so monkeypatch can work
        import session_mgmt_mcp.di as di_module

        di_module._configured = False

    except Exception:
        # If cleanup fails, we'll try again on next test
        pass

    # Test runs here
    yield

    # Clean up AFTER test as well for consistency
    try:
        from bevy import get_container

        container = get_container()

        # Same cleanup as above
        for cls in singleton_classes:
            if isinstance(cls, str):
                try:
                    if cls == "SessionLogger":
                        from session_mgmt_mcp.utils.logging import SessionLogger

                        cls = SessionLogger
                    elif cls == "SessionPermissionsManager":
                        from session_mgmt_mcp.server_core import (
                            SessionPermissionsManager,
                        )

                        cls = SessionPermissionsManager
                    elif cls == "SessionLifecycleManager":
                        from session_mgmt_mcp.core import SessionLifecycleManager

                        cls = SessionLifecycleManager
                    elif cls == "ApplicationMonitor":
                        from session_mgmt_mcp.app_monitor import ApplicationMonitor

                        cls = ApplicationMonitor
                    elif cls == "LLMManager":
                        from session_mgmt_mcp.llm_providers import LLMManager

                        cls = LLMManager
                    elif cls == "ServerlessSessionManager":
                        from session_mgmt_mcp.serverless_mode import (
                            ServerlessSessionManager,
                        )

                        cls = ServerlessSessionManager
                    elif cls == "ReflectionDatabase":
                        from session_mgmt_mcp.reflection_tools import ReflectionDatabase

                        cls = ReflectionDatabase
                    elif cls == "InterruptionManager":
                        from session_mgmt_mcp.interruption_manager import (
                            InterruptionManager,
                        )

                        cls = InterruptionManager
                except ImportError:
                    continue

            with suppress(KeyError, TypeError):
                container.instances.pop(cls, None)

        # Reset configuration flag again
        import session_mgmt_mcp.di as di_module

        di_module._configured = False

    except Exception:
        pass


@pytest.fixture
async def temp_db_path() -> AsyncGenerator[str]:
    """Provide temporary database path that's cleaned up after test."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
        db_path = tmp.name

    yield db_path

    # Cleanup
    try:
        db_path_obj = Path(db_path)
        if db_path_obj.exists():
            db_path_obj.unlink()
    except (OSError, PermissionError):
        # On Windows, file might still be locked
        pass


@pytest.fixture
async def temp_claude_dir() -> AsyncGenerator[Path]:
    """Provide temporary ~/.claude directory structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        claude_dir = Path(temp_dir) / ".claude"
        claude_dir.mkdir()

        # Create expected subdirectories
        (claude_dir / "data").mkdir()
        (claude_dir / "logs").mkdir()

        # Patch the home directory
        with patch.dict(os.environ, {"HOME": str(Path(temp_dir))}):
            with patch(
                "os.path.expanduser",
                lambda path: path.replace("~", str(Path(temp_dir))),
            ):
                yield claude_dir


@pytest.fixture
async def reflection_db(temp_db_path: str) -> AsyncGenerator[ReflectionDatabase]:
    """Provide initialized ReflectionDatabase instance."""
    db = ReflectionDatabase(db_path=temp_db_path)

    try:
        await db.initialize()
        yield db
    finally:
        db.close()


@pytest.fixture
async def reflection_db_with_data(
    reflection_db: ReflectionDatabase,
) -> AsyncGenerator[ReflectionDatabase]:
    """Provide ReflectionDatabase with test data."""
    # Add some test conversations
    test_conversations = [
        "How do I implement async/await patterns in Python?",
        "Setting up pytest fixtures for database testing",
        "Best practices for MCP server development",
        "DuckDB vector operations and similarity search",
        "FastMCP tool registration and async handlers",
    ]

    conversation_ids = []
    for content in test_conversations:
        conv_id = await reflection_db.store_conversation(
            content, {"project": "test-project"}
        )
        conversation_ids.append(conv_id)

    # Add some test reflections
    test_reflections = [
        (
            "Always use context managers for database connections",
            ["database", "patterns"],
        ),
        (
            "Async fixtures require careful setup in pytest",
            ["testing", "async", "pytest"],
        ),
        ("MCP tools should handle errors gracefully", ["mcp", "error-handling"]),
    ]

    reflection_ids = []
    for content, tags in test_reflections:
        refl_id = await reflection_db.store_reflection(content, tags)
        reflection_ids.append(refl_id)

    # Store IDs for test reference
    reflection_db._test_conversation_ids = conversation_ids
    reflection_db._test_reflection_ids = reflection_ids

    return reflection_db


@pytest.fixture
def mock_onnx_session() -> Mock:
    """Provide mock ONNX session for embedding tests."""
    mock_session = Mock()
    # Mock returns a 384-dimensional vector
    rng = np.random.default_rng(42)
    mock_session.run.return_value = [rng.random((1, 384)).astype(np.float32)]
    return mock_session


@pytest.fixture
def mock_tokenizer() -> Mock:
    """Provide mock tokenizer for embedding tests."""
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3, 4, 5]],
        "attention_mask": [[1, 1, 1, 1, 1]],
    }
    return mock_tokenizer


@pytest.fixture
async def mock_mcp_server() -> AsyncGenerator[Mock]:
    """Provide mock MCP server for testing."""
    mock_server = Mock()
    mock_server.tool = Mock()
    mock_server.prompt = Mock()

    # Mock async context manager behavior - assign directly to the mock object
    mock_server.__aenter__ = AsyncMock(return_value=mock_server)
    mock_server.__aexit__ = AsyncMock(return_value=None)

    return mock_server


@pytest.fixture
def clean_environment(tmp_path) -> Generator[dict[str, Any]]:
    """Provide clean environment with common patches."""
    original_env = os.environ.copy()

    # Set up test environment
    test_env = {
        "TESTING": "1",
        "LOG_LEVEL": "DEBUG",
    }

    # Remove potentially problematic env vars
    env_to_remove = ["OLDPWD", "VIRTUAL_ENV"]

    try:
        # Update environment
        os.environ.update(test_env)
        for key in env_to_remove:
            os.environ.pop(key, None)

        # Change to a safe temporary directory to avoid cwd issues
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        yield test_env

    finally:
        # Restore original environment and working directory
        os.environ.clear()
        os.environ.update(original_env)
        os.chdir(original_cwd)


@pytest.fixture
async def async_client() -> AsyncGenerator[Mock]:
    """Provide async client for MCP communication testing."""
    client = Mock()

    # Mock async methods
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.call_tool = AsyncMock()
    client.list_tools = AsyncMock(return_value=[])

    return client


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Provide sample embedding vector for testing."""
    # Create a consistent sample embedding
    rng = np.random.default_rng(42)
    return rng.random((384,)).astype(np.float32)


@pytest.fixture
def mock_embeddings_disabled():
    """Fixture to disable embeddings for testing fallback behavior."""
    with patch("session_mgmt_mcp.reflection_tools.ONNX_AVAILABLE", False):
        yield


@pytest.fixture
async def duckdb_connection() -> AsyncGenerator[duckdb.DuckDBPyConnection]:
    """Provide in-memory DuckDB connection for testing."""
    conn = duckdb.connect(":memory:")

    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def mock_file_operations():
    """Mock file system operations for testing."""
    mocks = {}

    with (
        patch("pathlib.Path.mkdir") as mock_mkdir,
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.unlink") as mock_unlink,
        patch("os.path.exists") as mock_os_exists,
    ):
        mock_exists.return_value = True
        mock_os_exists.return_value = True

        mocks["mkdir"] = mock_mkdir
        mocks["exists"] = mock_exists
        mocks["unlink"] = mock_unlink
        mocks["os_exists"] = mock_os_exists

        yield mocks


# Async test markers and utilities
@pytest.fixture(autouse=True)
def detect_asyncio_leaks():
    """Automatically detect asyncio task leaks in tests."""
    # Only check for leaks if there's a running event loop
    try:
        initial_tasks = len(asyncio.all_tasks())
    except RuntimeError:
        # No event loop running, skip leak detection
        yield
        return

    yield

    # Check for task leaks after test
    try:
        final_tasks = asyncio.all_tasks()
        if len(final_tasks) > initial_tasks:
            # Allow a small buffer for cleanup tasks
            if len(final_tasks) > initial_tasks + 2:
                task_names = [task.get_name() for task in final_tasks]
                pytest.fail(f"Potential task leak detected. Active tasks: {task_names}")
    except RuntimeError:
        # Event loop closed, no need to check
        pass


@pytest.fixture
def performance_baseline() -> dict[str, float]:
    """Provide performance baselines for benchmark tests."""
    return {
        "db_insert_time": 0.1,  # 100ms per insert
        "embedding_generation": 0.5,  # 500ms per embedding
        "search_query": 0.2,  # 200ms per search
        "bulk_operation": 1.0,  # 1s for bulk operations
    }


# Helper functions for test data generation
def generate_test_conversation(
    content: str = "Test conversation content",
    project: str = "test-project",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Generate test conversation data."""
    return {
        "content": content,
        "project": project,
        "timestamp": timestamp or "2024-01-01T12:00:00Z",
    }


def generate_test_reflection(
    content: str = "Test reflection content",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Generate test reflection data."""
    return {
        "content": content,
        "tags": tags or ["test"],
    }


# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "async_test: mark test as requiring async event loop"
    )
    config.addinivalue_line("markers", "db_test: mark test as requiring database")
    config.addinivalue_line(
        "markers", "embedding_test: mark test as requiring embeddings"
    )
    config.addinivalue_line("markers", "mcp_test: mark test as MCP server test")


# Performance Testing Fixtures
@pytest.fixture
def performance_monitor():
    """Monitor for performance testing with memory and timing metrics."""
    import time

    try:
        import psutil
    except ImportError:
        # Fallback monitor when psutil unavailable
        class SimplePerformanceMonitor:
            def __init__(self):
                self.start_time = None

            def start_monitoring(self):
                self.start_time = time.time()

            def stop_monitoring(self):
                return {
                    "duration": time.time() - self.start_time if self.start_time else 0,
                    "memory_delta": 0,
                    "peak_memory": 0,
                }

        return SimplePerformanceMonitor()

    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.start_time = None
            self.start_memory = None

        def start_monitoring(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        def stop_monitoring(self):
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

            return {
                "duration": end_time - self.start_time,
                "memory_delta": end_memory - self.start_memory,
                "peak_memory": end_memory,
            }

    return PerformanceMonitor()


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and environment."""
    # Add async marker to all async tests
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.async_test)

        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker("integration")
        elif "unit" in str(item.fspath):
            item.add_marker("unit")
        elif "functional" in str(item.fspath):
            item.add_marker("functional")
        elif "performance" in str(item.fspath):
            item.add_marker("performance")
        elif "security" in str(item.fspath):
            item.add_marker("security")


# Session management specific fixtures
@pytest.fixture
async def session_permissions():
    """Provide mock session permissions for testing."""
    class MockSessionPermissions:
        def __init__(self):
            self.trusted_operations = set()

        def add_trusted_operation(self, operation: str):
            self.trusted_operations.add(operation)

        def is_trusted(self, operation: str) -> bool:
            return operation in self.trusted_operations

    return MockSessionPermissions()


@pytest.fixture
async def temp_database(temp_db_path: str):
    """Provide temporary database for testing."""
    return temp_db_path


@pytest.fixture
async def temp_working_dir():
    """Provide temporary working directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# Async test timeout configuration
pytestmark = pytest.mark.asyncio(scope="function")
