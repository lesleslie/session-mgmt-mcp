"""Integration tests for ServerlessStorageAdapter with SessionStorageAdapter.

Tests the serverless storage bridge adapter that connects serverless_mode.py
to the new ACB-based SessionStorageAdapter.

Phase 2, Day 5: Backend consolidation and migration
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from session_buddy.adapters.serverless_storage_adapter import (
    ServerlessStorageAdapter,
    create_serverless_storage,
)
from session_buddy.backends.base import SessionState


class TestServerlessStorageAdapterBasics:
    """Test basic ServerlessStorageAdapter functionality."""

    def test_create_serverless_storage_default(self):
        """Test creating serverless storage with defaults."""
        storage = create_serverless_storage()

        assert isinstance(storage, ServerlessStorageAdapter)
        assert storage.backend == "file"

    def test_create_serverless_storage_custom_backend(self):
        """Test creating serverless storage with custom backend."""
        storage = create_serverless_storage(backend="memory")

        assert storage.backend == "memory"

    def test_create_serverless_storage_with_config(self):
        """Test creating serverless storage with config."""
        config = {"test": "value"}
        storage = create_serverless_storage(config=config)

        assert storage.config == config


class TestServerlessStorageAdapterOperations:
    """Test ServerlessStorageAdapter CRUD operations."""

    @pytest.fixture
    def mock_session_storage_adapter(self):
        """Mock SessionStorageAdapter."""
        mock = AsyncMock()
        mock.store_session = AsyncMock()
        mock.load_session = AsyncMock()
        mock.delete_session = AsyncMock(return_value=True)
        mock.session_exists = AsyncMock(return_value=True)
        return mock

    @pytest.fixture
    def sample_session_state(self):
        """Create sample session state for testing."""
        return SessionState(
            session_id="test_session_123",
            user_id="user_123",
            project_id="project_456",
            created_at=datetime.now().isoformat(),
            last_activity=datetime.now().isoformat(),
            permissions=["read", "write"],
            conversation_history=[{"role": "user", "content": "Hello"}],
            reflection_data={"key": "value"},
            app_monitoring_state={},
            llm_provider_configs={},
            metadata={"test": True},
        )

    @pytest.fixture
    async def adapter_with_mock(self, mock_session_storage_adapter):
        """Create ServerlessStorageAdapter with mocked SessionStorageAdapter."""
        adapter = ServerlessStorageAdapter(backend="memory")
        adapter._storage = mock_session_storage_adapter
        return adapter

    async def test_store_session_without_ttl(
        self, adapter_with_mock, sample_session_state, mock_session_storage_adapter
    ):
        """Test storing session without TTL."""
        success = await adapter_with_mock.store_session(sample_session_state)

        assert success is True
        mock_session_storage_adapter.store_session.assert_called_once()

        # Check stored data
        call_args = mock_session_storage_adapter.store_session.call_args
        session_id = call_args[0][0]
        state_dict = call_args[0][1]

        assert session_id == "test_session_123"
        assert state_dict["user_id"] == "user_123"
        assert "_ttl" not in state_dict  # No TTL when not provided

    async def test_store_session_with_ttl(
        self, adapter_with_mock, sample_session_state, mock_session_storage_adapter
    ):
        """Test storing session with TTL."""
        ttl_seconds = 3600  # 1 hour

        success = await adapter_with_mock.store_session(
            sample_session_state, ttl_seconds=ttl_seconds
        )

        assert success is True

        # Check stored data includes TTL metadata
        call_args = mock_session_storage_adapter.store_session.call_args
        state_dict = call_args[0][1]

        assert "_ttl" in state_dict
        assert state_dict["_ttl"]["ttl_seconds"] == 3600
        assert "expires_at" in state_dict["_ttl"]

    async def test_store_session_error_handling(
        self, adapter_with_mock, sample_session_state, mock_session_storage_adapter
    ):
        """Test store_session error handling."""
        mock_session_storage_adapter.store_session.side_effect = Exception(
            "Storage error"
        )

        success = await adapter_with_mock.store_session(sample_session_state)

        assert success is False

    async def test_retrieve_session_success(
        self, adapter_with_mock, sample_session_state, mock_session_storage_adapter
    ):
        """Test retrieving existing session."""
        # Mock load_session to return session data
        mock_session_storage_adapter.load_session = AsyncMock(
            return_value=sample_session_state.to_dict()
        )

        result = await adapter_with_mock.retrieve_session("test_session_123")

        assert result is not None
        assert result.session_id == "test_session_123"
        assert result.user_id == "user_123"

    async def test_retrieve_session_not_found(
        self, adapter_with_mock, mock_session_storage_adapter
    ):
        """Test retrieving non-existent session."""
        mock_session_storage_adapter.load_session = AsyncMock(return_value=None)

        result = await adapter_with_mock.retrieve_session("nonexistent_session")

        assert result is None

    async def test_retrieve_session_expired(
        self, adapter_with_mock, sample_session_state, mock_session_storage_adapter
    ):
        """Test retrieving expired session returns None."""
        # Create expired session data
        expired_data = sample_session_state.to_dict()
        expired_data["_ttl"] = {
            "ttl_seconds": 3600,
            "expires_at": "2020-01-01T00:00:00",  # Expired
        }

        mock_session_storage_adapter.load_session = AsyncMock(return_value=expired_data)
        mock_session_storage_adapter.delete_session = AsyncMock(return_value=True)

        result = await adapter_with_mock.retrieve_session("test_session_123")

        # Should return None and delete expired session
        assert result is None
        mock_session_storage_adapter.delete_session.assert_called_once_with(
            "test_session_123"
        )

    async def test_delete_session_success(
        self, adapter_with_mock, mock_session_storage_adapter
    ):
        """Test deleting session."""
        success = await adapter_with_mock.delete_session("test_session_123")

        assert success is True
        mock_session_storage_adapter.delete_session.assert_called_once_with(
            "test_session_123"
        )

    async def test_delete_session_not_found(
        self, adapter_with_mock, mock_session_storage_adapter
    ):
        """Test deleting non-existent session."""
        mock_session_storage_adapter.delete_session = AsyncMock(return_value=False)

        success = await adapter_with_mock.delete_session("nonexistent_session")

        assert success is False

    async def test_list_sessions_no_filter(self, adapter_with_mock):
        """Test listing sessions without filters."""
        # Add some metadata to cache
        adapter_with_mock._session_metadata = {
            "session_1": {
                "user_id": "user_1",
                "project_id": "project_1",
                "expires_at": None,
            },
            "session_2": {
                "user_id": "user_2",
                "project_id": "project_2",
                "expires_at": None,
            },
        }

        sessions = await adapter_with_mock.list_sessions()

        assert len(sessions) == 2
        assert "session_1" in sessions
        assert "session_2" in sessions

    async def test_list_sessions_filter_by_user(self, adapter_with_mock):
        """Test listing sessions filtered by user."""
        adapter_with_mock._session_metadata = {
            "session_1": {
                "user_id": "user_1",
                "project_id": "project_1",
                "expires_at": None,
            },
            "session_2": {
                "user_id": "user_2",
                "project_id": "project_2",
                "expires_at": None,
            },
        }

        sessions = await adapter_with_mock.list_sessions(user_id="user_1")

        assert len(sessions) == 1
        assert "session_1" in sessions

    async def test_cleanup_expired_sessions(self, adapter_with_mock):
        """Test cleaning up expired sessions."""
        # Add expired and non-expired sessions to metadata
        adapter_with_mock._session_metadata = {
            "expired_session": {
                "user_id": "user_1",
                "expires_at": "2020-01-01T00:00:00",  # Expired
            },
            "active_session": {
                "user_id": "user_2",
                "expires_at": "2099-01-01T00:00:00",  # Not expired
            },
        }

        count = await adapter_with_mock.cleanup_expired_sessions()

        assert count == 1
        assert "expired_session" not in adapter_with_mock._session_metadata
        assert "active_session" in adapter_with_mock._session_metadata

    async def test_is_available_success(
        self, adapter_with_mock, mock_session_storage_adapter
    ):
        """Test storage availability check success."""
        mock_session_storage_adapter.store_session = AsyncMock()
        mock_session_storage_adapter.load_session = AsyncMock(
            return_value={"test": True}
        )
        mock_session_storage_adapter.delete_session = AsyncMock(return_value=True)

        is_available = await adapter_with_mock.is_available()

        assert is_available is True

    async def test_is_available_failure(
        self, adapter_with_mock, mock_session_storage_adapter
    ):
        """Test storage availability check failure."""
        mock_session_storage_adapter.store_session.side_effect = Exception(
            "Storage error"
        )

        is_available = await adapter_with_mock.is_available()

        assert is_available is False
