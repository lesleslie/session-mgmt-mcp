"""Tests for serverless_mode module.

Tests serverless session management with external storage backends,
focusing on the new ACBCacheStorage adapter.

Phase: Week 5 Day 3 - Serverless Mode Coverage
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSessionState:
    """Test SessionState Pydantic model."""

    def test_session_state_initialization(self) -> None:
        """Should create SessionState with required fields."""
        from session_buddy.serverless_mode import SessionState

        session = SessionState(
            session_id="test-123",
            user_id="user-1",
            project_id="project-1",
            created_at="2025-01-01T12:00:00",
            last_activity="2025-01-01T12:00:00",
        )

        assert session.session_id == "test-123"
        assert session.user_id == "user-1"
        assert session.project_id == "project-1"
        assert isinstance(session.permissions, list)
        assert isinstance(session.conversation_history, list)

    def test_session_state_to_dict(self) -> None:
        """Should convert SessionState to dictionary."""
        from session_buddy.serverless_mode import SessionState

        session = SessionState(
            session_id="test-123",
            user_id="user-1",
            project_id="project-1",
            created_at="2025-01-01T12:00:00",
            last_activity="2025-01-01T12:00:00",
        )

        data = session.to_dict()

        assert isinstance(data, dict)
        assert data["session_id"] == "test-123"
        assert data["user_id"] == "user-1"

    def test_session_state_from_dict(self) -> None:
        """Should create SessionState from dictionary."""
        from session_buddy.serverless_mode import SessionState

        data = {
            "session_id": "test-123",
            "user_id": "user-1",
            "project_id": "project-1",
            "created_at": "2025-01-01T12:00:00",
            "last_activity": "2025-01-01T12:00:00",
            "permissions": [],
            "conversation_history": [],
            "reflection_data": {},
            "app_monitoring_state": {},
            "llm_provider_configs": {},
            "metadata": {},
        }

        session = SessionState.from_dict(data)

        assert session.session_id == "test-123"
        assert session.user_id == "user-1"


class TestACBCacheStorage:
    """Test ACBCacheStorage adapter (new implementation)."""

    @pytest.mark.asyncio
    async def test_store_session_success(self) -> None:
        """Should store session using aiocache."""
        from session_buddy.serverless_mode import ACBCacheStorage, SessionState

        mock_cache = AsyncMock()
        mock_cache.set = AsyncMock()
        mock_cache.get = AsyncMock(return_value=None)  # For index

        storage = ACBCacheStorage(mock_cache, namespace="test")

        session = SessionState(
            session_id="test-123",
            user_id="user-1",
            project_id="project-1",
            created_at="2025-01-01T12:00:00",
            last_activity="2025-01-01T12:00:00",
        )

        result = await storage.store_session(session, ttl_seconds=60)

        assert result is True
        # Should call cache.set for session data
        assert mock_cache.set.call_count >= 1

    @pytest.mark.asyncio
    async def test_retrieve_session_success(self) -> None:
        """Should retrieve session from aiocache."""
        from session_buddy.serverless_mode import ACBCacheStorage

        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(
            return_value={
                "session_id": "test-123",
                "user_id": "user-1",
                "project_id": "project-1",
                "created_at": "2025-01-01T12:00:00",
                "last_activity": "2025-01-01T12:00:00",
                "permissions": [],
                "conversation_history": [],
                "reflection_data": {},
                "app_monitoring_state": {},
                "llm_provider_configs": {},
                "metadata": {},
            }
        )

        storage = ACBCacheStorage(mock_cache, namespace="test")

        session = await storage.retrieve_session("test-123")

        assert session is not None
        assert session.session_id == "test-123"
        mock_cache.get.assert_called()

    @pytest.mark.asyncio
    async def test_retrieve_session_not_found(self) -> None:
        """Should return None when session not found."""
        from session_buddy.serverless_mode import ACBCacheStorage

        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=None)

        storage = ACBCacheStorage(mock_cache, namespace="test")

        session = await storage.retrieve_session("nonexistent")

        assert session is None

    @pytest.mark.asyncio
    async def test_delete_session_success(self) -> None:
        """Should delete session from aiocache."""
        from session_buddy.serverless_mode import ACBCacheStorage

        mock_cache = AsyncMock()
        mock_cache.delete = AsyncMock(return_value=True)
        mock_cache.get = AsyncMock(return_value={})  # For index

        storage = ACBCacheStorage(mock_cache, namespace="test")

        result = await storage.delete_session("test-123")

        assert result is True
        mock_cache.delete.assert_called()

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self) -> None:
        """Should return empty list when no sessions."""
        from session_buddy.serverless_mode import ACBCacheStorage

        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=None)  # No index

        storage = ACBCacheStorage(mock_cache, namespace="test")

        sessions = await storage.list_sessions()

        assert sessions == []

    @pytest.mark.asyncio
    async def test_list_sessions_with_filter(self) -> None:
        """Should filter sessions by user_id and project_id."""
        from session_buddy.serverless_mode import ACBCacheStorage

        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(
            return_value={
                "session-1": {"user_id": "user-1", "project_id": "project-1"},
                "session-2": {"user_id": "user-2", "project_id": "project-1"},
                "session-3": {"user_id": "user-1", "project_id": "project-2"},
            }
        )

        storage = ACBCacheStorage(mock_cache, namespace="test")

        # Filter by user_id
        sessions = await storage.list_sessions(user_id="user-1")
        assert len(sessions) == 2
        assert "session-1" in sessions
        assert "session-3" in sessions

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self) -> None:
        """Should cleanup stale index entries."""
        from session_buddy.serverless_mode import ACBCacheStorage

        mock_cache = AsyncMock()
        # Index has 2 sessions
        mock_cache.get = AsyncMock(return_value={"session-1": {}, "session-2": {}})
        # First session exists, second doesn't
        mock_cache.exists = AsyncMock(side_effect=[True, False])
        mock_cache.set = AsyncMock()

        storage = ACBCacheStorage(mock_cache, namespace="test")

        cleaned = await storage.cleanup_expired_sessions()

        assert cleaned == 1  # One session expired
        mock_cache.set.assert_called()  # Index updated

    @pytest.mark.asyncio
    async def test_is_available_success(self) -> None:
        """Should return True when cache is available."""
        from session_buddy.serverless_mode import ACBCacheStorage

        mock_cache = AsyncMock()
        mock_cache.set = AsyncMock()
        mock_cache.get = AsyncMock(return_value={"status": "ok"})
        mock_cache.delete = AsyncMock()

        storage = ACBCacheStorage(mock_cache, namespace="test")

        available = await storage.is_available()

        assert available is True


class TestServerlessSessionManager:
    """Test ServerlessSessionManager class."""

    @pytest.mark.asyncio
    async def test_create_session(self) -> None:
        """Should create new session with ServerlessSessionManager."""
        from session_buddy.serverless_mode import (
            ACBCacheStorage,
            ServerlessSessionManager,
        )

        mock_cache = AsyncMock()
        mock_cache.set = AsyncMock()
        mock_cache.get = AsyncMock(return_value=None)
        storage = ACBCacheStorage(mock_cache, namespace="test")

        manager = ServerlessSessionManager(storage)

        session_id = await manager.create_session(
            user_id="user-1", project_id="project-1"
        )

        assert isinstance(session_id, str)
        assert len(session_id) > 0

    @pytest.mark.asyncio
    async def test_get_session(self) -> None:
        """Should retrieve session state."""
        from session_buddy.serverless_mode import (
            ACBCacheStorage,
            ServerlessSessionManager,
        )

        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(
            return_value={
                "session_id": "test-123",
                "user_id": "user-1",
                "project_id": "project-1",
                "created_at": "2025-01-01T12:00:00",
                "last_activity": "2025-01-01T12:00:00",
                "permissions": [],
                "conversation_history": [],
                "reflection_data": {},
                "app_monitoring_state": {},
                "llm_provider_configs": {},
                "metadata": {},
            }
        )
        storage = ACBCacheStorage(mock_cache, namespace="test")

        manager = ServerlessSessionManager(storage)

        session = await manager.get_session("test-123")

        assert session is not None
        assert session.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_update_session(self) -> None:
        """Should update session state."""
        from session_buddy.serverless_mode import (
            ACBCacheStorage,
            ServerlessSessionManager,
        )

        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(
            return_value={
                "session_id": "test-123",
                "user_id": "user-1",
                "project_id": "project-1",
                "created_at": "2025-01-01T12:00:00",
                "last_activity": "2025-01-01T12:00:00",
                "permissions": [],
                "conversation_history": [],
                "reflection_data": {},
                "app_monitoring_state": {},
                "llm_provider_configs": {},
                "metadata": {},
            }
        )
        mock_cache.set = AsyncMock()
        storage = ACBCacheStorage(mock_cache, namespace="test")

        manager = ServerlessSessionManager(storage)

        result = await manager.update_session(
            "test-123", {"metadata": {"updated": True}}
        )

        assert result is True
        mock_cache.set.assert_called()

    @pytest.mark.asyncio
    async def test_delete_session(self) -> None:
        """Should delete session."""
        from session_buddy.serverless_mode import (
            ACBCacheStorage,
            ServerlessSessionManager,
        )

        mock_cache = AsyncMock()
        mock_cache.delete = AsyncMock(return_value=True)
        mock_cache.get = AsyncMock(return_value={})
        storage = ACBCacheStorage(mock_cache, namespace="test")

        manager = ServerlessSessionManager(storage)

        result = await manager.delete_session("test-123")

        assert result is True


class TestServerlessConfigManager:
    """Test ServerlessConfigManager factory methods."""

    def test_create_storage_backend_acb_default(self) -> None:
        """Should create ACBCacheStorage by default."""
        from session_buddy.serverless_mode import ServerlessConfigManager

        config = {"storage_backend": "acb", "backends": {"acb": {}}}

        storage = ServerlessConfigManager.create_storage_backend(config)

        assert storage.__class__.__name__ == "ACBCacheStorage"

    def test_create_storage_backend_legacy_redis_warns(self) -> None:
        """Should create RedisStorage with deprecation warning."""
        from session_buddy.serverless_mode import ServerlessConfigManager

        config = {
            "storage_backend": "redis",
            "backends": {"redis": {"host": "localhost"}},
        }

        with patch("logging.warning") as mock_warn:
            storage = ServerlessConfigManager.create_storage_backend(config)

            assert storage.__class__.__name__ == "RedisStorage"
            mock_warn.assert_called()
            assert "deprecated" in mock_warn.call_args[0][0].lower()

    @pytest.mark.asyncio
    async def test_test_storage_backends(self) -> None:
        """Should test all configured backends."""
        from session_buddy.serverless_mode import ServerlessConfigManager

        config = {
            "backends": {
                "acb": {"cache_type": "memory"},
                "local": {"storage_dir": "/tmp/test"},
            }
        }

        results = await ServerlessConfigManager.test_storage_backends(config)

        assert isinstance(results, dict)
        assert "acb" in results or "local" in results
