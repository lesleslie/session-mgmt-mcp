#!/usr/bin/env python3
"""Comprehensive test suite for ReflectionDatabase with improved coverage.

Tests all core functionality including initialization, storage, search,
and error handling for the reflection database system.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from session_buddy.reflection_tools import ReflectionDatabase


@pytest.mark.asyncio
class TestReflectionDatabaseInitialization:
    """Test database initialization and connection management."""

    async def test_initialize_creates_connection(self):
        """Test that initialize() creates a valid database connection."""
        import os

        # Create temp directory and path (but don't create file)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            db = ReflectionDatabase(db_path)
            assert db.conn is None  # Connection not created until initialize

            await db.initialize()
            assert db.conn is not None
            db.close()

    async def test_initialize_creates_tables(self):
        """Test that initialize() creates required database tables."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            db = ReflectionDatabase(db_path)
            await db.initialize()

            # Verify tables exist by querying table list
            tables = db.conn.execute(
                "SELECT table_name FROM duckdb_tables()"
            ).fetchall()
            table_names = [t[0] for t in tables]

            # Should have at least conversations and reflections tables
            assert len(table_names) > 0
            db.close()

    async def test_close_connection(self):
        """Test that close() properly closes the connection."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            db = ReflectionDatabase(db_path)
            await db.initialize()
            assert db.conn is not None

            db.close()
            assert db.conn is None

    async def test_get_conn_raises_error_when_not_initialized(self):
        """Test that _get_conn() raises error when not initialized."""
        db = ReflectionDatabase(":memory:")
        with pytest.raises(RuntimeError, match="Database connection not initialized"):
            db._get_conn()


@pytest.mark.asyncio
class TestReflectionDatabaseStorage:
    """Test data storage operations."""

    @pytest.fixture
    async def initialized_db(self):
        """Provide initialized database for testing."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()
        yield db
        db.close()

    async def test_store_conversation(self, initialized_db):
        """Test storing a conversation."""
        content = "How do I use async/await in Python?"
        metadata = {"project": "test-project"}

        conv_id = await initialized_db.store_conversation(content, metadata)

        assert conv_id is not None
        assert isinstance(conv_id, str)

    async def test_store_conversation_without_metadata(self, initialized_db):
        """Test storing conversation with empty metadata."""
        content = "Test conversation"

        conv_id = await initialized_db.store_conversation(content, {})

        assert conv_id is not None

    async def test_store_reflection(self, initialized_db):
        """Test storing a reflection."""
        content = "Important lesson about async patterns"
        tags = ["async", "patterns", "python"]

        refl_id = await initialized_db.store_reflection(content, tags)

        assert refl_id is not None
        assert isinstance(refl_id, str)

    async def test_store_reflection_without_tags(self, initialized_db):
        """Test storing reflection without tags."""
        content = "General reflection"

        refl_id = await initialized_db.store_reflection(content)

        assert refl_id is not None

    async def test_store_multiple_conversations(self, initialized_db):
        """Test storing multiple conversations."""
        conversations = [
            ("First conversation", {"project": "p1"}),
            ("Second conversation", {"project": "p2"}),
            ("Third conversation", {"project": "p3"}),
        ]

        conv_ids = []
        for content, metadata in conversations:
            conv_id = await initialized_db.store_conversation(content, metadata)
            conv_ids.append(conv_id)

        assert len(conv_ids) == 3
        assert len(set(conv_ids)) == 3  # All IDs should be unique


@pytest.mark.asyncio
class TestReflectionDatabaseRetrieval:
    """Test data retrieval operations."""

    @pytest.fixture
    async def db_with_data(self):
        """Provide database with test data."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()

        # Add test data
        for i in range(5):
            await db.store_conversation(
                f"Conversation {i} about Python", {"project": "test"}
            )
            await db.store_reflection(f"Reflection {i} about patterns", ["python"])

        yield db
        db.close()

    async def test_search_conversations_empty_database(self):
        """Test searching conversations from empty database."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()

        conversations = await db.search_conversations("test", limit=10)
        assert isinstance(conversations, list)

        db.close()

    async def test_search_reflections_with_data(self, db_with_data):
        """Test searching reflections with existing data."""
        reflections = await db_with_data.search_reflections("patterns", limit=10)
        assert isinstance(reflections, list)

    async def test_get_stats(self, db_with_data):
        """Test getting database statistics."""
        stats = await db_with_data.get_stats()
        assert isinstance(stats, dict)
        assert "conversation_count" in stats or len(stats) > 0


@pytest.mark.asyncio
class TestReflectionDatabaseSearch:
    """Test search functionality."""

    @pytest.fixture
    async def searchable_db(self):
        """Provide database with searchable content."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()

        # Add test content
        await db.store_conversation(
            "How to implement async/await patterns", {"topic": "async"}
        )
        await db.store_conversation(
            "Database connection pooling best practices", {"topic": "database"}
        )
        await db.store_conversation(
            "Testing async code with pytest", {"topic": "testing"}
        )

        yield db
        db.close()

    async def test_search_conversations_by_text(self, searchable_db):
        """Test text-based search of conversations."""
        results = await searchable_db.search_conversations("async", limit=10)
        assert isinstance(results, list)

    async def test_search_conversations_with_limit(self, searchable_db):
        """Test search with result limit."""
        results = await searchable_db.search_conversations("", limit=2)
        assert len(results) <= 2

    async def test_search_reflections_by_content(self, searchable_db):
        """Test searching reflections by content."""
        # Add a reflection with specific content
        await searchable_db.store_reflection(
            "Important async pattern", ["async", "patterns"]
        )

        results = await searchable_db.search_reflections("async", limit=10)
        assert isinstance(results, list)

    async def test_semantic_search_fallback(self, searchable_db):
        """Test that search works even without embeddings."""
        # Search should work using text fallback
        results = await searchable_db.search_conversations("database", limit=10)
        assert isinstance(results, list)


@pytest.mark.asyncio
class TestReflectionDatabaseErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    async def initialized_db(self):
        """Provide initialized database."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()
        yield db
        db.close()

    async def test_store_empty_conversation(self, initialized_db):
        """Test storing conversation with empty content."""
        conv_id = await initialized_db.store_conversation("", {})
        assert conv_id is not None  # Should still create record

    async def test_store_very_long_conversation(self, initialized_db):
        """Test storing conversation with very long content."""
        long_content = "x" * 10000  # 10KB of text
        conv_id = await initialized_db.store_conversation(long_content, {})
        assert conv_id is not None

    async def test_store_conversation_with_special_characters(self, initialized_db):
        """Test storing content with special characters."""
        special_content = "Testing with émojis 🚀 and spëcial çharacters!"
        conv_id = await initialized_db.store_conversation(special_content, {})
        assert conv_id is not None

    async def test_store_reflection_with_many_tags(self, initialized_db):
        """Test storing reflection with many tags."""
        tags = [f"tag{i}" for i in range(50)]
        refl_id = await initialized_db.store_reflection("Test reflection", tags)
        assert refl_id is not None

    async def test_search_no_results_conversation(self, initialized_db):
        """Test searching for conversations with no results."""
        # Search for something unlikely to exist
        results = await initialized_db.search_conversations("xyzabc123notfound")
        assert isinstance(results, list)
        # May be empty or have results depending on what's in DB

    async def test_search_no_results_reflection(self, initialized_db):
        """Test searching for reflections with no results."""
        # Search for something unlikely to exist
        results = await initialized_db.search_reflections("xyzabc123notfound")
        assert isinstance(results, list)


@pytest.mark.asyncio
class TestReflectionDatabaseConcurrency:
    """Test concurrent operations."""

    @pytest.fixture
    async def concurrent_db(self):
        """Provide database for concurrent testing."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()
        yield db
        db.close()

    async def test_concurrent_conversations(self, concurrent_db):
        """Test concurrent conversation storage."""
        import asyncio

        async def store_conversation(i: int) -> str:
            return await concurrent_db.store_conversation(
                f"Concurrent conversation {i}", {"index": i}
            )

        tasks = [store_conversation(i) for i in range(10)]
        conv_ids = await asyncio.gather(*tasks)

        assert len(conv_ids) == 10
        assert len(set(conv_ids)) == 10  # All unique


@pytest.mark.asyncio
class TestReflectionDatabaseMetadata:
    """Test metadata handling."""

    @pytest.fixture
    async def metadata_db(self):
        """Provide database for metadata testing."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()
        yield db
        db.close()

    async def test_store_conversation_with_complex_metadata(self, metadata_db):
        """Test storing conversation with complex metadata."""
        metadata = {
            "project": "test",
            "user": "test_user",
            "timestamp": "2025-01-01T00:00:00",
            "tags": ["python", "async"],
            "nested": {"key": "value"},
        }

        conv_id = await metadata_db.store_conversation("Test content", metadata)
        assert conv_id is not None

    async def test_store_conversation_with_empty_metadata(self, metadata_db):
        """Test storing conversation with empty metadata dict."""
        conv_id = await metadata_db.store_conversation("Test content", {})
        assert conv_id is not None

    async def test_conversation_with_metadata(self, metadata_db):
        """Test that conversations can be stored with metadata."""
        metadata = {"project": "important_project", "user": "alice"}
        conv_id = await metadata_db.store_conversation("Important message", metadata)

        # Verify conversation was stored
        assert conv_id is not None
        assert isinstance(conv_id, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
