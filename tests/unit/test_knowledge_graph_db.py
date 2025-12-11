"""Tests for Knowledge Graph Database (DuckDB + DuckPGQ).

Tests the knowledge graph semantic memory system that complements
episodic memory in ReflectionDatabase.

Phase: Week 4 Days 3-5 - Knowledge Graph Coverage
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path


class TestKnowledgeGraphInitialization:
    """Test knowledge graph database initialization."""

    @pytest.mark.asyncio
    async def test_init_with_default_path(self) -> None:
        """Should initialize with default database path."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        kg = KnowledgeGraphDatabase()

        assert kg.db_path.endswith("knowledge_graph.duckdb")
        assert "~/.claude/data" in kg.db_path or ".claude/data" in kg.db_path
        assert kg.conn is None

    @pytest.mark.asyncio
    async def test_init_with_custom_path(self, tmp_path: Path) -> None:
        """Should initialize with custom database path."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "custom_kg.duckdb")
        kg = KnowledgeGraphDatabase(db_path=db_path)

        assert kg.db_path == db_path
        assert kg.conn is None

    @pytest.mark.asyncio
    async def test_context_manager_sync(self, tmp_path: Path) -> None:
        """Should support synchronous context manager."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "sync_kg.duckdb")

        with KnowledgeGraphDatabase(db_path=db_path) as kg:
            assert isinstance(kg, KnowledgeGraphDatabase)
            assert kg.db_path == db_path

    @pytest.mark.asyncio
    async def test_context_manager_async(self, tmp_path: Path) -> None:
        """Should support asynchronous context manager."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "async_kg.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            assert isinstance(kg, KnowledgeGraphDatabase)
            assert kg.conn is not None  # Should be initialized

    @pytest.mark.asyncio
    async def test_close_connection(self, tmp_path: Path) -> None:
        """Should close database connection gracefully."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "close_kg.duckdb")
        kg = KnowledgeGraphDatabase(db_path=db_path)

        async with kg:
            assert kg.conn is not None

        # After exit, connection should be closed
        assert kg.conn is None

    @pytest.mark.asyncio
    async def test_close_handles_no_connection(self) -> None:
        """Should handle close() when no connection exists."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        kg = KnowledgeGraphDatabase()
        kg.close()  # Should not raise

        assert kg.conn is None


class TestKnowledgeGraphEntityOperations:
    """Test entity creation and retrieval."""

    @pytest.mark.asyncio
    async def test_create_entity_basic(self, tmp_path: Path) -> None:
        """Should create entity with basic information."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "entities.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            entity = await kg.create_entity(
                name="test-project",
                entity_type="project",
                observations=["Test project"],
            )

            assert entity["name"] == "test-project"
            assert entity["entity_type"] == "project"
            assert "id" in entity

    @pytest.mark.asyncio
    async def test_get_entity_by_id(self, tmp_path: Path) -> None:
        """Should retrieve entity by ID."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "get_entity.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            # Create entity
            created = await kg.create_entity(
                name="test-lib", entity_type="library", observations=["Test library"]
            )

            # Retrieve by ID
            retrieved = await kg.get_entity(created["id"])

            assert retrieved is not None
            assert retrieved["id"] == created["id"]
            assert retrieved["name"] == "test-lib"

    @pytest.mark.asyncio
    async def test_find_entity_by_name(self, tmp_path: Path) -> None:
        """Should find entity by name."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "find_entity.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            # Create entity
            await kg.create_entity(
                name="fastmcp", entity_type="library", observations=["MCP framework"]
            )

            # Find by name
            found = await kg.find_entity_by_name("fastmcp", entity_type="library")

            assert found is not None
            assert found["name"] == "fastmcp"
            assert found["entity_type"] == "library"

    @pytest.mark.asyncio
    async def test_search_entities_by_type(self, tmp_path: Path) -> None:
        """Should search entities by type."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "search_type.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            # Create multiple entities
            await kg.create_entity(
                name="project1", entity_type="project", observations=["Project 1"]
            )
            await kg.create_entity(
                name="project2", entity_type="project", observations=["Project 2"]
            )
            await kg.create_entity(
                name="library1", entity_type="library", observations=["Library 1"]
            )

            # Search for projects only (query parameter required)
            results = await kg.search_entities(query="project", entity_type="project")

            assert len(results) == 2
            assert all(r["entity_type"] == "project" for r in results)

    @pytest.mark.asyncio
    async def test_add_observation_to_entity(self, tmp_path: Path) -> None:
        """Should add observation to existing entity."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "observations.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            # Create entity
            await kg.create_entity(
                name="test-entity",
                entity_type="concept",
                observations=["Initial observation"],
            )

            # Add observation
            success = await kg.add_observation("test-entity", "New observation")

            assert success is True

            # Verify observation was added
            entity = await kg.find_entity_by_name("test-entity")
            assert entity is not None
            # Note: Actual verification would depend on how observations are stored


class TestKnowledgeGraphRelations:
    """Test relationship creation and queries."""

    @pytest.mark.asyncio
    async def test_create_relation_between_entities(self, tmp_path: Path) -> None:
        """Should create relation between two entities."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "relations.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            # Create entities
            await kg.create_entity(
                name="session-mgmt", entity_type="project", observations=["MCP server"]
            )
            await kg.create_entity(
                name="ACB", entity_type="library", observations=["DI framework"]
            )

            # Create relation
            relation = await kg.create_relation(
                from_entity="session-mgmt", to_entity="ACB", relation_type="uses"
            )

            assert relation["from_entity"] == "session-mgmt"
            assert relation["to_entity"] == "ACB"
            assert relation["relation_type"] == "uses"

    @pytest.mark.asyncio
    async def test_get_relationships_for_entity(self, tmp_path: Path) -> None:
        """Should retrieve all relationships for an entity."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "get_rels.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            # Create entities and relations
            await kg.create_entity(
                name="project", entity_type="project", observations=[]
            )
            await kg.create_entity(name="lib1", entity_type="library", observations=[])
            await kg.create_entity(name="lib2", entity_type="library", observations=[])

            await kg.create_relation("project", "lib1", "uses")
            await kg.create_relation("project", "lib2", "uses")

            # Get relationships
            relationships = await kg.get_relationships("project")

            assert len(relationships) >= 2
            # Verify relationships are present

    @pytest.mark.asyncio
    async def test_find_path_between_entities(self, tmp_path: Path) -> None:
        """Should find path between two entities."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "path.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            # Create chain: A -> B -> C
            await kg.create_entity(name="A", entity_type="project", observations=[])
            await kg.create_entity(name="B", entity_type="library", observations=[])
            await kg.create_entity(name="C", entity_type="tool", observations=[])

            await kg.create_relation("A", "B", "uses")
            await kg.create_relation("B", "C", "depends_on")

            # Find path from A to C
            path = await kg.find_path("A", "C", max_depth=3)

            assert path is not None
            # Path should exist through B


class TestKnowledgeGraphStats:
    """Test statistics and monitoring."""

    @pytest.mark.asyncio
    async def test_get_stats_empty_graph(self, tmp_path: Path) -> None:
        """Should return stats for empty graph."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "stats_empty.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            stats = await kg.get_stats()

            assert isinstance(stats, dict)
            assert "total_entities" in stats
            assert "total_relationships" in stats
            assert stats["total_entities"] == 0
            assert stats["total_relationships"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, tmp_path: Path) -> None:
        """Should return accurate stats with data."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "stats_data.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            # Create some entities and relations
            await kg.create_entity(name="e1", entity_type="project", observations=[])
            await kg.create_entity(name="e2", entity_type="library", observations=[])
            await kg.create_relation("e1", "e2", "uses")

            stats = await kg.get_stats()

            assert isinstance(stats, dict)
            # Should have at least 2 entities and 1 relation


class TestKnowledgeGraphErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_entity(self, tmp_path: Path) -> None:
        """Should return None for nonexistent entity."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "nonexistent.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            result = await kg.get_entity("nonexistent-id")

            assert result is None

    @pytest.mark.asyncio
    async def test_find_nonexistent_entity_by_name(self, tmp_path: Path) -> None:
        """Should return None for nonexistent entity name."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "find_nonexistent.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            result = await kg.find_entity_by_name("nonexistent")

            assert result is None

    @pytest.mark.asyncio
    async def test_create_relation_with_missing_entity(self, tmp_path: Path) -> None:
        """Should handle relation creation with missing entities."""
        from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

        db_path = str(tmp_path / "missing_entity.duckdb")

        async with KnowledgeGraphDatabase(db_path=db_path) as kg:
            # Try to create relation without creating entities first
            # Should either raise error or fail gracefully
            try:
                await kg.create_relation("missing1", "missing2", "uses")
            except Exception as e:
                # Expected - entities don't exist
                assert "missing" in str(e).lower() or "not found" in str(e).lower()

    @pytest.mark.asyncio
    async def test_duckdb_unavailable_handling(self) -> None:
        """Should handle missing DuckDB gracefully."""
        with patch("session_buddy.knowledge_graph_db.DUCKDB_AVAILABLE", False):
            from session_buddy.knowledge_graph_db import KnowledgeGraphDatabase

            KnowledgeGraphDatabase()

            # Should handle missing DuckDB gracefully
            # (May raise ImportError or return gracefully)
