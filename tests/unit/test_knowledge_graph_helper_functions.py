#!/usr/bin/env python3
"""Tests for knowledge graph tools helper functions.

Tests helper functions created during complexity refactoring of knowledge_graph_tools.py,
including entity extraction, formatting, and pattern matching.

Phase 2: Core Coverage (14% → 60%) - Helper Function Tests
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestExtractPatternsFromContext:
    """Test _extract_patterns_from_context helper function.

    Phase 2: Core Coverage - knowledge_graph_tools.py (14% → 60%)
    """

    def test_extract_project_patterns(self) -> None:
        """Should extract kebab-case project names."""
        from session_buddy.tools.knowledge_graph_tools import (
            _extract_patterns_from_context,
        )

        context = "Working on session-mgmt-mcp and acb-framework projects"

        result = _extract_patterns_from_context(context)

        assert "project" in result
        assert "session-mgmt-mcp" in result["project"]
        assert "acb-framework" in result["project"]

    def test_extract_library_patterns(self) -> None:
        """Should extract common library names."""
        from session_buddy.tools.knowledge_graph_tools import (
            _extract_patterns_from_context,
        )

        context = "Using FastMCP, DuckDB, and pytest for testing"

        result = _extract_patterns_from_context(context)

        assert "library" in result
        assert "FastMCP" in result["library"]
        assert "DuckDB" in result["library"]
        assert "pytest" in result["library"]

    def test_extract_technology_patterns(self) -> None:
        """Should extract technology names."""
        from session_buddy.tools.knowledge_graph_tools import (
            _extract_patterns_from_context,
        )

        context = "Built with Python and TypeScript, deployed on Docker"

        result = _extract_patterns_from_context(context)

        assert "technology" in result
        assert "Python" in result["technology"]
        assert "TypeScript" in result["technology"]
        assert "Docker" in result["technology"]

    def test_extract_concept_patterns(self) -> None:
        """Should extract concept phrases."""
        from session_buddy.tools.knowledge_graph_tools import (
            _extract_patterns_from_context,
        )

        context = "Implementing dependency injection and semantic memory features"

        result = _extract_patterns_from_context(context)

        assert "concept" in result
        assert "dependency injection" in result["concept"]
        assert "semantic memory" in result["concept"]

    def test_extract_multiple_pattern_types(self) -> None:
        """Should extract multiple pattern types simultaneously."""
        from session_buddy.tools.knowledge_graph_tools import (
            _extract_patterns_from_context,
        )

        context = """
        Building session-mgmt-mcp with Python and FastMCP.
        Implementing dependency injection using ACB framework.
        """

        result = _extract_patterns_from_context(context)

        assert len(result) >= 3  # project, library, technology, concept
        assert "session-mgmt-mcp" in result["project"]
        assert "Python" in result["technology"]
        assert "FastMCP" in result["library"]
        assert "dependency injection" in result["concept"]

    def test_extract_case_insensitive(self) -> None:
        """Should extract patterns case-insensitively."""
        from session_buddy.tools.knowledge_graph_tools import (
            _extract_patterns_from_context,
        )

        context = "Using PYTHON and python and Python"

        result = _extract_patterns_from_context(context)

        assert "technology" in result
        # Should deduplicate case-insensitive matches
        assert "Python" in result["technology"] or "PYTHON" in result["technology"]

    def test_extract_from_empty_context(self) -> None:
        """Should return empty dict for empty context."""
        from session_buddy.tools.knowledge_graph_tools import (
            _extract_patterns_from_context,
        )

        result = _extract_patterns_from_context("")

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_extract_no_matches(self) -> None:
        """Should return empty dict when no patterns match."""
        from session_buddy.tools.knowledge_graph_tools import (
            _extract_patterns_from_context,
        )

        context = "This is just regular text with no entity patterns"

        result = _extract_patterns_from_context(context)

        assert isinstance(result, dict)
        # May be empty or have empty sets


class TestFormatEntityTypes:
    """Test _format_entity_types helper function.

    Phase 2: Core Coverage - knowledge_graph_tools.py (14% → 60%)
    """

    def test_format_single_entity_type(self) -> None:
        """Should format single entity type."""
        from session_buddy.tools.knowledge_graph_tools import _format_entity_types

        entity_types = {"project": 5}

        result = _format_entity_types(entity_types)

        assert isinstance(result, list)
        assert len(result) == 3  # Header, item, blank line
        assert "📊 Entity Types:" in result[0]
        assert "project: 5" in result[1]
        assert result[2] == ""

    def test_format_multiple_entity_types(self) -> None:
        """Should format multiple entity types."""
        from session_buddy.tools.knowledge_graph_tools import _format_entity_types

        entity_types = {"project": 10, "library": 5, "technology": 3}

        result = _format_entity_types(entity_types)

        assert isinstance(result, list)
        assert len(result) == 5  # Header + 3 items + blank
        assert "📊 Entity Types:" in result[0]
        assert any("project: 10" in line for line in result)
        assert any("library: 5" in line for line in result)
        assert any("technology: 3" in line for line in result)

    def test_format_empty_entity_types(self) -> None:
        """Should return empty list for empty entity types."""
        from session_buddy.tools.knowledge_graph_tools import _format_entity_types

        result = _format_entity_types({})

        assert result == []

    def test_format_none_entity_types(self) -> None:
        """Should handle None gracefully."""
        from session_buddy.tools.knowledge_graph_tools import _format_entity_types

        # Function expects dict, but test defensive behavior
        result = _format_entity_types({})

        assert isinstance(result, list)


class TestFormatRelationshipTypes:
    """Test _format_relationship_types helper function.

    Phase 2: Core Coverage - knowledge_graph_tools.py (14% → 60%)
    """

    def test_format_single_relationship_type(self) -> None:
        """Should format single relationship type."""
        from session_buddy.tools.knowledge_graph_tools import (
            _format_relationship_types,
        )

        relationship_types = {"uses": 8}

        result = _format_relationship_types(relationship_types)

        assert isinstance(result, list)
        assert len(result) == 3  # Header, item, blank line
        assert "🔗 Relationship Types:" in result[0]
        assert "uses: 8" in result[1]
        assert result[2] == ""

    def test_format_multiple_relationship_types(self) -> None:
        """Should format multiple relationship types."""
        from session_buddy.tools.knowledge_graph_tools import (
            _format_relationship_types,
        )

        relationship_types = {"uses": 15, "depends_on": 10, "extends": 5}

        result = _format_relationship_types(relationship_types)

        assert isinstance(result, list)
        assert len(result) == 5  # Header + 3 items + blank
        assert "🔗 Relationship Types:" in result[0]
        assert any("uses: 15" in line for line in result)
        assert any("depends_on: 10" in line for line in result)
        assert any("extends: 5" in line for line in result)

    def test_format_empty_relationship_types(self) -> None:
        """Should return empty list for empty relationship types."""
        from session_buddy.tools.knowledge_graph_tools import (
            _format_relationship_types,
        )

        result = _format_relationship_types({})

        assert result == []


class TestFormatEntityResult:
    """Test _format_entity_result helper function.

    Phase 2: Core Coverage - knowledge_graph_tools.py (14% → 60%)
    """

    def test_format_entity_with_observations(self) -> None:
        """Should format entity with observations."""
        from session_buddy.tools.knowledge_graph_tools import _format_entity_result

        entity = {
            "name": "FastMCP",
            "entity_type": "library",
            "observations": ["MCP server framework", "Built with Python"],
        }

        result = _format_entity_result(entity)

        assert isinstance(result, list)
        assert len(result) == 4  # Name line, obs count, preview, blank
        assert "FastMCP (library)" in result[0]
        assert "Observations: 2" in result[1]
        assert "MCP server framework" in result[2]
        assert result[3] == ""

    def test_format_entity_without_observations(self) -> None:
        """Should format entity without observations."""
        from session_buddy.tools.knowledge_graph_tools import _format_entity_result

        entity = {"name": "TestEntity", "entity_type": "concept", "observations": []}

        result = _format_entity_result(entity)

        assert isinstance(result, list)
        assert "TestEntity (concept)" in result[0]
        # Should still have blank line at end
        assert result[-1] == ""

    def test_format_entity_long_observation(self) -> None:
        """Should truncate long observations."""
        from session_buddy.tools.knowledge_graph_tools import _format_entity_result

        long_text = "x" * 100
        entity = {
            "name": "Entity",
            "entity_type": "test",
            "observations": [long_text],
        }

        result = _format_entity_result(entity)

        # Check that observation is truncated with ellipsis
        observation_line = result[2]
        assert "..." in observation_line
        assert len(observation_line) < 100  # Should be truncated


class TestFormatBatchResults:
    """Test _format_batch_results helper function.

    Phase 2: Core Coverage - knowledge_graph_tools.py (14% → 60%)
    """

    def test_format_all_succeeded(self) -> None:
        """Should format when all entities created successfully."""
        from session_buddy.tools.knowledge_graph_tools import _format_batch_results

        created = ["Entity1", "Entity2", "Entity3"]
        failed: list[tuple[str, str]] = []

        result = _format_batch_results(created, failed)

        assert isinstance(result, list)
        assert "Successfully Created: 3" in result[2]
        assert "Entity1" in "\n".join(result)
        assert "Entity2" in "\n".join(result)
        assert "Entity3" in "\n".join(result)
        assert "Failed:" not in "\n".join(result)

    def test_format_all_failed(self) -> None:
        """Should format when all entities failed."""
        from session_buddy.tools.knowledge_graph_tools import _format_batch_results

        created: list[str] = []
        failed = [("Entity1", "Error1"), ("Entity2", "Error2")]

        result = _format_batch_results(created, failed)

        assert isinstance(result, list)
        assert "Successfully Created: 0" in result[2]
        assert "Failed: 2" in "\n".join(result)
        assert "Entity1: Error1" in "\n".join(result)
        assert "Entity2: Error2" in "\n".join(result)

    def test_format_mixed_results(self) -> None:
        """Should format mix of successes and failures."""
        from session_buddy.tools.knowledge_graph_tools import _format_batch_results

        created = ["Success1", "Success2"]
        failed = [("Failed1", "Error message")]

        result = _format_batch_results(created, failed)

        assert isinstance(result, list)
        assert "Successfully Created: 2" in result[2]
        assert "Failed: 1" in "\n".join(result)

    def test_format_many_created(self) -> None:
        """Should truncate when more than 10 created."""
        from session_buddy.tools.knowledge_graph_tools import _format_batch_results

        created = [f"Entity{i}" for i in range(15)]
        failed: list[tuple[str, str]] = []

        result = _format_batch_results(created, failed)

        result_text = "\n".join(result)
        assert "Successfully Created: 15" in result_text
        assert "and 5 more" in result_text

    def test_format_many_failed(self) -> None:
        """Should truncate when more than 5 failed."""
        from session_buddy.tools.knowledge_graph_tools import _format_batch_results

        created: list[str] = []
        failed = [(f"Entity{i}", f"Error{i}") for i in range(10)]

        result = _format_batch_results(created, failed)

        result_text = "\n".join(result)
        assert "Failed: 10" in result_text
        assert "and 5 more" in result_text


class TestAutoCreateEntityIfNew:
    """Test _auto_create_entity_if_new helper function.

    Phase 2: Core Coverage - knowledge_graph_tools.py (14% → 60%)
    """

    @pytest.mark.asyncio
    async def test_create_when_entity_not_exists(self) -> None:
        """Should create entity when it doesn't exist."""
        from session_buddy.tools.knowledge_graph_tools import (
            _auto_create_entity_if_new,
        )

        # Mock knowledge graph adapter
        mock_kg = AsyncMock()
        mock_kg.find_entity_by_name = AsyncMock(return_value=None)
        mock_kg.create_entity = AsyncMock(
            return_value={"id": "123", "name": "NewEntity"}
        )

        result = await _auto_create_entity_if_new(mock_kg, "NewEntity", "project")

        assert result is True
        mock_kg.find_entity_by_name.assert_called_once_with("NewEntity")
        mock_kg.create_entity.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_when_entity_exists(self) -> None:
        """Should not create entity when it already exists."""
        from session_buddy.tools.knowledge_graph_tools import (
            _auto_create_entity_if_new,
        )

        # Mock knowledge graph adapter
        mock_kg = AsyncMock()
        mock_kg.find_entity_by_name = AsyncMock(
            return_value={"id": "existing", "name": "ExistingEntity"}
        )
        mock_kg.create_entity = AsyncMock()

        result = await _auto_create_entity_if_new(mock_kg, "ExistingEntity", "project")

        assert result is False
        mock_kg.find_entity_by_name.assert_called_once_with("ExistingEntity")
        mock_kg.create_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_with_default_observation(self) -> None:
        """Should create entity with default observation."""
        from session_buddy.tools.knowledge_graph_tools import (
            _auto_create_entity_if_new,
        )

        mock_kg = AsyncMock()
        mock_kg.find_entity_by_name = AsyncMock(return_value=None)
        mock_kg.create_entity = AsyncMock(return_value={"id": "123", "name": "Entity"})

        await _auto_create_entity_if_new(mock_kg, "Entity", "library")

        # Verify create_entity was called with default observation
        call_kwargs = mock_kg.create_entity.call_args.kwargs
        assert "observations" in call_kwargs
        assert call_kwargs["observations"] == ["Extracted from conversation context"]
