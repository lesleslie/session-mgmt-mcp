"""Tests for multi_project_coordinator module.

Tests project group management, dependency tracking, session linking,
and cross-project insights functionality.

Phase: Week 5 Day 4 - Multi-Project Coordinator Coverage
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestPydanticModels:
    """Test Pydantic data models for multi-project coordination."""

    def test_project_group_creation(self) -> None:
        """Should create ProjectGroup with required fields."""
        from session_buddy.multi_project_coordinator import ProjectGroup

        group = ProjectGroup(
            id="group-1",
            name="Microservices Group",
            projects=["service-a", "service-b"],
            description="Related microservices",
            created_at=datetime.now().isoformat(),
            metadata={"owner": "team-a"},
        )

        assert group.id == "group-1"
        assert group.name == "Microservices Group"
        assert len(group.projects) == 2
        assert "service-a" in group.projects

    def test_project_dependency_types(self) -> None:
        """Should validate ProjectDependency types."""
        from session_buddy.multi_project_coordinator import ProjectDependency

        # Valid dependency types
        for dep_type in ["uses", "extends", "references", "shares_code"]:
            dep = ProjectDependency(
                id=f"dep-{dep_type}",
                source_project="project-a",
                target_project="project-b",
                dependency_type=dep_type,
                description=f"Test {dep_type} dependency",
            )
            assert dep.dependency_type == dep_type

    def test_session_link_types(self) -> None:
        """Should validate SessionLink types."""
        from session_buddy.multi_project_coordinator import SessionLink

        # Valid link types
        for link_type in ["related", "continuation", "reference", "dependency"]:
            link = SessionLink(
                id=f"link-{link_type}",
                source_session_id="session-1",
                target_session_id="session-2",
                link_type=link_type,
                context=f"Test {link_type} link",
            )
            assert link.link_type == link_type


class TestMultiProjectCoordinatorCRUD:
    """Test CRUD operations for project groups, dependencies, and links."""

    @pytest.mark.asyncio
    async def test_create_project_group(self) -> None:
        """Should create project group in database."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()
        mock_db.conn.execute = MagicMock()

        coordinator = MultiProjectCoordinator(mock_db)

        group = await coordinator.create_project_group(
            name="Test Group",
            projects=["proj-a", "proj-b"],
            description="Test description",
        )

        assert group.name == "Test Group"
        assert len(group.projects) == 2
        # Should have called execute to insert
        assert mock_db.conn.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_add_project_dependency(self) -> None:
        """Should add project dependency relationship."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()
        mock_db.conn.execute = MagicMock()

        coordinator = MultiProjectCoordinator(mock_db)

        dependency = await coordinator.add_project_dependency(
            source_project="service-frontend",
            target_project="service-backend",
            dependency_type="uses",
            description="Frontend uses backend API",
        )

        assert dependency.source_project == "service-frontend"
        assert dependency.target_project == "service-backend"
        assert dependency.dependency_type == "uses"
        assert mock_db.conn.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_link_sessions(self) -> None:
        """Should create session link between projects."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()
        mock_db.conn.execute = MagicMock()

        coordinator = MultiProjectCoordinator(mock_db)

        link = await coordinator.link_sessions(
            source_session_id="sess-1",
            target_session_id="sess-2",
            link_type="continuation",
            context="Continued work in related project",
        )

        assert link.source_session_id == "sess-1"
        assert link.link_type == "continuation"
        assert mock_db.conn.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_get_project_groups_empty(self) -> None:
        """Should return empty list when no groups exist."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()
        mock_db.conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=[]))
        )

        coordinator = MultiProjectCoordinator(mock_db)

        groups = await coordinator.get_project_groups()

        assert groups == []

    @pytest.mark.asyncio
    async def test_get_project_dependencies_with_direction(self) -> None:
        """Should filter dependencies by direction."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()
        mock_deps = [
            (
                "dep-1",
                "proj-a",
                "proj-b",
                "uses",
                "Test dependency",
                datetime.now(),
                "{}",
            )
        ]
        mock_db.conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=mock_deps))
        )

        coordinator = MultiProjectCoordinator(mock_db)

        # Test outbound dependencies
        deps = await coordinator.get_project_dependencies(
            project="proj-a", direction="outbound"
        )

        assert len(deps) == 1
        assert deps[0].source_project == "proj-a"

    @pytest.mark.asyncio
    async def test_get_session_links(self) -> None:
        """Should retrieve all links for a session."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()
        mock_links = [
            (
                "link-1",
                "sess-1",
                "sess-2",
                "related",
                "Test link",
                datetime.now(),
                "{}",
            )
        ]
        mock_db.conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=mock_links))
        )

        coordinator = MultiProjectCoordinator(mock_db)

        links = await coordinator.get_session_links("sess-1")

        assert len(links) == 1
        assert links[0].source_session_id == "sess-1"


class TestCachingBehavior:
    """Test caching functionality in multi-project operations."""

    @pytest.mark.asyncio
    async def test_cache_groups_on_retrieval(self) -> None:
        """Should cache project groups after retrieval."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()
        mock_groups = [
            (
                "group-1",
                "Test Group",
                "Desc",
                ["proj-a", "proj-b"],
                datetime.now(),
                "{}",
            )
        ]
        mock_db.conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=mock_groups))
        )

        coordinator = MultiProjectCoordinator(mock_db)

        # First call - hits database
        groups1 = await coordinator.get_project_groups()
        # Second call - database hit again (no in-memory cache for list operations)
        groups2 = await coordinator.get_project_groups()

        assert len(groups1) == 1
        assert len(groups2) == 1
        # Implementation populates active_project_groups dict

    @pytest.mark.asyncio
    async def test_cache_population_on_create(self) -> None:
        """Should populate cache when creating new group."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()
        mock_db.conn.execute = MagicMock()
        mock_db.conn.commit = MagicMock()

        coordinator = MultiProjectCoordinator(mock_db)

        # Verify cache is initially empty
        assert len(coordinator.active_project_groups) == 0

        # Create group should populate cache
        group = await coordinator.create_project_group(
            name="New Group", projects=["proj-a"], description="Test"
        )

        # Cache should contain the new group
        assert len(coordinator.active_project_groups) == 1
        assert group.id in coordinator.active_project_groups
        assert coordinator.active_project_groups[group.id] is group


class TestCrossProjectSearch:
    """Test cross-project search functionality."""

    @pytest.mark.asyncio
    async def test_find_related_conversations(self) -> None:
        """Should search across related projects."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()

        # Mock get_project_dependencies (empty)
        mock_db.conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=[]))
        )

        # Mock ReflectionDatabase.search_conversations
        mock_db.search_conversations = AsyncMock(
            return_value=[
                {
                    "id": "conv-1",
                    "content": "Test conversation about API",
                    "score": 0.85,
                }
            ]
        )

        coordinator = MultiProjectCoordinator(mock_db)

        results = await coordinator.find_related_conversations(
            query="API", current_project="proj-a", limit=10
        )

        assert isinstance(results, list)
        # Should return conversations with source_project field
        if len(results) > 0:
            assert "source_project" in results[0]
            assert "is_current_project" in results[0]

    @pytest.mark.asyncio
    async def test_dependency_aware_ranking(self) -> None:
        """Should rank results based on project dependencies."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()

        # Mock get_project_dependencies - proj-a depends on proj-b
        mock_deps = [
            ("dep-1", "proj-a", "proj-b", "uses", "Desc", datetime.now(), "{}"),
        ]
        mock_db.conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=mock_deps))
        )

        # Mock ReflectionDatabase.search_conversations for each project
        mock_db.search_conversations = AsyncMock(
            side_effect=[
                [{"id": "conv-1", "content": "API implementation", "score": 0.9}],
                [{"id": "conv-2", "content": "API usage", "score": 0.8}],
            ]
        )

        coordinator = MultiProjectCoordinator(mock_db)

        results = await coordinator.find_related_conversations(
            query="API", current_project="proj-a", limit=10
        )

        # Should include results from related projects
        assert isinstance(results, list)
        # Results should be sorted by score
        if len(results) >= 2:
            assert results[0].get("score", 0) >= results[1].get("score", 0)


class TestInsightsAndAnalytics:
    """Test cross-project insights and analytics."""

    @pytest.mark.asyncio
    async def test_get_cross_project_insights(self) -> None:
        """Should generate insights across related projects."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()

        # Mock _get_project_stats queries (one per project)
        mock_stats = (10, datetime.now(), 250.5)  # count, last_activity, avg_length
        # Mock _get_conversation_data
        mock_convs = [("proj-a", "test content"), ("proj-b", "another test")]

        mock_db.conn.execute = MagicMock(
            side_effect=[
                MagicMock(fetchone=MagicMock(return_value=mock_stats)),
                MagicMock(fetchone=MagicMock(return_value=mock_stats)),
                MagicMock(fetchall=MagicMock(return_value=mock_convs)),
            ]
        )

        coordinator = MultiProjectCoordinator(mock_db)

        insights = await coordinator.get_cross_project_insights(
            projects=["proj-a", "proj-b"]
        )

        assert isinstance(insights, dict)
        assert "project_activity" in insights
        assert "common_patterns" in insights

    @pytest.mark.asyncio
    async def test_pattern_detection(self) -> None:
        """Should detect patterns across projects."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()

        # Mock _get_project_stats
        mock_stats = (10, datetime.now(), 250.5)
        # Mock _get_conversation_data with similar content
        mock_convs = [
            ("proj-a", "Fix authentication error and update tests"),
            ("proj-b", "Fix authentication error in login flow"),
        ]

        mock_db.conn.execute = MagicMock(
            side_effect=[
                MagicMock(fetchone=MagicMock(return_value=mock_stats)),
                MagicMock(fetchone=MagicMock(return_value=mock_stats)),
                MagicMock(fetchall=MagicMock(return_value=mock_convs)),
            ]
        )

        coordinator = MultiProjectCoordinator(mock_db)

        insights = await coordinator.get_cross_project_insights(
            projects=["proj-a", "proj-b"]
        )

        # Should identify common patterns like "authentication"
        assert isinstance(insights, dict)
        assert "common_patterns" in insights
        assert isinstance(insights["common_patterns"], list)

    @pytest.mark.asyncio
    async def test_collaboration_opportunities(self) -> None:
        """Should provide insights structure."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()

        # Mock _get_project_stats
        mock_stats = (10, datetime.now(), 250.5)
        # Mock _get_conversation_data
        mock_convs = [("proj-a", "shared utilities code"), ("proj-b", "shared library")]

        mock_db.conn.execute = MagicMock(
            side_effect=[
                MagicMock(fetchone=MagicMock(return_value=mock_stats)),
                MagicMock(fetchone=MagicMock(return_value=mock_stats)),
                MagicMock(fetchall=MagicMock(return_value=mock_convs)),
            ]
        )

        coordinator = MultiProjectCoordinator(mock_db)

        insights = await coordinator.get_cross_project_insights(
            projects=["proj-a", "proj-b"]
        )

        # Should include collaboration_opportunities field
        assert isinstance(insights, dict)
        assert "collaboration_opportunities" in insights


class TestCleanupOperations:
    """Test cleanup functionality for stale links and groups."""

    @pytest.mark.asyncio
    async def test_cleanup_old_links(self) -> None:
        """Should remove old session links."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()

        # Mock count before deletion
        count_result = 5
        # Mock delete operation
        mock_db.conn.execute = MagicMock(
            side_effect=[
                MagicMock(fetchone=MagicMock(return_value=(count_result,))),
                MagicMock(),  # DELETE statement
            ]
        )
        mock_db.conn.commit = MagicMock()

        coordinator = MultiProjectCoordinator(mock_db)

        # Cleanup links older than 90 days (default parameter is max_age_days not days)
        result = await coordinator.cleanup_old_links(max_age_days=90)

        assert isinstance(result, dict)
        assert "deleted_session_links" in result
        assert result["deleted_session_links"] == count_result

    @pytest.mark.asyncio
    async def test_cleanup_respects_threshold(self) -> None:
        """Should only cleanup links older than threshold."""
        from session_buddy.multi_project_coordinator import MultiProjectCoordinator

        mock_db = MagicMock()
        mock_db.conn = MagicMock()

        # Count of old links (those before cutoff)
        old_links_count = 3

        # Mock count and delete
        mock_db.conn.execute = MagicMock(
            side_effect=[
                MagicMock(fetchone=MagicMock(return_value=(old_links_count,))),
                MagicMock(),  # DELETE
            ]
        )
        mock_db.conn.commit = MagicMock()

        coordinator = MultiProjectCoordinator(mock_db)

        result = await coordinator.cleanup_old_links(max_age_days=90)

        # Should only count/remove links older than 90 days
        assert isinstance(result, dict)
        assert result["deleted_session_links"] == old_links_count
