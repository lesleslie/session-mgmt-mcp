#!/usr/bin/env python3
"""Unit tests for Advanced Search Engine."""

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from session_mgmt_mcp.advanced_search import (
    AdvancedSearchEngine,
    SearchFacet,
    SearchFilter,
    SearchResult,
)
from session_mgmt_mcp.reflection_tools import ReflectionDatabase


@pytest.fixture
async def temp_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    db = ReflectionDatabase(db_path)
    await db.initialize()

    yield db

    # Cleanup
    db.close()
    Path(db_path).unlink()


@pytest.fixture
async def search_engine(temp_db):
    """Create a search engine with test database."""
    return AdvancedSearchEngine(temp_db)


@pytest.fixture
async def sample_data(temp_db):
    """Add sample conversations and reflections to test database."""
    # Sample conversations
    conversations = [
        {
            "content": "Implementing user authentication with JWT tokens in Python Flask. Need to handle token expiration and refresh logic.",
            "project": "webapp-backend",
            "metadata": {
                "type": "development",
                "language": "python",
                "framework": "flask",
            },
        },
        {
            "content": "Frontend React component for user login form. Using axios for API calls to backend authentication endpoint.",
            "project": "webapp-frontend",
            "metadata": {
                "type": "development",
                "language": "javascript",
                "framework": "react",
            },
        },
        {
            "content": "Database schema design for user management. Created users table with proper indexes for performance.",
            "project": "webapp-backend",
            "metadata": {
                "type": "database",
                "database": "postgresql",
                "performance": "indexes",
            },
        },
        {
            "content": "Error: JWT token validation failed. TokenExpiredError: Signature has expired. Need to implement token refresh.",
            "project": "webapp-backend",
            "metadata": {
                "type": "error",
                "error_type": "TokenExpiredError",
                "component": "auth",
            },
        },
        {
            "content": "DevOps: Setting up CI/CD pipeline with Docker containers. Automated testing and deployment to production.",
            "project": "devops-pipeline",
            "metadata": {
                "type": "infrastructure",
                "tool": "docker",
                "environment": "production",
            },
        },
    ]

    for conv in conversations:
        await temp_db.store_conversation(
            content=conv["content"],
            metadata={"project": conv["project"], **conv["metadata"]},
        )

    # Sample reflections
    reflections = [
        {
            "content": "Authentication patterns: Always use secure JWT implementation with proper expiration handling",
            "tags": ["authentication", "jwt", "security", "best-practices"],
        },
        {
            "content": "Database performance tip: Index frequently queried columns, especially foreign keys",
            "tags": ["database", "performance", "postgresql", "optimization"],
        },
        {
            "content": "React component patterns: Use functional components with hooks for better performance",
            "tags": ["react", "frontend", "performance", "hooks"],
        },
    ]

    for refl in reflections:
        await temp_db.store_reflection(content=refl["content"], tags=refl["tags"])

    return conversations, reflections


class TestSearchIndexing:
    """Test search index creation and management."""

    @pytest.mark.asyncio
    async def test_index_conversations(self, search_engine, sample_data):
        """Test indexing of conversations."""
        await search_engine._rebuild_search_index()

        # Check that conversations were indexed
        sql = "SELECT COUNT(*) FROM search_index WHERE content_type = 'conversation'"
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_engine.reflection_db.conn.execute(sql).fetchone(),
        )

        conversation_count = result[0]
        assert conversation_count >= 5  # We added 5 conversations

    @pytest.mark.asyncio
    async def test_index_reflections(self, search_engine, sample_data):
        """Test indexing of reflections."""
        await search_engine._rebuild_search_index()

        # Check that reflections were indexed
        sql = "SELECT COUNT(*) FROM search_index WHERE content_type = 'reflection'"
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_engine.reflection_db.conn.execute(sql).fetchone(),
        )

        reflection_count = result[0]
        assert reflection_count >= 3  # We added 3 reflections

    @pytest.mark.asyncio
    async def test_technical_term_extraction(self, search_engine, sample_data):
        """Test extraction of technical terms from content."""
        # Test technical term extraction
        test_content = """
        def authenticate_user(username, password):
            try:
                user = User.objects.get(username=username)
                if user.check_password(password):
                    return generate_jwt_token(user)
                else:
                    raise ValueError("Invalid credentials")
            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                return None
        """

        terms = search_engine._extract_technical_terms(test_content)

        # Should detect Python language
        assert "python" in terms

        # Should detect function definition
        function_terms = [t for t in terms if t.startswith("function:")]
        assert len(function_terms) >= 1

        # Should detect error handling
        assert "error" in terms or "Error" in terms

    @pytest.mark.asyncio
    async def test_facet_generation(self, search_engine, sample_data):
        """Test automatic facet generation."""
        await search_engine._rebuild_search_index()
        await search_engine._update_search_facets()

        # Check that facets were created
        sql = "SELECT DISTINCT facet_name FROM search_facets"
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_engine.reflection_db.conn.execute(sql).fetchall(),
        )

        facet_names = [row[0] for row in results]

        # Should have project facets
        assert "project" in facet_names

        # Should have content type facets
        assert "content_type" in facet_names


class TestBasicSearch:
    """Test basic search functionality."""

    @pytest.mark.asyncio
    async def test_simple_text_search(self, search_engine, sample_data):
        """Test simple text-based search."""
        await search_engine._rebuild_search_index()

        # Search for authentication-related content
        results = await search_engine.search(
            query="authentication",
            limit=5,
            include_highlights=True,
        )

        assert len(results["results"]) > 0

        # Should find authentication-related content
        auth_result = next(
            (r for r in results["results"] if "authentication" in r.content.lower()),
            None,
        )
        assert auth_result is not None

        # Should have highlights
        if auth_result.highlights:
            highlight_text = " ".join(auth_result.highlights)
            assert "authentication" in highlight_text.lower()

    @pytest.mark.asyncio
    async def test_project_filtering(self, search_engine, sample_data):
        """Test filtering by project."""
        await search_engine._rebuild_search_index()

        # Create project filter
        project_filter = SearchFilter(
            field="project",
            operator="eq",
            value="webapp-backend",
        )

        results = await search_engine.search(
            query="*",  # Match all
            filters=[project_filter],
            limit=10,
        )

        # All results should be from webapp-backend project
        for result in results["results"]:
            assert result.project == "webapp-backend"

    @pytest.mark.asyncio
    async def test_content_type_filtering(self, search_engine, sample_data):
        """Test filtering by content type."""
        await search_engine._rebuild_search_index()

        # Filter for conversations only
        content_filter = SearchFilter(
            field="content_type",
            operator="eq",
            value="conversation",
        )

        results = await search_engine.search(
            query="*",
            filters=[content_filter],
            limit=10,
        )

        # All results should be conversations
        for result in results["results"]:
            assert result.content_type == "conversation"

    @pytest.mark.asyncio
    async def test_timeframe_filtering(self, search_engine, sample_data):
        """Test filtering by time range."""
        await search_engine._rebuild_search_index()

        # Create time filter for last 24 hours
        now = datetime.now(UTC)
        yesterday = now - timedelta(hours=24)

        time_filter = SearchFilter(
            field="timestamp",
            operator="range",
            value=(yesterday, now),
        )

        results = await search_engine.search(query="*", filters=[time_filter], limit=10)

        # Should return results (all our test data is recent)
        assert len(results["results"]) > 0

    @pytest.mark.asyncio
    async def test_multiple_filters(self, search_engine, sample_data):
        """Test combining multiple filters."""
        await search_engine._rebuild_search_index()

        # Combine project and content type filters
        filters = [
            SearchFilter(field="project", operator="eq", value="webapp-backend"),
            SearchFilter(field="content_type", operator="eq", value="conversation"),
        ]

        results = await search_engine.search(query="*", filters=filters, limit=10)

        # Results should match both filters
        for result in results["results"]:
            assert result.project == "webapp-backend"
            assert result.content_type == "conversation"


class TestAdvancedSearch:
    """Test advanced search features."""

    @pytest.mark.asyncio
    async def test_faceted_search(self, search_engine, sample_data):
        """Test faceted search with result counts."""
        await search_engine._rebuild_search_index()

        # Search with facets
        results = await search_engine.search(
            query="authentication",
            facets=["project", "content_type"],
            limit=5,
        )

        assert "facets" in results
        facets = results["facets"]

        # Should have requested facets
        if "project" in facets:
            project_facet = facets["project"]
            assert isinstance(project_facet, SearchFacet)
            assert project_facet.name == "project"
            assert len(project_facet.values) > 0

    @pytest.mark.asyncio
    async def test_search_suggestions(self, search_engine, sample_data):
        """Test search completion suggestions."""
        await search_engine._rebuild_search_index()

        # Get suggestions for partial query
        suggestions = await search_engine.suggest_completions(
            query="auth",
            field="content",
            limit=5,
        )

        assert isinstance(suggestions, list)

        if suggestions:
            # Should have frequency information
            assert "frequency" in suggestions[0]
            assert "text" in suggestions[0]

    @pytest.mark.asyncio
    async def test_similar_content(self, search_engine, sample_data):
        """Test finding similar content."""
        await search_engine._rebuild_search_index()

        # First, get a conversation ID from the index
        sql = "SELECT content_id FROM search_index WHERE content_type = 'conversation' LIMIT 1"
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_engine.reflection_db.conn.execute(sql).fetchone(),
        )

        if result:
            content_id = result[0]

            similar = await search_engine.get_similar_content(
                content_id=content_id,
                content_type="conversation",
                limit=3,
            )

            assert isinstance(similar, list)
            # Should not include the original content
            similar_ids = [s.content_id for s in similar]
            assert content_id not in similar_ids

    @pytest.mark.asyncio
    async def test_timeframe_search(self, search_engine, sample_data):
        """Test timeframe-based search."""
        await search_engine._rebuild_search_index()

        # Search for content from last day
        results = await search_engine.search_by_timeframe(
            timeframe="1d",
            query="authentication",
            limit=5,
        )

        assert isinstance(results, list)

        # All results should be recent (our test data is all recent)
        for result in results:
            assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_sorting_options(self, search_engine, sample_data):
        """Test different sorting options."""
        await search_engine._rebuild_search_index()

        # Test relevance sorting (default)
        relevance_results = await search_engine.search(
            query="authentication",
            sort_by="relevance",
            limit=3,
        )

        # Test date sorting
        date_results = await search_engine.search(
            query="authentication",
            sort_by="date",
            limit=3,
        )

        # Both should return results
        assert len(relevance_results["results"]) > 0
        assert len(date_results["results"]) > 0

        # Results might be in different order
        [r.content_id for r in relevance_results["results"]]
        [r.content_id for r in date_results["results"]]

        # Not necessarily different order with small dataset, but should not error


class TestSearchMetrics:
    """Test search metrics and analytics."""

    @pytest.mark.asyncio
    async def test_activity_metrics(self, search_engine, sample_data):
        """Test activity metrics calculation."""
        await search_engine._rebuild_search_index()

        metrics = await search_engine.aggregate_metrics(
            metric_type="activity",
            timeframe="30d",
        )

        assert "metric_type" in metrics
        assert metrics["metric_type"] == "activity"
        assert "data" in metrics
        assert isinstance(metrics["data"], list)

    @pytest.mark.asyncio
    async def test_project_metrics(self, search_engine, sample_data):
        """Test project-based metrics."""
        await search_engine._rebuild_search_index()

        metrics = await search_engine.aggregate_metrics(
            metric_type="projects",
            timeframe="30d",
        )

        assert metrics["metric_type"] == "projects"

        if metrics["data"]:
            # Should have project information
            project_data = metrics["data"][0]
            assert "key" in project_data
            assert "value" in project_data

    @pytest.mark.asyncio
    async def test_content_type_metrics(self, search_engine, sample_data):
        """Test content type metrics."""
        await search_engine._rebuild_search_index()

        metrics = await search_engine.aggregate_metrics(
            metric_type="content_types",
            timeframe="30d",
        )

        assert metrics["metric_type"] == "content_types"

        if metrics["data"]:
            content_types = [item["key"] for item in metrics["data"]]
            assert "conversation" in content_types or "reflection" in content_types


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_query(self, search_engine, sample_data):
        """Test handling of empty queries."""
        await search_engine._rebuild_search_index()

        # Empty query should still work
        results = await search_engine.search(query="", limit=5)

        assert "results" in results
        assert isinstance(results["results"], list)

    @pytest.mark.asyncio
    async def test_invalid_filters(self, search_engine, sample_data):
        """Test handling of invalid filters."""
        await search_engine._rebuild_search_index()

        # Invalid field name
        invalid_filter = SearchFilter(
            field="nonexistent_field",
            operator="eq",
            value="test",
        )

        # Should not crash, might return empty results
        results = await search_engine.search(
            query="test",
            filters=[invalid_filter],
            limit=5,
        )

        assert "results" in results
        assert isinstance(results["results"], list)

    @pytest.mark.asyncio
    async def test_unknown_metric_type(self, search_engine, sample_data):
        """Test handling of unknown metric types."""
        await search_engine._rebuild_search_index()

        metrics = await search_engine.aggregate_metrics(
            metric_type="unknown_metric",
            timeframe="30d",
        )

        # Should return error message
        assert "error" in metrics
        assert "unknown_metric" in metrics["error"]

    @pytest.mark.asyncio
    async def test_malformed_timeframe(self, search_engine):
        """Test handling of malformed timeframe strings."""
        # Should handle invalid timeframe gracefully
        start_time, end_time = search_engine._parse_timeframe("invalid_timeframe")

        # Should fallback to 30 days
        assert start_time is not None
        assert end_time is not None
        assert end_time > start_time


class TestPerformance:
    """Test search performance characteristics."""

    @pytest.mark.asyncio
    async def test_large_result_set_handling(self, search_engine, temp_db):
        """Test handling of large result sets."""
        # Add many conversations
        for i in range(100):
            await temp_db.store_conversation(
                f"Test conversation {i} about authentication and security",
                {"project": f"project-{i % 5}"},
            )

        await search_engine._rebuild_search_index()

        # Search with large limit
        results = await search_engine.search(query="authentication", limit=50)

        # Should handle large result sets
        assert len(results["results"]) <= 50
        assert "total" in results

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, search_engine, sample_data):
        """Test concurrent search operations."""
        await search_engine._rebuild_search_index()

        # Run multiple searches concurrently
        tasks = [
            search_engine.search("authentication", limit=3),
            search_engine.search("database", limit=3),
            search_engine.search("frontend", limit=3),
            search_engine.search("error", limit=3),
        ]

        results = await asyncio.gather(*tasks)

        # All searches should complete successfully
        assert len(results) == 4
        for result in results:
            assert "results" in result
            assert isinstance(result["results"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
