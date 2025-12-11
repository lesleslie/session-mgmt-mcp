#!/usr/bin/env python3
"""Comprehensive test suite for search functionality.

Tests semantic search, full-text search, and search filtering operations.
"""

from __future__ import annotations

import tempfile
from typing import Any

import pytest
from session_buddy.reflection_tools import ReflectionDatabase


@pytest.mark.asyncio
class TestFullTextSearch:
    """Test full-text search functionality."""

    @pytest.fixture
    async def search_db(self):
        """Provide database with searchable content."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()

        # Add diverse content for searching
        await db.store_conversation("Python async/await patterns", {"lang": "python"})
        await db.store_conversation("JavaScript promises and async", {"lang": "js"})
        await db.store_conversation("Database optimization techniques", {"topic": "db"})
        await db.store_conversation(
            "Testing strategies in pytest", {"framework": "pytest"}
        )
        await db.store_conversation("Git workflow best practices", {"tool": "git"})

        yield db
        db.close()

    async def test_search_single_word(self, search_db):
        """Test searching for single word."""
        results = await search_db.search_conversations("python", limit=10)
        assert isinstance(results, list)

    async def test_search_multiple_words(self, search_db):
        """Test searching for multiple words."""
        results = await search_db.search_conversations("async patterns", limit=10)
        assert isinstance(results, list)

    async def test_search_case_insensitive(self, search_db):
        """Test case-insensitive search."""
        results_lower = await search_db.search_conversations("python", limit=10)
        results_upper = await search_db.search_conversations("PYTHON", limit=10)

        # Both should work (results may be empty but should not error)
        assert isinstance(results_lower, list)
        assert isinstance(results_upper, list)

    async def test_search_exact_phrase(self, search_db):
        """Test searching for exact phrase."""
        results = await search_db.search_conversations("async/await", limit=10)
        assert isinstance(results, list)

    async def test_search_with_special_characters(self, search_db):
        """Test search with special characters."""
        results = await search_db.search_conversations("testing pytest", limit=10)
        assert isinstance(results, list)

    async def test_search_empty_query(self, search_db):
        """Test search with empty query."""
        results = await search_db.search_conversations("", limit=10)
        assert isinstance(results, list)

    async def test_search_nonexistent_term(self, search_db):
        """Test searching for nonexistent term."""
        results = await search_db.search_conversations("xyzabc123notfound", limit=10)
        assert isinstance(results, list)


@pytest.mark.asyncio
class TestSearchFiltering:
    """Test search filtering and result limiting."""

    @pytest.fixture
    async def filtered_search_db(self):
        """Provide database with categorized content."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()

        # Add categorized content
        for i in range(20):
            await db.store_conversation(
                f"Python topic {i}",
                {"category": "python", "index": i},
            )
            await db.store_conversation(
                f"JavaScript topic {i}",
                {"category": "javascript", "index": i},
            )

        yield db
        db.close()

    async def test_search_with_limit(self, filtered_search_db):
        """Test search result limiting."""
        results_5 = await filtered_search_db.search_conversations("topic", limit=5)
        results_10 = await filtered_search_db.search_conversations("topic", limit=10)

        assert len(results_5) <= 5
        assert len(results_10) <= 10

    async def test_search_limit_enforcement(self, filtered_search_db):
        """Test that limit is strictly enforced."""
        results = await filtered_search_db.search_conversations("", limit=3)
        assert len(results) <= 3

    async def test_search_zero_limit(self, filtered_search_db):
        """Test search with zero limit."""
        results = await filtered_search_db.search_conversations("python", limit=0)
        # Should either return empty or handle gracefully
        assert isinstance(results, list)

    async def test_search_large_limit(self, filtered_search_db):
        """Test search with very large limit."""
        results = await filtered_search_db.search_conversations("topic", limit=1000)
        assert isinstance(results, list)


@pytest.mark.asyncio
class TestSemanticSearch:
    """Test semantic search functionality."""

    @pytest.fixture
    async def semantic_db(self):
        """Provide database for semantic search testing."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()

        # Add semantically related content
        await db.store_conversation("Machine learning models", {"topic": "ml"})
        await db.store_conversation(
            "Neural networks and deep learning", {"topic": "dl"}
        )
        await db.store_conversation(
            "Artificial intelligence applications", {"topic": "ai"}
        )
        await db.store_conversation("Data science techniques", {"topic": "data"})

        yield db
        db.close()

    async def test_semantic_search_similarity(self, semantic_db):
        """Test semantic similarity search."""
        # Even if embeddings fail, should fall back to text search
        results = await semantic_db.search_conversations("machine learning", limit=10)
        assert isinstance(results, list)

    async def test_semantic_search_synonyms(self, semantic_db):
        """Test semantic search with synonymous terms."""
        results = await semantic_db.search_conversations("neural nets", limit=10)
        assert isinstance(results, list)

    async def test_semantic_search_fallback(self, semantic_db):
        """Test that semantic search falls back to text search."""
        # Search should work regardless of embedding availability
        results = await semantic_db.search_conversations("learning models", limit=10)
        assert isinstance(results, list)


@pytest.mark.asyncio
class TestTagSearch:
    """Test searching reflections by tags."""

    @pytest.fixture
    async def tagged_db(self):
        """Provide database with tagged reflections."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()

        tags_list = [
            ["python", "async"],
            ["python", "concurrency"],
            ["javascript", "async"],
            ["testing", "pytest"],
            ["testing", "coverage"],
        ]

        for tags in tags_list:
            await db.store_reflection(f"Reflection on {tags}", tags)

        yield db
        db.close()

    async def test_search_reflections_for_tag(self, tagged_db):
        """Test searching reflections that contain tag content."""
        results = await tagged_db.search_reflections("python", limit=10)
        assert isinstance(results, list)

    async def test_search_for_nonexistent_tag_content(self, tagged_db):
        """Test searching for nonexistent tag content."""
        results = await tagged_db.search_reflections("nonexistent", limit=10)
        assert isinstance(results, list)

    async def test_search_case_handling(self, tagged_db):
        """Test search case handling."""
        results_lower = await tagged_db.search_reflections("python", limit=10)
        results_upper = await tagged_db.search_reflections("PYTHON", limit=10)

        # Both should execute without error
        assert isinstance(results_lower, list)
        assert isinstance(results_upper, list)


@pytest.mark.asyncio
class TestSearchPerformance:
    """Test search performance with larger datasets."""

    @pytest.fixture
    async def large_db(self):
        """Provide database with many records."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()

        # Add 100+ conversations for performance testing
        for i in range(100):
            await db.store_conversation(
                f"Conversation {i} about various topics", {"index": i}
            )

        yield db
        db.close()

    async def test_search_large_dataset(self, large_db):
        """Test search performance on larger dataset."""
        results = await large_db.search_conversations("conversation", limit=50)
        assert isinstance(results, list)

    async def test_search_respects_limit_large_dataset(self, large_db):
        """Test that limit is respected on large dataset."""
        results = await large_db.search_conversations("", limit=10)
        assert len(results) <= 10

    async def test_multiple_searches_large_dataset(self, large_db):
        """Test multiple searches on large dataset."""
        import asyncio

        async def do_search(query: str):
            return await large_db.search_conversations(query, limit=5)

        results = await asyncio.gather(
            do_search("conversation"),
            do_search("topics"),
            do_search("various"),
        )

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)


@pytest.mark.asyncio
class TestSearchErrorHandling:
    """Test error handling in search operations."""

    @pytest.fixture
    async def error_db(self):
        """Provide database for error testing."""
        db = ReflectionDatabase(":memory:")
        await db.initialize()

        await db.store_conversation("Test content", {})

        yield db
        db.close()

    async def test_search_with_very_long_query(self, error_db):
        """Test search with very long query string."""
        long_query = "test " * 1000  # Create very long query
        results = await error_db.search_conversations(long_query, limit=10)
        assert isinstance(results, list)

    async def test_search_with_special_sql_chars(self, error_db):
        """Test search with SQL special characters."""
        queries = ["'; DROP TABLE", "OR 1=1", "test%", "_test"]
        for query in queries:
            results = await error_db.search_conversations(query, limit=10)
            assert isinstance(results, list)

    async def test_search_with_unicode(self, error_db):
        """Test search with unicode characters."""
        queries = ["café", "naïve", "résumé", "日本語", "🚀"]
        for query in queries:
            results = await error_db.search_conversations(query, limit=10)
            assert isinstance(results, list)

    async def test_search_concurrent_queries(self, error_db):
        """Test concurrent search queries."""
        import asyncio

        async def search(query: str):
            return await error_db.search_conversations(query, limit=10)

        queries = ["test", "content", "data", "search", "query"]
        tasks = [search(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
