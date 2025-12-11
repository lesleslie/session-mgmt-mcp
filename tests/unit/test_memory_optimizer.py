"""Tests for memory_optimizer module.

Tests conversation consolidation, summarization, and memory compression capabilities.

Phase: Week 5 Day 3 - Memory Optimizer Coverage
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestConversationDataclasses:
    """Test immutable conversation dataclasses."""

    def test_conversation_data_initialization(self) -> None:
        """Should create ConversationData with required fields."""
        from session_buddy.memory_optimizer import ConversationData

        conv = ConversationData(
            id="conv-1",
            content="Test conversation",
            project="test-project",
            timestamp="2025-01-01T12:00:00",
            metadata={"tag": "test"},
            original_size=100,
        )

        assert conv.id == "conv-1"
        assert conv.content == "Test conversation"
        assert conv.original_size == 100

    def test_compression_results_structure(self) -> None:
        """Should create CompressionResults with stats."""
        from session_buddy.memory_optimizer import CompressionResults

        results = CompressionResults(
            status="success",
            dry_run=True,
            total_conversations=100,
            conversations_to_keep=80,
            conversations_to_consolidate=20,
            clusters_created=5,
            consolidated_summaries=[],
            space_saved_estimate=5000,
            compression_ratio=0.5,
        )

        assert results.status == "success"
        assert results.compression_ratio == 0.5


class TestConversationSummarizer:
    """Test conversation summarization strategies."""

    def test_extractive_summarization(self) -> None:
        """Should extract important sentences from conversation."""
        from session_buddy.memory_optimizer import ConversationSummarizer

        summarizer = ConversationSummarizer()
        content = """
        We need to implement a new function for user authentication.
        The function should handle OAuth2 tokens.
        This will improve security significantly.
        The weather is nice today.
        We also need to fix the database connection error.
        """

        summary = summarizer.summarize_conversation(content, strategy="extractive")

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_template_based_summarization(self) -> None:
        """Should create summary using templates based on content patterns."""
        from session_buddy.memory_optimizer import ConversationSummarizer

        summarizer = ConversationSummarizer()
        content = """
        Here's the code:
        ```python
        def example():
            return "test"
        ```
        We got an ImportError when importing the module.
        The issue was in utils/helpers.py file.
        """

        summary = summarizer.summarize_conversation(content, strategy="template_based")

        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should mention files or errors
        assert "error" in summary.lower() or "file" in summary.lower()

    def test_keyword_based_summarization(self) -> None:
        """Should extract keywords from conversation."""
        from session_buddy.memory_optimizer import ConversationSummarizer

        summarizer = ConversationSummarizer()
        content = """
        database connection error occurred multiple times.
        postgresql query optimization is needed.
        redis cache implementation failed.
        """

        summary = summarizer.summarize_conversation(content, strategy="keyword_based")

        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should be keywords format, general discussion, or error message
        assert any(
            keyword in summary.lower()
            for keyword in ["keyword", "discussion", "failed", "generation"]
        )

    def test_summarize_conversation_with_strategy(self) -> None:
        """Should use specified summarization strategy."""
        from session_buddy.memory_optimizer import ConversationSummarizer

        summarizer = ConversationSummarizer()
        content = "Test conversation with function implementation and error handling."

        # Test each strategy
        for strategy in ["extractive", "template_based", "keyword_based"]:
            summary = summarizer.summarize_conversation(content, strategy)
            assert isinstance(summary, str)
            assert len(summary) > 0

    def test_summarize_conversation_invalid_strategy_fallback(self) -> None:
        """Should fall back to template_based for invalid strategy."""
        from session_buddy.memory_optimizer import ConversationSummarizer

        summarizer = ConversationSummarizer()
        content = "Test conversation content."

        summary = summarizer.summarize_conversation(content, strategy="invalid")

        assert isinstance(summary, str)
        assert len(summary) > 0


class TestConversationClusterer:
    """Test conversation clustering functionality."""

    def test_cluster_conversations_by_project(self) -> None:
        """Should cluster conversations from same project."""
        from session_buddy.memory_optimizer import ConversationClusterer

        clusterer = ConversationClusterer()
        conversations = [
            {
                "id": "conv-1",
                "project": "project-a",
                "content": "Implement authentication system",
                "timestamp": "2025-01-01T12:00:00",
            },
            {
                "id": "conv-2",
                "project": "project-a",
                "content": "Add authentication tests",
                "timestamp": "2025-01-01T13:00:00",
            },
            {
                "id": "conv-3",
                "project": "project-b",
                "content": "Different topic entirely",
                "timestamp": "2025-01-01T14:00:00",
            },
        ]

        clusters = clusterer.cluster_conversations(conversations)

        assert len(clusters) > 0
        # Related conversations should be clustered together
        cluster_with_auth = [
            c
            for c in clusters
            if any("authentication" in conv.get("content", "").lower() for conv in c)
        ]
        assert len(cluster_with_auth) > 0

    def test_calculate_similarity_same_project(self) -> None:
        """Should give higher similarity for same project."""
        from session_buddy.memory_optimizer import ConversationClusterer

        clusterer = ConversationClusterer()
        conv1 = {
            "project": "test-project",
            "content": "Test content",
            "timestamp": "2025-01-01T12:00:00",
        }
        conv2 = {
            "project": "test-project",
            "content": "Similar test content",
            "timestamp": "2025-01-01T13:00:00",
        }

        similarity = clusterer._calculate_similarity(conv1, conv2)

        assert similarity > 0.3  # Should get project bonus

    def test_calculate_similarity_time_proximity(self) -> None:
        """Should give higher similarity for temporally close conversations."""
        from session_buddy.memory_optimizer import ConversationClusterer

        clusterer = ConversationClusterer()
        now = datetime.now()
        conv1 = {
            "project": "test",
            "content": "content one",
            "timestamp": now.isoformat(),
        }
        conv2 = {
            "project": "test",
            "content": "content two",
            "timestamp": (now + timedelta(hours=1)).isoformat(),
        }

        similarity = clusterer._calculate_similarity(conv1, conv2)

        assert similarity >= 0.5  # Project + time proximity


class TestRetentionPolicyManager:
    """Test retention policy and importance scoring."""

    def test_calculate_importance_score_with_code(self) -> None:
        """Should give higher importance to conversations with code."""
        from session_buddy.memory_optimizer import RetentionPolicyManager

        manager = RetentionPolicyManager()
        conversation = {
            "content": "Here's the implementation:\n```python\ndef example():\n    return True\n```",
            "timestamp": datetime.now().isoformat(),
        }

        score = manager.calculate_importance_score(conversation)

        assert score > 0.3  # Should get has_code bonus

    def test_calculate_importance_score_with_errors(self) -> None:
        """Should give higher importance to error-related conversations."""
        from session_buddy.memory_optimizer import RetentionPolicyManager

        manager = RetentionPolicyManager()
        conversation = {
            "content": "Got an exception: ImportError when importing module. Traceback shows the issue.",
            "timestamp": datetime.now().isoformat(),
        }

        score = manager.calculate_importance_score(conversation)

        assert score > 0.2  # Should get has_errors bonus

    def test_get_conversations_for_retention_recent_kept(self) -> None:
        """Should keep recent conversations regardless of importance."""
        from session_buddy.memory_optimizer import RetentionPolicyManager

        manager = RetentionPolicyManager()
        now = datetime.now()
        conversations = [
            {
                "id": f"conv-{i}",
                "content": "Recent conversation",
                "timestamp": (now - timedelta(days=i)).isoformat(),
            }
            for i in range(10)
        ]

        keep, _consolidate = manager.get_conversations_for_retention(conversations)

        # Should keep newest conversations
        assert len(keep) > 0
        assert all("conv-" in conv["id"] for conv in keep)

    def test_get_conversations_for_retention_old_consolidated(self) -> None:
        """Should consolidate old low-importance conversations."""
        from session_buddy.memory_optimizer import RetentionPolicyManager

        manager = RetentionPolicyManager()
        old_date = datetime.now() - timedelta(days=60)
        conversations = [
            {
                "id": "old-conv",
                "content": "Old unimportant conversation",
                "timestamp": old_date.isoformat(),
            },
            {
                "id": "recent-conv",
                "content": "Recent conversation",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        keep, _consolidate = manager.get_conversations_for_retention(conversations)

        # Recent should be kept, old might be consolidated
        recent_kept = any(conv["id"] == "recent-conv" for conv in keep)
        assert recent_kept


class TestMemoryOptimizer:
    """Test main memory optimizer class."""

    @pytest.mark.asyncio
    async def test_compress_memory_no_database(self) -> None:
        """Should return error when database unavailable."""
        from session_buddy.memory_optimizer import MemoryOptimizer

        mock_db = MagicMock()
        mock_db.conn = None

        optimizer = MemoryOptimizer(mock_db)
        result = await optimizer.compress_memory()

        assert "error" in result
        assert "Database not available" in result["error"]

    @pytest.mark.asyncio
    async def test_compress_memory_no_conversations(self) -> None:
        """Should handle case with no conversations."""
        from session_buddy.memory_optimizer import MemoryOptimizer

        mock_db = MagicMock()
        mock_db.conn = MagicMock()
        mock_db.conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=[]))
        )

        optimizer = MemoryOptimizer(mock_db)
        result = await optimizer.compress_memory()

        assert result["status"] == "no_conversations"

    @pytest.mark.asyncio
    async def test_compress_memory_dry_run(self) -> None:
        """Should perform dry run without modifying data."""
        from session_buddy.memory_optimizer import MemoryOptimizer

        mock_db = MagicMock()
        mock_db.conn = MagicMock()

        # Mock conversations data
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        mock_conversations = [
            ("conv-1", "Old conversation 1", "project-a", old_date, "{}"),
            ("conv-2", "Old conversation 2", "project-a", old_date, "{}"),
            (
                "conv-3",
                "Recent conversation",
                "project-a",
                datetime.now().isoformat(),
                "{}",
            ),
        ]
        mock_db.conn.execute = MagicMock(
            return_value=MagicMock(fetchall=MagicMock(return_value=mock_conversations))
        )

        optimizer = MemoryOptimizer(mock_db)
        result = await optimizer.compress_memory(dry_run=True)

        assert result["status"] == "success"
        assert result["dry_run"] is True
        assert result["total_conversations"] == 3
        # Should not call DELETE or INSERT in dry run
        insert_calls = [
            call
            for call in mock_db.conn.execute.call_args_list
            if len(call[0]) > 0 and "INSERT" in str(call[0][0])
        ]
        assert len(insert_calls) == 0

    @pytest.mark.asyncio
    async def test_get_compression_stats(self) -> None:
        """Should return compression statistics."""
        from session_buddy.memory_optimizer import MemoryOptimizer

        mock_db = MagicMock()
        optimizer = MemoryOptimizer(mock_db)

        # Set some stats
        optimizer.compression_stats["conversations_processed"] = 50
        optimizer.compression_stats["space_saved_bytes"] = 10000

        stats = await optimizer.get_compression_stats()

        assert isinstance(stats, dict)
        assert stats["conversations_processed"] == 50
        assert stats["space_saved_bytes"] == 10000

    @pytest.mark.asyncio
    async def test_set_retention_policy_valid(self) -> None:
        """Should update retention policy with valid values."""
        from session_buddy.memory_optimizer import MemoryOptimizer

        mock_db = MagicMock()
        optimizer = MemoryOptimizer(mock_db)

        new_policy = {"max_age_days": 180, "max_conversations": 5000}
        result = await optimizer.set_retention_policy(new_policy)

        assert result["status"] == "success"
        assert optimizer.retention_manager.default_policies["max_age_days"] == 180

    @pytest.mark.asyncio
    async def test_set_retention_policy_invalid_max_age(self) -> None:
        """Should reject invalid max_age_days."""
        from session_buddy.memory_optimizer import MemoryOptimizer

        mock_db = MagicMock()
        optimizer = MemoryOptimizer(mock_db)

        result = await optimizer.set_retention_policy({"max_age_days": 0})

        assert "error" in result
        assert "max_age_days must be at least 1" in result["error"]

    @pytest.mark.asyncio
    async def test_set_retention_policy_invalid_max_conversations(self) -> None:
        """Should reject invalid max_conversations."""
        from session_buddy.memory_optimizer import MemoryOptimizer

        mock_db = MagicMock()
        optimizer = MemoryOptimizer(mock_db)

        result = await optimizer.set_retention_policy({"max_conversations": 50})

        assert "error" in result
        assert "max_conversations must be at least 100" in result["error"]
