#!/usr/bin/env python3
"""Tests for the reflection_utils module."""

import tempfile
from pathlib import Path

import pytest
from session_buddy.utils.reflection_utils import (
    AutoStoreDecision,
    CheckpointReason,
    format_auto_store_summary,
    should_auto_store_checkpoint,
)


def test_format_auto_store_summary_should_store():
    """Test formatting of auto-store summary when storing."""
    decision = AutoStoreDecision(
        should_store=True,
        reason=CheckpointReason.QUALITY_IMPROVEMENT,
        metadata={
            "quality_score": 85,
            "delta": 15,
        },
    )

    summary = format_auto_store_summary(decision)

    assert "Quality improved significantly" in summary
    assert "quality: 85/100" in summary
    assert "+15 points" in summary


def test_format_auto_store_summary_should_not_store():
    """Test formatting of auto-store summary when not storing."""
    decision = AutoStoreDecision(
        should_store=False, reason=CheckpointReason.ROUTINE_SKIP, metadata={}
    )

    summary = format_auto_store_summary(decision)

    assert "skipped" in summary
    assert "signal-to-noise ratio" in summary


def test_should_auto_store_checkpoint_quality_improvement():
    """Test auto-store decision for quality improvement."""
    decision = should_auto_store_checkpoint(
        quality_score=85, previous_score=70, is_manual=False, session_phase="checkpoint"
    )

    assert decision.should_store is True
    assert decision.reason == CheckpointReason.QUALITY_IMPROVEMENT
    assert "delta" in decision.metadata


def test_should_auto_store_checkpoint_no_previous_score():
    """Test auto-store decision when no previous score exists."""
    decision = should_auto_store_checkpoint(
        quality_score=75,
        previous_score=None,
        is_manual=False,
        session_phase="checkpoint",
    )

    # When there's no previous score, it won't trigger quality improvement logic
    # but might still be skipped based on other criteria
    assert isinstance(decision.should_store, bool)
    assert isinstance(decision.reason, CheckpointReason)


def test_should_auto_store_checkpoint_manual():
    """Test auto-store decision for manual checkpoint."""
    decision = should_auto_store_checkpoint(
        quality_score=75, previous_score=70, is_manual=True, session_phase="checkpoint"
    )

    assert decision.should_store is True
    assert decision.reason == CheckpointReason.MANUAL_CHECKPOINT


def test_should_auto_store_checkpoint_session_end():
    """Test auto-store decision for session end."""
    decision = should_auto_store_checkpoint(
        quality_score=75, previous_score=70, is_manual=False, session_phase="end"
    )

    assert decision.should_store is True
    assert decision.reason == CheckpointReason.SESSION_END


def test_should_auto_store_checkpoint_exceptional_quality():
    """Test auto-store decision for exceptional quality."""
    decision = should_auto_store_checkpoint(
        quality_score=95,  # Exceptional quality
        previous_score=70,
        is_manual=False,
        session_phase="checkpoint",
    )

    assert decision.should_store is True
    assert decision.reason == CheckpointReason.EXCEPTIONAL_QUALITY


if __name__ == "__main__":
    pytest.main([__file__])
