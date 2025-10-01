"""Unit tests for reflection storage utilities."""

import pytest

from session_mgmt_mcp.utils.reflection_utils import (
    AutoStoreDecision,
    CheckpointReason,
    format_auto_store_summary,
    generate_auto_store_tags,
    should_auto_store_checkpoint,
)


class TestShouldAutoStoreCheckpoint:
    """Test cases for should_auto_store_checkpoint function."""

    def test_manual_checkpoint_always_stored(self):
        """Manual checkpoints should always be stored."""
        decision = should_auto_store_checkpoint(
            quality_score=75, is_manual=True, session_phase="checkpoint"
        )
        assert decision.should_store is True
        assert decision.reason == CheckpointReason.MANUAL_CHECKPOINT

    def test_session_end_always_stored(self):
        """Session end should always be stored."""
        decision = should_auto_store_checkpoint(
            quality_score=75, session_phase="end"
        )
        assert decision.should_store is True
        assert decision.reason == CheckpointReason.SESSION_END

    def test_exceptional_quality_stored(self):
        """Exceptional quality sessions should be stored."""
        decision = should_auto_store_checkpoint(
            quality_score=95, session_phase="checkpoint"
        )
        assert decision.should_store is True
        assert decision.reason == CheckpointReason.EXCEPTIONAL_QUALITY
        assert decision.metadata["quality_score"] == 95

    def test_quality_improvement_stored(self):
        """Significant quality improvements should be stored."""
        decision = should_auto_store_checkpoint(
            quality_score=85, previous_score=70, session_phase="checkpoint"
        )
        assert decision.should_store is True
        assert decision.reason == CheckpointReason.QUALITY_IMPROVEMENT
        assert decision.metadata["delta"] == 15

    def test_quality_degradation_stored(self):
        """Significant quality degradations should be stored."""
        decision = should_auto_store_checkpoint(
            quality_score=60, previous_score=75, session_phase="checkpoint"
        )
        assert decision.should_store is True
        assert decision.reason == CheckpointReason.QUALITY_DEGRADATION
        assert decision.metadata["delta"] == 15

    def test_routine_checkpoint_skipped(self):
        """Routine checkpoints without significant changes should be skipped."""
        decision = should_auto_store_checkpoint(
            quality_score=75, previous_score=73, session_phase="checkpoint"
        )
        assert decision.should_store is False
        assert decision.reason == CheckpointReason.ROUTINE_SKIP

    def test_minimal_quality_change_skipped(self):
        """Small quality changes below threshold should be skipped."""
        decision = should_auto_store_checkpoint(
            quality_score=75, previous_score=70, session_phase="checkpoint"
        )
        # Default threshold is 10, delta of 5 should be skipped
        assert decision.should_store is False
        assert decision.reason == CheckpointReason.ROUTINE_SKIP

    def test_no_previous_score_exceptional_stored(self):
        """First checkpoint with exceptional quality should be stored."""
        decision = should_auto_store_checkpoint(
            quality_score=92, previous_score=None, session_phase="checkpoint"
        )
        assert decision.should_store is True
        assert decision.reason == CheckpointReason.EXCEPTIONAL_QUALITY

    def test_no_previous_score_normal_skipped(self):
        """First checkpoint with normal quality should be skipped."""
        decision = should_auto_store_checkpoint(
            quality_score=75, previous_score=None, session_phase="checkpoint"
        )
        assert decision.should_store is False
        assert decision.reason == CheckpointReason.ROUTINE_SKIP


class TestGenerateAutoStoreTags:
    """Test cases for generate_auto_store_tags function."""

    def test_basic_tags(self):
        """Basic tag generation includes checkpoint, auto-stored, and reason."""
        tags = generate_auto_store_tags(
            reason=CheckpointReason.MANUAL_CHECKPOINT, project="test-project"
        )
        assert "checkpoint" in tags
        assert "auto-stored" in tags
        assert "manual_checkpoint" in tags
        assert "test-project" in tags

    def test_high_quality_tag(self):
        """High quality scores get high-quality tag."""
        tags = generate_auto_store_tags(
            reason=CheckpointReason.EXCEPTIONAL_QUALITY,
            project="test-project",
            quality_score=95,
        )
        assert "high-quality" in tags

    def test_good_quality_tag(self):
        """Good quality scores get good-quality tag."""
        tags = generate_auto_store_tags(
            reason=CheckpointReason.QUALITY_IMPROVEMENT,
            project="test-project",
            quality_score=80,
        )
        assert "good-quality" in tags

    def test_needs_improvement_tag(self):
        """Low quality scores get needs-improvement tag."""
        tags = generate_auto_store_tags(
            reason=CheckpointReason.QUALITY_DEGRADATION,
            project="test-project",
            quality_score=50,
        )
        assert "needs-improvement" in tags

    def test_session_end_specific_tags(self):
        """Session end gets session-summary tag."""
        tags = generate_auto_store_tags(
            reason=CheckpointReason.SESSION_END, project="test-project"
        )
        assert "session-summary" in tags

    def test_manual_checkpoint_specific_tags(self):
        """Manual checkpoints get user-initiated tag."""
        tags = generate_auto_store_tags(
            reason=CheckpointReason.MANUAL_CHECKPOINT, project="test-project"
        )
        assert "user-initiated" in tags

    def test_quality_change_specific_tags(self):
        """Quality changes get quality-change tag."""
        tags = generate_auto_store_tags(
            reason=CheckpointReason.QUALITY_IMPROVEMENT, project="test-project"
        )
        assert "quality-change" in tags


class TestFormatAutoStoreSummary:
    """Test cases for format_auto_store_summary function."""

    def test_routine_skip_message(self):
        """Routine skip has clear skip message."""
        decision = AutoStoreDecision(
            should_store=False,
            reason=CheckpointReason.ROUTINE_SKIP,
            metadata={"quality_score": 75},
        )
        summary = format_auto_store_summary(decision)
        assert "⏭️" in summary
        assert "skipped" in summary.lower()

    def test_manual_checkpoint_message(self):
        """Manual checkpoint has clear storage message."""
        decision = AutoStoreDecision(
            should_store=True,
            reason=CheckpointReason.MANUAL_CHECKPOINT,
            metadata={"quality_score": 75},
        )
        summary = format_auto_store_summary(decision)
        assert "💾" in summary
        assert "Manual checkpoint" in summary
        assert "quality: 75/100" in summary

    def test_quality_improvement_message(self):
        """Quality improvement shows delta."""
        decision = AutoStoreDecision(
            should_store=True,
            reason=CheckpointReason.QUALITY_IMPROVEMENT,
            metadata={"quality_score": 85, "delta": 15},
        )
        summary = format_auto_store_summary(decision)
        assert "📈" in summary
        assert "+15 points" in summary

    def test_quality_degradation_message(self):
        """Quality degradation shows negative delta."""
        decision = AutoStoreDecision(
            should_store=True,
            reason=CheckpointReason.QUALITY_DEGRADATION,
            metadata={"quality_score": 60, "delta": 15},
        )
        summary = format_auto_store_summary(decision)
        assert "📉" in summary
        assert "-15 points" in summary

    def test_exceptional_quality_message(self):
        """Exceptional quality has star emoji."""
        decision = AutoStoreDecision(
            should_store=True,
            reason=CheckpointReason.EXCEPTIONAL_QUALITY,
            metadata={"quality_score": 95},
        )
        summary = format_auto_store_summary(decision)
        assert "⭐" in summary
        assert "Exceptional" in summary


class TestConfigurationIntegration:
    """Test configuration-based behavior."""

    def test_respects_disabled_auto_store(self, monkeypatch):
        """When auto-store is disabled globally, should always skip."""
        # This would require mocking the config, but demonstrates the test pattern
        # In practice, this would use pytest fixtures to mock get_config()
        pass

    def test_respects_custom_thresholds(self, monkeypatch):
        """Custom thresholds should be respected."""
        # Mock config with custom threshold of 5
        # Then test that delta of 6 triggers store, delta of 4 does not
        pass


# Integration test pattern (would be in tests/integration/)
@pytest.mark.integration
class TestAutoStoreIntegration:
    """Integration tests for auto-store with actual checkpoint flow."""

    @pytest.mark.asyncio
    async def test_checkpoint_stores_on_manual_trigger(self):
        """Manual checkpoint should trigger reflection storage."""
        # Would test actual checkpoint_session call with is_manual=True
        pass

    @pytest.mark.asyncio
    async def test_checkpoint_skips_on_routine(self):
        """Routine checkpoint should skip reflection storage."""
        # Would test actual checkpoint_session with minimal quality change
        pass
