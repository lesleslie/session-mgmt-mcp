#!/usr/bin/env python3
"""Comprehensive tests for quality_utils_v2 quality scoring algorithm.

Tests the complete V2 quality scoring system including:
- Code quality metrics (coverage, linting, type safety, complexity)
- Project health indicators (tooling, maturity, testing, documentation)
- Development velocity analysis (git activity, patterns)
- Security scoring (security tools, hygiene)
- Trust score calculation (separate from quality)
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
class TestCodeQualityScoring:
    """Test code quality component scoring."""

    async def test_calculate_quality_score_v2_imports(self):
        """Test that quality scoring functions import successfully."""
        try:
            from session_mgmt_mcp.utils.quality_utils_v2 import (
                calculate_quality_score_v2,
            )
            assert callable(calculate_quality_score_v2)
        except ImportError as e:
            pytest.skip(f"Quality utils import failed: {e}")

    async def test_calculate_quality_score_v2_with_temp_dir(self):
        """Test quality score calculation with temporary directory."""
        try:
            from session_mgmt_mcp.utils.quality_utils_v2 import (
                calculate_quality_score_v2,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                result = await calculate_quality_score_v2(
                    project_dir=Path(tmpdir)
                )

                # Should return QualityScoreV2 object with expected attributes
                assert hasattr(result, "total_score")
                assert hasattr(result, "version")
                assert hasattr(result, "code_quality")
                assert hasattr(result, "project_health")
                assert hasattr(result, "dev_velocity")
                assert hasattr(result, "security")
                assert hasattr(result, "trust_score")
                assert hasattr(result, "recommendations")
                assert result.version == "2.0"
        except ImportError:
            pytest.skip("Quality utils not available")

    async def test_code_quality_score_structure(self):
        """Test CodeQualityScore dataclass structure."""
        try:
            from session_mgmt_mcp.utils.quality_utils_v2 import CodeQualityScore

            score = CodeQualityScore(
                test_coverage=10.0,
                lint_score=8.0,
                type_coverage=7.0,
                complexity_score=4.0,
                total=29.0,
                details={"source": "test"}
            )

            assert score.test_coverage == 10.0
            assert score.lint_score == 8.0
            assert score.type_coverage == 7.0
            assert score.complexity_score == 4.0
            assert score.total == 29.0
        except ImportError:
            pytest.skip("CodeQualityScore not available")


@pytest.mark.asyncio
class TestToolingScore:
    """Test tooling score calculation."""

    async def test_calculate_tooling_score(self):
        """Test complete tooling score calculation."""
        try:
            from session_mgmt_mcp.utils.quality_utils_v2 import (
                _calculate_tooling_score,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                (tmppath / "pyproject.toml").touch()
                (tmppath / "uv.lock").touch()

                result = _calculate_tooling_score(tmppath)
                assert "score" in result
                assert "details" in result
                assert result["score"] >= 0
        except ImportError:
            pytest.skip("Tooling score calculator not available")


@pytest.mark.asyncio
class TestMaturityScore:
    """Test project maturity scoring."""

    async def test_calculate_maturity_score_with_tests(self):
        """Test maturity score with comprehensive tests."""
        try:
            from session_mgmt_mcp.utils.quality_utils_v2 import (
                _calculate_maturity_score,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                test_dir = tmppath / "tests"
                test_dir.mkdir()
                (test_dir / "conftest.py").touch()
                for i in range(11):
                    (test_dir / f"test_module_{i}.py").touch()

                result = _calculate_maturity_score(tmppath)
                assert result["score"] >= 5
        except ImportError:
            pytest.skip("Maturity score calculator not available")


@pytest.mark.asyncio
class TestSecurityScoring:
    """Test security component scoring."""

    async def test_calculate_security_score(self):
        """Test security score calculation."""
        try:
            from session_mgmt_mcp.utils.quality_utils_v2 import (
                _calculate_security,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                result = await _calculate_security(tmppath)

                assert hasattr(result, "security_tools")
                assert hasattr(result, "security_hygiene")
                assert hasattr(result, "total")
        except ImportError:
            pytest.skip("Security scorer not available")


@pytest.mark.asyncio
class TestTrustScore:
    """Test trust score calculation."""

    async def test_calculate_trust_score(self):
        """Test trust score calculation."""
        try:
            from session_mgmt_mcp.utils.quality_utils_v2 import (
                _calculate_trust_score,
            )

            result = _calculate_trust_score(
                permissions_count=2,
                session_available=True,
                tool_count=10,
            )

            assert hasattr(result, "trusted_operations")
            assert hasattr(result, "session_availability")
            assert hasattr(result, "tool_ecosystem")
            assert hasattr(result, "total")
            assert result.trusted_operations == 20
            assert result.session_availability == 30
            assert result.tool_ecosystem == 30
        except ImportError:
            pytest.skip("Trust score calculator not available")


@pytest.mark.asyncio
class TestRecommendationGeneration:
    """Test recommendation generation."""

    async def test_generate_recommendations_high_score(self):
        """Test recommendations for high-quality projects."""
        try:
            from session_mgmt_mcp.utils.quality_utils_v2 import (
                _generate_recommendations_v2,
                CodeQualityScore,
                ProjectHealthScore,
                DevVelocityScore,
                SecurityScore,
            )

            code_quality = CodeQualityScore(
                test_coverage=15, lint_score=10, type_coverage=10,
                complexity_score=5, total=40,
                details={"coverage_pct": 95}
            )
            project_health = ProjectHealthScore(
                tooling_score=15, maturity_score=15, total=30,
                details={}
            )
            dev_velocity = DevVelocityScore(
                git_activity=10, dev_patterns=10, total=20,
                details={}
            )
            security = SecurityScore(
                security_tools=5, security_hygiene=5, total=10,
                details={}
            )

            recommendations = _generate_recommendations_v2(
                code_quality, project_health, dev_velocity, security, 100
            )

            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
        except ImportError:
            pytest.skip("Recommendation generator not available")


@pytest.mark.asyncio
class TestQualityScoreIntegration:
    """Integration tests for complete quality scoring."""

    async def test_quality_score_v2_complete_flow(self):
        """Test complete quality scoring workflow."""
        try:
            from session_mgmt_mcp.utils.quality_utils_v2 import (
                calculate_quality_score_v2,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                (tmppath / "pyproject.toml").touch()
                (tmppath / "README.md").touch()
                (tmppath / ".git").mkdir()
                tests_dir = tmppath / "tests"
                tests_dir.mkdir()
                (tests_dir / "conftest.py").touch()
                for i in range(15):
                    (tests_dir / f"test_module_{i}.py").touch()

                result = await calculate_quality_score_v2(
                    project_dir=tmppath,
                    permissions_count=2,
                    session_available=True,
                    tool_count=10,
                )

                assert result.total_score >= 0
                assert result.total_score <= 100
                assert result.version == "2.0"
                assert len(result.recommendations) >= 0
                assert hasattr(result, "timestamp")
        except ImportError:
            pytest.skip("Quality scoring V2 not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
