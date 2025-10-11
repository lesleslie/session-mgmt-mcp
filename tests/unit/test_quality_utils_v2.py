#!/usr/bin/env python3
"""Tests for the quality_utils_v2 module."""

import tempfile
from pathlib import Path

import pytest
from session_mgmt_mcp.utils.quality_utils_v2 import calculate_quality_score_v2


@pytest.mark.asyncio
async def test_calculate_quality_score_v2_basic():
    """Test basic functionality of calculate_quality_score_v2."""
    # Create a temporary project directory
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create basic project structure
        (project_path / "pyproject.toml").write_text("[tool.poetry]\nname = \"test\"\n")
        (project_path / "README.md").touch()
        
        # Create a test with basic parameters
        result = await calculate_quality_score_v2(
            project_dir=project_path,
            permissions_count=2,
            session_available=True,
            tool_count=5
        )
        
        # Verify the result structure
        assert hasattr(result, 'total_score')
        assert 0 <= result.total_score <= 100
        assert result.version == "2.0"
        assert hasattr(result, 'code_quality')
        assert hasattr(result, 'project_health')
        assert hasattr(result, 'dev_velocity')
        assert hasattr(result, 'security')
        assert hasattr(result, 'trust_score')
        assert isinstance(result.recommendations, list)
        assert isinstance(result.timestamp, str)


if __name__ == "__main__":
    pytest.main([__file__])