"""Quality recommendations generator.

This module provides functions for generating quality improvement recommendations
based on session metrics, project context, and development patterns.
"""

from __future__ import annotations

from typing import Any


def generate_quality_recommendations(
    score: int,
    project_context: dict[str, Any],
    permissions_count: int,
    uv_available: bool,
) -> list[str]:
    """Generate quality improvement recommendations based on score factors."""
    recommendations = []

    if score < 50:
        recommendations.append(
            "Session needs attention - multiple areas for improvement",
        )
    elif score < 75:
        recommendations.append("Good session health - minor optimizations available")
    else:
        recommendations.append("Excellent session quality - maintain current practices")

    # Project-specific recommendations
    if not project_context.get("has_tests"):
        recommendations.append("Consider adding tests to improve project structure")
    if not project_context.get("has_docs"):
        recommendations.append("Documentation would enhance project maturity")

    # Permissions recommendations
    if permissions_count == 0:
        recommendations.append(
            "No trusted operations yet - permissions will be granted on first use",
        )
    elif permissions_count > 5:
        recommendations.append(
            "Many trusted operations - consider reviewing for security",
        )

    # Tools recommendations
    if not uv_available:
        recommendations.append(
            "Install UV package manager for better dependency management",
        )

    return recommendations
