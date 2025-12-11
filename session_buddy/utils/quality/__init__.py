"""Quality analysis utilities.

This package provides modular utilities for quality analysis including:
- Compaction analysis for context optimization
- Quality recommendations generation
- Conversation summary creation and processing

All modules are designed to be reusable components that reduce code duplication
in the quality_engine module.
"""

from session_buddy.utils.quality.compaction import (
    check_git_activity,
    count_significant_files,
    evaluate_git_activity_heuristic,
    evaluate_large_project_heuristic,
    evaluate_python_project_heuristic,
    get_default_compaction_reason,
    get_fallback_compaction_reason,
)
from session_buddy.utils.quality.recommendations import (
    generate_quality_recommendations,
)
from session_buddy.utils.quality.summary import (
    create_empty_summary,
    ensure_summary_defaults,
    extract_decisions_from_content,
    extract_next_steps_from_content,
    extract_topics_from_content,
    get_error_summary,
    get_fallback_summary,
    process_recent_reflections,
)

__all__ = [
    # Compaction
    "check_git_activity",
    "count_significant_files",
    # Summary
    "create_empty_summary",
    "ensure_summary_defaults",
    "evaluate_git_activity_heuristic",
    "evaluate_large_project_heuristic",
    "evaluate_python_project_heuristic",
    "extract_decisions_from_content",
    "extract_next_steps_from_content",
    "extract_topics_from_content",
    # Recommendations
    "generate_quality_recommendations",
    "get_default_compaction_reason",
    "get_error_summary",
    "get_fallback_compaction_reason",
    "get_fallback_summary",
    "process_recent_reflections",
]
