"""Session lifecycle management utilities.

This package provides modular utilities for session lifecycle operations:
- Handoff documentation generation
- Project context analysis
- Session information parsing

All modules are designed to be reusable components that reduce code duplication
in the session_manager module.
"""

from session_buddy.core.lifecycle.handoff import (
    build_handoff_header,
    build_quality_section,
    build_recommendations_section,
    build_static_sections,
    generate_handoff_documentation,
    save_handoff_documentation,
)
from session_buddy.core.lifecycle.project_context import (
    add_python_context_indicators,
    analyze_project_context,
    check_ci_cd_exists,
    check_docs_exist,
    check_framework_imports,
    check_readme_exists,
    check_tests_exist,
    check_venv_exists,
    detect_python_frameworks,
    get_basic_project_indicators,
)
from session_buddy.core.lifecycle.session_info import (
    SessionInfo,
    discover_session_files,
    extract_session_metadata,
    extract_session_recommendations,
    find_latest_handoff_file,
    parse_session_file,
    read_file_safely,
    read_previous_session_info,
)

__all__ = [
    # Session info
    "SessionInfo",
    # Project context
    "add_python_context_indicators",
    "analyze_project_context",
    # Handoff
    "build_handoff_header",
    "build_quality_section",
    "build_recommendations_section",
    "build_static_sections",
    "check_ci_cd_exists",
    "check_docs_exist",
    "check_framework_imports",
    "check_readme_exists",
    "check_tests_exist",
    "check_venv_exists",
    "detect_python_frameworks",
    "discover_session_files",
    "extract_session_metadata",
    "extract_session_recommendations",
    "find_latest_handoff_file",
    "generate_handoff_documentation",
    "get_basic_project_indicators",
    "parse_session_file",
    "read_file_safely",
    "read_previous_session_info",
    "save_handoff_documentation",
]
