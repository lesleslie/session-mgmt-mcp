"""Server Utility and Formatting Functions.

This module provides helper functions extracted from server.py including:
- Display formatting functions (26 functions)
- Session initialization helpers (6 functions)
- Additional helper functions (8 functions)

Phase 2.2 Migration: Extracted utility functions for modularity.
Total: 40 functions, ~900 lines
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ============================================================================
# Display Formatting Functions (26 functions)
# ============================================================================


def _format_metrics_summary(session_stats: dict[str, Any]) -> str:
    """Format session metrics summary."""
    detail_summary = (
        f"Session metrics - Duration: {session_stats.get('duration_minutes', 0)}min, "
    )
    detail_summary += f"Success rate: {session_stats.get('success_rate', 0):.1f}%, "
    detail_summary += f"Checkpoints: {session_stats.get('total_checkpoints', 0)}"
    return detail_summary


def _format_project_maturity_section(context_score: int, max_score: int) -> list[str]:
    """Format the project maturity section."""
    return [f"\n\x75 Project maturity: {context_score}/{max_score}"]


def _format_git_worktree_header() -> str:
    """Format the git worktree information header."""
    return "\n\x72 Git Worktree Information:"


def _format_current_worktree_info(worktree_info: Any) -> list[str]:
    """Format current worktree information."""
    output = []
    if worktree_info.is_main_worktree:
        output.append(
            f"   \x73 Current: Main repository on '{worktree_info.branch}'",
        )
    else:
        output.append(f"   \x73 Current: Worktree on '{worktree_info.branch}'")
        output.append(f"   \x72 Path: {worktree_info.path}")
    return output


def _format_worktree_count_info(all_worktrees: list[Any]) -> list[str]:
    """Format worktree count information."""
    output = []
    if len(all_worktrees) > 1:
        output.append(f"   \x74 Total worktrees: {len(all_worktrees)}")
    return output


def _format_other_branches_info(
    all_worktrees: list[Any], worktree_info: Any
) -> list[str]:
    """Format information about other branches."""
    output = []
    other_branches = [
        wt.branch for wt in all_worktrees if wt.path != worktree_info.path
    ]
    if other_branches:
        output.append(
            f"   \x75 Other branches: {', '.join(other_branches[:3])}",
        )
        if len(other_branches) > 3:
            output.append(f"   ... and {len(other_branches) - 3} more")
    return output


def _format_worktree_suggestions(all_worktrees: list[Any]) -> list[str]:
    """Format worktree-related suggestions."""
    output = []
    if len(all_worktrees) > 1:
        output.append("   \x74 Use 'git_worktree_list' to see all worktrees")
    else:
        output.append(
            "   \x74 Use 'git_worktree_add <branch> <path>' to create parallel worktrees",
        )
    return output


def _format_detached_head_warning(worktree_info: Any) -> list[str]:
    """Format detached HEAD warning if applicable."""
    output = []
    if worktree_info.is_detached:
        output.append("   \x77 Detached HEAD - consider checking out a branch")
    return output


def _format_no_reminders_message(user_id: str, project_id: str | None) -> list[str]:
    """Format message when no reminders are found."""
    output = []
    output.append("📋 No pending reminders found")
    output.append(f"👤 User: {user_id}")
    if project_id:
        output.append(f"📁 Project: {project_id}")
    output.append(
        "💡 Use 'create_natural_reminder' to set up time-based reminders",
    )
    return output


def _format_reminders_header(
    reminders: list[dict[str, Any]], user_id: str, project_id: str | None
) -> list[str]:
    """Format header for reminders list."""
    output = []
    output.append(f"⏰ Found {len(reminders)} pending reminders")
    output.append(f"👤 User: {user_id}")
    if project_id:
        output.append(f"📁 Project: {project_id}")
    output.append("=" * 50)
    return output


def _format_single_reminder(reminder: dict[str, Any], index: int) -> list[str]:
    """Format a single reminder for display."""
    output = []
    output.append(f"\n#{index}")
    output.append(f"🆔 ID: {reminder['id']}")
    output.append(f"📝 Title: {reminder['title']}")


def _format_reminders_list(
    reminders: list[dict[str, Any]], user_id: str, project_id: str | None
) -> list[str]:
    """Format the complete reminders list."""
    _format_reminders_header(reminders, user_id, project_id)


def _format_reminder_basic_info(reminder: dict[str, Any], index: int) -> list[str]:
    """Format basic reminder information."""
    [
        f"\n🔥 #{index} OVERDUE",
        f"🆔 ID: {reminder['id']}",
        f"📝 Title: {reminder['title']}",
    ]


def _format_project_insights(insights: dict[str, Any], time_range_days: int) -> str:
    """Format project insights for display."""


def _format_project_activity_section(project_activity: dict[str, Any]) -> list[str]:
    """Format project activity section."""
    output = ["**📈 Project Activity:**"]
    for project, stats in project_activity.items():
        output.append(
            f"• **{project}:** {stats['conversation_count']} conversations, last active: {stats.get('last_activity', 'Unknown')}"
        )
    output.append("")
    return output


def _format_common_patterns_section(common_patterns: list[dict[str, Any]]) -> list[str]:
    """Format common patterns section."""
    output = ["**🔍 Common Patterns:**"]
    for pattern in common_patterns[:5]:  # Top 5
        projects_str = ", ".join(pattern["projects"])
        output.append(
            f"• **{pattern['pattern']}** across {projects_str} (frequency: {pattern['frequency']})"
        )
    output.append("")
    return output


def _format_advanced_search_results(results: list[Any]) -> str:
    """Format advanced search results for display."""
    [f"🔍 **Advanced Search Results** ({len(results)} found)\n"]


def _format_worktree_status(wt: dict[str, Any]) -> str:
    """Format worktree status items."""
    status_items = []
    if wt["locked"]:
        status_items.append("🔒 locked")
    if wt["prunable"]:
        status_items.append("🗑️ prunable")
    if not wt["exists"]:
        status_items.append("❌ missing")
    if wt["has_session"]:
        status_items.append("🧠 has session")


def _format_worktree_list_header(
    total_count: int, repo_name: str, current_worktree: str
) -> list[str]:
    """Format the header for the worktree list output."""
    return [
        f"🌿 **Git Worktrees** ({total_count} total)\\n",
        f"📂 Repository: {repo_name}",
        f"🎯 Current: {current_worktree}\\n",
    ]


def _format_single_worktree(wt: dict[str, Any]) -> list[str]:
    """Format a single worktree entry."""


def _format_session_summary(result: dict[str, Any]) -> list[str]:
    """Format session summary across all worktrees."""
    session_summary = result["session_summary"]
    return [
        "📊 **Multi-Worktree Summary:**",
        f"• Total worktrees: {result['total_worktrees']}",
        f"• Active sessions: {session_summary['active_sessions']}",
        f"• Unique branches: {session_summary['unique_branches']}",
        f"• Branches: {', '.join(session_summary['branches'])}\n",
    ]


def _format_worktree_status_display(
    status_info: dict[str, Any], working_dir: Path
) -> str:
    """Format worktree status information for display."""
    _format_basic_worktree_info(status_info, working_dir)
    _format_session_info(status_info.get("session_info"))


def _format_basic_worktree_info(
    status_info: dict[str, Any], working_dir: Path
) -> list[str]:
    """Format basic worktree information."""
    return [
        f"📂 Repository: {working_dir.name}",
        f"🎯 Current worktree: {status_info['branch']}",
        f"📁 Path: {status_info['path']}",
        f"🧠 Has session: {'Yes' if status_info['has_session'] else 'No'}",
        f"🔸 Detached HEAD: {'Yes' if status_info['is_detached'] else 'No'}\n",
    ]


def _format_session_info(session_info: dict[str, Any] | None) -> list[str]:
    """Format session information if available."""
    if not session_info:
        return []
    return None


def _format_interruption_statistics(interruptions: list[dict[str, Any]]) -> list[str]:
    """Format interruption statistics for display."""
    if not interruptions:
        return ["📊 **Interruption Patterns**: No recent interruptions"]
    return None


def _format_snapshot_statistics(snapshots: list[dict[str, Any]]) -> list[str]:
    """Format snapshot statistics for display."""
    if not snapshots:
        return ["💾 **Context Snapshots**: No snapshots available"]
    return None


# ============================================================================
# Session Setup & Helper Functions (14 functions)
# ============================================================================


def _setup_claude_directory(output: list[str]) -> dict[str, Any]:
    """Setup Claude directory and return validation results."""
    output.append("\n📋 Phase 1: Claude directory setup...")


def _setup_uv_dependencies(output: list[str], current_dir: Path) -> None:
    """Setup UV dependencies and package management."""
    output.append("\n🔧 Phase 2: UV dependency management & session setup...")


def _handle_uv_operations(
    output: list[str],
    current_dir: Path,
    uv_trusted: bool,
) -> None:
    """Handle UV operations for dependency management."""
    (current_dir / "pyproject.toml").exists()


def _run_uv_sync_and_compile(output: list[str], current_dir: Path) -> None:
    """Run UV sync and compile operations."""
    # Sync dependencies
    sync_result = subprocess.run(
        ["uv", "sync"], check=False, capture_output=True, text=True
    )
    if sync_result.returncode == 0:
        output.append("✅ UV sync completed successfully")


def _setup_session_management(output: list[str]) -> None:
    """Setup session management functionality."""
    output.append("\n🔧 Phase 3: Session management setup...")
    output.append("✅ Session management functionality ready")
    output.append("   📊 Conversation memory system enabled")
    output.append("   🔍 Semantic search capabilities available")


def _add_final_summary(
    output: list[str],
    current_project: str,
    context_score: int,
    project_context: dict[str, Any],
    claude_validation: dict[str, Any],
) -> None:
    """Add final summary information to output."""
    output.append("\n" + "=" * 60)
    output.append(f"🎯 {current_project.upper()} SESSION INITIALIZATION COMPLETE")
    output.append("=" * 60)


def _add_permissions_and_tools_summary(output: list[str], current_project: str) -> None:
    """Add permissions summary and available tools."""
    # Permissions Summary
    permissions_status = permissions_manager.get_permission_status()
    output.append("\n🔐 Session Permissions Summary:")
    output.append(
        f"   📊 Trusted operations: {permissions_status['trusted_operations_count']}",
    )


def _add_session_health_insights(insights: list[str], quality_score: float) -> None:
    """Add session health indicators to insights."""
    if quality_score >= 80:
        insights.append("Excellent session progress with optimal workflow patterns")
    elif quality_score >= 60:
        insights.append("Good session progress with minor optimization opportunities")
    else:
        insights.append(
            "Session requires attention - potential workflow improvements needed",
        )


def _add_current_session_context(summary: dict[str, Any]) -> None:
    """Add current session context to summary."""
    current_dir = Path(os.environ.get("PWD", Path.cwd()))
    if (current_dir / "session_mgmt_mcp").exists():
        summary["key_topics"].append("session-mgmt-mcp development")


def _add_permissions_info(output: list[str]) -> None:
    """Add permissions information to output."""
    permissions_status = permissions_manager.get_permission_status()
    output.append("\n🔐 Session Permissions:")
    output.append(
        f"   📊 Trusted operations: {permissions_status['trusted_operations_count']}",
    )
    if permissions_status["trusted_operations"]:
        for op in permissions_status["trusted_operations"]:
            output.append(f"   ✅ {op.replace('_', ' ').title()}")
    else:
        output.append("   ⚠️ No trusted operations yet - will prompt for permissions")


def _add_basic_tools_info(output: list[str]) -> None:
    """Add basic MCP tools information to output."""
    output.append("\n🛠️ Available MCP Tools:")
    output.append("• init - Full session initialization")
    output.append("• checkpoint - Quality monitoring")
    output.append("• end - Complete cleanup")
    output.append("• status - This status report with health checks")
    output.append("• permissions - Manage trusted operations")
    output.append("• git_worktree_list - List all git worktrees")
    output.append("• git_worktree_add - Create new worktrees")
    output.append("• git_worktree_remove - Remove worktrees")
    output.append("• git_worktree_status - Comprehensive worktree status")
    output.append("• git_worktree_prune - Clean up stale references")


def _add_feature_status_info(output: list[str]) -> None:
    """Add feature status information to output."""
    # Token Optimization Status
    if TOKEN_OPTIMIZER_AVAILABLE:
        output.append("\n⚡ Token Optimization:")
        output.append("• get_cached_chunk - Retrieve chunked response data")
        output.append("• get_token_usage_stats - Token usage and savings metrics")
        output.append("• optimize_memory_usage - Consolidate old conversations")
        output.append("• Built-in response chunking and truncation")


def _add_configuration_info(output: list[str]) -> None:
    """Add configuration information to output."""
    if CONFIG_AVAILABLE:
        output.append("\n⚙️ Configuration:")
        output.append("• pyproject.toml configuration support")
        output.append("• Environment variable overrides")
        output.append("• Configurable database, search, and optimization settings")


def _add_crackerjack_integration_info(output: list[str]) -> None:
    """Add Crackerjack integration information to output."""
    if CRACKERJACK_INTEGRATION_AVAILABLE:
        output.append("\n🔧 Crackerjack Integration (Enhanced):")
        output.append("\n🎯 RECOMMENDED COMMANDS (Enhanced with Memory & Analytics):")
        output.append(
            "• /session-mgmt:crackerjack-run <command> - Smart execution with insights",
        )
        output.append("• /session-mgmt:crackerjack-history - View trends and patterns")
        output.append("• /session-mgmt:crackerjack-metrics - Quality metrics over time")
        output.append("• /session-mgmt:crackerjack-patterns - Test failure analysis")
        output.append("• /session-mgmt:crackerjack-help - Complete command guide")
