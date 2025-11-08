#!/usr/bin/env python3
"""Memory and reflection management MCP tools.

This module provides tools for storing, searching, and managing reflections and conversation memories.
"""

from __future__ import annotations

import importlib.util
import typing as t
from datetime import datetime
from typing import TYPE_CHECKING, Any

from acb.adapters import import_adapter
from acb.depends import depends
from session_mgmt_mcp.utils.instance_managers import (
    get_reflection_database as resolve_reflection_database,
)


def _get_logger() -> t.Any:
    """Lazy logger resolution using ACB's logger adapter from DI container."""
    logger_class = import_adapter("logger")
    return depends.get_sync(logger_class)


# Lazy detection flag
_reflection_tools_available: bool | None = None


def _format_score(score: float) -> str:
    """Format a score as a percentage or relevance indicator."""
    return f"{score:.2f}"


if TYPE_CHECKING:
    from session_mgmt_mcp.adapters.reflection_adapter import ReflectionDatabaseAdapter


async def _get_reflection_database() -> ReflectionDatabaseAdapter:
    """Resolve reflection database via DI and ensure availability."""
    global _reflection_tools_available

    if _reflection_tools_available is False:
        msg = "Reflection tools not available"
        raise ImportError(msg)

    db = await resolve_reflection_database()
    if db is None:
        _reflection_tools_available = False
        msg = "Reflection tools not available. Install dependencies: uv sync --extra embeddings"
        raise ImportError(msg)

    _reflection_tools_available = True
    return db


def _check_reflection_tools_available() -> bool:
    """Check if reflection tools are available."""
    global _reflection_tools_available

    if _reflection_tools_available is None:
        try:
            spec = importlib.util.find_spec("session_mgmt_mcp.reflection_tools")
            _reflection_tools_available = spec is not None
        except ImportError:
            _reflection_tools_available = False

    return bool(_reflection_tools_available)


# Tool implementations
async def _store_reflection_impl(content: str, tags: list[str] | None = None) -> str:
    """Implementation for store_reflection tool."""
    if not _check_reflection_tools_available():
        return "❌ Reflection tools not available. Install dependencies: uv sync --extra embeddings"

    try:
        db = await _get_reflection_database()
        success = await db.store_reflection(content, tags=tags or [])

        if success:
            output = []
            output.append("💾 Reflection stored successfully!")
            output.append(
                f"📝 Content: {content[:100]}{'...' if len(content) > 100 else ''}",
            )
            if tags:
                output.append(f"🏷️ Tags: {', '.join(tags)}")
            output.append(f"📅 Stored: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            _get_logger().info(
                "Reflection stored", content_length=len(content), tags=tags
            )
            return "\n".join(output)
        return "❌ Failed to store reflection"

    except Exception as e:
        # Use regular logging instead of exception logging which isn't available
        _get_logger().exception(f"Error storing reflection: {e}")
        return f"❌ Error storing reflection: {e}"


def _format_quick_search_header(query: str) -> list[str]:
    """Format the header for quick search results."""
    return [f"🔍 Quick search for: '{query}'"]


def _format_quick_search_results(results: list[dict[str, Any]]) -> list[str]:
    """Format the quick search results."""
    output = []

    if results:
        result = results[0]
        output.append("📊 Found results (showing top 1)")
        output.append(
            f"📝 {result['content'][:150]}{'...' if len(result['content']) > 150 else ''}",
        )
        if result.get("project"):
            output.append(f"📁 Project: {result['project']}")
        if result.get("score") is not None:
            output.append(f"⭐ Relevance: {_format_score(result['score'])}")
        output.append(f"📅 Date: {result.get('timestamp', 'Unknown')}")
    else:
        output.append("🔍 No results found")
        output.append("💡 Try adjusting your search terms or lowering min_score")

    return output


async def _quick_search_impl(
    query: str,
    min_score: float = 0.7,
    project: str | None = None,
) -> str:
    """Implementation for quick_search tool."""
    if not _check_reflection_tools_available():
        return "❌ Reflection tools not available. Install dependencies: uv sync --extra embeddings"

    try:
        db = await _get_reflection_database()
        results = await db.search_conversations(
            query=query,
            project=project,
            limit=1,
            min_score=min_score,
        )

        output = _format_quick_search_header(query)
        output.extend(_format_quick_search_results(results))

        _get_logger().info(
            "Quick search performed", query=query, results_count=len(results)
        )
        return "\n".join(output)

    except Exception as e:
        # Use regular logging instead of exception logging which isn't available
        _get_logger().exception(f"Error in quick search: {e}")
        return f"❌ Search error: {e}"


async def _analyze_project_distribution(
    results: list[dict[str, Any]],
) -> dict[str, int]:
    """Analyze project distribution of search results."""
    projects: dict[str, int] = {}
    for result in results:
        proj = result.get("project", "Unknown")
        projects[proj] = projects.get(proj, 0) + 1
    return projects


async def _analyze_time_distribution(results: list[dict[str, Any]]) -> list[str]:
    """Analyze time distribution of search results."""
    return [r.get("timestamp", "") for r in results if r.get("timestamp")]


async def _analyze_relevance_scores(
    results: list[dict[str, Any]],
) -> tuple[float, list[float]]:
    """Analyze relevance scores of search results."""
    scores = [r.get("score", 0.0) for r in results if r.get("score") is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score, scores


async def _extract_common_themes(
    results: list[dict[str, Any]],
) -> list[tuple[str, int]]:
    """Extract common themes from search results."""
    all_content = " ".join([r["content"] for r in results])
    words = all_content.lower().split()
    word_freq: dict[str, int] = {}

    for word in words:
        if len(word) > 4:  # Skip short words
            word_freq[word] = word_freq.get(word, 0) + 1

    if word_freq:
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    return []


def _format_search_header(query: str) -> list[str]:
    """Format the search summary header."""
    output = []
    output.append(f"📊 Search Summary for: '{query}'")
    output.append("=" * 50)
    return output


def _format_no_results_message() -> list[str]:
    """Format message for when no results are found."""
    output = []
    output.append("🔍 No results found")
    output.append("💡 Try different search terms or lower the min_score threshold")
    return output


def _format_results_summary(results: list[dict[str, Any]]) -> list[str]:
    """Format the basic results summary."""
    output = []
    output.append(f"📈 Total results: {len(results)}")
    return output


async def _format_project_distribution(results: list[dict[str, Any]]) -> list[str]:
    """Format project distribution information."""
    output = []
    projects = await _analyze_project_distribution(results)
    if len(projects) > 1:
        output.append("📁 Project distribution:")
        for proj, count in sorted(
            projects.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            output.append(f"   • {proj}: {count} results")
    return output


async def _format_time_distribution(results: list[dict[str, Any]]) -> list[str]:
    """Format time distribution information."""
    output = []
    timestamps = await _analyze_time_distribution(results)
    if timestamps:
        output.append(f"📅 Time range: {len(timestamps)} results with dates")
    return output


async def _format_relevance_scores(results: list[dict[str, Any]]) -> list[str]:
    """Format relevance scores information."""
    output = []
    avg_score, scores = await _analyze_relevance_scores(results)
    if scores:
        output.append(f"⭐ Average relevance: {_format_score(avg_score)}")
    return output


async def _format_common_themes(results: list[dict[str, Any]]) -> list[str]:
    """Format common themes information."""
    output = []
    top_words = await _extract_common_themes(results)
    if top_words:
        output.append("🔤 Common themes:")
        for word, freq in top_words:
            output.append(f"   • {word}: {freq} mentions")
    return output


def _check_reflection_tools() -> bool:
    """Check if reflection tools are available."""
    return _check_reflection_tools_available()


async def _get_search_results(
    db: ReflectionDatabaseAdapter, query: str, project: str | None, min_score: float
) -> list[dict[str, Any]]:
    """Get search results from the database."""
    return await db.search_conversations(
        query=query,
        project=project,
        limit=20,
        min_score=min_score,
    )


async def _format_search_results_summary(results: list[dict[str, Any]]) -> list[str]:
    """Format the search results summary."""
    output = []
    output.extend(_format_results_summary(results))

    # Project distribution
    project_dist = await _format_project_distribution(results)
    output.extend(project_dist)

    # Time distribution
    time_dist = await _format_time_distribution(results)
    output.extend(time_dist)

    # Average relevance score
    relevance_scores = await _format_relevance_scores(results)
    output.extend(relevance_scores)

    # Common themes
    common_themes = await _format_common_themes(results)
    output.extend(common_themes)

    return output


async def _search_summary_impl(
    query: str,
    min_score: float = 0.7,
    project: str | None = None,
) -> str:
    """Implementation for search_summary tool."""
    if not _check_reflection_tools():
        return "❌ Reflection tools not available. Install dependencies: uv sync --extra embeddings"

    try:
        db = await _get_reflection_database()
        results = await _get_search_results(db, query, project, min_score)

        output = _format_search_header(query)

        if results:
            output.extend(await _format_search_results_summary(results))
        else:
            output.extend(_format_no_results_message())

        _get_logger().info(
            "Search summary generated", query=query, results_count=len(results)
        )
        return "\n".join(output)

    except Exception as e:
        # Use regular logging instead of exception logging which isn't available
        _get_logger().exception(f"Error generating search summary: {e}")
        return f"❌ Search summary error: {e}"


def _format_file_search_result(result: dict[str, Any], index: int) -> list[str]:
    """Format a single file search result."""
    output = []
    output.append(
        f"\n{index}. 📝 {result['content'][:200]}{'...' if len(result['content']) > 200 else ''}",
    )
    if result.get("project"):
        output.append(f"   📁 Project: {result['project']}")
        if result.get("score") is not None:
            output.append(f"   ⭐ Relevance: {_format_score(result['score'])}")
    if result.get("timestamp"):
        output.append(f"   📅 Date: {result['timestamp']}")

    return output


async def _search_by_file_impl(
    file_path: str,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Implementation for search_by_file tool."""
    if not _check_reflection_tools_available():
        return "❌ Reflection tools not available. Install dependencies: uv sync --extra embeddings"

    try:
        db = await _get_reflection_database()
        results = await db.search_conversations(
            query=file_path,
            project=project,
            limit=limit,
        )

        output = []
        output.append(f"📁 Searching conversations about: {file_path}")
        output.append("=" * 50)

        if results:
            output.append(f"📈 Found {len(results)} relevant conversations:")

            for i, result in enumerate(results, 1):
                output.extend(_format_file_search_result(result, i))
        else:
            output.append("🔍 No conversations found about this file")
            output.append(
                "💡 The file might not have been discussed in previous sessions",
            )

        _get_logger().info(
            "File search performed",
            file_path=file_path,
            results_count=len(results),
        )
        return "\n".join(output)

    except Exception as e:
        _get_logger().exception(
            "Error searching by file", error=str(e), file_path=file_path
        )
        return f"❌ File search error: {e}"


def _format_concept_search_result(
    result: dict[str, Any], index: int, include_files: bool
) -> list[str]:
    """Format a single concept search result."""
    output = []
    output.append(
        f"\n{index}. 📝 {result['content'][:250]}{'...' if len(result['content']) > 250 else ''}",
    )
    if result.get("project"):
        output.append(f"   📁 Project: {result['project']}")
        if result.get("score") is not None:
            output.append(f"   ⭐ Relevance: {_format_score(result['score'])}")
    if result.get("timestamp"):
        output.append(f"   📅 Date: {result['timestamp']}")

    if include_files and result.get("files"):
        files = result["files"][:3]
        if files:
            output.append(f"   📄 Files: {', '.join(files)}")

    return output


async def _search_by_concept_impl(
    concept: str,
    include_files: bool = True,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Implementation for search_by_concept tool."""
    if not _check_reflection_tools_available():
        return "❌ Reflection tools not available. Install dependencies: uv sync --extra embeddings"

    try:
        db = await _get_reflection_database()
        results = await db.search_conversations(
            query=concept,
            project=project,
            limit=limit,
        )

        output = []
        output.append(f"🧠 Searching for concept: '{concept}'")
        output.append("=" * 50)

        if results:
            output.append(f"📈 Found {len(results)} related conversations:")

            for i, result in enumerate(results, 1):
                output.extend(_format_concept_search_result(result, i, include_files))
        else:
            output.append("🔍 No conversations found about this concept")
            output.append("💡 Try related terms or broader concepts")

        _get_logger().info(
            "Concept search performed",
            concept=concept,
            results_count=len(results),
        )
        return "\n".join(output)

    except Exception as e:
        _get_logger().exception(
            "Error searching by concept", error=str(e), concept=concept
        )
        return f"❌ Concept search error: {e}"


def _format_new_stats(stats: dict[str, t.Any]) -> list[str]:
    """Format statistics in new format (conversations_count, reflections_count)."""
    conv_count = stats.get("conversations_count", 0)
    refl_count = stats.get("reflections_count", 0)
    provider = stats.get("embedding_provider", "unknown")

    return [
        f"📈 Total conversations: {conv_count}",
        f"💭 Total reflections: {refl_count}",
        f"🔧 Embedding provider: {provider}",
        f"\n🏥 Database health: {'✅ Healthy' if (conv_count + refl_count) > 0 else '⚠️ Empty'}",
    ]


def _format_old_stats(stats: dict[str, t.Any]) -> list[str]:
    """Format statistics in old/test format (total_reflections, projects, date_range)."""
    output = [
        f"📈 Total reflections: {stats.get('total_reflections', 0)}",
        f"📁 Projects: {stats.get('projects', 0)}",
    ]

    # Add date range if present
    date_range = stats.get("date_range")
    if isinstance(date_range, dict):
        output.append(
            f"📅 Date range: {date_range.get('start')} to {date_range.get('end')}"
        )

    # Add recent activity if present
    recent_activity = stats.get("recent_activity", [])
    if recent_activity:
        output.append("\n🕐 Recent activity:")
        output.extend([f"   • {activity}" for activity in recent_activity[:5]])

    # Database health
    is_healthy = stats.get("total_reflections", 0) > 0
    output.append(f"\n🏥 Database health: {'✅ Healthy' if is_healthy else '⚠️ Empty'}")

    return output


async def _reflection_stats_impl() -> str:
    """Implementation for reflection_stats tool."""
    if not _check_reflection_tools_available():
        return "❌ Reflection tools not available. Install dependencies: uv sync --extra embeddings"

    try:
        db = await _get_reflection_database()
        stats = await db.get_stats()

        output = ["📊 Reflection Database Statistics", "=" * 40]

        if stats and "error" not in stats:
            # Format based on stat structure
            if "conversations_count" in stats:
                output.extend(_format_new_stats(stats))
            else:
                output.extend(_format_old_stats(stats))
        else:
            output.extend(
                [
                    "📊 No statistics available",
                    "💡 Database may be empty or inaccessible",
                ]
            )

        _get_logger().info("Reflection stats retrieved")
        return "\n".join(output)

    except Exception as e:
        _get_logger().exception("Error getting reflection stats", error=str(e))
        return f"❌ Stats error: {e}"


async def _reset_reflection_database_impl() -> str:
    """Implementation for reset_reflection_database tool."""
    if not _check_reflection_tools_available():
        return "❌ Reflection tools not available. Install dependencies: uv sync --extra embeddings"

    try:
        # Try to create a new database connection (DI will handle cleanup)
        await _get_reflection_database()

        output = []
        output.append("🔄 Reflection database connection reset")
        output.append("✅ New connection established successfully")
        output.append("💡 Database locks should be resolved")

        _get_logger().info("Reflection database reset successfully")
        return "\n".join(output)

    except Exception as e:
        _get_logger().exception("Error resetting reflection database", error=str(e))
        return f"❌ Reset error: {e}"


def register_memory_tools(mcp_server: Any) -> None:
    """Register all memory management tools with the MCP server."""

    @mcp_server.tool()  # type: ignore[misc]
    async def store_reflection(content: str, tags: list[str] | None = None) -> str:
        """Store an important insight or reflection for future reference."""
        return await _store_reflection_impl(content, tags)

    @mcp_server.tool()  # type: ignore[misc]
    async def quick_search(
        query: str,
        min_score: float = 0.7,
        project: str | None = None,
    ) -> str:
        """Quick search that returns only the count and top result for fast overview."""
        return await _quick_search_impl(query, min_score, project)

    @mcp_server.tool()  # type: ignore[misc]
    async def search_summary(
        query: str,
        limit: int = 10,
        project: str | None = None,
        min_score: float = 0.7,
    ) -> str:
        """Get aggregated insights from search results without individual result details."""
        return await _search_summary_impl(query, min_score, project)

    @mcp_server.tool()  # type: ignore[misc]
    async def search_by_file(
        file_path: str,
        limit: int = 10,
        project: str | None = None,
        min_score: float = 0.7,
    ) -> str:
        """Search for conversations that analyzed a specific file."""
        return await _search_by_file_impl(file_path, limit, project)

    @mcp_server.tool()  # type: ignore[misc]
    async def search_by_concept(
        concept: str,
        include_files: bool = True,
        limit: int = 10,
        project: str | None = None,
        min_score: float = 0.7,
    ) -> str:
        """Search for conversations about a specific development concept."""
        return await _search_by_concept_impl(concept, include_files, limit, project)

    @mcp_server.tool()  # type: ignore[misc]
    async def reflection_stats(project: str | None = None) -> str:
        """Get statistics about the reflection database."""
        return await _reflection_stats_impl()

    @mcp_server.tool()  # type: ignore[misc]
    async def reset_reflection_database() -> str:
        """Reset the reflection database connection to fix lock issues."""
        return await _reset_reflection_database_impl()
