#!/usr/bin/env python3
"""Search and reflection tools for session-mgmt-mcp.

Following crackerjack architecture patterns with focused, single-responsibility tools
for conversation memory, semantic search, and knowledge retrieval.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from session_mgmt_mcp.reflection_tools import get_reflection_database

if TYPE_CHECKING:
    from session_mgmt_mcp.reflection_tools import ReflectionDatabase

logger = logging.getLogger(__name__)


async def _optimize_search_results_impl(
    results: list[dict[str, Any]],
    optimize_tokens: bool,
    max_tokens: int,
    query: str,
) -> dict[str, Any]:
    """Apply token optimization to search results if available."""
    try:
        from session_mgmt_mcp.token_optimizer import TokenOptimizer

        if optimize_tokens and results:
            optimizer = TokenOptimizer()
            optimized_results, optimization_info = optimizer.optimize_search_results(
                results, "truncate_old", max_tokens
            )
            return {
                "results": optimized_results,
                "optimized": True,
                "optimization_info": optimization_info,
            }

        return {"results": results, "optimized": False, "token_count": 0}
    except ImportError:
        logger.info("Token optimizer not available, returning results as-is")
        return {"results": results, "optimized": False, "token_count": 0}
    except Exception as e:
        logger.exception(f"Search optimization failed: {e}")
        return {"results": results, "optimized": False, "error": str(e)}


async def _store_reflection_impl(content: str, tags: list[str] | None = None) -> str:
    """Store an important insight or reflection for future reference."""
    try:
        db = await get_reflection_database()
        if not db:
            return "❌ Reflection system not available. Install optional dependencies with `uv sync --extra embeddings`"

        async with db:
            reflection_id = await db.store_reflection(content, tags or [])
            tag_text = f" (tags: {', '.join(tags)})" if tags else ""
            return (
                f"✅ Reflection stored successfully with ID: {reflection_id}{tag_text}"
            )

    except Exception as e:
        logger.exception(f"Failed to store reflection: {e}")
        return f"❌ Error storing reflection: {e!s}"


async def _quick_search_impl(
    query: str,
    project: str | None = None,
    min_score: float = 0.7,
) -> str:
    """Quick search that returns only the count and top result for fast overview."""
    try:
        db = await get_reflection_database()
        if not db:
            return "❌ Search system not available. Install optional dependencies with `uv sync --extra embeddings`"

        async with db:
            total_results = await db.search_conversations(
                query=query, project=project, min_score=min_score, limit=100
            )

            if not total_results:
                return f"🔍 No results found for '{query}'"

            top_result = total_results[0]
            result = f"🔍 **{len(total_results)} results** for '{query}'\n\n"
            result += (
                f"**Top Result** (score: {top_result.get('similarity', 'N/A')}):\n"
            )
            result += f"{top_result.get('content', '')[:200]}..."

            if len(total_results) > 1:
                result += f"\n\n💡 Use get_more_results to see additional {len(total_results) - 1} results"

            return result

    except Exception as e:
        logger.exception(f"Quick search failed: {e}")
        return f"❌ Search error: {e!s}"


def _extract_key_terms(all_content: str) -> list[str]:
    """Extract key terms from content."""
    word_freq: dict[str, int] = {}
    for word in all_content.split():
        if len(word) > 4:  # Skip short words
            word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1

    if word_freq:
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [w[0] for w in top_words]
    return []


def _format_search_summary_header(query: str) -> str:
    """Format the header for search summary results."""
    return f"🔍 **Search Summary for '{query}'**\\n\\n"


def _format_search_summary_basic_info(results_count: int) -> str:
    """Format basic information for search summary."""
    return f"**Found**: {results_count} relevant conversations\\n"


def _analyze_time_distribution(results: list[dict[str, Any]]) -> str:
    """Analyze and format time distribution of search results."""
    if results:
        dates = [r.get("timestamp", "") for r in results if r.get("timestamp")]
        if dates:
            return f"**Time Range**: {min(dates)} to {max(dates)}\\n"
    return ""


def _extract_key_themes(results: list[dict[str, Any]]) -> str:
    """Extract and format key themes from search results."""
    all_content = " ".join([r.get("content", "")[:100] for r in results])
    key_terms = _extract_key_terms(all_content)
    if key_terms:
        return f"**Key Terms**: {', '.join(key_terms)}\\n"
    return ""


def _format_search_summary_footer() -> str:
    """Format the footer for search summary results."""
    return "\\n💡 Use search with same query to see individual results"


async def _search_summary_impl(
    query: str,
    project: str | None = None,
    min_score: float = 0.7,
) -> str:
    """Get aggregated insights from search results without individual result details."""
    try:
        db = await get_reflection_database()
        if not db:
            return "❌ Search system not available. Install optional dependencies with `uv sync --extra embeddings`"

        async with db:
            results = await db.search_conversations(
                query=query, project=project, min_score=min_score, limit=20
            )

            if not results:
                return f"🔍 No results found for '{query}'"

            summary = _format_search_summary_header(query)
            summary += _format_search_summary_basic_info(len(results))

            # Analyze time distribution
            summary += _analyze_time_distribution(results)

            # Key themes (basic)
            summary += _extract_key_themes(results)

            summary += _format_search_summary_footer()
            return summary

    except Exception as e:
        logger.exception(f"Search summary failed: {e}")
        return f"❌ Search summary error: {e!s}"


async def _get_more_results_impl(
    query: str,
    offset: int = 3,
    limit: int = 3,
    project: str | None = None,
) -> str:
    """Get additional search results after an initial search (pagination support)."""
    try:
        db = await get_reflection_database()
        if not db:
            return "❌ Search system not available. Install optional dependencies with `uv sync --extra embeddings`"

        async with db:
            results = await db.search_conversations(
                query=query, project=project, limit=limit + offset
            )

            paginated_results = results[offset : offset + limit]
            if not paginated_results:
                return f"🔍 No more results for '{query}' (offset: {offset})"

            return _build_pagination_output(
                query, offset, paginated_results, len(results), limit
            )

    except Exception as e:
        logger.exception(f"Get more results failed: {e}")
        return f"❌ Pagination error: {e!s}"


def _format_paginated_result(result: dict[str, Any], index: int) -> str:
    """Format a single paginated search result."""
    output = f"**{index}.** "
    if result.get("timestamp"):
        output += f"({result['timestamp']}) "
    output += f"{result.get('content', '')[:150]}...\n\n"
    return output


def _build_pagination_output(
    query: str,
    offset: int,
    paginated_results: list[dict[str, Any]],
    total_results: int,
    limit: int,
) -> str:
    """Build the complete output for paginated results."""
    output = f"🔍 **Results {offset + 1}-{offset + len(paginated_results)}** for '{query}'\n\n"

    for i, result in enumerate(paginated_results, offset + 1):
        output += _format_paginated_result(result, i)

    if offset + limit < total_results:
        remaining = total_results - (offset + limit)
        output += f"💡 {remaining} more results available"

    return output


def _extract_file_excerpt(content: str, file_path: str) -> str:
    """Extract a relevant excerpt from content based on the file path."""
    if file_path in content:
        start = max(0, content.find(file_path) - 50)
        end = min(len(content), content.find(file_path) + len(file_path) + 100)
        excerpt = content[start:end]
    else:
        excerpt = content[:150]
    return excerpt


def _format_file_search_result(result: dict[str, Any], file_path: str) -> str:
    """Format a single file search result."""
    output = "**1.** "  # Default numbering will be handled by caller
    if result.get("timestamp"):
        output += f"({result['timestamp']}) "

    content = result.get("content", "")
    excerpt = _extract_file_excerpt(content, file_path)
    output += f"{excerpt}...\n\n"

    return output


def _format_file_search_header(results_count: int, file_path: str) -> str:
    """Format the header for file search results."""
    return f"🔍 **{results_count} conversations** about `{file_path}`\n\n"


async def _search_by_file_impl(
    file_path: str,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Search for conversations that analyzed a specific file."""
    try:
        db = await get_reflection_database()
        if not db:
            return "❌ Search system not available. Install optional dependencies with `uv sync --extra embeddings`"

        async with db:
            results = await db.search_conversations(
                query=file_path, project=project, limit=limit
            )

            if not results:
                return f"🔍 No conversations found about file: {file_path}"

            output = _format_file_search_header(len(results), file_path)

            for i, result in enumerate(results, 1):
                single_result = _format_file_search_result(result, file_path)
                # Replace the default "1." with the correct number
                single_result = single_result.replace("**1.**", f"**{i}.**")
                output += single_result

            return output

    except Exception as e:
        logger.exception(f"File search failed: {e}")
        return f"❌ File search error: {e!s}"


def _format_concept_result_header(concept: str, results_count: int) -> str:
    """Format the header for concept search results."""
    return f"🔍 **{results_count} conversations** about `{concept}`\n\n"


def _extract_relevant_excerpt(content: str, concept: str) -> str:
    """Extract a relevant excerpt from content based on the concept."""
    if concept.lower() in content.lower():
        start = max(0, content.lower().find(concept.lower()) - 75)
        end = min(len(content), start + 200)
        excerpt = content[start:end]
    else:
        excerpt = content[:150]
    return excerpt


def _format_single_concept_result(
    result: dict[str, Any], index: int, concept: str
) -> str:
    """Format a single concept search result."""
    output = f"**{index}.** "
    if result.get("timestamp"):
        output += f"({result['timestamp']}) "
    if result.get("similarity"):
        output += f"(relevance: {result['similarity']:.2f}) "

    content = result.get("content", "")
    excerpt = _extract_relevant_excerpt(content, concept)
    output += f"{excerpt}...\n\n"

    return output


def _extract_mentioned_files(results: list[dict[str, Any]]) -> list[str]:
    """Extract mentioned files from search results."""
    # Extract mentioned files
    all_content = " ".join([r.get("content", "") for r in results])
    from session_mgmt_mcp.utils.regex_patterns import SAFE_PATTERNS

    files = []
    for pattern_name in (
        "python_files",
        "javascript_files",
        "config_files",
        "documentation_files",
    ):
        pattern = SAFE_PATTERNS[pattern_name]
        matches = pattern.findall(all_content)
        files.extend(matches)

    if files:
        return list(set(files))[:10]

    return []


def _format_related_files(files: list[str]) -> str:
    """Format related files section."""
    return f"📁 **Related Files**: {', '.join(files)}"


async def _get_concept_search_database() -> ReflectionDatabase | None:
    """Get database connection for concept search."""
    db = await get_reflection_database()
    if not db:
        return None
    return db


async def _perform_concept_search(
    db: ReflectionDatabase, concept: str, project: str | None, limit: int
) -> list[dict[str, Any]]:
    """Perform the actual concept search query."""
    async with db:
        return await db.search_conversations(
            query=concept, project=project, limit=limit, min_score=0.6
        )


def _build_concept_output(
    concept: str, results: list[dict[str, Any]], include_files: bool
) -> str:
    """Build the formatted output for concept search results."""
    output = _format_concept_result_header(concept, len(results))

    for i, result in enumerate(results, 1):
        output += _format_single_concept_result(result, i, concept)

    if include_files and results:
        files = _extract_mentioned_files(results)
        if files:
            output += _format_related_files(files)

    return output


async def _search_by_concept_impl(
    concept: str,
    include_files: bool = True,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Search for conversations about a specific development concept."""
    try:
        db = await _get_concept_search_database()
        if not db:
            return "❌ Search system not available. Install optional dependencies with `uv sync --extra embeddings`"

        results = await _perform_concept_search(db, concept, project, limit)
        if not results:
            return f"🔍 No conversations found about concept: {concept}"

        return _build_concept_output(concept, results, include_files)

    except Exception as e:
        logger.exception(f"Concept search failed: {e}")
        return f"❌ Concept search error: {e!s}"


async def _reset_reflection_database_impl() -> str:
    """Reset the reflection database connection to fix lock issues."""
    try:
        db = await get_reflection_database()
        if not db:
            return "❌ Reflection database not available"

        async with db:
            # Database connection is managed by async context manager
            return "✅ Reflection database connection verified successfully"

    except Exception as e:
        logger.exception(f"Database reset failed: {e}")
        return f"❌ Database reset error: {e!s}"


async def _reflection_stats_impl() -> str:
    """Get statistics about the reflection database."""
    try:
        db = await get_reflection_database()
        if not db:
            return "❌ Reflection database not available. Install optional dependencies with `uv sync --extra embeddings`"

        async with db:
            stats = await db.get_stats()
            output = "📊 **Reflection Database Statistics**\n\n"
            for key, value in stats.items():
                output += f"**{key.replace('_', ' ').title()}**: {value}\n"
            return output

    except Exception as e:
        logger.exception(f"Stats collection failed: {e}")
        return f"❌ Stats error: {e!s}"


def _extract_code_blocks_from_content(content: str) -> list[str]:
    """Extract code blocks from content using regex patterns."""
    try:
        from session_mgmt_mcp.utils.regex_patterns import SAFE_PATTERNS

        code_pattern = SAFE_PATTERNS["generic_code_block"]
        matches = code_pattern.findall(content)
        return matches if matches is not None else []
    except Exception:
        return []


def _format_code_result(result: dict[str, Any], query: str) -> str:
    """Format a single code search result."""
    output = ""
    if result.get("timestamp"):
        output += f"({result['timestamp']}) "

    content = result.get("content", "")
    code_blocks = _extract_code_blocks_from_content(content)

    if code_blocks:
        code = code_blocks[0][:200]
        output += f"\n```\n{code}...\n```\n\n"
    else:
        if query.lower() in content.lower():
            start = max(0, content.lower().find(query.lower()) - 50)
            end = min(len(content), start + 150)
            excerpt = content[start:end]
        else:
            excerpt = content[:100]
        output += f"{excerpt}...\n\n"

    return output


def _build_code_query(query: str, pattern_type: str | None) -> str:
    """Build the code query string for searching."""
    code_query = f"code {query}"
    if pattern_type:
        code_query += f" {pattern_type}"
    return code_query


def _format_code_search_header(
    results_count: int, query: str, pattern_type: str | None
) -> str:
    """Format the header for code search results."""
    output = f"🔍 **{results_count} code patterns** for `{query}`"
    if pattern_type:
        output += f" (type: {pattern_type})"
    output += "\\n\\n"
    return output


def _format_single_code_result(index: int, result: dict[str, Any], query: str) -> str:
    """Format a single code search result."""
    return f"**{index}.** " + _format_code_result(result, query)


async def _search_code_impl(
    query: str,
    pattern_type: str | None = None,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Search for code patterns in conversations using AST parsing."""
    try:
        db = await get_reflection_database()
        if not db:
            return "❌ Search system not available. Install optional dependencies with `uv sync --extra embeddings`"

        code_query = _build_code_query(query, pattern_type)

        async with db:
            results = await db.search_conversations(
                query=code_query, project=project, limit=limit, min_score=0.5
            )

            if not results:
                return f"🔍 No code patterns found for: {query}"

            output = _format_code_search_header(len(results), query, pattern_type)

            for i, result in enumerate(results, 1):
                output += _format_single_code_result(i, result, query)

            return output

    except Exception as e:
        logger.exception(f"Code search failed: {e}")
        return f"❌ Code search error: {e!s}"


def _find_best_error_excerpt(content: str) -> str:
    """Find the most relevant excerpt from content based on error keywords."""
    error_keywords = ["error", "exception", "traceback", "failed", "fix"]
    best_excerpt = ""
    best_score = 0

    for keyword in error_keywords:
        if keyword in content.lower():
            start = max(0, content.lower().find(keyword) - 75)
            end = min(len(content), start + 200)
            excerpt = content[start:end]
            score = content.lower().count(keyword)
            if score > best_score:
                best_score = score
                best_excerpt = excerpt

    if not best_excerpt:
        best_excerpt = content[:150]

    return best_excerpt


def _format_error_result(result: dict[str, Any]) -> str:
    """Format a single error search result."""
    output = ""
    if result.get("timestamp"):
        output += f"({result['timestamp']}) "

    content = result.get("content", "")
    best_excerpt = _find_best_error_excerpt(content)
    output += f"{best_excerpt}...\n\n"

    return output


def _build_error_query(query: str, error_type: str | None) -> str:
    """Build the error query string for searching."""
    error_query = f"error {query}"
    if error_type:
        error_query += f" {error_type}"
    return error_query


def _format_error_search_header(
    results_count: int, query: str, error_type: str | None
) -> str:
    """Format the header for error search results."""
    output = f"🔍 **{results_count} error contexts** for `{query}`"
    if error_type:
        output += f" (type: {error_type})"
    output += "\\n\\n"
    return output


def _process_error_search_results(
    results: list[dict[str, Any]], query: str, error_type: str | None
) -> str:
    """Process and format error search results."""
    output = _format_error_search_header(len(results), query, error_type)

    for i, result in enumerate(results, 1):
        output += f"**{i}.** "
        output += _format_error_result(result)

    return output


async def _search_errors_impl(
    query: str,
    error_type: str | None = None,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Search for error patterns and debugging contexts in conversations."""
    try:
        db = await get_reflection_database()
        if not db:
            return "❌ Search system not available. Install optional dependencies with `uv sync --extra embeddings`"

        error_query = _build_error_query(query, error_type)

        async with db:
            results = await db.search_conversations(
                query=error_query, project=project, limit=limit, min_score=0.4
            )

            if not results:
                return f"🔍 No error patterns found for: {query}"

            return _process_error_search_results(results, query, error_type)

    except Exception as e:
        logger.exception(f"Error search failed: {e}")
        return f"❌ Error search failed: {e!s}"


def _parse_time_expression(time_expression: str) -> datetime | None:
    """Parse natural language time expression into datetime."""
    now = datetime.now()

    if "yesterday" in time_expression.lower():
        return now - timedelta(days=1)
    if "last week" in time_expression.lower():
        return now - timedelta(days=7)
    if "last month" in time_expression.lower():
        return now - timedelta(days=30)
    if "today" in time_expression.lower():
        return now - timedelta(hours=24)

    return None


def _format_temporal_search_header(
    results_count: int, time_expression: str, query: str | None
) -> str:
    """Format the header for temporal search results."""
    output = f"🔍 **{results_count} conversations** from `{time_expression}`"
    if query:
        output += f" matching `{query}`"
    output += "\n\n"
    return output


def _format_single_temporal_result(result: dict[str, Any]) -> str:
    """Format a single temporal search result."""
    output = "**1.** "  # Default numbering will be handled by caller

    if result.get("timestamp"):
        output += f"({result['timestamp']}) "

    content = result.get("content", "")
    output += f"{content[:150]}...\n\n"

    return output


async def _search_temporal_impl(
    time_expression: str,
    query: str | None = None,
    limit: int = 10,
    project: str | None = None,
) -> str:
    """Search conversations within a specific time range using natural language."""
    try:
        db = await get_reflection_database()
        if not db:
            return "❌ Search system not available. Install optional dependencies with `uv sync --extra embeddings`"

        start_time = _parse_time_expression(time_expression)

        async with db:
            search_query = query or ""
            results = await db.search_conversations(
                query=search_query, project=project, limit=limit * 2
            )

            if start_time:
                # This is a simplified filter - would need proper timestamp parsing
                filtered_results = results.copy()
                results = filtered_results[:limit]

            if not results:
                return f"🔍 No conversations found for time period: {time_expression}"

            output = _format_temporal_search_header(
                len(results), time_expression, query
            )

            for i, result in enumerate(results, 1):
                single_result = _format_single_temporal_result(result)
                # Replace the default "1." with the correct number
                single_result = single_result.replace("**1.**", f"**{i}.**")
                output += single_result

            return output

    except Exception as e:
        logger.exception(f"Temporal search failed: {e}")
        return f"❌ Temporal search error: {e!s}"


def register_search_tools(mcp: Any) -> None:
    """Register all search-related MCP tools.

    Args:
        mcp: FastMCP server instance

    """

    # Register tools with proper decorator syntax
    @mcp.tool()  # type: ignore[misc]  # type: ignore[misc]
    async def _optimize_search_results(
        results: list[dict[str, Any]],
        optimize_tokens: bool,
        max_tokens: int,
        query: str,
    ) -> dict[str, Any]:
        return await _optimize_search_results_impl(
            results, optimize_tokens, max_tokens, query
        )

    @mcp.tool()  # type: ignore[misc]  # type: ignore[misc]
    async def store_reflection(content: str, tags: list[str] | None = None) -> str:
        return await _store_reflection_impl(content, tags)

    @mcp.tool()  # type: ignore[misc]
    async def quick_search(
        query: str, project: str | None = None, min_score: float = 0.7
    ) -> str:
        return await _quick_search_impl(query, project, min_score)

    @mcp.tool()  # type: ignore[misc]
    async def search_summary(
        query: str, project: str | None = None, min_score: float = 0.7
    ) -> str:
        return await _search_summary_impl(query, project, min_score)

    @mcp.tool()  # type: ignore[misc]
    async def get_more_results(
        query: str, offset: int = 3, limit: int = 3, project: str | None = None
    ) -> str:
        return await _get_more_results_impl(query, offset, limit, project)

    @mcp.tool()  # type: ignore[misc]
    async def search_by_file(
        file_path: str, limit: int = 10, project: str | None = None
    ) -> str:
        return await _search_by_file_impl(file_path, limit, project)

    @mcp.tool()  # type: ignore[misc]
    async def search_by_concept(
        concept: str,
        include_files: bool = True,
        limit: int = 10,
        project: str | None = None,
    ) -> str:
        return await _search_by_concept_impl(concept, include_files, limit, project)

    @mcp.tool()  # type: ignore[misc]
    async def reset_reflection_database() -> str:
        return await _reset_reflection_database_impl()

    @mcp.tool()  # type: ignore[misc]
    async def reflection_stats() -> str:
        return await _reflection_stats_impl()

    @mcp.tool()  # type: ignore[misc]
    async def search_code(
        query: str,
        pattern_type: str | None = None,
        limit: int = 10,
        project: str | None = None,
    ) -> str:
        return await _search_code_impl(query, pattern_type, limit, project)

    @mcp.tool()  # type: ignore[misc]
    async def search_errors(
        query: str,
        error_type: str | None = None,
        limit: int = 10,
        project: str | None = None,
    ) -> str:
        return await _search_errors_impl(query, error_type, limit, project)

    @mcp.tool()  # type: ignore[misc]
    async def search_temporal(
        time_expression: str,
        query: str | None = None,
        limit: int = 10,
        project: str | None = None,
    ) -> str:
        return await _search_temporal_impl(time_expression, query, limit, project)
