from __future__ import annotations

import typing as t
from types import SimpleNamespace

import pytest
from session_buddy.tools.feature_flags_tools import register_feature_flags_tools


class DummyMCP:
    def __init__(self) -> None:
        self.tools: dict[str, t.Callable[..., t.Any]] = {}

    def tool(self) -> t.Callable[[t.Callable[..., t.Any]], t.Callable[..., t.Any]]:
        def decorator(fn: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
            self.tools[fn.__name__] = fn
            return fn

        return decorator


@pytest.mark.asyncio
async def test_feature_flags_status_shape(monkeypatch: t.Any) -> None:
    # No env settings override; just ensure keys are present
    mcp = DummyMCP()
    register_feature_flags_tools(mcp)
    res = await mcp.tools["feature_flags_status"]()
    keys = {
        "use_schema_v2",
        "enable_llm_entity_extraction",
        "enable_anthropic",
        "enable_ollama",
        "enable_conscious_agent",
        "enable_filesystem_extraction",
    }
    assert keys.issubset(res.keys())
