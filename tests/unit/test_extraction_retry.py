from __future__ import annotations

import json
import typing as t

import pytest
from session_buddy.memory.entity_extractor import EntityExtractionEngine


class DummyResponse:
    def __init__(self, content: str) -> None:
        self.content = content


@pytest.mark.asyncio
async def test_cascade_retries_timeout(monkeypatch: t.Any) -> None:
    engine = EntityExtractionEngine()

    class DummyManager:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def generate(
            self, messages: list[t.Any], provider: str, **_: t.Any
        ) -> DummyResponse:
            self.calls.append(provider)
            if provider == "openai":
                msg = "timeout"
                raise TimeoutError(msg)
            if provider == "anthropic":
                pm = {
                    "category": "facts",
                    "importance_score": 0.7,
                    "summary": "s",
                    "searchable_content": "c",
                    "reasoning": "r",
                    "entities": [],
                    "relationships": [],
                    "suggested_tier": "long_term",
                    "tags": [],
                }
                return DummyResponse(json.dumps(pm))
            msg = "unexpected provider"
            raise RuntimeError(msg)

    # Force timeout=1s, retries=0
    monkeypatch.setattr(engine, "timeout_s", 1)
    monkeypatch.setattr(engine, "retries", 0)
    monkeypatch.setattr(engine, "manager", DummyManager())

    res = await engine.extract_entities("u", "a")
    assert res.llm_provider == "anthropic"
    assert res.processed_memory.category == "facts"
