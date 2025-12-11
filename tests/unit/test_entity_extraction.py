from __future__ import annotations

import json
import typing as t
from types import SimpleNamespace

import pytest
from session_buddy.memory.entity_extractor import EntityExtractionEngine


class DummyResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyManager:
    def __init__(self, seq: list[str]) -> None:
        # seq is a list of provider names that will succeed in order
        self._seq = seq
        self._calls: list[str] = []

    async def generate(
        self, messages: list[t.Any], provider: str, **_: t.Any
    ) -> DummyResponse:
        self._calls.append(provider)
        if provider in self._seq:
            # Minimal valid ProcessedMemory JSON
            content = json.dumps(
                {
                    "category": "facts",
                    "importance_score": 0.8,
                    "summary": "ok",
                    "searchable_content": "text",
                    "reasoning": "",
                    "entities": [],
                    "relationships": [],
                    "suggested_tier": "long_term",
                    "tags": [],
                }
            )
            return DummyResponse(content)
        msg = "provider failed"
        raise RuntimeError(msg)


@pytest.mark.asyncio
async def test_cascade_tries_providers_in_order(monkeypatch: t.Any) -> None:
    engine = EntityExtractionEngine()

    # Replace manager with dummy that succeeds on anthropic
    dummy = DummyManager(["anthropic"])  # openai fails, anthropic succeeds
    monkeypatch.setattr(engine, "manager", dummy)

    res = await engine.extract_entities("hi", "there")
    assert res.llm_provider == "anthropic"
    assert res.processed_memory.category == "facts"
