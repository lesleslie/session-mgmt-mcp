from __future__ import annotations

import json
import typing as t

import pytest
from session_buddy.memory.entity_extractor import EntityExtractionEngine


@pytest.mark.asyncio
async def test_extractor_pattern_fallback(monkeypatch: t.Any) -> None:
    engine = EntityExtractionEngine()

    class DummyManager:
        async def generate(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
            msg = "no providers"
            raise RuntimeError(msg)

    monkeypatch.setattr(engine, "manager", DummyManager())

    res = await engine.extract_entities("user msg", "ai reply")
    assert res.llm_provider == "pattern"
    assert res.processed_memory.category in {
        "facts",
        "preferences",
        "skills",
        "rules",
        "context",
    }
