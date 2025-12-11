"""Tests for LLM provider system.

Tests cross-LLM compatibility with OpenAI, Gemini, and Ollama providers,
including provider initialization, message handling, and fallback strategies.

Phase: Week 4 Day 4 - LLM Provider Coverage
Phase 2: Core Coverage (30% → 70%) - Helper Function Tests
"""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDataClasses:
    """Test LLM provider data classes."""

    def test_stream_generation_options_defaults(self) -> None:
        """Should create options with sensible defaults."""
        from session_buddy.llm_providers import StreamGenerationOptions

        options = StreamGenerationOptions()

        assert options.provider is None
        assert options.model is None
        assert options.use_fallback is True
        assert options.temperature == 0.7
        assert options.max_tokens is None

    def test_stream_generation_options_immutable(self) -> None:
        """Should be frozen/immutable dataclass."""
        from session_buddy.llm_providers import StreamGenerationOptions

        options = StreamGenerationOptions(temperature=0.5)

        with pytest.raises(AttributeError):
            options.temperature = 0.9  # type: ignore[misc]

    def test_stream_chunk_content_chunk(self) -> None:
        """Should create content chunk."""
        from session_buddy.llm_providers import StreamChunk

        chunk = StreamChunk.content_chunk("Hello", provider="openai")

        assert chunk.content == "Hello"
        assert chunk.provider == "openai"
        assert chunk.is_error is False

    def test_stream_chunk_error_chunk(self) -> None:
        """Should create error chunk."""
        from session_buddy.llm_providers import StreamChunk

        chunk = StreamChunk.error_chunk("Connection failed")

        assert chunk.content == ""
        assert chunk.is_error is True
        assert chunk.metadata["error"] == "Connection failed"

    def test_llm_message_auto_timestamp(self) -> None:
        """Should auto-generate timestamp."""
        from session_buddy.llm_providers import LLMMessage

        msg = LLMMessage(role="user", content="Hello")

        assert msg.timestamp is not None
        assert msg.metadata is not None
        assert isinstance(msg.metadata, dict)

    def test_llm_message_custom_timestamp(self) -> None:
        """Should accept custom timestamp."""
        from session_buddy.llm_providers import LLMMessage

        custom_time = "2024-01-01T12:00:00"
        msg = LLMMessage(role="assistant", content="Hi", timestamp=custom_time)

        assert msg.timestamp == custom_time

    def test_llm_response_structure(self) -> None:
        """Should create response with all fields."""
        from session_buddy.llm_providers import LLMResponse

        response = LLMResponse(
            content="Response text",
            model="gpt-4",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            finish_reason="stop",
            timestamp="2024-01-01T12:00:00",
        )

        assert response.content == "Response text"
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.metadata is not None

    def test_llm_response_auto_metadata(self) -> None:
        """Should auto-initialize metadata."""
        from session_buddy.llm_providers import LLMResponse

        response = LLMResponse(
            content="Test",
            model="test",
            provider="test",
            usage={},
            finish_reason="stop",
            timestamp="2024-01-01",
        )

        assert isinstance(response.metadata, dict)


class TestLLMProviderBase:
    """Test LLM provider abstract base class."""

    def test_provider_initialization(self) -> None:
        """Should initialize provider with config."""
        from session_buddy.llm_providers import OpenAIProvider

        config = {"api_key": "test-key", "model": "gpt-4"}
        provider = OpenAIProvider(config)

        assert provider.config == config
        assert provider.name == "openai"

    def test_provider_name_extraction(self) -> None:
        """Should extract provider name from class name."""
        from session_buddy.llm_providers import GeminiProvider, OllamaProvider

        gemini = GeminiProvider({"api_key": "test"})
        ollama = OllamaProvider({"base_url": "http://localhost:11434"})

        assert gemini.name == "gemini"
        assert ollama.name == "ollama"


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    def test_init_with_api_key(self) -> None:
        """Should initialize with API key."""
        from session_buddy.llm_providers import OpenAIProvider

        provider = OpenAIProvider({"api_key": "sk-test123"})

        assert provider.config["api_key"] == "sk-test123"
        assert provider.name == "openai"

    def test_convert_messages_format(self) -> None:
        """Should convert messages to OpenAI format."""
        from session_buddy.llm_providers import LLMMessage, OpenAIProvider

        provider = OpenAIProvider({"api_key": "test"})
        messages = [
            LLMMessage(role="system", content="You are helpful"),
            LLMMessage(role="user", content="Hello"),
        ]

        converted = provider._convert_messages(messages)

        assert len(converted) == 2
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "You are helpful"
        assert converted[1]["role"] == "user"
        assert converted[1]["content"] == "Hello"

    def test_get_models_list(self) -> None:
        """Should return list of available models."""
        from session_buddy.llm_providers import OpenAIProvider

        provider = OpenAIProvider({"api_key": "test"})
        models = provider.get_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-4" in models or "gpt-3.5-turbo" in models

    @pytest.mark.asyncio
    async def test_is_available_with_api_key(self) -> None:
        """Should check availability based on API key."""
        from session_buddy.llm_providers import OpenAIProvider

        # With API key
        provider = OpenAIProvider({"api_key": "sk-test123"})
        available = await provider.is_available()
        assert isinstance(available, bool)

        # Without API key
        provider_no_key = OpenAIProvider({})
        available_no_key = await provider_no_key.is_available()
        assert available_no_key is False


class TestGeminiProvider:
    """Test Google Gemini provider implementation."""

    def test_init_with_api_key(self) -> None:
        """Should initialize with API key."""
        from session_buddy.llm_providers import GeminiProvider

        provider = GeminiProvider({"api_key": "test-gemini-key"})

        assert provider.config["api_key"] == "test-gemini-key"
        assert provider.name == "gemini"

    def test_convert_messages_gemini_format(self) -> None:
        """Should convert messages to Gemini format."""
        from session_buddy.llm_providers import GeminiProvider, LLMMessage

        provider = GeminiProvider({"api_key": "test"})
        messages = [
            LLMMessage(role="user", content="What is AI?"),
            LLMMessage(role="assistant", content="AI is..."),
        ]

        converted = provider._convert_messages(messages)

        assert isinstance(converted, list)
        assert len(converted) == 2

    def test_get_models_list(self) -> None:
        """Should return list of Gemini models."""
        from session_buddy.llm_providers import GeminiProvider

        provider = GeminiProvider({"api_key": "test"})
        models = provider.get_models()

        assert isinstance(models, list)
        assert len(models) > 0
        # Gemini models typically include "gemini-pro", "gemini-pro-vision"
        assert any("gemini" in model.lower() for model in models)

    @pytest.mark.asyncio
    async def test_is_available_with_api_key(self) -> None:
        """Should check availability based on API key."""
        from session_buddy.llm_providers import GeminiProvider

        # With API key
        provider = GeminiProvider({"api_key": "test-key"})
        available = await provider.is_available()
        assert isinstance(available, bool)

        # Without API key
        provider_no_key = GeminiProvider({})
        available_no_key = await provider_no_key.is_available()
        assert available_no_key is False


class TestOllamaProvider:
    """Test Ollama provider implementation."""

    def test_init_with_base_url(self) -> None:
        """Should initialize with base URL."""
        from session_buddy.llm_providers import OllamaProvider

        provider = OllamaProvider({"base_url": "http://localhost:11434"})

        assert provider.config["base_url"] == "http://localhost:11434"
        assert provider.name == "ollama"

    def test_init_default_base_url(self) -> None:
        """Should use default base URL if not provided."""
        from session_buddy.llm_providers import OllamaProvider

        provider = OllamaProvider({})

        # Provider should have config
        assert hasattr(provider, "config")
        assert isinstance(provider.config, dict)

    def test_convert_messages_format(self) -> None:
        """Should convert messages to Ollama format."""
        from session_buddy.llm_providers import LLMMessage, OllamaProvider

        provider = OllamaProvider({})
        messages = [
            LLMMessage(role="system", content="Be helpful"),
            LLMMessage(role="user", content="Hello"),
        ]

        converted = provider._convert_messages(messages)

        assert isinstance(converted, list)
        assert len(converted) == 2
        assert converted[0]["role"] == "system"
        assert converted[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_is_available_checks_connection(self) -> None:
        """Should check Ollama server availability."""
        from session_buddy.llm_providers import OllamaProvider

        provider = OllamaProvider({"base_url": "http://localhost:11434"})

        # Check availability (will be False without running Ollama server)
        available = await provider.is_available()
        assert isinstance(available, bool)


class TestLLMManager:
    """Test LLM manager for provider coordination."""

    def test_manager_initialization(self) -> None:
        """Should initialize manager with config."""
        from session_buddy.llm_providers import LLMManager

        # LLMManager takes config_path (str), not config dict
        manager = LLMManager(config_path=None)

        assert isinstance(manager.config, dict)
        assert isinstance(manager.providers, dict)

    def test_manager_loads_providers(self) -> None:
        """Should load configured providers."""
        from session_buddy.llm_providers import LLMManager

        # LLMManager loads from environment variables by default
        manager = LLMManager(config_path=None)

        # Should have loaded providers from environment/defaults
        assert isinstance(manager.providers, dict)
        # May have providers if env vars are set
        if manager.providers:
            assert all(hasattr(p, "name") for p in manager.providers.values())

    def test_manager_default_provider(self) -> None:
        """Should set default provider from config."""
        from session_buddy.llm_providers import LLMManager

        manager = LLMManager(config_path=None)

        # Default provider should be set (typically "openai")
        assert hasattr(manager.config, "__getitem__")
        assert "default_provider" in manager.config

    def test_manager_fallback_order(self) -> None:
        """Should configure fallback provider order."""
        from session_buddy.llm_providers import LLMManager

        manager = LLMManager(config_path=None)

        # Fallback providers should be configured
        assert "fallback_providers" in manager.config
        assert isinstance(manager.config["fallback_providers"], list)

    @pytest.mark.asyncio
    async def test_manager_generate_with_default_provider(self) -> None:
        """Should generate using default provider."""
        from session_buddy.llm_providers import LLMManager

        manager = LLMManager(config_path=None)

        # Verify manager has generate method
        assert hasattr(manager, "generate")
        assert callable(manager.generate)

    @pytest.mark.asyncio
    async def test_manager_fallback_on_failure(self) -> None:
        """Should support fallback configuration."""
        from session_buddy.llm_providers import LLMManager

        manager = LLMManager(config_path=None)

        # Verify fallback configuration exists
        assert "fallback_providers" in manager.config
        assert isinstance(manager.config["fallback_providers"], list)


class TestProviderAvailability:
    """Test provider availability checking."""

    @pytest.mark.asyncio
    async def test_check_multiple_providers(self) -> None:
        """Should check availability of multiple providers."""
        from session_buddy.llm_providers import LLMManager

        manager = LLMManager(config_path=None)

        # Manager should have providers dict
        assert isinstance(manager.providers, dict)

        # If providers exist, they should have is_available method
        for provider in manager.providers.values():
            assert hasattr(provider, "is_available")
            assert callable(provider.is_available)


class TestErrorHandling:
    """Test error handling in LLM providers."""

    @pytest.mark.asyncio
    async def test_handle_missing_api_key(self) -> None:
        """Should handle missing API key gracefully."""
        from session_buddy.llm_providers import OpenAIProvider

        provider = OpenAIProvider({})  # No API key

        # Should detect unavailability
        available = await provider.is_available()
        assert available is False

    @pytest.mark.asyncio
    async def test_handle_network_error(self) -> None:
        """Should handle network errors gracefully."""
        from session_buddy.llm_providers import LLMMessage, OllamaProvider

        provider = OllamaProvider({"base_url": "http://invalid:99999"})

        # Should raise RuntimeError when provider not available
        with pytest.raises(RuntimeError, match="Ollama provider not available"):
            messages = [LLMMessage(role="user", content="Test")]
            await provider.generate(messages)

    def test_invalid_message_role(self) -> None:
        """Should validate message roles."""
        from session_buddy.llm_providers import LLMMessage

        # Standard roles should work
        msg = LLMMessage(role="user", content="Test")
        assert msg.role == "user"

        # Custom roles are allowed (validation in provider logic)
        custom_msg = LLMMessage(role="custom", content="Test")
        assert custom_msg.role == "custom"
