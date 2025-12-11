"""LLM provider system for session management.

This package provides a unified interface for working with multiple LLM providers
including OpenAI, Google Gemini, and Ollama. It includes:
- Standardized data models for messages and responses
- Abstract base class for provider implementations
- Individual provider implementations
- Security utilities for API key validation
- Manager class for multi-provider orchestration
"""

from session_buddy.llm.base import LLMProvider
from session_buddy.llm.models import (
    LLMMessage,
    LLMResponse,
    StreamChunk,
    StreamGenerationOptions,
)
from session_buddy.llm.providers import (
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
)
from session_buddy.llm.security import (
    get_masked_api_key,
    validate_llm_api_keys_at_startup,
)

__all__ = [
    "GeminiProvider",
    # Data models
    "LLMMessage",
    # Base classes
    "LLMProvider",
    "LLMResponse",
    "OllamaProvider",
    # Provider implementations
    "OpenAIProvider",
    "StreamChunk",
    "StreamGenerationOptions",
    # Security utilities
    "get_masked_api_key",
    "validate_llm_api_keys_at_startup",
]
