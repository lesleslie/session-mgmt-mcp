#!/usr/bin/env python3
"""Cross-LLM Compatibility for Session Management MCP Server.

Provides unified interface for multiple LLM providers including OpenAI, Google Gemini, and Ollama.
"""

import contextlib
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StreamGenerationOptions:
    """Immutable streaming generation options."""

    provider: str | None = None
    model: str | None = None
    use_fallback: bool = True
    temperature: float = 0.7
    max_tokens: int | None = None


@dataclass
class StreamChunk:
    """Immutable streaming response chunk."""

    content: str = field(default="")
    is_error: bool = field(default=False)
    provider: str = field(default="")
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def content_chunk(cls, content: str, provider: str = "") -> "StreamChunk":
        """Create content chunk."""
        return cls(content=content, provider=provider)  # type: ignore[call-arg]

    @classmethod
    def error_chunk(cls, error: str) -> "StreamChunk":
        """Create error chunk."""
        return cls(content="", is_error=True, metadata={"error": error})  # type: ignore[call-arg]


@dataclass
class LLMMessage:
    """Standardized message format across LLM providers."""

    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMResponse:
    """Standardized response format from LLM providers."""

    content: str
    model: str
    provider: str
    usage: dict[str, Any]
    finish_reason: str
    timestamp: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.name = self.__class__.__name__.replace("Provider", "").lower()
        self.logger = logging.getLogger(f"llm_providers.{self.name}")

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM."""

    @abstractmethod
    async def stream_generate(  # type: ignore[override]
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str]:
        """Generate a streaming response from the LLM."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""

    @abstractmethod
    def get_models(self) -> list[str]:
        """Get list of available models for this provider."""


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.default_model = config.get("default_model", "gpt-4")
        self._client = None

    async def _get_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai

                self._client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                msg = "OpenAI package not installed. Install with: pip install openai"
                raise ImportError(
                    msg,
                )
        return self._client

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, str]]:
        """Convert LLMMessage objects to OpenAI format."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    async def generate(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not await self.is_available():
            msg = "OpenAI provider not available"
            raise RuntimeError(msg)

        client = await self._get_client()
        model_name = model or self.default_model

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=self._convert_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=model_name,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens
                    if response.usage
                    else 0,
                    "completion_tokens": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "total_tokens": response.usage.total_tokens
                    if response.usage
                    else 0,
                },
                finish_reason=response.choices[0].finish_reason,
                timestamp=datetime.now().isoformat(),
                metadata={"response_id": response.id},
            )

        except Exception as e:
            self.logger.exception(f"OpenAI generation failed: {e}")
            raise

    async def stream_generate(  # type: ignore[override]
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str]:
        """Stream response using OpenAI API."""
        if not await self.is_available():
            msg = "OpenAI provider not available"
            raise RuntimeError(msg)

        client = await self._get_client()
        model_name = model or self.default_model

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=self._convert_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self.logger.exception(f"OpenAI streaming failed: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.api_key:
            return False

        try:
            client = await self._get_client()
            # Test with a simple request
            await client.models.list()
            return True
        except Exception:
            return False

    def get_models(self) -> list[str]:
        """Get available OpenAI models."""
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.api_key = (
            config.get("api_key")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.default_model = config.get("default_model", "gemini-pro")
        self._client = None

    async def _get_client(self) -> Any:
        """Get or create Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                msg = "Google Generative AI package not installed. Install with: pip install google-generativeai"
                raise ImportError(
                    msg,
                )
        return self._client

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert LLMMessage objects to Gemini format using modern pattern matching."""
        converted: list[dict[str, Any]] = []

        for msg in messages:
            match msg.role:
                case "system":
                    # Gemini doesn't have system role, prepend to first user message
                    if converted and converted[-1]["role"] == "user":
                        converted[-1]["parts"] = [
                            f"System: {msg.content}\n\nUser: {converted[-1]['parts'][0]}",
                        ]
                    else:
                        converted.append(
                            {"role": "user", "parts": [f"System: {msg.content}"]},
                        )
                case "user":
                    converted.append({"role": "user", "parts": [msg.content]})
                case "assistant":
                    converted.append({"role": "model", "parts": [msg.content]})
                case _:
                    # Unknown role - default to user for safety
                    converted.append({"role": "user", "parts": [msg.content]})

        return converted

    async def generate(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Gemini API."""
        if not await self.is_available():
            msg = "Gemini provider not available"
            raise RuntimeError(msg)

        genai = await self._get_client()
        model_name = model or self.default_model

        try:
            model_instance = genai.GenerativeModel(model_name)

            # Convert messages to Gemini chat format
            chat_messages = self._convert_messages(messages)

            # Create chat or generate single response
            if len(chat_messages) > 1:
                chat = model_instance.start_chat(history=chat_messages[:-1])
                response = await chat.send_message_async(
                    chat_messages[-1]["parts"][0],
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )
            else:
                response = await model_instance.generate_content_async(
                    chat_messages[0]["parts"][0],
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )

            return LLMResponse(
                content=response.text,
                model=model_name,
                provider="gemini",
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count
                    if hasattr(response, "usage_metadata")
                    else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count
                    if hasattr(response, "usage_metadata")
                    else 0,
                    "total_tokens": response.usage_metadata.total_token_count
                    if hasattr(response, "usage_metadata")
                    else 0,
                },
                finish_reason="stop",  # Gemini doesn't provide detailed finish reasons
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            self.logger.exception(f"Gemini generation failed: {e}")
            raise

    async def stream_generate(  # type: ignore[override]
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str]:
        """Stream response using Gemini API."""
        if not await self.is_available():
            msg = "Gemini provider not available"
            raise RuntimeError(msg)

        genai = await self._get_client()
        model_name = model or self.default_model

        try:
            model_instance = genai.GenerativeModel(model_name)
            chat_messages = self._convert_messages(messages)

            if len(chat_messages) > 1:
                chat = model_instance.start_chat(history=chat_messages[:-1])
                response = chat.send_message(
                    chat_messages[-1]["parts"][0],
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                    stream=True,
                )
            else:
                response = model_instance.generate_content(
                    chat_messages[0]["parts"][0],
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                    stream=True,
                )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            self.logger.exception(f"Gemini streaming failed: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if Gemini API is available."""
        if not self.api_key:
            return False

        try:
            genai = await self._get_client()
            # Test with a simple model list request
            list(genai.list_models())
            return True
        except Exception:
            return False

    def get_models(self) -> list[str]:
        """Get available Gemini models."""
        return [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.default_model = config.get("default_model", "llama2")
        self._available_models: list[str] = []

    async def _make_api_request(
        self,
        endpoint: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Make API request to Ollama service."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/{endpoint}",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    return await response.json()  # type: ignore[no-any-return]
        except ImportError:
            msg = "aiohttp package not installed. Install with: pip install aiohttp"
            raise ImportError(
                msg,
            )  # type: ignore[no-any-return]

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, str]]:
        """Convert LLMMessage objects to Ollama format."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    async def generate(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Ollama API."""
        if not await self.is_available():
            msg = "Ollama provider not available"
            raise RuntimeError(msg)

        model_name = model or self.default_model

        try:
            data: dict[str, Any] = {
                "model": model_name,
                "messages": self._convert_messages(messages),
                "options": {"temperature": temperature},
            }

            if max_tokens:
                data["options"]["num_predict"] = max_tokens

            response = await self._make_api_request("api/chat", data)

            return LLMResponse(
                content=response.get("message", {}).get("content", ""),
                model=model_name,
                provider="ollama",
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0)
                    + response.get("eval_count", 0),
                },
                finish_reason=response.get("done_reason", "stop"),
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            self.logger.exception(f"Ollama generation failed: {e}")
            raise

    def _prepare_stream_data(
        self,
        model_name: str,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Prepare data payload for streaming request."""
        data: dict[str, Any] = {
            "model": model_name,
            "messages": self._convert_messages(messages),
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            data["options"]["num_predict"] = max_tokens
        return data

    def _extract_chunk_content(self, line: bytes) -> str | None:
        """Extract content from a streaming chunk line."""
        if not line:
            return None

        try:
            chunk_data = json.loads(line.decode("utf-8"))
            if isinstance(chunk_data, dict) and "message" in chunk_data:
                message = chunk_data["message"]
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"])
        except json.JSONDecodeError:
            pass
        return None

    async def _stream_from_response(self, response: Any) -> AsyncGenerator[str]:
        """Process streaming response and yield content chunks."""
        async for line in response.content:
            content = self._extract_chunk_content(line)
            if content:
                yield content

    async def stream_generate(  # type: ignore[override]
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str]:
        """Stream response using Ollama API."""
        if not await self.is_available():
            msg = "Ollama provider not available"
            raise RuntimeError(msg)

        model_name = model or self.default_model

        try:
            import aiohttp

            data = self._prepare_stream_data(
                model_name, messages, temperature, max_tokens
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    async for chunk in self._stream_from_response(response):
                        yield chunk

        except Exception as e:
            self.logger.exception(f"Ollama streaming failed: {e}")
            raise

    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._available_models = [
                            model["name"] for model in data.get("models", [])
                        ]
                        return True
            return False
        except Exception:
            return False

    def get_models(self) -> list[str]:
        """Get available Ollama models."""
        return self._available_models or [
            "llama2",
            "llama2:13b",
            "llama2:70b",
            "codellama",
            "mistral",
            "mixtral",
        ]


class LLMManager:
    """Manager for multiple LLM providers with fallback support."""

    def __init__(self, config_path: str | None = None) -> None:
        self.providers: dict[str, LLMProvider] = {}
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger("llm_providers.manager")
        self._initialize_providers()

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """Load configuration from file or environment."""
        config: dict[str, Any] = {
            "providers": {},
            "default_provider": "openai",
            "fallback_providers": ["gemini", "ollama"],
        }

        if config_path and Path(config_path).exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                with open(config_path) as f:
                    file_config = json.load(f)
                    config.update(file_config)

        # Add environment-based provider configs
        if not config["providers"].get("openai"):
            config["providers"]["openai"] = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "default_model": "gpt-4",
            }

        if not config["providers"].get("gemini"):
            config["providers"]["gemini"] = {
                "api_key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
                "default_model": "gemini-pro",
            }

        if not config["providers"].get("ollama"):
            config["providers"]["ollama"] = {
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "default_model": "llama2",
            }

        return config

    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        provider_classes = {
            "openai": OpenAIProvider,
            "gemini": GeminiProvider,
            "ollama": OllamaProvider,
        }

        for provider_name, provider_config in self.config["providers"].items():
            if provider_name in provider_classes:
                try:
                    self.providers[provider_name] = provider_classes[provider_name](
                        provider_config,
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to initialize {provider_name} provider: {e}",
                    )

    async def get_available_providers(self) -> list[str]:
        """Get list of available providers."""
        return [
            name
            for name, provider in self.providers.items()
            if await provider.is_available()
        ]

    async def generate(
        self,
        messages: list[LLMMessage],
        provider: str | None = None,
        model: str | None = None,
        use_fallback: bool = True,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response with optional fallback."""
        target_provider = provider or self.config["default_provider"]

        # Try primary provider
        result = await self._try_primary_provider_generate(
            target_provider, messages, model, **kwargs
        )
        if result is not None:
            return result

        # Try fallback providers if enabled
        if use_fallback:
            result = await self._try_fallback_providers_generate(
                target_provider, messages, model, **kwargs
            )
            if result is not None:
                return result

        msg = "No available LLM providers"
        raise RuntimeError(msg)

    async def _try_primary_provider_generate(
        self,
        target_provider: str,
        messages: list[LLMMessage],
        model: str | None,
        **kwargs: Any,
    ) -> LLMResponse | None:
        """Try generating with primary provider."""
        if target_provider not in self.providers:
            return None

        try:
            provider_instance = self.providers[target_provider]
            if await provider_instance.is_available():
                return await provider_instance.generate(messages, model, **kwargs)
        except Exception as e:
            self.logger.warning(f"Provider {target_provider} failed: {e}")
        return None

    async def _try_fallback_providers_generate(
        self,
        target_provider: str,
        messages: list[LLMMessage],
        model: str | None,
        **kwargs: Any,
    ) -> LLMResponse | None:
        """Try generating with fallback providers."""
        for fallback_name in self.config.get("fallback_providers", []):
            if fallback_name in self.providers and fallback_name != target_provider:
                try:
                    provider_instance = self.providers[fallback_name]
                    if await provider_instance.is_available():
                        self.logger.info(f"Falling back to {fallback_name}")
                        return await provider_instance.generate(
                            messages, model, **kwargs
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Fallback provider {fallback_name} failed: {e}"
                    )
        return None

    def _get_fallback_providers(self, target_provider: str) -> list[str]:
        """Get list of fallback providers excluding the target provider."""
        return [
            name
            for name in self.config.get("fallback_providers", [])
            if name in self.providers and name != target_provider
        ]

    def _is_valid_provider(self, provider_name: str) -> bool:
        """Check if a provider is valid and available."""
        return provider_name in self.providers

    async def _get_provider_stream(
        self,
        provider_name: str,
        messages: list[LLMMessage],
        model: str | None,
        **kwargs: Any,
    ) -> AsyncGenerator[str]:
        """Get stream from provider (assumes provider is available)."""
        provider_instance = self.providers[provider_name]
        async for chunk in provider_instance.stream_generate(  # type: ignore[attr-defined]
            messages, model, **kwargs
        ):
            yield chunk

    async def _try_provider_streaming(
        self,
        provider_name: str,
        messages: list[LLMMessage],
        model: str | None,
        **kwargs: Any,
    ) -> AsyncGenerator[str]:
        """Try streaming from a provider with error handling."""
        try:
            provider_instance = self.providers[provider_name]
            if await provider_instance.is_available():
                async for chunk in self._get_provider_stream(
                    provider_name, messages, model, **kwargs
                ):
                    yield chunk
        except Exception as e:
            self.logger.warning(f"Provider {provider_name} failed: {e}")

    async def _select_primary_provider(self, options: StreamGenerationOptions) -> str:
        """Select primary provider. Target complexity: ≤3."""
        target_provider = options.provider or self.config["default_provider"]
        if not self._is_valid_provider(target_provider):
            msg = f"Invalid provider: {target_provider}"
            raise RuntimeError(msg)
        return target_provider

    async def _try_streaming_from_provider(
        self,
        provider_name: str,
        messages: list[LLMMessage],
        options: StreamGenerationOptions,
    ) -> AsyncGenerator[StreamChunk]:
        """Try streaming from a specific provider. Target complexity: ≤6."""
        try:
            stream_started = False
            async for chunk_content in self._try_provider_streaming(
                provider_name,
                messages,
                options.model,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
            ):
                stream_started = True
                yield StreamChunk.content_chunk(chunk_content, provider_name)

            if not stream_started:
                yield StreamChunk.error_chunk(f"No response from {provider_name}")

        except Exception as e:
            self.logger.warning(f"Provider {provider_name} failed: {e}")
            yield StreamChunk.error_chunk(str(e))

    async def _stream_from_primary_provider(
        self,
        primary_provider: str,
        messages: list[LLMMessage],
        options: StreamGenerationOptions,
    ) -> AsyncGenerator[str]:
        """Stream from primary provider. Target complexity: ≤4."""
        has_content = False
        async for chunk in self._try_streaming_from_provider(
            primary_provider, messages, options
        ):
            if chunk.is_error:
                if not has_content:  # Log errors only if no content received
                    self.logger.warning(
                        f"Primary provider error: {chunk.metadata.get('error', 'Unknown')}"
                    )
                continue

            has_content = True
            yield chunk.content

        if not has_content:
            self.logger.debug(
                f"Primary provider {primary_provider} produced no content"
            )

    async def _stream_from_fallback_providers(
        self,
        primary_provider: str,
        messages: list[LLMMessage],
        options: StreamGenerationOptions,
    ) -> AsyncGenerator[str]:
        """Stream from fallback providers. Target complexity: ≤5."""
        if not options.use_fallback:
            return

        fallback_providers = self._get_fallback_providers(primary_provider)
        for fallback_name in fallback_providers:
            self.logger.info(f"Falling back to {fallback_name}")
            has_content = False
            async for chunk in self._try_streaming_from_provider(
                fallback_name, messages, options
            ):
                if chunk.is_error:
                    continue
                has_content = True
                yield chunk.content
            if has_content:
                return

    async def stream_generate(  # type: ignore[override]
        self,
        messages: list[LLMMessage],
        provider: str | None = None,
        model: str | None = None,
        use_fallback: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[str]:
        """Stream generate response with optional fallback. Target complexity: ≤8."""
        options = StreamGenerationOptions(
            provider=provider,
            model=model,
            use_fallback=use_fallback,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens"),
        )

        try:
            # Try primary provider first
            primary_provider = await self._select_primary_provider(options)
            async for chunk_content in self._stream_from_primary_provider(
                primary_provider, messages, options
            ):
                yield chunk_content
                return  # Success - exit early

            # Try fallback providers if primary failed
            async for chunk_content in self._stream_from_fallback_providers(
                primary_provider, messages, options
            ):
                yield chunk_content
                return  # Success - exit early

            # All providers failed
            msg = "No available LLM providers"
            raise RuntimeError(msg)

        except Exception as e:
            self.logger.exception(f"Stream generation failed: {e}")
            raise

    def get_provider_info(self) -> dict[str, Any]:
        """Get information about all providers."""
        info: dict[str, Any] = {
            "providers": {},
            "config": {
                "default_provider": self.config["default_provider"],
                "fallback_providers": self.config.get("fallback_providers", []),
            },
        }

        for name, provider in self.providers.items():
            info["providers"][name] = {
                "models": provider.get_models(),
                "config": {
                    k: v for k, v in provider.config.items() if "key" not in k.lower()
                },
            }

        return info

    async def test_providers(self) -> dict[str, Any]:
        """Test all providers and return status."""
        test_message = [
            LLMMessage(role="user", content='Hello, respond with just "OK"'),
        ]
        results = {}

        for name, provider in self.providers.items():
            try:
                available = await provider.is_available()
                if available:
                    # Quick test generation
                    response = await provider.generate(test_message, max_tokens=10)
                    results[name] = {
                        "available": True,
                        "test_successful": True,
                        "response_length": len(response.content),
                        "model": response.model,
                    }
                else:
                    results[name] = {
                        "available": False,
                        "test_successful": False,
                        "error": "Provider not available",
                    }
            except Exception as e:
                results[name] = {
                    "available": False,
                    "test_successful": False,
                    "error": str(e),
                }

        return results
