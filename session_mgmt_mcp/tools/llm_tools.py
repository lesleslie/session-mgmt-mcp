#!/usr/bin/env python3
"""LLM provider management MCP tools.

This module provides tools for managing and interacting with LLM providers
following crackerjack architecture patterns.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy loading for optional LLM dependencies
_llm_manager = None
_llm_available = None


async def _get_llm_manager() -> Any:
    """Get LLM manager instance with lazy loading."""
    global _llm_manager, _llm_available

    if _llm_available is False:
        return None

    if _llm_manager is None:
        try:
            from session_mgmt_mcp.llm_providers import LLMManager

            _llm_manager = LLMManager()
            _llm_available = True
        except ImportError as e:
            logger.warning(f"LLM providers not available: {e}")
            _llm_available = False
            return None

    return _llm_manager


def _check_llm_available() -> bool:
    """Check if LLM providers are available."""
    global _llm_available

    if _llm_available is None:
        try:
            import importlib.util

            spec = importlib.util.find_spec("session_mgmt_mcp.llm_providers")
            _llm_available = spec is not None
        except ImportError:
            _llm_available = False

    return _llm_available


async def _list_llm_providers_impl() -> str:
    """List all available LLM providers and their models."""
    if not _check_llm_available():
        return "❌ LLM providers not available. Install dependencies: pip install openai google-generativeai aiohttp"

    try:
        manager = await _get_llm_manager()
        if not manager:
            return "❌ Failed to initialize LLM manager"

        available_providers = await manager.get_available_providers()
        provider_info = manager.get_provider_info()

        output = ["🤖 Available LLM Providers", ""]

        for provider_name, info in provider_info["providers"].items():
            status = "✅" if provider_name in available_providers else "❌"
            output.append(f"{status} {provider_name.title()}")

            if provider_name in available_providers:
                models = info["models"][:5]  # Show first 5 models
                for model in models:
                    output.append(f"   • {model}")
                if len(info["models"]) > 5:
                    output.append(f"   • ... and {len(info['models']) - 5} more")
            output.append("")

        config = provider_info["config"]
        output.extend(
            [
                f"🎯 Default Provider: {config['default_provider']}",
                f"🔄 Fallback Providers: {', '.join(config['fallback_providers'])}",
            ]
        )

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error listing LLM providers: {e}")
        return f"❌ Error listing providers: {e}"


async def _test_llm_providers_impl() -> str:
    """Test all LLM providers to check their availability and functionality."""
    if not _check_llm_available():
        return "❌ LLM providers not available. Install dependencies: pip install openai google-generativeai aiohttp"

    try:
        manager = await _get_llm_manager()
        if not manager:
            return "❌ Failed to initialize LLM manager"

        test_results = await manager.test_all_providers()

        output = ["🧪 LLM Provider Test Results", ""]

        for provider, result in test_results.items():
            status = "✅" if result["success"] else "❌"
            output.append(f"{status} {provider.title()}")

            if result["success"]:
                output.append(
                    f"   ⚡ Response time: {result['response_time_ms']:.0f}ms"
                )
                output.append(f"   🎯 Model: {result['model']}")
            else:
                output.append(f"   ❌ Error: {result['error']}")
            output.append("")

        working_count = sum(1 for r in test_results.values() if r["success"])
        total_count = len(test_results)
        output.append(f"📊 Summary: {working_count}/{total_count} providers working")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error testing LLM providers: {e}")
        return f"❌ Error testing providers: {e}"


async def _generate_with_llm_impl(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    use_fallback: bool = True,
) -> str:
    """Generate text using specified LLM provider.

    Args:
        prompt: The text prompt to generate from
        provider: LLM provider to use (openai, gemini, ollama)
        model: Specific model to use
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        use_fallback: Whether to use fallback providers if primary fails

    """
    if not _check_llm_available():
        return "❌ LLM providers not available. Install dependencies: pip install openai google-generativeai aiohttp"

    try:
        manager = await _get_llm_manager()
        if not manager:
            return "❌ Failed to initialize LLM manager"

        result = await manager.generate_text(
            prompt=prompt,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_fallback=use_fallback,
        )

        if result["success"]:
            output = ["✨ LLM Generation Result", ""]
            output.append(f"🤖 Provider: {result['metadata']['provider']}")
            output.append(f"🎯 Model: {result['metadata']['model']}")
            output.append(
                f"⚡ Response time: {result['metadata']['response_time_ms']:.0f}ms"
            )
            output.append(f"📊 Tokens: {result['metadata'].get('tokens_used', 'N/A')}")
            output.append("")
            output.append("💬 Generated text:")
            output.append("─" * 40)
            output.append(result["text"])

            return "\n".join(output)
        return f"❌ Generation failed: {result['error']}"

    except Exception as e:
        logger.exception(f"Error generating with LLM: {e}")
        return f"❌ Error generating text: {e}"


async def _chat_with_llm_impl(
    messages: list[dict[str, str]],
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Have a conversation with an LLM provider.

    Args:
        messages: List of messages in format [{"role": "user/assistant/system", "content": "text"}]
        provider: LLM provider to use (openai, gemini, ollama)
        model: Specific model to use
        temperature: Generation temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate

    """
    if not _check_llm_available():
        return "❌ LLM providers not available. Install dependencies: pip install openai google-generativeai aiohttp"

    try:
        manager = await _get_llm_manager()
        if not manager:
            return "❌ Failed to initialize LLM manager"

        result = await manager.chat(
            messages=messages,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if result["success"]:
            output = ["💬 LLM Chat Result", ""]
            output.append(f"🤖 Provider: {result['metadata']['provider']}")
            output.append(f"🎯 Model: {result['metadata']['model']}")
            output.append(
                f"⚡ Response time: {result['metadata']['response_time_ms']:.0f}ms"
            )
            output.append(f"📊 Messages: {len(messages)} → 1")
            output.append("")
            output.append("🎭 Assistant response:")
            output.append("─" * 40)
            output.append(result["response"])

            return "\n".join(output)
        return f"❌ Chat failed: {result['error']}"

    except Exception as e:
        logger.exception(f"Error chatting with LLM: {e}")
        return f"❌ Error in chat: {e}"


def _format_provider_config_output(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    default_model: str | None = None,
) -> list[str]:
    """Format the provider configuration output."""
    output = ["⚙️ Provider Configuration Updated", ""]
    output.append(f"🤖 Provider: {provider}")

    if api_key:
        # Don't show the full API key for security
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        output.append(f"🔑 API Key: {masked_key}")

    if base_url:
        output.append(f"🌐 Base URL: {base_url}")

    if default_model:
        output.append(f"🎯 Default Model: {default_model}")

    output.append("")
    output.append("✅ Configuration saved successfully!")
    output.append("💡 Use `test_llm_providers` to verify the configuration")

    return output


async def _build_provider_config_data(
    api_key: str | None = None,
    base_url: str | None = None,
    default_model: str | None = None,
) -> dict[str, Any]:
    """Build provider configuration data."""
    config_data = {}
    if api_key:
        config_data["api_key"] = api_key
    if base_url:
        config_data["base_url"] = base_url
    if default_model:
        config_data["default_model"] = default_model
    return config_data


async def _configure_llm_provider_impl(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    default_model: str | None = None,
) -> str:
    """Configure an LLM provider with API credentials and settings.

    Args:
        provider: Provider name (openai, gemini, ollama)
        api_key: API key for the provider
        base_url: Base URL for the provider API
        default_model: Default model to use

    """
    if not _check_llm_available():
        return "❌ LLM providers not available. Install dependencies: pip install openai google-generativeai aiohttp"

    try:
        manager = await _get_llm_manager()
        if not manager:
            return "❌ Failed to initialize LLM manager"

        config_data = await _build_provider_config_data(
            api_key, base_url, default_model
        )
        result = await manager.configure_provider(provider, config_data)

        if result["success"]:
            output = _format_provider_config_output(
                provider, api_key, base_url, default_model
            )
            return "\n".join(output)
        return f"❌ Configuration failed: {result['error']}"

    except Exception as e:
        logger.exception(f"Error configuring LLM provider: {e}")
        return f"❌ Error configuring provider: {e}"


def register_llm_tools(mcp) -> None:
    """Register all LLM provider management MCP tools.

    Args:
        mcp: FastMCP server instance

    """

    @mcp.tool()
    async def list_llm_providers() -> str:
        """List all available LLM providers and their models."""
        return await _list_llm_providers_impl()

    @mcp.tool()
    async def test_llm_providers() -> str:
        """Test all LLM providers to check their availability and functionality."""
        return await _test_llm_providers_impl()

    @mcp.tool()
    async def generate_with_llm(
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        use_fallback: bool = True,
    ) -> str:
        """Generate text using specified LLM provider.

        Args:
            prompt: The text prompt to generate from
            provider: LLM provider to use (openai, gemini, ollama)
            model: Specific model to use
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            use_fallback: Whether to use fallback providers if primary fails

        """
        return await _generate_with_llm_impl(
            prompt, provider, model, temperature, max_tokens, use_fallback
        )

    @mcp.tool()
    async def chat_with_llm(
        messages: list[dict[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Have a conversation with an LLM provider.

        Args:
            messages: List of messages in format [{"role": "user/assistant/system", "content": "text"}]
            provider: LLM provider to use (openai, gemini, ollama)
            model: Specific model to use
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        """
        return await _chat_with_llm_impl(
            messages, provider, model, temperature, max_tokens
        )

    @mcp.tool()
    async def configure_llm_provider(
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ) -> str:
        """Configure an LLM provider with API credentials and settings.

        Args:
            provider: Provider name (openai, gemini, ollama)
            api_key: API key for the provider
            base_url: Base URL for the provider API
            default_model: Default model to use

        """
        return await _configure_llm_provider_impl(
            provider, api_key, base_url, default_model
        )
