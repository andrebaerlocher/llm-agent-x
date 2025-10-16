"""
Factory for creating LLM provider instances based on configuration.
"""

from os import getenv
from llm_agent_x.llm_providers.base import LLMProvider
from llm_agent_x.llm_providers.openai_provider import OpenAIProvider
from llm_agent_x.llm_providers.lmstudio_provider import LMStudioProvider
from llm_agent_x.llm_providers.ollama_provider import OllamaProvider


def create_provider(
    provider_type: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.5,
) -> LLMProvider:
    """
    Create an LLM provider instance based on configuration.
    
    Args:
        provider_type: Type of provider ("openai", "lmstudio", "ollama")
                      If None, uses LLM_PROVIDER env var (defaults to "openai")
        model: Model name/identifier
              If None, uses DEFAULT_LLM env var (defaults vary by provider)
        base_url: Base URL for the API
                 If None, uses provider-specific env var or default
        api_key: API key for authentication
                If None, uses provider-specific env var
        temperature: Temperature parameter for generation
    
    Returns:
        LLMProvider: An instance of the appropriate provider
        
    Raises:
        ValueError: If provider_type is not supported
    """
    # Get provider type from env if not specified
    if provider_type is None:
        provider_type = getenv("LLM_PROVIDER", "openai").lower()
    
    provider_type = provider_type.lower()
    
    # Get model name from env if not specified
    if model is None:
        model = getenv("DEFAULT_LLM")
    
    # Create the appropriate provider
    if provider_type == "openai":
        return OpenAIProvider(
            model=model or "gpt-4o-mini",
            api_key=api_key or getenv("OPENAI_API_KEY"),
            base_url=base_url or getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            temperature=temperature,
        )
    elif provider_type == "lmstudio":
        return LMStudioProvider(
            model=model or getenv("LMSTUDIO_MODEL", "local-model"),
            base_url=base_url or getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            api_key=api_key or getenv("LMSTUDIO_API_KEY", "lm-studio"),
            temperature=temperature,
        )
    elif provider_type == "ollama":
        return OllamaProvider(
            model=model or getenv("OLLAMA_MODEL", "llama2"),
            base_url=base_url or getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unsupported provider type: {provider_type}. "
            f"Supported types: openai, lmstudio, ollama"
        )
