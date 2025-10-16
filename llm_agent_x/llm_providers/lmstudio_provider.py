"""
LM Studio LLM Provider implementation.
"""

from typing import Any
from .base import LLMProvider


class LMStudioProvider(LLMProvider):
    """
    Provider for LM Studio's locally hosted models.
    
    LM Studio provides an OpenAI-compatible API for running models locally.
    Default endpoint is typically http://localhost:1234/v1
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        temperature: float = 0.5,
    ):
        """
        Initialize the LM Studio provider.
        
        Args:
            model: The model identifier (can be any string, LM Studio uses whatever is loaded)
            base_url: Base URL for LM Studio API (default: http://localhost:1234/v1)
            api_key: API key (LM Studio doesn't require real auth, but needs some value)
            temperature: Temperature parameter for generation
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model

    def get_model_config(self) -> dict[str, Any]:
        """
        Get the configuration for pydantic_ai.Agent.
        
        LM Studio uses OpenAI-compatible API, so we use the openai: prefix.
        """
        return {
            "model": f"openai:{self.model}",
            "temperature": self.temperature,
        }

    def get_base_url(self) -> str | None:
        """Get the base URL."""
        return self.base_url

    def get_api_key(self) -> str | None:
        """Get the API key."""
        return self.api_key
