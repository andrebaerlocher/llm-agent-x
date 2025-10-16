"""
OpenAI LLM Provider implementation.
"""

from typing import Any
from llm_agent_x.llm_providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    Provider for OpenAI's language models.
    
    This provider supports standard OpenAI models via the OpenAI API.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.5,
    ):
        """
        Initialize the OpenAI provider.
        
        Args:
            model: The model name (e.g., "gpt-4o-mini", "gpt-4")
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            base_url: Base URL for OpenAI API
            temperature: Temperature parameter for generation
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model

    def get_model_config(self) -> dict[str, Any]:
        """
        Get the configuration for pydantic_ai.Agent.
        
        Returns a model string in the format "openai:model_name" for pydantic_ai.
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
