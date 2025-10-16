"""
Ollama LLM Provider implementation.
"""

from typing import Any
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """
    Provider for Ollama's locally hosted models.
    
    Ollama provides a local API for running open-source models.
    Default endpoint is typically http://localhost:11434
    """

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.5,
    ):
        """
        Initialize the Ollama provider.
        
        Args:
            model: The model name (e.g., "llama2", "mistral", "codellama")
            base_url: Base URL for Ollama API (default: http://localhost:11434)
            temperature: Temperature parameter for generation
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model

    def get_model_config(self) -> dict[str, Any]:
        """
        Get the configuration for pydantic_ai.Agent.
        
        Ollama has native support in pydantic_ai via the "ollama:" prefix.
        """
        return {
            "model": f"ollama:{self.model}",
            "temperature": self.temperature,
        }

    def get_base_url(self) -> str | None:
        """Get the base URL."""
        return self.base_url

    def get_api_key(self) -> str | None:
        """Ollama doesn't require an API key."""
        return None
