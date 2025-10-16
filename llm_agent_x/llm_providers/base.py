"""
Base interface for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This interface allows the system to work with different LLM backends
    (OpenAI, LM Studio, Ollama, etc.) in a consistent way.
    """

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the model name/identifier for this provider.
        
        Returns:
            str: The model name or identifier
        """
        pass

    @abstractmethod
    def get_model_config(self) -> dict[str, Any]:
        """
        Get the configuration dictionary for pydantic_ai.Agent.
        
        This should return a dictionary that can be used to instantiate
        a pydantic_ai.Agent with the appropriate model parameter.
        
        Returns:
            dict: Configuration including 'model' key and any provider-specific settings
        """
        pass

    @abstractmethod
    def get_base_url(self) -> str | None:
        """
        Get the base URL for the API endpoint, if applicable.
        
        Returns:
            str | None: The base URL or None if using default
        """
        pass

    @abstractmethod
    def get_api_key(self) -> str | None:
        """
        Get the API key for authentication, if applicable.
        
        Returns:
            str | None: The API key or None if not required
        """
        pass
