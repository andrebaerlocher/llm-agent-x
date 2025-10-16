"""
LLM Manager module for managing language model instances across the application.

This module provides a centralized way to create and configure LLM providers
for use with pydantic_ai Agents throughout the application.
"""

from os import getenv
from llm_agent_x.llm_providers.factory import create_provider
from llm_agent_x.llm_providers.base import LLMProvider


# Create the default provider based on environment configuration
default_provider = create_provider()

# Create a tiny/small model provider for less demanding tasks
tiny_model = getenv("DEFAULT_TINY_LLM", "gpt-4o-mini")
tiny_provider = create_provider(model=tiny_model)


def get_default_model_name() -> str:
    """
    Get the default model name for use with pydantic_ai.Agent.
    
    Returns:
        str: The model name in pydantic_ai format (e.g., "openai:gpt-4o-mini")
    """
    config = default_provider.get_model_config()
    return config["model"]


def get_tiny_model_name() -> str:
    """
    Get the tiny/small model name for less demanding tasks.
    
    Returns:
        str: The model name in pydantic_ai format
    """
    config = tiny_provider.get_model_config()
    return config["model"]


def get_provider() -> LLMProvider:
    """
    Get the default LLM provider instance.
    
    Returns:
        LLMProvider: The configured default provider
    """
    return default_provider


def get_tiny_provider() -> LLMProvider:
    """
    Get the tiny/small LLM provider instance.
    
    Returns:
        LLMProvider: The configured tiny provider
    """
    return tiny_provider
