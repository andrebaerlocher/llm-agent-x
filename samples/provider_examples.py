#!/usr/bin/env python3
"""
Example script demonstrating how to use different LLM providers with LLM Agent X.

This script shows how to:
1. Use different providers (OpenAI, LM Studio, Ollama)
2. Switch providers programmatically
3. Configure providers with custom settings
"""

import os
from llm_agent_x.llm_providers import (
    create_provider,
    OpenAIProvider,
    LMStudioProvider,
    OllamaProvider,
)


def example_openai():
    """Example: Using OpenAI provider."""
    print("\n=== OpenAI Provider Example ===")
    
    # Create OpenAI provider with custom settings
    provider = OpenAIProvider(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
    )
    
    # Get configuration for use with pydantic_ai
    config = provider.get_model_config()
    print(f"Model: {config['model']}")
    print(f"Temperature: {config['temperature']}")
    print(f"Base URL: {provider.get_base_url()}")


def example_lmstudio():
    """Example: Using LM Studio provider."""
    print("\n=== LM Studio Provider Example ===")
    
    # Create LM Studio provider
    provider = LMStudioProvider(
        model="local-llama",
        base_url="http://localhost:1234/v1",
        temperature=0.5,
    )
    
    config = provider.get_model_config()
    print(f"Model: {config['model']}")
    print(f"Temperature: {config['temperature']}")
    print(f"Base URL: {provider.get_base_url()}")
    print("\nNote: Make sure LM Studio is running with a model loaded!")


def example_ollama():
    """Example: Using Ollama provider."""
    print("\n=== Ollama Provider Example ===")
    
    # Create Ollama provider
    provider = OllamaProvider(
        model="llama2",
        base_url="http://localhost:11434",
        temperature=0.6,
    )
    
    config = provider.get_model_config()
    print(f"Model: {config['model']}")
    print(f"Temperature: {config['temperature']}")
    print(f"Base URL: {provider.get_base_url()}")
    print("\nNote: Make sure Ollama is installed and the model is pulled!")


def example_factory():
    """Example: Using the factory with environment variables."""
    print("\n=== Factory Example (from environment) ===")
    
    # Set environment variables
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["OLLAMA_MODEL"] = "mistral"
    
    # Create provider using factory
    provider = create_provider()
    
    config = provider.get_model_config()
    print(f"Provider type: {type(provider).__name__}")
    print(f"Model: {config['model']}")
    print(f"Temperature: {config['temperature']}")


def example_dag_agent():
    """Example: Using providers with DAGAgent."""
    print("\n=== DAGAgent Integration Example ===")
    print("""
To use a specific provider with DAGAgent:

# Option 1: Set environment variables before running
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama2
python -m llm_agent_x.cli dag "your task"

# Option 2: Pass model directly with CLI
llm-agent-x dag "your task" --model "ollama:llama2"

# Option 3: In Python code
from llm_agent_x.agents.dag_agent import DAGAgent

# Uses default provider from environment
agent = DAGAgent()

# Or specify explicitly
agent = DAGAgent(llm_model="ollama:llama2")
    """)


def main():
    """Run all examples."""
    print("=" * 60)
    print("LLM Agent X - Provider Examples")
    print("=" * 60)
    
    example_openai()
    example_lmstudio()
    example_ollama()
    example_factory()
    example_dag_agent()
    
    print("\n" + "=" * 60)
    print("For more information, see docs/llm_providers.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
