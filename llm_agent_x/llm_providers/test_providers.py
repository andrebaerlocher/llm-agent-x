"""
Unit tests for LLM provider implementations.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add parent directory to sys.path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import using relative path from parent
import llm_providers.base as base_module
import llm_providers.openai_provider as openai_module
import llm_providers.lmstudio_provider as lmstudio_module
import llm_providers.ollama_provider as ollama_module
import llm_providers.factory as factory_module

LLMProvider = base_module.LLMProvider
OpenAIProvider = openai_module.OpenAIProvider
LMStudioProvider = lmstudio_module.LMStudioProvider
OllamaProvider = ollama_module.OllamaProvider
create_provider = factory_module.create_provider


class TestOpenAIProvider(unittest.TestCase):
    """Test OpenAI provider implementation."""

    def test_initialization(self):
        """Test OpenAI provider initialization with default values."""
        provider = OpenAIProvider()
        self.assertEqual(provider.get_model_name(), "gpt-4o-mini")
        self.assertEqual(provider.get_base_url(), "https://api.openai.com/v1")
        self.assertIsNone(provider.get_api_key())

    def test_custom_initialization(self):
        """Test OpenAI provider initialization with custom values."""
        provider = OpenAIProvider(
            model="gpt-4",
            api_key="test-key",
            base_url="https://custom.openai.com/v1",
            temperature=0.7,
        )
        self.assertEqual(provider.get_model_name(), "gpt-4")
        self.assertEqual(provider.get_api_key(), "test-key")
        self.assertEqual(provider.get_base_url(), "https://custom.openai.com/v1")

    def test_model_config(self):
        """Test OpenAI provider model configuration."""
        provider = OpenAIProvider(model="gpt-4", temperature=0.8)
        config = provider.get_model_config()
        self.assertEqual(config["model"], "openai:gpt-4")
        self.assertEqual(config["temperature"], 0.8)


class TestLMStudioProvider(unittest.TestCase):
    """Test LM Studio provider implementation."""

    def test_initialization(self):
        """Test LM Studio provider initialization with default values."""
        provider = LMStudioProvider()
        self.assertEqual(provider.get_model_name(), "local-model")
        self.assertEqual(provider.get_base_url(), "http://localhost:1234/v1")
        self.assertEqual(provider.get_api_key(), "lm-studio")

    def test_custom_initialization(self):
        """Test LM Studio provider initialization with custom values."""
        provider = LMStudioProvider(
            model="my-local-model",
            base_url="http://192.168.1.100:1234/v1",
            api_key="custom-key",
            temperature=0.3,
        )
        self.assertEqual(provider.get_model_name(), "my-local-model")
        self.assertEqual(provider.get_api_key(), "custom-key")
        self.assertEqual(provider.get_base_url(), "http://192.168.1.100:1234/v1")

    def test_model_config(self):
        """Test LM Studio provider model configuration."""
        provider = LMStudioProvider(model="llama-2-7b", temperature=0.6)
        config = provider.get_model_config()
        # LM Studio uses OpenAI-compatible API, so it uses the openai: prefix
        self.assertEqual(config["model"], "openai:llama-2-7b")
        self.assertEqual(config["temperature"], 0.6)


class TestOllamaProvider(unittest.TestCase):
    """Test Ollama provider implementation."""

    def test_initialization(self):
        """Test Ollama provider initialization with default values."""
        provider = OllamaProvider()
        self.assertEqual(provider.get_model_name(), "llama2")
        self.assertEqual(provider.get_base_url(), "http://localhost:11434")
        self.assertIsNone(provider.get_api_key())

    def test_custom_initialization(self):
        """Test Ollama provider initialization with custom values."""
        provider = OllamaProvider(
            model="mistral",
            base_url="http://192.168.1.100:11434",
            temperature=0.4,
        )
        self.assertEqual(provider.get_model_name(), "mistral")
        self.assertEqual(provider.get_base_url(), "http://192.168.1.100:11434")
        self.assertIsNone(provider.get_api_key())

    def test_model_config(self):
        """Test Ollama provider model configuration."""
        provider = OllamaProvider(model="codellama", temperature=0.2)
        config = provider.get_model_config()
        self.assertEqual(config["model"], "ollama:codellama")
        self.assertEqual(config["temperature"], 0.2)


class TestProviderFactory(unittest.TestCase):
    """Test provider factory functionality."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"})
    def test_create_openai_provider_from_env(self):
        """Test creating OpenAI provider from environment variables."""
        provider = create_provider()
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.get_api_key(), "test-key")

    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "lmstudio",
            "LMSTUDIO_BASE_URL": "http://localhost:1234/v1",
            "LMSTUDIO_MODEL": "my-model",
        },
    )
    def test_create_lmstudio_provider_from_env(self):
        """Test creating LM Studio provider from environment variables."""
        provider = create_provider()
        self.assertIsInstance(provider, LMStudioProvider)
        self.assertEqual(provider.get_base_url(), "http://localhost:1234/v1")
        self.assertEqual(provider.get_model_name(), "my-model")

    @patch.dict(
        os.environ,
        {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "OLLAMA_MODEL": "llama2",
        },
    )
    def test_create_ollama_provider_from_env(self):
        """Test creating Ollama provider from environment variables."""
        provider = create_provider()
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.get_base_url(), "http://localhost:11434")
        self.assertEqual(provider.get_model_name(), "llama2")

    def test_create_provider_with_explicit_type(self):
        """Test creating provider with explicit type parameter."""
        provider = create_provider(provider_type="lmstudio", model="test-model")
        self.assertIsInstance(provider, LMStudioProvider)
        self.assertEqual(provider.get_model_name(), "test-model")

    def test_create_provider_unsupported_type(self):
        """Test creating provider with unsupported type raises ValueError."""
        with self.assertRaises(ValueError) as context:
            create_provider(provider_type="unsupported")
        self.assertIn("Unsupported provider type", str(context.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_create_provider_default_fallback(self):
        """Test that factory defaults to OpenAI when no env vars are set."""
        provider = create_provider()
        self.assertIsInstance(provider, OpenAIProvider)

    def test_create_provider_with_custom_temperature(self):
        """Test creating provider with custom temperature."""
        provider = create_provider(provider_type="ollama", temperature=0.9)
        config = provider.get_model_config()
        self.assertEqual(config["temperature"], 0.9)


class TestProviderInterface(unittest.TestCase):
    """Test that all providers implement the LLMProvider interface."""

    def test_openai_implements_interface(self):
        """Test that OpenAI provider implements LLMProvider interface."""
        provider = OpenAIProvider()
        self.assertIsInstance(provider, LLMProvider)
        # Test all required methods are implemented
        self.assertTrue(hasattr(provider, "get_model_name"))
        self.assertTrue(hasattr(provider, "get_model_config"))
        self.assertTrue(hasattr(provider, "get_base_url"))
        self.assertTrue(hasattr(provider, "get_api_key"))

    def test_lmstudio_implements_interface(self):
        """Test that LM Studio provider implements LLMProvider interface."""
        provider = LMStudioProvider()
        self.assertIsInstance(provider, LLMProvider)
        self.assertTrue(hasattr(provider, "get_model_name"))
        self.assertTrue(hasattr(provider, "get_model_config"))
        self.assertTrue(hasattr(provider, "get_base_url"))
        self.assertTrue(hasattr(provider, "get_api_key"))

    def test_ollama_implements_interface(self):
        """Test that Ollama provider implements LLMProvider interface."""
        provider = OllamaProvider()
        self.assertIsInstance(provider, LLMProvider)
        self.assertTrue(hasattr(provider, "get_model_name"))
        self.assertTrue(hasattr(provider, "get_model_config"))
        self.assertTrue(hasattr(provider, "get_base_url"))
        self.assertTrue(hasattr(provider, "get_api_key"))


if __name__ == "__main__":
    unittest.main()
