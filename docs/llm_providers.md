# LLM Provider Configuration Guide

LLM Agent X now supports multiple LLM providers, allowing you to use different language model backends including OpenAI, LM Studio, and Ollama.

## Supported Providers

### OpenAI (Default)
The OpenAI provider connects to OpenAI's API for models like GPT-4, GPT-4o-mini, etc.

**Configuration:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional, defaults to OpenAI's API
DEFAULT_LLM=gpt-4o-mini  # Model to use
```

### LM Studio
LM Studio allows you to run models locally with an OpenAI-compatible API.

**Configuration:**
```env
LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1  # Default LM Studio endpoint
LMSTUDIO_MODEL=local-model  # Can be any identifier
LMSTUDIO_API_KEY=lm-studio  # LM Studio doesn't require real auth
```

**Setup Steps:**
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load a model in LM Studio
3. Start the local server (usually on port 1234)
4. Configure your `.env` file as shown above
5. Run LLM Agent X

### Ollama
Ollama is a lightweight framework for running open-source LLMs locally.

**Configuration:**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434  # Default Ollama endpoint
OLLAMA_MODEL=llama2  # e.g., llama2, mistral, codellama
```

**Setup Steps:**
1. Install [Ollama](https://ollama.ai/)
2. Pull a model: `ollama pull llama2`
3. Configure your `.env` file as shown above
4. Run LLM Agent X (Ollama serves models automatically)

## Configuration File

Update your `.env` file in the project root:

```env
# LLM Provider Configuration
# Supported providers: openai, lmstudio, ollama
LLM_PROVIDER=openai

# OpenAI Configuration (when LLM_PROVIDER=openai)
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
DEFAULT_LLM=gpt-4o-mini

# LM Studio Configuration (when LLM_PROVIDER=lmstudio)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model
LMSTUDIO_API_KEY=lm-studio

# Ollama Configuration (when LLM_PROVIDER=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Optional: Smaller model for less demanding tasks
DEFAULT_TINY_LLM=gpt-4o-mini
```

## Advanced Usage

### Programmatic Provider Selection

You can also configure providers programmatically in your Python code:

```python
from llm_agent_x.llm_providers import create_provider, OpenAIProvider, OllamaProvider

# Create a specific provider
openai_provider = OpenAIProvider(
    model="gpt-4",
    api_key="your-key",
    temperature=0.7
)

# Or use the factory with custom settings
ollama_provider = create_provider(
    provider_type="ollama",
    model="mistral",
    base_url="http://localhost:11434",
    temperature=0.5
)

# Get model name for use with pydantic_ai
model_name = openai_provider.get_model_config()["model"]
# Returns: "openai:gpt-4"
```

### Using Providers with DAGAgent

The DAGAgent automatically uses the configured provider:

```python
from llm_agent_x.agents.dag_agent import DAGAgent

# Uses default provider from environment
agent = DAGAgent()

# Or specify a model explicitly (with provider prefix)
agent = DAGAgent(llm_model="ollama:llama2")
```

## Model Name Format

When specifying models directly, use the provider prefix:
- OpenAI: `openai:gpt-4o-mini`
- LM Studio: `openai:local-model` (uses OpenAI-compatible API)
- Ollama: `ollama:llama2`

## Troubleshooting

### Connection Errors

**LM Studio:**
- Ensure LM Studio is running and the server is started
- Check the port in your configuration matches LM Studio's server port
- Verify a model is loaded in LM Studio

**Ollama:**
- Ensure Ollama is installed: `ollama --version`
- Pull the model first: `ollama pull llama2`
- Check if Ollama is running: `curl http://localhost:11434/api/tags`

### API Key Issues

**OpenAI:**
- Verify your API key is valid
- Check your OpenAI account has credits

**LM Studio/Ollama:**
- These don't require real API keys, any value works for LM Studio
- Ollama doesn't use API keys at all

### Performance Considerations

- **OpenAI**: Fast but requires internet and incurs costs
- **LM Studio**: Runs locally, speed depends on your hardware (GPU recommended)
- **Ollama**: Lightweight and optimized for local execution

## Example Configurations

### Development with Local Models
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=mistral
```

### Production with OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
DEFAULT_LLM=gpt-4o-mini
```

### Custom OpenAI-Compatible Service
```env
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://your-custom-endpoint.com/v1
OPENAI_API_KEY=your-key
DEFAULT_LLM=custom-model
```
