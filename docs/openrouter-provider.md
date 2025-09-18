# OpenRouter Provider Guide

OpenRouter is a unified API that provides access to 200+ AI models from multiple providers including OpenAI, Anthropic, Google, Meta, Microsoft, and more. The chimeric OpenRouter provider gives you seamless access to this vast ecosystem of models through a single, consistent interface.

## 🌟 Key Features

- **200+ Models**: Access models from OpenAI, Anthropic, Google, Meta, Microsoft, Perplexity, and more
- **Cost Optimization**: Often cheaper than direct provider APIs with transparent pricing
- **Unified Interface**: Same API for all models, no need to learn different SDKs
- **Model Fallbacks**: Automatic failover if a model is unavailable
- **Free Models**: Access to free models for testing and development
- **Higher Rate Limits**: Better rate limits than individual provider APIs
- **Streaming Support**: Real-time response streaming for all compatible models
- **Tool Calling**: Function calling support for models that support it
- **Async Support**: Full async/await support for high-performance applications

## 📦 Installation

```bash
# Install chimeric with OpenAI support (OpenRouter is OpenAI-compatible)
pip install "chimeric[openai]"

# Or with uv (recommended)
uv add "chimeric[openai]"

# For development with all extras
uv add "chimeric[all]"
```

## 🔐 Authentication

### Get Your API Key

1. Visit [OpenRouter.ai](https://openrouter.ai)
2. Sign up for a free account
3. Navigate to the API Keys section
4. Generate a new API key

### Set Up Environment Variable

```bash
# Set environment variable (recommended)
export OPENROUTER_API_KEY="your-api-key-here"

# Or add to your .env file
echo "OPENROUTER_API_KEY=your-api-key-here" >> .env
```

## 🚀 Quick Start

### Basic Usage

```python
from chimeric import Chimeric

# Initialize with environment variable
client = Chimeric()

# Or pass API key directly
client = Chimeric(openrouter_api_key="your-api-key")

# Simple text generation
response = client.generate(
    model="openai/gpt-4o-mini",
    messages="Hello! Explain quantum computing in simple terms."
)

print(response.content)
```

### List Available Models

```python
# Get all available models
models = client.list_models("openrouter")
print(f"Total models available: {len(models)}")

# Show first few models
for model in models[:5]:
    print(f"- {model.id}: {model.name}")
```

## 🎯 Model Selection

OpenRouter provides access to models from many providers. Here are some popular choices:

### OpenAI Models
```python
# GPT-4o models (latest and most capable)
response = client.generate(
    model="openai/gpt-4o",
    messages="Write a Python function to calculate fibonacci numbers"
)

# GPT-4o-mini (faster, cheaper)
response = client.generate(
    model="openai/gpt-4o-mini", 
    messages="Summarize the benefits of renewable energy"
)
```

### Anthropic Models
```python
# Claude 3.5 Sonnet (excellent for reasoning and coding)
response = client.generate(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages="Debug this Python code and explain the issues"
)

# Claude 3.5 Haiku (fast and efficient)
response = client.generate(
    model="anthropic/claude-3-5-haiku-20241022",
    messages="Translate this text to Spanish"
)
```

### Google Models
```python
# Gemini 2.0 Flash (Google's latest)
response = client.generate(
    model="google/gemini-2.0-flash-exp",
    messages="Analyze this data and provide insights"
)
```

### Free Models
```python
# Meta Llama 3.1 8B (free tier)
response = client.generate(
    model="meta-llama/llama-3.1-8b-instruct:free",
    messages="Write a short story about a robot"
)

# Other free models
free_models = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "openchat/openchat-7b:free"
]
```

## 🌊 Streaming Responses

Get real-time responses as they're generated:

```python
# Sync streaming
print("Response: ", end="")
for chunk in client.generate(
    model="openai/gpt-4o-mini",
    messages="Tell me a story about space exploration",
    stream=True
):
    print(chunk.content, end="", flush=True)
print()  # New line at end
```

### Async Streaming

```python
import asyncio

async def stream_response():
    client = Chimeric()
    
    print("Async Response: ", end="")
    async for chunk in client.agenerate(
        model="anthropic/claude-3-5-haiku-20241022",
        messages="Explain machine learning concepts",
        stream=True
    ):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(stream_response())
```

## 🛠️ Tool Calling (Function Calling)

Enable models to call functions and interact with external systems:

### Basic Tool Usage

```python
from chimeric import Chimeric

client = Chimeric()

# Define tools using decorators
@client.tool()
def get_weather(city: str, units: str = "fahrenheit") -> str:
    """Get current weather information for a city.
    
    Args:
        city: Name of the city
        units: Temperature units (fahrenheit or celsius)
    """
    # In a real implementation, call a weather API
    return f"Weather in {city}: 72°{units[0].upper()}, partly cloudy"

@client.tool()
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> dict:
    """Calculate tip and total bill amount.
    
    Args:
        bill_amount: Original bill amount in dollars
        tip_percentage: Tip percentage (default 15%)
    """
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    return {
        "bill_amount": bill_amount,
        "tip_percentage": tip_percentage,
        "tip_amount": round(tip, 2),
        "total_amount": round(total, 2)
    }

# Use tools in conversation
response = client.generate(
    model="openai/gpt-4o-mini",  # Supports tool calling
    messages="What's the weather in New York and calculate a 20% tip on a $85 bill?"
)

print(response.content)
```

### Async Tools

```python
import asyncio
import aiohttp

@client.tool()
async def fetch_news(topic: str, count: int = 3) -> str:
    """Fetch latest news articles about a topic.
    
    Args:
        topic: News topic to search for
        count: Number of articles to return
    """
    # Simulate async API call
    await asyncio.sleep(0.1)
    return f"Found {count} articles about {topic}: [Article 1], [Article 2], [Article 3]"

async def main():
    response = await client.agenerate(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages="Get me the latest news about artificial intelligence"
    )
    print(response.content)

asyncio.run(main())
```

## ⚙️ Configuration Options

### Custom Headers (Recommended)

OpenRouter supports custom headers for better analytics and potentially better rates:

```python
client = Chimeric(
    openrouter_api_key="your-api-key",
    default_headers={
        "HTTP-Referer": "https://your-website.com",  # Your website URL
        "X-Title": "Your Application Name"            # Your app name
    }
)
```

### Model Parameters

Control model behavior with various parameters:

```python
response = client.generate(
    model="openai/gpt-4o-mini",
    messages="Write a formal business email",
    
    # Control randomness
    temperature=0.3,        # Lower = more deterministic (0.0-2.0)
    top_p=0.9,             # Nucleus sampling (0.0-1.0)
    
    # Control length
    max_tokens=500,        # Maximum response tokens
    
    # Control formatting
    response_format={"type": "json_object"},  # For JSON responses
    
    # Other parameters
    frequency_penalty=0.1,  # Reduce repetition
    presence_penalty=0.1,   # Encourage new topics
)
```

### Timeout and Retries

```python
client = Chimeric(
    openrouter_api_key="your-api-key",
    timeout=60,            # Request timeout in seconds
    max_retries=3,         # Number of retries on failure
)
```

## 🔄 Multiple Models and Fallbacks

Compare responses from different models or implement fallbacks:

```python
async def compare_models(prompt: str):
    client = Chimeric()
    
    models = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-haiku-20241022", 
        "google/gemini-2.0-flash-exp"
    ]
    
    tasks = []
    for model in models:
        task = client.agenerate(
            model=model,
            messages=prompt,
            max_tokens=100
        )
        tasks.append(task)
    
    # Get responses concurrently
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for model, response in zip(models, responses):
        if isinstance(response, Exception):
            print(f"{model}: Error - {response}")
        else:
            print(f"{model}: {response.content[:100]}...")

# Run comparison
asyncio.run(compare_models("Explain the theory of relativity"))
```

## 🏷️ Advanced Use Cases

### Content Generation Pipeline

```python
class ContentGenerator:
    def __init__(self):
        self.client = Chimeric()
    
    async def generate_blog_post(self, topic: str) -> dict:
        """Generate a complete blog post with title, outline, and content."""
        
        # Generate title
        title_response = await self.client.agenerate(
            model="openai/gpt-4o-mini",
            messages=f"Generate a compelling blog post title about: {topic}",
            max_tokens=50
        )
        
        # Generate outline  
        outline_response = await self.client.agenerate(
            model="anthropic/claude-3-5-haiku-20241022",
            messages=f"Create a detailed outline for a blog post titled: {title_response.content}",
            max_tokens=200
        )
        
        # Generate full content
        content_response = await self.client.agenerate(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=f"""Write a comprehensive blog post:
Title: {title_response.content}
Outline: {outline_response.content}
Make it engaging and informative.""",
            max_tokens=1500
        )
        
        return {
            "title": title_response.content.strip(),
            "outline": outline_response.content,
            "content": content_response.content,
            "word_count": len(content_response.content.split())
        }

# Usage
generator = ContentGenerator()
blog_post = asyncio.run(generator.generate_blog_post("sustainable technology"))
print(f"Generated: {blog_post['title']}")
print(f"Word count: {blog_post['word_count']}")
```

### Code Analysis and Generation

```python
@client.tool()
def run_python_code(code: str) -> str:
    """Execute Python code safely and return the output.
    
    Args:
        code: Python code to execute
    """
    # In production, use a sandboxed environment
    try:
        # Simple example - use proper sandboxing in production
        exec_globals = {"__builtins__": {}}
        exec(code, exec_globals)
        return "Code executed successfully"
    except Exception as e:
        return f"Error: {str(e)}"

# Code generation and testing
response = client.generate(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages="""
    Write a Python function to find the longest common subsequence between two strings.
    Then test it with examples.
    """,
    tools=[run_python_code]
)

print(response.content)
```

## 🚨 Error Handling

Robust error handling for production applications:

```python
from chimeric.exceptions import (
    ModelNotSupportedError, 
    ProviderError, 
    ChimericError
)
import asyncio
from typing import Optional

class RobustOpenRouterClient:
    def __init__(self):
        self.client = Chimeric()
        self.fallback_models = [
            "openai/gpt-4o-mini",
            "anthropic/claude-3-5-haiku-20241022",
            "meta-llama/llama-3.1-8b-instruct:free"
        ]
    
    async def generate_with_fallback(
        self, 
        messages: str, 
        preferred_model: str = "openai/gpt-4o"
    ) -> Optional[str]:
        """Generate response with automatic fallback to alternative models."""
        
        models_to_try = [preferred_model] + self.fallback_models
        
        for model in models_to_try:
            try:
                response = await self.client.agenerate(
                    model=model,
                    messages=messages,
                    timeout=30
                )
                print(f"✅ Success with {model}")
                return response.content
                
            except ModelNotSupportedError:
                print(f"❌ Model {model} not supported, trying next...")
                continue
                
            except ProviderError as e:
                print(f"⚠️ Provider error with {model}: {e}, trying next...")
                continue
                
            except asyncio.TimeoutError:
                print(f"⏰ Timeout with {model}, trying next...")
                continue
                
            except Exception as e:
                print(f"🚫 Unexpected error with {model}: {e}")
                continue
        
        print("❌ All models failed")
        return None

# Usage
client = RobustOpenRouterClient()
response = asyncio.run(
    client.generate_with_fallback(
        "Explain quantum computing",
        preferred_model="anthropic/claude-3-5-sonnet-20241022"
    )
)

if response:
    print(f"Response: {response[:100]}...")
else:
    print("Failed to generate response")
```

## 💰 Cost Optimization

### Monitor Usage

```python
def track_usage(client):
    """Track token usage across requests."""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Make requests and track usage
    response = client.generate(
        model="openai/gpt-4o-mini",
        messages="Write a short poem about programming"
    )
    
    if hasattr(response, 'usage') and response.usage:
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens
        
        print(f"Request used:")
        print(f"- Prompt tokens: {response.usage.prompt_tokens}")
        print(f"- Completion tokens: {response.usage.completion_tokens}")
        print(f"- Total tokens: {response.usage.total_tokens}")
    
    return response.content

content = track_usage(client)
print(f"Generated: {content}")
```

### Choose Cost-Effective Models

```python
# Model cost tiers (approximate)
COST_TIERS = {
    "free": [
        "meta-llama/llama-3.1-8b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "huggingfaceh4/zephyr-7b-beta:free"
    ],
    "cheap": [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-haiku-20241022",
        "google/gemini-1.5-flash"
    ],
    "premium": [
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet-20241022",
        "google/gemini-2.0-flash-exp"
    ]
}

def choose_model_by_budget(task_complexity: str) -> str:
    """Choose appropriate model based on task complexity and budget."""
    if task_complexity == "simple":
        return COST_TIERS["free"][0]
    elif task_complexity == "medium":
        return COST_TIERS["cheap"][0] 
    else:
        return COST_TIERS["premium"][0]

# Usage
model = choose_model_by_budget("simple")
response = client.generate(
    model=model,
    messages="What is 2+2?"
)
```

## 🏭 Production Best Practices

### Rate Limiting

```python
import asyncio
from asyncio import Semaphore

class RateLimitedClient:
    def __init__(self, max_concurrent_requests: int = 5):
        self.client = Chimeric()
        self.semaphore = Semaphore(max_concurrent_requests)
    
    async def generate_with_rate_limit(self, **kwargs):
        async with self.semaphore:
            return await self.client.agenerate(**kwargs)

# Usage
rate_limited_client = RateLimitedClient(max_concurrent_requests=3)

async def batch_generate(prompts: list[str]):
    tasks = [
        rate_limited_client.generate_with_rate_limit(
            model="openai/gpt-4o-mini",
            messages=prompt
        )
        for prompt in prompts
    ]
    
    return await asyncio.gather(*tasks, return_exceptions=True)

prompts = [
    "Explain photosynthesis",
    "Write a haiku about technology", 
    "Describe machine learning",
]

responses = asyncio.run(batch_generate(prompts))
for i, response in enumerate(responses):
    if isinstance(response, Exception):
        print(f"Prompt {i+1}: Error - {response}")
    else:
        print(f"Prompt {i+1}: {response.content[:50]}...")
```

### Configuration Management

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class OpenRouterConfig:
    api_key: str
    default_model: str = "openai/gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> 'OpenRouterConfig':
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            default_model=os.getenv("OPENROUTER_DEFAULT_MODEL", "openai/gpt-4o-mini"),
            max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("OPENROUTER_TIMEOUT", "30")),
            max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))
        )

def create_configured_client() -> Chimeric:
    config = OpenRouterConfig.from_env()
    return Chimeric(
        openrouter_api_key=config.api_key,
        timeout=config.timeout,
        max_retries=config.max_retries
    )

# Usage
client = create_configured_client()
```

### Logging and Monitoring

```python
import logging
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_openrouter_requests(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        model = kwargs.get('model', 'unknown')
        
        try:
            logger.info(f"Starting request to {model}")
            result = await func(*args, **kwargs)
            
            duration = time.time() - start_time
            logger.info(f"Request to {model} completed in {duration:.2f}s")
            
            if hasattr(result, 'usage') and result.usage:
                logger.info(f"Token usage - Prompt: {result.usage.prompt_tokens}, "
                          f"Completion: {result.usage.completion_tokens}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Request to {model} failed after {duration:.2f}s: {e}")
            raise
    
    return wrapper

# Apply to client methods
client = Chimeric()
client.agenerate = log_openrouter_requests(client.agenerate)
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. API Key Issues
```python
# Test API key validity
try:
    client = Chimeric(openrouter_api_key="test-key")
    models = client.list_models("openrouter")
    print("✅ API key is valid")
except Exception as e:
    print(f"❌ API key error: {e}")
```

#### 2. Model Availability
```python
# Check if a specific model is available
def check_model_availability(model_name: str) -> bool:
    try:
        models = client.list_models("openrouter")
        available_models = [m.id for m in models]
        return model_name in available_models
    except:
        return False

# Usage
if check_model_availability("openai/gpt-4o"):
    print("✅ Model is available")
else:
    print("❌ Model not available")
```

#### 3. Timeout Issues
```python
# Increase timeout for long-running requests
client = Chimeric(
    openrouter_api_key="your-key",
    timeout=120  # 2 minutes
)

# Or handle timeouts gracefully
import asyncio

async def generate_with_timeout(prompt: str, timeout: int = 60):
    try:
        response = await asyncio.wait_for(
            client.agenerate(
                model="anthropic/claude-3-5-sonnet-20241022",
                messages=prompt
            ),
            timeout=timeout
        )
        return response.content
    except asyncio.TimeoutError:
        return "Request timed out. Try a shorter prompt or increase timeout."

result = asyncio.run(generate_with_timeout("Write a very long essay about AI"))
```

## 📚 Additional Resources

### Official Links
- [OpenRouter Website](https://openrouter.ai)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter Models](https://openrouter.ai/models)
- [OpenRouter API Reference](https://openrouter.ai/docs/api)

### Model Comparisons
Visit [OpenRouter Models](https://openrouter.ai/models) to compare:
- Model capabilities and pricing
- Context length limits
- Special features (vision, function calling, etc.)
- Performance benchmarks

### Community and Support
- Join the [OpenRouter Discord](https://discord.gg/openrouter) 
- Check the [OpenRouter GitHub](https://github.com/OpenRouterTeam)
- Read the [chimeric documentation](https://github.com/Johnson-f/chimeric)

## 🎯 Next Steps

1. **Start Simple**: Begin with basic text generation using free models
2. **Experiment**: Try different models for your specific use case
3. **Add Tools**: Implement function calling for interactive applications
4. **Scale Up**: Use async for high-performance applications
5. **Monitor**: Track usage and optimize for cost and performance
6. **Production**: Implement proper error handling, logging, and rate limiting

The OpenRouter provider in chimeric gives you access to the entire ecosystem of AI models through a single, consistent interface. Start experimenting and building amazing AI-powered applications!
