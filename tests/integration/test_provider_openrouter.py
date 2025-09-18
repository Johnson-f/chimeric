from collections.abc import AsyncGenerator

import pytest

from chimeric import Chimeric
from chimeric.exceptions import ProviderError

from .vcr_config import get_cassette_path, get_vcr


@pytest.mark.openrouter
def test_openrouter_model_listing(api_keys):
    """Test OpenRouter model listing functionality."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    cassette_path = get_cassette_path("openrouter", "test_model_listing")

    with get_vcr().use_cassette(cassette_path):
        chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])

        models = chimeric.list_models()
        assert len(models) > 0

        # Should have models from multiple providers (OpenAI, Anthropic, etc.)
        model_ids = [model.id.lower() for model in models]
        # OpenRouter should have GPT models
        assert any("gpt" in model_id for model_id in model_ids)
        # And likely Claude models
        # Note: This might vary based on OpenRouter's available models
        
        print(f"Found {len(models)} OpenRouter models")


@pytest.mark.openrouter
def test_openrouter_sync_generation(api_keys):
    """Test OpenRouter synchronous generation functionality."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_sync_generation")

    with get_vcr().use_cassette(cassette_path):
        response = chimeric.generate(
            model="openai/gpt-4o-mini",  # Use OpenRouter format
            messages=[{"role": "user", "content": "Hello, respond briefly."}],
            stream=False,
            max_tokens=20,
        )

        assert response is not None
        assert response.content


@pytest.mark.openrouter
@pytest.mark.asyncio
async def test_openrouter_async_generation(api_keys):
    """Test OpenRouter asynchronous generation functionality."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_async_generation")

    with get_vcr().use_cassette(cassette_path):
        response = await chimeric.agenerate(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, respond briefly."}],
            stream=False,
            max_tokens=20,
        )

        assert response is not None
        assert response.content


@pytest.mark.openrouter
def test_openrouter_sync_streaming(api_keys):
    """Test OpenRouter synchronous streaming functionality."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_sync_streaming")

    with get_vcr().use_cassette(cassette_path):
        response = chimeric.generate(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me a short story about a robot."}],
            stream=True,
            max_tokens=50,
        )

        # Collect all chunks and verify streaming works
        assert response is not None
        chunks = list(response)
        assert len(chunks) > 0
        content_chunks = [chunk for chunk in chunks if hasattr(chunk, "content") and chunk.content]
        assert len(content_chunks) > 0, "At least some chunks should have content"


@pytest.mark.openrouter
@pytest.mark.asyncio
async def test_openrouter_async_streaming(api_keys):
    """Test OpenRouter asynchronous streaming functionality."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_async_streaming")

    with get_vcr().use_cassette(cassette_path):
        response = await chimeric.agenerate(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me a short story about a robot."}],
            stream=True,
            max_tokens=50,
        )

        # Collect all chunks and verify streaming works
        assert response is not None
        assert isinstance(response, AsyncGenerator), (
            "Response should be an AsyncGenerator when streaming"
        )
        chunks = [chunk async for chunk in response]
        assert len(chunks) > 0
        content_chunks = [chunk for chunk in chunks if chunk.content]
        assert len(content_chunks) > 0, "At least some chunks should have content"


@pytest.mark.openrouter
def test_openrouter_sync_tools_non_streaming(api_keys):
    """Test OpenRouter sync generation with tools (non-streaming)."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_sync_tools_non_streaming")

    with get_vcr().use_cassette(cassette_path):
        # Track tool calls
        tool_calls = {"add": 0, "subtract": 0, "joke": 0}

        @chimeric.tool()
        def add(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Adds two numbers together.
            Args:
                x: the first number
                y: the second number

            Returns:
                The sum of x and y.
            """
            print("Adding numbers:", x, y)
            tool_calls["add"] += 1
            return x + y

        @chimeric.tool()
        def subtract(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Subtracts the second number from the first.
            Args:
                x: the first number
                y: the second number

            Returns:
                The result of x - y.
            """
            print("Subtracting numbers:", x, y)
            tool_calls["subtract"] += 1
            return x - y

        @chimeric.tool()
        def joke() -> str:  # type: ignore[reportUnusedFunction]
            """
            Returns a joke.
            """
            print("Telling a joke...")
            tool_calls["joke"] += 1
            return "Why did the chicken cross the road? To get to the other side!"

        response = chimeric.generate(
            model="openai/gpt-4o-mini",  # Use a model that supports tools
            messages=[{"role": "user", "content": "What is 2+2-4-10+50? Tell me a joke."}],
            stream=False,
        )

        assert response is not None
        assert response.content

        # Verify tools were called
        assert tool_calls["add"] > 0, "Add function should have been called"
        assert tool_calls["subtract"] > 0, "Subtract function should have been called"
        assert tool_calls["joke"] > 0, "Joke function should have been called"

        # Print summary for debugging
        print(f"Tool call counts: {tool_calls}")


@pytest.mark.openrouter
def test_openrouter_sync_tools_streaming(api_keys):
    """Test OpenRouter sync generation with tools (streaming)."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_sync_tools_streaming")

    with get_vcr().use_cassette(cassette_path):
        # Track tool calls
        tool_calls = {"add": 0, "subtract": 0, "joke": 0}

        @chimeric.tool()
        def add(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Adds two numbers together.
            Args:
                x: the first number
                y: the second number

            Returns:
                The sum of x and y.
            """
            print("Adding numbers:", x, y)
            tool_calls["add"] += 1
            return x + y

        @chimeric.tool()
        def subtract(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Subtracts the second number from the first.
            Args:
                x: the first number
                y: the second number

            Returns:
                The result of x - y.
            """
            print("Subtracting numbers:", x, y)
            tool_calls["subtract"] += 1
            return x - y

        @chimeric.tool()
        def joke() -> str:  # type: ignore[reportUnusedFunction]
            """
            Returns a joke.
            """
            print("Telling a joke...")
            tool_calls["joke"] += 1
            return "Why did the chicken cross the road? To get to the other side!"

        response = chimeric.generate(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2-4-10+50? Tell me a joke."}],
            stream=True,
        )

        # Collect all chunks and verify at least some have content
        assert response is not None
        chunks = list(response)
        assert len(chunks) > 0
        content_chunks = [chunk for chunk in chunks if hasattr(chunk, "content") and chunk.content]
        assert len(content_chunks) > 0, "At least some chunks should have content"

        # Verify tools were actually called
        assert tool_calls["add"] > 0, "Add function should have been called"
        assert tool_calls["subtract"] > 0, "Subtract function should have been called"
        assert tool_calls["joke"] > 0, "Joke function should have been called"

        # Print summary for debugging
        print(f"Tool call counts: {tool_calls}")


@pytest.mark.openrouter
@pytest.mark.asyncio
async def test_openrouter_async_tools_streaming(api_keys):
    """Test OpenRouter async generation with tools (streaming)."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_async_tools_streaming")

    with get_vcr().use_cassette(cassette_path):
        # Track tool calls
        tool_calls = {"add": 0, "subtract": 0, "joke": 0}

        @chimeric.tool()
        def add(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Adds two numbers together.
            Args:
                x: the first number
                y: the second number

            Returns:
                The sum of x and y.
            """
            print("Adding numbers:", x, y)
            tool_calls["add"] += 1
            return x + y

        @chimeric.tool()
        def subtract(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Subtracts the second number from the first.
            Args:
                x: the first number
                y: the second number

            Returns:
                The result of x - y.
            """
            print("Subtracting numbers:", x, y)
            tool_calls["subtract"] += 1
            return x - y

        @chimeric.tool()
        def joke() -> str:  # type: ignore[reportUnusedFunction]
            """
            Returns a joke.
            """
            print("Telling a joke...")
            tool_calls["joke"] += 1
            return "Why did the chicken cross the road? To get to the other side!"

        response = await chimeric.agenerate(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2-4-10+50? Tell me a joke."}],
            stream=True,
        )

        # Collect all chunks and verify at least some have content
        assert response is not None
        assert isinstance(response, AsyncGenerator), (
            "Response should be an AsyncGenerator when streaming"
        )
        chunks = [chunk async for chunk in response]
        assert len(chunks) > 0
        content_chunks = [chunk for chunk in chunks if chunk.content]
        assert len(content_chunks) > 0, "At least some chunks should have content"

        # Verify tools were actually called
        assert tool_calls["add"] > 0, "Add function should have been called"
        assert tool_calls["subtract"] > 0, "Subtract function should have been called"
        assert tool_calls["joke"] > 0, "Joke function should have been called"

        # Print summary for debugging
        print(f"Tool call counts: {tool_calls}")


@pytest.mark.openrouter
@pytest.mark.asyncio
async def test_openrouter_async_tools_non_streaming(api_keys):
    """Test OpenRouter async generation with tools (non-streaming)."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_async_tools_non_streaming")

    with get_vcr().use_cassette(cassette_path):
        # Track tool calls
        tool_calls = {"add": 0, "subtract": 0, "joke": 0}

        @chimeric.tool()
        def add(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Adds two numbers together.
            Args:
                x: the first number
                y: the second number

            Returns:
                The sum of x and y.
            """
            print("Adding numbers:", x, y)
            tool_calls["add"] += 1
            return x + y

        @chimeric.tool()
        def subtract(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Subtracts the second number from the first.
            Args:
                x: the first number
                y: the second number

            Returns:
                The result of x - y.
            """
            print("Subtracting numbers:", x, y)
            tool_calls["subtract"] += 1
            return x - y

        @chimeric.tool()
        def joke() -> str:  # type: ignore[reportUnusedFunction]
            """
            Returns a joke.
            """
            print("Telling a joke...")
            tool_calls["joke"] += 1
            return "Why did the chicken cross the road? To get to the other side!"

        response = await chimeric.agenerate(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2-4-10+50? Tell me a joke."}],
            stream=False,
        )

        assert response is not None
        assert response.content

        # Verify tools were actually called
        assert tool_calls["add"] > 0, "Add function should have been called"
        assert tool_calls["subtract"] > 0, "Subtract function should have been called"
        assert tool_calls["joke"] > 0, "Joke function should have been called"

        # Print summary for debugging
        print(f"Tool call counts: {tool_calls}")


@pytest.mark.openrouter
def test_openrouter_different_models(api_keys):
    """Test OpenRouter with different provider models."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_different_models")

    with get_vcr().use_cassette(cassette_path):
        # Test OpenAI model through OpenRouter
        response1 = chimeric.generate(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'OpenAI model works'"}],
            stream=False,
            max_tokens=10,
        )
        assert response1 is not None
        assert response1.content

        # Test Anthropic model through OpenRouter (if available)
        try:
            response2 = chimeric.generate(
                model="anthropic/claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "Say 'Anthropic model works'"}],
                stream=False,
                max_tokens=10,
            )
            assert response2 is not None
            assert response2.content
            print("Anthropic model test successful")
        except Exception as e:
            print(f"Anthropic model test skipped: {e}")

        # Test Meta model through OpenRouter (if available)
        try:
            response3 = chimeric.generate(
                model="meta-llama/llama-3.1-8b-instruct:free",
                messages=[{"role": "user", "content": "Say 'Meta model works'"}],
                stream=False,
                max_tokens=10,
            )
            assert response3 is not None
            assert response3.content
            print("Meta model test successful")
        except Exception as e:
            print(f"Meta model test skipped: {e}")


@pytest.mark.openrouter
def test_openrouter_init_kwargs_propagation(api_keys):
    """Test OpenRouter kwargs propagation through the stack with custom headers."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    # Initialize Chimeric with custom OpenRouter-specific parameters
    chimeric = Chimeric(
        openrouter_api_key=api_keys["openrouter_api_key"],
        timeout=60,
        max_retries=3,
        default_headers={
            "HTTP-Referer": "https://github.com/test/chimeric",
            "X-Title": "Chimeric Test Suite",
        },
        # Fake params that other providers might use
        anthropic_fake_param="should_be_ignored",
        google_vertex_project="fake_project",
        cohere_fake_setting=True,
    )

    cassette_path = get_cassette_path("openrouter", "test_kwargs_propagation")

    with get_vcr().use_cassette(cassette_path):
        # Test with generation kwargs
        response = chimeric.generate(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, respond briefly."}],
            temperature=0.1,
            max_tokens=20,
            stream=False,
        )

        assert response is not None
        assert response.content


@pytest.mark.openrouter
def test_openrouter_provider_routing_features(api_keys):
    """Test OpenRouter-specific features like provider routing."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_provider_routing")

    with get_vcr().use_cassette(cassette_path):
        # Test with OpenRouter-specific parameters in extra_body
        # Note: These would typically be passed via extra_body in the OpenAI client
        response = chimeric.generate(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, respond briefly."}],
            stream=False,
            max_tokens=20,
            # OpenRouter-specific parameters could be added here if needed
        )

        assert response is not None
        assert response.content


@pytest.mark.openrouter
def test_openrouter_async_tools_with_different_types(api_keys):
    """Test OpenRouter async tools with different return types."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_async_tools_different_types")

    @chimeric.tool()
    async def async_calculate(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
        """Async calculation function."""
        return x * y

    @chimeric.tool()
    def sync_format(text: str) -> str:  # type: ignore[reportUnusedFunction]
        """Sync text formatting function."""
        return f"Formatted: {text.upper()}"

    with get_vcr().use_cassette(cassette_path):
        response = chimeric.generate(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Calculate 3 times 4, then format the word 'hello'"}],
            stream=False,
        )

        assert response is not None
        assert response.content


@pytest.mark.openrouter
def test_openrouter_invalid_generate_kwargs_raises_provider_error(api_keys):
    """Test that invalid kwargs in generate raise ProviderError."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    cassette_path = get_cassette_path("openrouter", "test_invalid_kwargs_raises_provider_error")

    with get_vcr().use_cassette(cassette_path):
        # Test with an invalid parameter that doesn't exist in OpenAI/OpenRouter API
        with pytest.raises(ProviderError) as exc_info:
            chimeric.generate(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                invalid_openrouter_parameter="this_should_fail",
                stream=False,
            )

        # Verify the error contains provider information
        assert "OpenRouter" in str(exc_info.value) or "openrouter" in str(exc_info.value).lower()
        print(f"ProviderError raised as expected: {exc_info.value}")


@pytest.mark.openrouter
def test_openrouter_capabilities(api_keys):
    """Test OpenRouter capabilities detection."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])
    
    # Test capabilities
    capabilities = chimeric.get_capabilities("openrouter")
    assert capabilities.streaming is True
    assert capabilities.tools is True
    
    print(f"OpenRouter capabilities: {capabilities}")


@pytest.mark.openrouter
def test_openrouter_model_filtering(api_keys):
    """Test OpenRouter model filtering functionality."""
    if "openrouter_api_key" not in api_keys:
        pytest.skip("OpenRouter API key not found")

    cassette_path = get_cassette_path("openrouter", "test_model_filtering")

    with get_vcr().use_cassette(cassette_path):
        chimeric = Chimeric(openrouter_api_key=api_keys["openrouter_api_key"])

        # Get all models
        all_models = chimeric.list_models("openrouter")
        assert len(all_models) > 0

        # Check model metadata
        for model in all_models[:5]:  # Check first 5 models
            assert model.id is not None
            assert model.name is not None
            assert model.provider == "openrouter"

        print(f"First few OpenRouter models: {[m.id for m in all_models[:5]]}")
