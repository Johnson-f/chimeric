from unittest.mock import AsyncMock, MagicMock, Mock, patch

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from chimeric.providers.openrouter import OpenRouterAsyncClient, OpenRouterClient
from chimeric.types import Capability, Message, Tool, ToolCall, ToolExecutionResult, ToolParameters
from chimeric.utils import StreamProcessor
from conftest import BaseProviderTestSuite


class TestOpenRouterClient(BaseProviderTestSuite):
    """Test suite for OpenRouter sync client"""

    client_class = OpenRouterClient
    provider_name = "OpenRouter"
    mock_client_path = "chimeric.providers.openrouter.client.OpenAI"

    @property
    def sample_response(self):
        """Create a sample OpenRouter response."""
        response = Mock(spec=ChatCompletion)
        
        # Mock the choice
        choice = Mock()
        choice.message = Mock()
        choice.message.content = "Hello there"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        
        response.choices = [choice]
        response.usage = Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response.model = "openai/gpt-4o-mini"
        response.id = "chatcmpl-123"
        
        return response

    @property
    def sample_stream_events(self):
        """Create sample OpenRouter stream events."""
        events = []

        # Content delta event
        chunk1 = Mock(spec=ChatCompletionChunk)
        choice1 = Mock()
        choice1.delta = Mock()
        choice1.delta.content = "Hello"
        choice1.delta.tool_calls = None
        choice1.finish_reason = None
        chunk1.choices = [choice1]
        events.append(chunk1)

        # Tool call start event
        chunk2 = Mock(spec=ChatCompletionChunk)
        choice2 = Mock()
        choice2.delta = Mock()
        choice2.delta.content = None
        tool_call_delta = Mock()
        tool_call_delta.id = "call_123"
        tool_call_delta.function = Mock()
        tool_call_delta.function.name = "test_tool"
        tool_call_delta.function.arguments = None
        tool_call_delta.index = 0
        choice2.delta.tool_calls = [tool_call_delta]
        choice2.finish_reason = None
        chunk2.choices = [choice2]
        events.append(chunk2)

        # Tool call arguments event
        chunk3 = Mock(spec=ChatCompletionChunk)
        choice3 = Mock()
        choice3.delta = Mock()
        choice3.delta.content = None
        tool_call_delta2 = Mock()
        tool_call_delta2.id = None
        tool_call_delta2.function = Mock()
        tool_call_delta2.function.name = None
        tool_call_delta2.function.arguments = '{"x": 10}'
        tool_call_delta2.index = 0
        choice3.delta.tool_calls = [tool_call_delta2]
        choice3.finish_reason = None
        chunk3.choices = [choice3]
        events.append(chunk3)

        # Finish event
        chunk4 = Mock(spec=ChatCompletionChunk)
        choice4 = Mock()
        choice4.delta = Mock()
        choice4.delta.content = None
        choice4.delta.tool_calls = None
        choice4.finish_reason = "stop"
        chunk4.choices = [choice4]
        events.append(chunk4)

        return events

    # ===== Initialization Tests =====

    def test_client_initialization_success(self):
        """Test successful client initialization with all parameters."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_openai:
            client = self.client_class(
                api_key="test-key", 
                tool_manager=tool_manager, 
                timeout=30,
                default_headers={"X-Title": "Test"}
            )

            assert client.api_key == "test-key"
            assert client.tool_manager == tool_manager
            assert client._provider_name == self.provider_name
            # Should be called with OpenRouter base URL
            mock_openai.assert_called_once_with(
                api_key="test-key", 
                base_url="https://openrouter.ai/api/v1",
                timeout=30,
                default_headers={"X-Title": "Test"}
            )

    def test_client_initialization_custom_base_url(self):
        """Test client initialization with custom base URL."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_openai:
            client = self.client_class(
                api_key="test-key", 
                tool_manager=tool_manager,
                base_url="https://custom.openrouter.ai/api/v1"
            )
            
            # Should use the custom base URL
            mock_openai.assert_called_once_with(
                api_key="test-key", 
                base_url="https://custom.openrouter.ai/api/v1"
            )

    # ===== Capability Tests =====

    def test_capabilities(self):
        """Test provider capabilities."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            capabilities = client._get_capabilities()

            assert isinstance(capabilities, Capability)
            assert capabilities.streaming is True
            assert capabilities.tools is True

    # ===== Model Listing Tests =====

    def test_list_models_success(self):
        """Test successful model listing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Mock model list response
            mock_model = Mock(id="openai/gpt-4o-mini")
            mock_model.name = "GPT-4o Mini"
            mock_model.owned_by = "openai"
            mock_model.created = 1234567890
            mock_client.models.list.return_value = [mock_model]

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "openai/gpt-4o-mini"
            assert models[0].name == "GPT-4o Mini"
            assert models[0].owned_by == "openai"

    # ===== Message Formatting Tests =====

    def test_messages_to_provider_format_basic(self):
        """Test basic message formatting."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            messages = [
                Message(role="system", content="You are helpful"),
                Message(role="user", content="Hello"),
            ]
            
            formatted = client._messages_to_provider_format(messages)
            
            assert len(formatted) == 2
            assert formatted[0]["role"] == "system"
            assert formatted[0]["content"] == "You are helpful"
            assert formatted[1]["role"] == "user"
            assert formatted[1]["content"] == "Hello"

    def test_messages_to_provider_format_with_tool_calls(self):
        """Test message formatting with tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            tool_call = ToolCall(call_id="call_123", name="test_tool", arguments='{"x": 10}')
            messages = [
                Message(role="assistant", content="I'll help you", tool_calls=[tool_call])
            ]
            
            formatted = client._messages_to_provider_format(messages)
            
            assert len(formatted) == 1
            assert formatted[0]["role"] == "assistant"
            assert formatted[0]["content"] == "I'll help you"
            assert len(formatted[0]["tool_calls"]) == 1
            assert formatted[0]["tool_calls"][0]["id"] == "call_123"
            assert formatted[0]["tool_calls"][0]["function"]["name"] == "test_tool"

    def test_messages_to_provider_format_tool_result(self):
        """Test message formatting with tool result."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            messages = [
                Message(role="tool", content="Result: 42", tool_call_id="call_123")
            ]
            
            formatted = client._messages_to_provider_format(messages)
            
            assert len(formatted) == 1
            assert formatted[0]["role"] == "tool"
            assert formatted[0]["content"] == "Result: 42"
            assert formatted[0]["tool_call_id"] == "call_123"

    # ===== Tool Formatting Tests =====

    def test_tools_to_provider_format(self):
        """Test tool formatting."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            tool_params = ToolParameters(
                type="object",
                properties={"x": {"type": "integer"}},
                required=["x"]
            )
            tools = [
                Tool(name="test_tool", description="A test tool", parameters=tool_params)
            ]
            
            formatted = client._tools_to_provider_format(tools)
            
            assert len(formatted) == 1
            assert formatted[0]["type"] == "function"
            assert formatted[0]["function"]["name"] == "test_tool"
            assert formatted[0]["function"]["description"] == "A test tool"
            assert formatted[0]["function"]["parameters"]["type"] == "object"

    # ===== Provider Request Tests =====

    def test_make_provider_request(self):
        """Test making provider request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            messages = [{"role": "user", "content": "Hello"}]
            tools = [{"type": "function", "function": {"name": "test"}}]
            
            client._make_provider_request(
                messages=messages,
                model="openai/gpt-4o-mini",
                stream=False,
                tools=tools,
                temperature=0.7
            )
            
            mock_client.chat.completions.create.assert_called_once_with(
                model="openai/gpt-4o-mini",
                messages=messages,
                stream=False,
                tools=tools,
                temperature=0.7
            )

    # ===== Response Extraction Tests =====

    def test_extract_usage_from_response(self):
        """Test usage extraction from response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            response = self.sample_response
            usage = client._extract_usage_from_response(response)
            
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30

    def test_extract_content_from_response(self):
        """Test content extraction from response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            response = self.sample_response
            content = client._extract_content_from_response(response)
            
            assert content == "Hello there"

    def test_extract_tool_calls_from_response(self):
        """Test tool call extraction from response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            response = Mock(spec=ChatCompletion)
            choice = Mock()
            choice.message = Mock()
            tool_call = Mock()
            tool_call.id = "call_123"
            tool_call.function = Mock()
            tool_call.function.name = "test_tool"
            tool_call.function.arguments = '{"x": 10}'
            choice.message.tool_calls = [tool_call]
            response.choices = [choice]
            
            tool_calls = client._extract_tool_calls_from_response(response)
            
            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "call_123"
            assert tool_calls[0].name == "test_tool"
            assert tool_calls[0].arguments == '{"x": 10}'

    # ===== Stream Processing Tests =====

    def test_process_provider_stream_event_content(self):
        """Test processing stream event with content."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()
            
            chunk = Mock(spec=ChatCompletionChunk)
            choice = Mock()
            choice.delta = Mock()
            choice.delta.content = "Hello"
            choice.delta.tool_calls = None
            choice.finish_reason = None
            chunk.choices = [choice]
            
            result = client._process_provider_stream_event(chunk, processor)
            
            assert result is not None
            assert result.common.content == "Hello"

    def test_process_provider_stream_event_no_choices(self):
        """Test processing stream event with no choices."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()
            
            chunk = Mock(spec=ChatCompletionChunk)
            chunk.choices = []
            
            result = client._process_provider_stream_event(chunk, processor)
            
            assert result is None

    def test_process_provider_stream_event_tool_call(self):
        """Test processing stream event with tool call."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()
            
            chunk = Mock(spec=ChatCompletionChunk)
            choice = Mock()
            choice.delta = Mock()
            choice.delta.content = None
            tool_call_delta = Mock()
            tool_call_delta.id = "call_123"
            tool_call_delta.function = Mock()
            tool_call_delta.function.name = "test_tool"
            tool_call_delta.function.arguments = None
            tool_call_delta.index = 0
            choice.delta.tool_calls = [tool_call_delta]
            choice.finish_reason = None
            chunk.choices = [choice]
            
            result = client._process_provider_stream_event(chunk, processor)
            
            assert result is not None

    def test_process_provider_stream_event_finish(self):
        """Test processing stream event with finish reason."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()
            
            chunk = Mock(spec=ChatCompletionChunk)
            choice = Mock()
            choice.delta = Mock()
            choice.delta.content = None
            choice.delta.tool_calls = None
            choice.finish_reason = "stop"
            chunk.choices = [choice]
            
            result = client._process_provider_stream_event(chunk, processor)
            
            assert result is not None
            assert result.common.finish_reason == "stop"

    def test_update_messages_with_tool_calls(self):
        """Test updating messages with tool calls and results."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            messages = [{"role": "user", "content": "Hello"}]
            response = self.sample_response
            tool_calls = [ToolCall(call_id="call_123", name="test_tool", arguments='{"x": 10}')]
            tool_results = [ToolExecutionResult(call_id="call_123", name="test_tool", arguments='{"x": 10}', result="Result: 10")]
            
            updated = client._update_messages_with_tool_calls(messages, response, tool_calls, tool_results)
            
            assert len(updated) == 3  # original + assistant + tool result
            assert updated[1]["role"] == "assistant"
            assert updated[2]["role"] == "tool"

    def test_extract_tool_calls_from_response_no_tool_calls_sync(self):
        """Test tool call extraction when no tool calls present."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            response = Mock(spec=ChatCompletion)
            choice = Mock()
            choice.message = Mock()
            choice.message.tool_calls = None
            response.choices = [choice]
            
            tool_calls = client._extract_tool_calls_from_response(response)
            
            assert tool_calls is None

    def test_extract_tool_calls_from_response_empty_choices_sync(self):
        """Test tool call extraction with empty choices."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            response = Mock(spec=ChatCompletion)
            response.choices = []
            
            tool_calls = client._extract_tool_calls_from_response(response)
            
            assert tool_calls is None


class TestOpenRouterAsyncClient(BaseProviderTestSuite):
    """Test suite for OpenRouter async client"""

    client_class = OpenRouterAsyncClient
    provider_name = "OpenRouter"
    mock_client_path = "chimeric.providers.openrouter.client.AsyncOpenAI"

    @property
    def sample_response(self):
        """Create a sample OpenRouter response."""
        response = Mock(spec=ChatCompletion)
        
        choice = Mock()
        choice.message = Mock()
        choice.message.content = "Hello there"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        
        response.choices = [choice]
        response.usage = Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response.model = "openai/gpt-4o-mini"
        response.id = "chatcmpl-123"
        
        return response

    @property
    def sample_stream_events(self):
        """Create sample OpenRouter stream events."""
        events = []

        chunk = Mock(spec=ChatCompletionChunk)
        choice = Mock()
        choice.delta = Mock()
        choice.delta.content = "Hello"
        choice.delta.tool_calls = None
        choice.finish_reason = None
        chunk.choices = [choice]
        events.append(chunk)

        return events

    # ===== Basic Async Tests =====

    def test_async_client_initialization(self):
        """Test async client initialization."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_async_openai:
            client = self.client_class(
                api_key="test-key", 
                tool_manager=tool_manager
            )

            assert client.api_key == "test-key"
            assert client._provider_name == self.provider_name
            mock_async_openai.assert_called_once_with(
                api_key="test-key", 
                base_url="https://openrouter.ai/api/v1"
            )

    def test_async_list_models(self):
        """Test async model listing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_async_openai:
            mock_client = AsyncMock()
            mock_async_openai.return_value = mock_client

            mock_model = Mock()
            mock_model.id = "openai/gpt-4o-mini"
            mock_model.name = "GPT-4o Mini"
            mock_model.owned_by = "openai"
            mock_model.created = 1234567890
            mock_client.models.list.return_value = [mock_model]

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            import asyncio
            async def test():
                models = await client._list_models_impl()
                assert len(models) == 1
                assert models[0].id == "openai/gpt-4o-mini"
                assert models[0].name == "GPT-4o Mini"
                assert models[0].owned_by == "openai"
            
            asyncio.run(test())

    def test_async_make_provider_request(self):
        """Test async provider request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_async_openai:
            mock_client = AsyncMock()
            mock_async_openai.return_value = mock_client

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            import asyncio
            async def test():
                await client._make_async_provider_request(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="openai/gpt-4o-mini",
                    stream=False,
                    tools=None
                )
                
                mock_client.chat.completions.create.assert_called_once()
                _, kwargs = mock_client.chat.completions.create.call_args
                from openai import NOT_GIVEN
                assert kwargs.get("tools") == NOT_GIVEN
            
            asyncio.run(test())
