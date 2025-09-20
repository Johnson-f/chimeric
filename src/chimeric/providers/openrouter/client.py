from typing import Any

from openai import NOT_GIVEN, AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from chimeric.base import ChimericAsyncClient, ChimericClient
from chimeric.types import (
    Capability,
    ChimericStreamChunk,
    Message,
    ModelSummary,
    Tool,
    ToolCall,
    ToolExecutionResult,
    Usage,
)
from chimeric.utils import StreamProcessor, create_stream_chunk


class OpenRouterClient(ChimericClient[OpenAI, ChatCompletion, ChatCompletionChunk]):
    """Synchronous OpenRouter Client for interacting with multiple LLM providers via OpenRouter API.

    This client provides a unified interface for synchronous interactions with
    various LLM providers through OpenRouter's API gateway. OpenRouter provides
    access to models from OpenAI, Anthropic, Google, Meta, Cohere, and many other
    providers through a single API interface that's compatible with OpenAI's format.

    The client supports:
        - Access to 200+ models from different providers
        - Advanced text generation with GPT-4, Claude, Gemini, Llama, and more
        - Function/tool calling with automatic execution
        - Streaming responses with real-time processing
        - Model routing and fallback capabilities
        - Cost optimization through provider selection
        - Model listing and metadata retrieval

    Note:
        OpenRouter acts as a gateway, providing access to the best models from
        different providers while handling authentication, routing, and billing.
        This allows you to use the best model for each specific task.

    Example:
        ```python
        from chimeric.providers.openrouter import OpenRouterClient
        from chimeric.tools import ToolManager

        tool_manager = ToolManager()
        client = OpenRouterClient(api_key="your-openrouter-key", tool_manager=tool_manager)

        response = client.chat_completion(
            messages="What's the capital of France?",
            model="openai/gpt-4o"  # or "anthropic/claude-3-5-sonnet-20241022"
        )
        print(response.common.content)
        ```

    Attributes:
        api_key (str): The OpenRouter API key for authentication.
        tool_manager (ToolManager): Manager for handling tool registration and execution.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initialize the synchronous OpenRouter client.

        Args:
            api_key: The OpenRouter API key for authentication.
            tool_manager: The tool manager instance for handling function calls.
            **kwargs: Additional keyword arguments to pass to the OpenAI client
                constructor. OpenRouter supports most OpenAI client parameters.
                Common parameters include:
                - base_url: Automatically set to OpenRouter's endpoint
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retries
                - default_headers: Additional headers (e.g., HTTP-Referer, X-Title)

        Raises:
            ValueError: If api_key is None or empty.
            ProviderError: If client initialization fails.
        """
        self._provider_name = "OpenRouter"
        # Set OpenRouter's base URL if not provided
        if "base_url" not in kwargs:
            kwargs["base_url"] = "https://openrouter.ai/api/v1"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    # ====================================================================
    # Required abstract method implementations
    # ====================================================================

    def _get_client_type(self) -> type:
        """Get the synchronous OpenAI client class type.

        Returns:
            The OpenAI client class from the openai library.
        """
        return OpenAI

    def _init_client(self, client_type: type, **kwargs: Any) -> OpenAI:
        """Initializes the synchronous OpenAI client pointed to OpenRouter."""
        return OpenAI(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Get supported features for the OpenRouter provider.

        Returns:
            Capability object indicating which features are supported:
                - streaming: True (supports real-time streaming responses)
                - tools: True (supports function calling and tool use)

        Note:
            OpenRouter supports the full OpenAI API interface including
            streaming and tool calling across most models.
        """
        return Capability(streaming=True, tools=True)

    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the OpenRouter API.

        Returns:
            A list of ModelSummary objects containing model information
            from OpenRouter's model registry.
        """
        return [
            ModelSummary(
                id=model.id,
                name=getattr(model, "name", model.id),
                owned_by=getattr(model, "owned_by", "openrouter"),
                created_at=getattr(model, "created", None),
                metadata=model.__dict__ if hasattr(model, "__dict__") else {},
            )
            for model in self.client.models.list()
        ]

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to OpenRouter/OpenAI chat format.

        OpenRouter uses the same message format as OpenAI's chat completions API.
        """
        formatted_messages = []
        for msg in messages:
            if msg.role == "tool":
                # Tool result message - format for OpenAI chat completions
                formatted_messages.append(
                    {
                        "role": "tool",
                        "content": str(msg.content),
                        "tool_call_id": msg.tool_call_id,
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls
                tool_calls_formatted = [
                    {
                        "id": tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                    for tool_call in msg.tool_calls
                ]
                formatted_messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": tool_calls_formatted,
                    }
                )
            else:
                # Regular message - convert to standard format
                formatted_messages.append(msg.model_dump(exclude_none=True))

        return formatted_messages

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to OpenRouter/OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters.model_dump() if tool.parameters else {},
                },
            }
            for tool in tools
        ]

    def _make_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Makes the actual API request to OpenRouter."""
        tools_param = NOT_GIVEN if tools is None else tools

        return self.client.chat.completions.create(
            model=model, messages=messages, stream=stream, tools=tools_param, **kwargs
        )

    def _process_provider_stream_event(
        self, event: ChatCompletionChunk, processor: StreamProcessor
    ) -> ChimericStreamChunk[ChatCompletionChunk] | None:
        """Processes an OpenRouter stream event using the standardized processor."""
        if not event.choices:
            return None

        choice = event.choices[0]
        delta = choice.delta

        # Handle content deltas
        if delta.content:
            return create_stream_chunk(
                native_event=event, processor=processor, content_delta=delta.content
            )

        # Handle tool call deltas
        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                if tool_call_delta.id:
                    # New tool call starting
                    processor.process_tool_call_start(
                        tool_call_delta.id,
                        tool_call_delta.function.name if tool_call_delta.function else "",
                        tool_call_delta.id,
                    )
                elif tool_call_delta.function and tool_call_delta.function.arguments:
                    # Tool call arguments delta - need the tool call ID from index
                    # For OpenAI streaming, we track by index
                    tool_call_id = f"call_{tool_call_delta.index}"
                    processor.process_tool_call_delta(
                        tool_call_id, tool_call_delta.function.arguments
                    )

        # Handle completion
        if choice.finish_reason:
            return create_stream_chunk(
                native_event=event,
                processor=processor,
                finish_reason=event.choices[0].finish_reason,
            )

        return None

    def _extract_usage_from_response(self, response: ChatCompletion) -> Usage:
        """Extracts usage information from OpenRouter response."""
        if not response.usage:
            return Usage()

        return Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _extract_content_from_response(self, response: ChatCompletion) -> str | list[Any]:
        """Extracts content from OpenRouter response."""
        if not response.choices:
            return ""

        choice = response.choices[0]
        return choice.message.content or ""

    def _extract_tool_calls_from_response(self, response: ChatCompletion) -> list[ToolCall] | None:
        """Extracts tool calls from OpenRouter response."""
        if not response.choices:
            return None

        choice = response.choices[0]
        if not choice.message.tool_calls:
            return None

        return [
            ToolCall(
                call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in choice.message.tool_calls
        ]

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: ChatCompletion | ChatCompletionChunk | Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results.

        For OpenRouter/OpenAI, this follows the chat completions format where:
        1. Assistant message with tool calls is added
        2. Tool result messages are added for each tool execution
        """
        updated_messages = list(messages)

        # Add the assistant message with tool calls
        if isinstance(assistant_response, ChatCompletion) and assistant_response.choices:
            choice = assistant_response.choices[0]
            if choice.message.tool_calls:
                updated_messages.append(
                    {
                        "role": "assistant",
                        "content": choice.message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ],
                    }
                )
        else:
            # For streaming responses, reconstruct the assistant message
            tool_calls_formatted = [
                {
                    "id": tool_call.call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
                for tool_call in tool_calls
            ]
            updated_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls_formatted,
                }
            )

        # Add the tool result messages
        for result in tool_results:
            updated_messages.append(
                {
                    "role": "tool",
                    "content": result.result if not result.is_error else f"Error: {result.error}",
                    "tool_call_id": result.call_id,
                }
            )

        return updated_messages


class OpenRouterAsyncClient(ChimericAsyncClient[AsyncOpenAI, ChatCompletion, ChatCompletionChunk]):
    """Asynchronous OpenRouter Client for interacting with multiple LLM providers via OpenRouter API.

    This client provides a unified interface for asynchronous interactions with
    various LLM providers through OpenRouter's API gateway. OpenRouter provides
    access to models from OpenAI, Anthropic, Google, Meta, Cohere, and many other
    providers through a single API interface that's compatible with OpenAI's format.

    The async client supports all the same features as the synchronous client:
        - Asynchronous access to 200+ models from different providers
        - Asynchronous advanced text generation with concurrent request processing
        - Asynchronous function/tool calling with automatic execution
        - Asynchronous streaming responses with real-time processing
        - Model routing and fallback capabilities
        - Cost optimization through provider selection
        - Model listing and metadata retrieval

    Note:
        OpenRouter acts as a gateway, providing access to the best models from
        different providers while handling authentication, routing, and billing.
        The async client is ideal for high-throughput applications and concurrent
        request processing across multiple models.

    Example:
        ```python
        import asyncio
        from chimeric.providers.openrouter import OpenRouterAsyncClient
        from chimeric.tools import ToolManager

        async def main():
            tool_manager = ToolManager()
            client = OpenRouterAsyncClient(api_key="your-openrouter-key", tool_manager=tool_manager)

            response = await client.chat_completion(
                messages="What's the capital of France?",
                model="openai/gpt-4o"  # or "anthropic/claude-3-5-sonnet-20241022"
            )
            print(response.common.content)

        asyncio.run(main())
        ```

    Attributes:
        api_key (str): The OpenRouter API key for authentication.
        tool_manager (ToolManager): Manager for handling tool registration and execution.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initialize the asynchronous OpenRouter client.

        Args:
            api_key: The OpenRouter API key for authentication.
            tool_manager: The tool manager instance for handling function calls.
            **kwargs: Additional keyword arguments to pass to the AsyncOpenAI client
                constructor. OpenRouter supports most OpenAI client parameters.
                Common parameters include:
                - base_url: Automatically set to OpenRouter's endpoint
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retries
                - default_headers: Additional headers (e.g., HTTP-Referer, X-Title)

        Raises:
            ValueError: If api_key is None or empty.
            ProviderError: If client initialization fails.
        """
        self._provider_name = "OpenRouter"
        # Set OpenRouter's base URL if not provided
        if "base_url" not in kwargs:
            kwargs["base_url"] = "https://openrouter.ai/api/v1"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    # ====================================================================
    # Required abstract method implementations
    # ====================================================================

    def _get_async_client_type(self) -> type:
        """Get the asynchronous OpenAI client class type.

        Returns:
            The AsyncOpenAI client class from the openai library.
        """
        return AsyncOpenAI

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncOpenAI:
        """Initializes the asynchronous OpenAI client pointed to OpenRouter."""
        return AsyncOpenAI(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Get supported features for the OpenRouter provider.

        Returns:
            Capability object indicating which features are supported:
                - streaming: True (supports real-time streaming responses)
                - tools: True (supports function calling and tool use)

        Note:
            OpenRouter supports the full OpenAI API interface including
            streaming and tool calling across most models.
        """
        return Capability(streaming=True, tools=True)

    async def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the OpenRouter API asynchronously.

        Returns:
            A list of ModelSummary objects containing model information
            from OpenRouter's model registry.
        """
        models_response = await self._async_client.models.list()
        return [
            ModelSummary(
                id=model.id,
                name=getattr(model, "name", model.id),
                owned_by=getattr(model, "owned_by", "openrouter"),
                created_at=getattr(model, "created", None),
                metadata=model.__dict__ if hasattr(model, "__dict__") else {},
            )
            for model in models_response
        ]

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to OpenRouter/OpenAI chat format.

        OpenRouter uses the same message format as OpenAI's chat completions API.
        """
        formatted_messages = []
        for msg in messages:
            if msg.role == "tool":
                # Tool result message - format for OpenAI chat completions
                formatted_messages.append(
                    {
                        "role": "tool",
                        "content": str(msg.content),
                        "tool_call_id": msg.tool_call_id,
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls
                tool_calls_formatted = [
                    {
                        "id": tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                    for tool_call in msg.tool_calls
                ]
                formatted_messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": tool_calls_formatted,
                    }
                )
            else:
                # Regular message - convert to standard format
                formatted_messages.append(msg.model_dump(exclude_none=True))

        return formatted_messages

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to OpenRouter/OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters.model_dump() if tool.parameters else {},
                },
            }
            for tool in tools
        ]

    async def _make_async_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Makes the actual async API request to OpenRouter."""
        tools_param = NOT_GIVEN if tools is None else tools

        return await self._async_client.chat.completions.create(
            model=model, messages=messages, stream=stream, tools=tools_param, **kwargs
        )

    def _process_provider_stream_event(
        self, event: ChatCompletionChunk, processor: StreamProcessor
    ) -> ChimericStreamChunk[ChatCompletionChunk] | None:
        """Processes an OpenRouter async stream event using the standardized processor.

        This is the same implementation as the sync client since event processing
        is identical.
        """
        if event.choices and event.choices[0].delta.content:
            delta = event.choices[0].delta.content
            return create_stream_chunk(native_event=event, processor=processor, content_delta=delta)

        # Handle tool calls in streaming
        if event.choices and event.choices[0].delta.tool_calls:
            for tool_call_delta in event.choices[0].delta.tool_calls:
                call_id = tool_call_delta.id or f"tool_call_{getattr(tool_call_delta, 'index', 0)}"

                if tool_call_delta.function and tool_call_delta.function.name:
                    processor.process_tool_call_start(call_id, tool_call_delta.function.name)

                if tool_call_delta.function and tool_call_delta.function.arguments:
                    processor.process_tool_call_delta(call_id, tool_call_delta.function.arguments)

        # Handle completion
        if event.choices and event.choices[0].finish_reason:
            # Mark any streaming tool calls as complete
            for call_id in processor.state.tool_calls:
                processor.process_tool_call_complete(call_id)

            return create_stream_chunk(
                native_event=event,
                processor=processor,
                finish_reason=event.choices[0].finish_reason,
            )

        return None

    def _extract_usage_from_response(self, response: ChatCompletion) -> Usage:
        """Extracts usage information from OpenRouter response."""
        if not response.usage:
            return Usage()

        return Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _extract_content_from_response(self, response: ChatCompletion) -> str | list[Any]:
        """Extracts content from OpenRouter response."""
        if not response.choices:
            return ""

        choice = response.choices[0]
        return choice.message.content or ""

    def _extract_tool_calls_from_response(self, response: ChatCompletion) -> list[ToolCall] | None:
        """Extracts tool calls from OpenRouter response."""
        if not response.choices:
            return None

        choice = response.choices[0]
        if not choice.message.tool_calls:
            return None

        return [
            ToolCall(
                call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in choice.message.tool_calls
        ]

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: ChatCompletion | ChatCompletionChunk | Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results asynchronously.

        For OpenRouter/OpenAI, this follows the chat completions format where:
        1. Assistant message with tool calls is added
        2. Tool result messages are added for each tool execution
        """
        updated_messages = list(messages)

        # Add the assistant message with tool calls
        if isinstance(assistant_response, ChatCompletion) and assistant_response.choices:
            choice = assistant_response.choices[0]
            if choice.message.tool_calls:
                updated_messages.append(
                    {
                        "role": "assistant",
                        "content": choice.message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ],
                    }
                )
        else:
            # For streaming responses, reconstruct the assistant message
            tool_calls_formatted = [
                {
                    "id": tool_call.call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
                for tool_call in tool_calls
            ]
            updated_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls_formatted,
                }
            )

        # Add the tool result messages
        for result in tool_results:
            updated_messages.append(
                {
                    "role": "tool",
                    "content": result.result if not result.is_error else f"Error: {result.error}",
                    "tool_call_id": result.call_id,
                }
            )

        return updated_messages
