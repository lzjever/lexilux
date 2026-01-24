"""
Example Wolo GLM Client Adapter using lexilux.

This file demonstrates how to create a wolo-specific adapter that leverages
lexilux's standard OpenAI client for HTTP/SSE/error handling while providing
wolo-specific functionality.

This is a reference implementation for the wolo project integration.
"""

import json
import logging
import os
import platform
import time
from typing import AsyncIterator

from lexilux.chat import Chat
from lexilux.chat.params import ChatParams

logger = logging.getLogger(__name__)


class WoloConfig:
    """Example wolo configuration class."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = 0.7
        self.max_tokens = None
        self.enable_think = False  # GLM thinking mode
        self.debug_llm_file = None
        self.debug_full_dir = None


class WoloGLMClient:
    """
    Example Wolo GLM client adapter using lexilux.

    This adapter demonstrates the corrected approach:
    - Uses lexilux for standard OpenAI client functionality
    - Focuses on wolo product features (headers, debugging, event conversion)
    - ✅ NO LONGER includes topP/topK error parameters
    - ✅ Uses standard OpenAI parameters only
    - ✅ Supports reasoning models via lexilux's include_reasoning
    """

    def __init__(
        self,
        config: WoloConfig,
        session_id: str | None = None,
        agent_display_name: str | None = None,
    ):
        """Initialize wolo GLM client adapter."""
        # Build opencode headers (wolo product feature)
        headers = self._build_opencode_headers(session_id, agent_display_name)

        # Initialize lexilux Chat client
        self._lexilux_chat = Chat(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            headers=headers,
        )

        # ✅ CORRECTED: Store parameters separately for use in calls
        self._default_params = ChatParams(
            temperature=config.temperature or 0.7,
            max_tokens=config.max_tokens,
            # ✅ NO MORE: topP, topK, maxOutputTokens error parameters
            # ✅ Only use extra for legitimate provider-specific features if needed
            extra=None,  # Keep clean unless real provider features needed
        )

        # Wolo product configuration
        self._enable_think = config.enable_think
        self._debug_llm_file = config.debug_llm_file
        self._debug_full_dir = config.debug_full_dir
        self._request_count = 0

    def _build_opencode_headers(
        self, session_id: str | None, agent_display_name: str | None
    ) -> dict[str, str]:
        """Build opencode-specific headers (wolo product feature)."""
        try:
            current_dir = os.path.basename(os.getcwd())
        except OSError:
            current_dir = "unknown"

        return {
            "x-opencode-project": current_dir,
            "x-opencode-session": session_id or "unknown",
            "x-opencode-request": "user",
            "x-opencode-client": "cli",
            "User-Agent": f"opencode/1.0.0 ({platform.system()} {platform.machine()})",
        }

    def _log_request(self, messages: list[dict]) -> None:
        """Wolo product-specific debugging logs."""
        if not self._debug_llm_file and not self._debug_full_dir:
            return

        self._request_count += 1

        try:
            # Incremental debug log
            if self._debug_llm_file:
                with open(self._debug_llm_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"Request #{self._request_count}: {self._lexilux_chat.model}\n"
                    )
                    f.write(f"Messages: {len(messages)}\n")
                    if messages:
                        last_content = messages[-1].get("content", "")
                        preview = (
                            last_content[:100] + "..."
                            if len(last_content) > 100
                            else last_content
                        )
                        f.write(f"Last message: {preview}\n")
                    f.write("---\n")

            # Full debug log with complete request/response
            if self._debug_full_dir:
                debug_file = os.path.join(
                    self._debug_full_dir, f"request_{self._request_count}.json"
                )
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "request_id": self._request_count,
                            "model": self._lexilux_chat.model,
                            "messages": messages,
                            "timestamp": time.time(),
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

        except Exception as e:
            logger.warning(f"Failed to write debug logs: {e}")

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[dict]:
        """
        Main chat completion method.

        Converts lexilux events to wolo format, focusing on:
        - Standard OpenAI functionality via lexilux
        - Reasoning support via include_reasoning
        - Product-specific event format conversion
        - ✅ NO MORE topP/topK error parameter handling
        """
        # 1. Product-level debugging
        self._log_request(messages)

        # 2. Format conversion (lexilux and wolo should be compatible)
        lexilux_messages = self._to_lexilux_messages(messages)
        lexilux_tools = self._convert_tools(tools)

        # 3. ✅ Call lexilux with reasoning support (NEW FEATURE)
        try:
            lexilux_stream = self._lexilux_chat.astream(
                messages=lexilux_messages,
                tools=lexilux_tools,
                params=self._default_params,  # ✅ Use stored parameters
                include_reasoning=self._enable_think,  # ✅ Use lexilux reasoning support
            )

            # 4. Convert lexilux events to wolo format
            async for lexilux_chunk in lexilux_stream:
                # ✅ NEW: Reasoning content (standard OpenAI/Claude/DeepSeek feature)
                if lexilux_chunk.reasoning_content:
                    yield {
                        "type": "reasoning-delta",
                        "text": lexilux_chunk.reasoning_content,
                    }

                # Standard text content
                if lexilux_chunk.delta:
                    yield {"type": "text-delta", "text": lexilux_chunk.delta}

                # Tool calls
                if lexilux_chunk.tool_calls:
                    for tc in lexilux_chunk.tool_calls:
                        yield {
                            "type": "tool-call",
                            "tool": tc.name,
                            "input": tc.get_arguments(),
                            "id": tc.id,
                        }

                # Completion
                if lexilux_chunk.done:
                    yield {
                        "type": "finish",
                        "reason": lexilux_chunk.finish_reason,
                        "usage": {
                            "input_tokens": lexilux_chunk.usage.input_tokens or 0,
                            "output_tokens": lexilux_chunk.usage.output_tokens or 0,
                            "total_tokens": lexilux_chunk.usage.total_tokens or 0,
                        },
                    }

        except Exception as e:
            # Convert lexilux exceptions to wolo format
            logger.error(f"Chat completion failed: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def _to_lexilux_messages(self, messages: list[dict]) -> list[dict]:
        """Convert wolo message format to lexilux format."""
        # Most formats should be compatible, add conversion logic if needed
        return messages

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Convert wolo tool format to lexilux format."""
        if not tools:
            return None
        # Convert if needed, likely formats are compatible
        return tools


# ✅ EXAMPLE USAGE DEMONSTRATION
async def example_usage():
    """Demonstrate how to use the corrected wolo adapter."""
    config = WoloConfig(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="your-api-key",
        model="glm-4",
    )
    config.enable_think = True  # Enable reasoning mode

    client = WoloGLMClient(config, session_id="test-session")

    messages = [{"role": "user", "content": "Solve this math problem: What is 2^10?"}]

    print("🚀 Using corrected wolo adapter with lexilux:")
    print("✅ Standard OpenAI parameters only")
    print("✅ Reasoning support via include_reasoning")
    print("✅ Product features (headers, debugging) preserved")
    print("❌ NO MORE topP/topK error parameters")
    print()

    async for event in client.chat_completion(messages):
        if event["type"] == "reasoning-delta":
            print(f"🤔 Reasoning: {event['text']}")
        elif event["type"] == "text-delta":
            print(f"💬 Response: {event['text']}")
        elif event["type"] == "finish":
            print(f"✅ Done: {event['reason']}")
            print(f"📊 Usage: {event['usage']}")


if __name__ == "__main__":
    print(__doc__)
    # asyncio.run(example_usage())  # Uncomment to run example
