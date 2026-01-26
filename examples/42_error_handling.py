#!/usr/bin/env python
"""
42 Error Handling - Robust Error Management

Learn how to handle various error types and implement retry logic.

Level: Expert Topic
"""

from config_loader import get_chat_config, parse_args

from lexilux import Chat
from lexilux.exceptions import (
    APIError,
    AuthenticationError,
    ConnectionError as LexiluxConnectionError,
    InvalidRequestError,
    RateLimitError,
    TimeoutError as LexiluxTimeoutError,
)


def main():
    """Demonstrate comprehensive error handling."""
    args = parse_args()
    try:
        config = get_chat_config(config_path=args.config)
    except (FileNotFoundError, KeyError) as e:
        print(f"Configuration error: {e}")
        print("\nUsing placeholder values. Please configure test_endpoints.json")
        config = {
            "base_url": "https://api.example.com/v1",
            "api_key": "your-api-key",
            "model": "gpt-4",
        }

    chat = Chat(**config)

    # Example 1: Basic error handling
    print("=" * 50)
    print("Example 1: Basic Error Handling")
    print("=" * 50)

    try:
        result = chat("Hello, world!")
        print(f"Response: {result.text}")
    except AuthenticationError as e:
        print(f"Authentication failed: {e.message}")
        print("  Check your API key")
    except RateLimitError as e:
        print(f"Rate limited: {e.message}")
        print(f"  Retryable: {e.retryable}")
    except LexiluxTimeoutError as e:
        print(f"Request timed out: {e.message}")
        print(f"  Retryable: {e.retryable}")
    except APIError as e:
        print(f"API error: {e.code} - {e.message}")

    # Example 2: Specific error handling
    print("\n" + "=" * 50)
    print("Example 2: Handle Specific Errors Differently")
    print("=" * 50)

    def safe_chat(prompt: str) -> str | None:
        """Chat with comprehensive error handling."""
        try:
            result = chat(prompt)
            return result.text
        except AuthenticationError:
            print("❌ Authentication failed - check your API key")
            return None
        except RateLimitError:
            print("⏱️ Rate limited - please wait a moment")
            return None
        except InvalidRequestError as e:
            print(f"❌ Invalid request: {e.message}")
            return None
        except LexiluxConnectionError:
            print("❌ Connection failed - check your network")
            return None
        except LexiluxTimeoutError:
            print("⏱️ Request timed out - try again")
            return None
        except APIError as e:
            print(f"❌ API error ({e.code}): {e.message}")
            return None

    response = safe_chat("Hello!")
    if response:
        print(f"✓ Got response: {response[:50]}...\n")

    # Example 3: Retry logic
    print("=" * 50)
    print("Example 3: Implement Retry Logic")
    print("=" * 50)

    import time

    def chat_with_retry(
        prompt: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> str | None:
        """Chat with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                result = chat(prompt)
                return result.text
            except RateLimitError as e:
                if e.retryable and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    print(
                        f"Rate limited. Waiting {delay:.1f}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    print(f"Rate limit exceeded after {max_retries} attempts")
                    return None
            except APIError as e:
                if e.retryable and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    print(
                        f"Temporary error. Retrying in {delay:.1f}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    print(f"Failed after {max_retries} attempts: {e.message}")
                    return None
        return None

    response = chat_with_retry("Hello with retry!")
    if response:
        print(f"✓ Success: {response[:50]}...\n")

    # Example 4: Streaming error handling
    print("=" * 50)
    print("Example 4: Handle Errors in Streaming")
    print("=" * 50)

    print("Streaming with error handling:\n")

    try:
        got_response = False
        for chunk in chat.stream("Tell me a joke"):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
                got_response = True

            if chunk.done:
                print("\n✓ Streaming completed successfully")
                print(f"Tokens: {chunk.usage.total_tokens}\n")
                break
        else:
            # Loop completed without done=True
            if got_response:
                print("\n⚠️ Stream interrupted before completion")
            else:
                print("\n❌ No content received")

    except AuthenticationError as e:
        print(f"\n❌ Authentication failed: {e.message}")
    except RateLimitError as e:
        print(f"\n❌ Rate limited: {e.message}")
    except APIError as e:
        print(f"\n❌ API error: {e.code} - {e.message}")

    # Example 5: Validate input before request
    print("=" * 50)
    print("Example 5: Input Validation")
    print("=" * 50)

    def validated_chat(prompt: str, max_length: int = 10000) -> str | None:
        """Chat with input validation."""
        # Validate input
        if not prompt or not prompt.strip():
            print("❌ Empty prompt")
            return None

        if len(prompt) > max_length:
            print(f"❌ Prompt too long ({len(prompt)} > {max_length})")
            return None

        # Make request
        try:
            result = chat(prompt)
            return result.text
        except APIError as e:
            print(f"❌ Request failed: {e.message}")
            return None

    # Test validation
    result = validated_chat("Valid prompt")
    if result:
        print(f"✓ Valid request succeeded: {result[:50]}...\n")

    result = validated_chat("")
    print("✓ Empty prompt caught\n")

    # Example 6: Error context for debugging
    print("=" * 50)
    print("Example 6: Extract Error Context")
    print("=" * 50)

    try:
        # Use invalid model to trigger error
        bad_chat = Chat(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model="nonexistent-model-12345",
        )
        bad_chat("This will fail")
    except APIError as e:
        print("Error details:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Code: {e.code}")
        print(f"  Message: {e.message}")
        print(f"  Retryable: {e.retryable}")
        if hasattr(e, "response"):
            print(f"  Status code: {getattr(e.response, 'status_code', 'N/A')}")

    print("\n✓ Error handling completed!")


if __name__ == "__main__":
    main()
