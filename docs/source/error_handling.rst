Error Handling and Network Interruptions
==========================================

This guide explains how to handle errors using Lexilux's unified exception system
and distinguish between network problems and normal API completions.

.. note::
   As of v2.2.0, Lexilux provides a comprehensive exception hierarchy with error codes
   and retryable flags. See :ref:`lexilux-exceptions` for details.

Understanding finish_reason
---------------------------

The ``finish_reason`` field indicates why a chat completion stopped. It is only
available when the API successfully returns a response:

- **"stop"**: Model stopped naturally or hit a stop sequence
- **"length"**: Reached max_tokens limit
- **"content_filter"**: Content was filtered
- **None**: Unknown or not provided (some APIs may not provide this)

**Important**: ``finish_reason`` is **NOT** set when network errors occur.

Distinguishing Network Errors from Normal Completion
----------------------------------------------------

### Non-Streaming Requests

For non-streaming requests (``chat()`` method):

**Network Error**:
- A Lexilux exception is raised (``ConnectionError``, ``TimeoutError``, etc.)
- No ``ChatResult`` is returned
- No ``finish_reason`` is available

**Normal Completion**:
- ``ChatResult`` is returned successfully
- ``finish_reason`` is set to a valid value ("stop", "length", "content_filter", or None)

Example:

.. code-block:: python

   from lexilux import Chat, LexiluxError, ConnectionError, TimeoutError

   chat = Chat(base_url="https://api.example.com/v1", api_key="key", model="gpt-4")

   try:
       result = chat("Hello, world!")
       # Success: finish_reason indicates why generation stopped
       print(f"Completed: {result.finish_reason}")
       print(f"Text: {result.text}")
   except ConnectionError as e:
       # Network error: no finish_reason available
       print(f"Connection failed: {e.message}")
       print(f"Error code: {e.code}")  # "connection_failed"
       print(f"Can retry: {e.retryable}")  # True
   except TimeoutError as e:
       print(f"Request timeout: {e.message}")
       print(f"Can retry: {e.retryable}")  # True
   except LexiluxError as e:
       print(f"Error: {e.code} - {e.message}")

### Streaming Requests

For streaming requests (``chat.stream()`` method):

**Network Error**:
- An exception is raised during iteration
- The iterator stops yielding chunks
- If interrupted before receiving a ``done=True`` chunk, no ``finish_reason`` is available

**Normal Completion**:
- A chunk with ``done=True`` is received
- ``finish_reason`` is set in that chunk (or may be None for [DONE] messages)

**Incomplete Stream**:
- Exception raised after receiving some chunks
- Check if any chunk has ``done=True`` to determine if completion occurred before interruption

Example:

.. code-block:: python

   from lexilux import Chat, ConnectionError, LexiluxError

   chat = Chat(base_url="https://api.example.com/v1", api_key="key", model="gpt-4")

   try:
       chunks = []
       for chunk in chat.stream("Write a long story"):
           print(chunk.delta, end="", flush=True)
           chunks.append(chunk)

       # Check if we received a completion
       done_chunks = [c for c in chunks if c.done]
       if done_chunks:
           final_chunk = done_chunks[-1]
           print(f"\nCompleted: {final_chunk.finish_reason}")
       else:
           print("\nStream ended without completion signal")

   except ConnectionError as e:
       # Network error during streaming
       print(f"\nConnection lost: {e.message}")

       # Check if we got any completion before the error
       done_chunks = [c for c in chunks if c.done]
       if done_chunks:
           print("Completion occurred before network error")
           print(f"Finish reason: {done_chunks[-1].finish_reason}")
       else:
           print("No completion received - stream was interrupted")

.. _lexilux-exceptions:

Lexilux Exception Hierarchy
----------------------------

All Lexilux exceptions inherit from ``LexiluxError`` and provide three properties:

- **``code``**: Machine-readable error code (e.g., "authentication_failed")
- **``message``**: Human-readable error message
- **``retryable``**: Boolean indicating if the error can be retried

Exception Hierarchy:

.. code-block:: text

   LexiluxError (base class)
   ├── APIError
   │   ├── AuthenticationError (401, not retryable)
   │   ├── RateLimitError (429, retryable)
   │   ├── TimeoutError (retryable)
   │   ├── ConnectionError (retryable)
   │   ├── ValidationError (400, not retryable)
   │   ├── NotFoundError (404, not retryable)
   │   └── ServerError (5xx, retryable)
   ├── InvalidRequestError (alias for ValidationError)
   ├── ConfigurationError (not retryable)
   └── NetworkError (base class for network issues)

Common Exceptions:

``AuthenticationError``
~~~~~~~~~~~~~~~~~~~~~~~

Authentication/authorization failures (HTTP 401).

.. code-block:: python

   from lexilux import Chat, AuthenticationError

   chat = Chat(base_url="https://api.example.com/v1", api_key="invalid-key")

   try:
       result = chat("Hello")
   except AuthenticationError as e:
       print(f"Auth failed: {e.message}")
       print(f"Check your API key")
       # Don't retry - won't work without valid key

``RateLimitError``
~~~~~~~~~~~~~~~~~~~

Rate limit exceeded (HTTP 429).

.. code-block:: python

   from lexilux import Chat, RateLimitError

   chat = Chat(base_url="https://api.example.com/v1", api_key="key")

   try:
       result = chat("Hello")
   except RateLimitError as e:
       print(f"Rate limited: {e.message}")
       print(f"Wait and retry, or upgrade your plan")
       # Can retry after waiting

``TimeoutError``
~~~~~~~~~~~~~~~~~

Request timed out.

.. code-block:: python

   from lexilux import Chat, TimeoutError

   # Increase timeout
   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="key",
       timeout_s=120,  # 2 minutes
   )

   try:
       result = chat("Write a long response")
   except TimeoutError as e:
       print(f"Request timeout: {e.message}")
       # Can retry with longer timeout

``ConnectionError``
~~~~~~~~~~~~~~~~~~~~

Failed to establish connection.

.. code-block:: python

   from lexilux import Chat, ConnectionError

   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="key",
       max_retries=3,  # Enable automatic retry
   )

   try:
       result = chat("Hello")
   except ConnectionError as e:
       print(f"Connection failed: {e.message}")
       print(f"Check your network connection")

``ServerError``
~~~~~~~~~~~~~~~

Internal server errors (HTTP 5xx).

.. code-block:: python

   from lexilux import Chat, ServerError

   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="key",
       max_retries=3,  # Auto-retry on 5xx errors
   )

   try:
       result = chat("Hello")
   except ServerError as e:
       print(f"Server error: {e.message}")
       # Server might be temporarily unavailable

``ValidationError``
~~~~~~~~~~~~~~~~~~~

Invalid input (HTTP 400).

.. code-block:: python

   from lexilux import Chat, ValidationError

   chat = Chat(base_url="https://api.example.com/v1", api_key="key")

   try:
       result = chat("Hello", temperature=2.5)  # Invalid: temperature must be 0-2
   except ValidationError as e:
       print(f"Invalid input: {e.message}")
       # Fix the input and retry

``NotFoundError``
~~~~~~~~~~~~~~~~~

Resource not found (HTTP 404).

.. code-block:: python

   from lexilux import Chat, NotFoundError

   chat = Chat(base_url="https://api.example.com/v1", api_key="key")

   try:
       result = chat("Hello", model="nonexistent-model")
   except NotFoundError as e:
       print(f"Not found: {e.message}")
       # Check the model name and endpoint

Automatic Retry Logic
---------------------

.. versionadded:: 2.2.0

Lexilux supports automatic retry with exponential backoff for transient failures:

.. code-block:: python

   from lexilux import Chat

   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="key",
       max_retries=3,  # Automatically retry retryable errors
   )

   result = chat("Hello")
   # Will automatically retry on:
   # - 429 (rate limit)
   # - 500, 502, 503, 504 (server errors)
   # - Timeout
   # - Connection failures

**Retry Behavior**:

- Exponential backoff: 0.1s, 0.2s, 0.4s...
- Only retries errors with ``retryable=True``
- Up to ``max_retries`` attempts
- Does not retry authentication or validation errors

.. warning::
   Automatic retry is disabled by default (``max_retries=0``).
   Enable it if you want automatic recovery from transient failures.

Manual Retry with Retryable Flag
-----------------------------------

For more control, use the ``retryable`` flag to implement custom retry logic:

.. code-block:: python

   import time
   from lexilux import Chat, LexiluxError

   chat = Chat(base_url="https://api.example.com/v1", api_key="key")

   max_retries = 5
   for attempt in range(max_retries):
       try:
           result = chat("Hello, world!")
           break  # Success
       except LexiluxError as e:
           if e.retryable and attempt < max_retries - 1:
               wait_time = 2 ** attempt  # Exponential backoff
               print(f"Attempt {attempt + 1} failed: {e.code}")
               print(f"Retrying in {wait_time}s...")
               time.sleep(wait_time)
           else:
               raise  # Non-retryable or max retries reached

.. tip::
   Check ``error.retryable`` before implementing retry logic.
   This prevents unnecessary retries on permanent failures (like invalid API keys).

Enabling Logging for Debugging
-------------------------------

.. versionadded:: 2.2.0

Enable logging to see request timing and errors:

.. code-block:: python

   import logging

   # Enable INFO level logging
   logging.basicConfig(level=logging.INFO)

   from lexilux import Chat
   chat = Chat(base_url="https://api.example.com/v1", api_key="key")

   result = chat("Hello")
   # Logs: "Request completed in 0.52s with status 200: https://..."

Log levels:

- **DEBUG**: Request URL, timeout configuration
- **INFO**: Request success, timing, status code
- **WARNING**: HTTP errors (4xx, 5xx)
- **ERROR**: Timeout, connection failures

.. seealso::
   :doc:`troubleshooting` - Guide for diagnosing and fixing common issues.

Handling Incomplete Responses
------------------------------

When using ``chat.complete()`` or continuation functionality, you may encounter
``ChatIncompleteResponse`` if the response is still truncated after maximum continues.

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError

   chat = Chat(...)
   history = ChatHistory()

   try:
       result = chat.complete("Very long response", history=history, max_tokens=30, max_continues=2)
   except ChatIncompleteResponse as e:
       print(f"Still incomplete after {e.continue_count} continues")
       print(f"Received: {len(e.final_result.text)} chars")
       # Use partial result if acceptable
       result = e.final_result

   # Or allow partial results
   result = chat.complete(
       "Very long response",
       history=history,
       max_tokens=30,
       max_continues=2,
       ensure_complete=False  # Returns partial result instead of raising
   )
   if result.finish_reason == "length":
       print("Warning: Response was truncated")

Handling Streaming Interruptions
---------------------------------

When streaming is interrupted, partial content is preserved in history:

.. code-block:: python

   from lexilux import Chat, ChatHistory

   chat = Chat(...)
   history = ChatHistory()

   iterator = chat.stream("Long response", history=history)
   try:
       for chunk in iterator:
           print(chunk.delta, end="")
   except requests.RequestException as e:
       print(f"\nStream interrupted: {e}")
       
       # Partial content is preserved in history
       if history.messages:
           last_msg = history.messages[-1]
           if last_msg.get("role") == "assistant":
               print(f"Partial content: {len(last_msg['content'])} chars")
       
       # Clean up if needed
       chat.clear_last_assistant_message()

Lexilux-Specific Exceptions
----------------------------

ChatIncompleteResponse
~~~~~~~~~~~~~~~~~~~~~~

Raised when a response is still incomplete after maximum continuation attempts.

.. code-block:: python

   from lexilux.chat.exceptions import ChatIncompleteResponse

   try:
       result = chat.complete("Very long response", max_tokens=30, max_continues=2)
   except ChatIncompleteResponse as e:
       print(f"Final result: {e.final_result.text}")
       print(f"Continue count: {e.continue_count}")
       print(f"Max continues: {e.max_continues}")

ChatStreamInterrupted
~~~~~~~~~~~~~~~~~~~~~

Raised when a streaming request is interrupted before completion (if implemented).

.. code-block:: python

   from lexilux.chat.exceptions import ChatStreamInterrupted

   try:
       iterator = chat.stream("Long response")
       for chunk in iterator:
           print(chunk.delta, end="")
   except ChatStreamInterrupted as e:
       print(f"Interrupted. Received: {len(e.get_partial_text())} chars")
       partial_result = e.get_partial_result()
       # Can try to recover using ChatContinue or retry

Best Practices
--------------

1. **Always use try-except blocks** when making API calls:

   .. code-block:: python

      try:
          result = chat("Hello")
          if result.finish_reason:
              print(f"Normal completion: {result.finish_reason}")
      except requests.RequestException as e:
          print(f"Network error: {e}")

2. **Use chat.complete() for guaranteed complete responses**:

   .. code-block:: python

      from lexilux import Chat, ChatHistory
      from lexilux.chat.exceptions import ChatIncompleteResponseError

      chat = Chat(...)
      history = ChatHistory()

      try:
          result = chat.complete("Extract JSON", history=history, max_tokens=100)
          json_data = json.loads(result.text)  # Guaranteed complete
      except ChatIncompleteResponseError as e:
          print(f"Still incomplete: {e.final_result.text}")
          # Handle partial result

3. **For streaming, track completion status and clean up on error**:

   .. code-block:: python

      completed = False
      try:
          for chunk in chat.stream("Hello"):
              print(chunk.delta, end="")
              if chunk.done:
                  completed = True
                  print(f"\nFinished: {chunk.finish_reason}")
      except requests.RequestException as e:
          if completed:
              print(f"\nCompleted before error: {e}")
          else:
              print(f"\nInterrupted: {e}")
              # Clean up partial response if needed
              # You manage history explicitly
              if history.messages and history.messages[-1].get("role") == "assistant":
                  history.remove_last()

4. **Handle history behavior on errors**:

   .. code-block:: python

      from lexilux import Chat, ChatHistory

      chat = Chat(...)
      history = ChatHistory()

      # Non-streaming: Exception means no assistant response in history
      try:
          result = chat("Hello", history=history)
      except Exception:
          # History contains user message but NO assistant response
          # (user message is added before request, assistant only on success)
          # Only user messages, no assistant responses for failed calls

      # Streaming: Partial content is preserved, clean up if needed
      iterator = chat.stream("Long response")
      try:
          for chunk in iterator:
              print(chunk.delta)
      except Exception:
          # Partial content is in history
          chat.clear_last_assistant_message()  # Clean up

5. **Check finish_reason only after successful response**:

   .. code-block:: python

      # Correct: finish_reason is only available on success
      result = chat("Hello")
      if result.finish_reason == "length":
          print("Hit token limit")
      
      # Incorrect: finish_reason won't exist if exception is raised
      # try:
      #     result = chat("Hello")
      # except Exception:
      #     print(result.finish_reason)  # ERROR: result may not exist

6. **Use retry logic for network errors**:

   .. code-block:: python

      import time
      from requests import RequestException

      max_retries = 3
      for attempt in range(max_retries):
          try:
              result = chat("Hello")
              break  # Success
          except RequestException as e:
              if attempt < max_retries - 1:
                  wait_time = 2 ** attempt  # Exponential backoff
                  time.sleep(wait_time)
                  continue
              raise  # Last attempt failed

