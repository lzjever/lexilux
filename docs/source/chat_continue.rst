Continue Generation
===================

Lexilux provides functionality to continue generation when responses are cut off
due to token limits, allowing you to seamlessly extend incomplete responses.

.. important::
   **History Immutability**: All methods that receive a `history` parameter create a clone internally
   and **never modify the original history object**. You must manually update your history after each API call.

Overview
--------

When a chat completion is stopped due to ``max_tokens`` limit (``finish_reason == "length"``),
you may want to continue the generation. Lexilux provides multiple ways to handle this:

1. **Chat.complete()** - Recommended for most cases, ensures complete response
2. **ChatContinue.continue_request()** - Advanced control with full flexibility
3. **Streaming versions** - ``complete_stream()`` and ``continue_request_stream()``

Key Features
------------

1. **History Immutability**: All methods clone history internally, never modify original
2. **Optional History**: `complete()` methods can work without history (creates internally)
3. **Multiple Continues**: Automatically continue multiple times if needed
4. **Result Merging**: Automatically merge all continuation results
5. **Usage Aggregation**: Automatically combine token usage from multiple requests
6. **Streaming Support**: Stream continuation chunks in real-time
7. **Customizable Strategy**: Progress tracking, custom prompts, delays, error handling

When to Use
-----------

Use continuation when:

* A response has ``finish_reason == "length"`` (cut off due to token limit)
* You need complete responses (e.g., JSON extraction)
* You're working with long-form content generation
* You want to ensure response completeness

Recommended Approach: Chat.complete()
-------------------------------------

The simplest and most recommended way to ensure complete responses:

**Single-turn conversation (no history needed):**

.. code-block:: python

   from lexilux import Chat
   import json

   chat = Chat(...)

   # Automatically handles truncation, returns complete result
   # No history needed for single-turn conversations
   result = chat.complete("Write a long JSON response", max_tokens=100)
   json_data = json.loads(result.text)  # Guaranteed complete

**Multi-turn conversation (with history):**

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError

   chat = Chat(...)
   history = ChatHistory()

   # First turn
   result1 = chat.complete("First question", history=history, max_tokens=100)
   # Manually update history (history is immutable)
   history.add_user("First question")
   history.append_result(result1)

   # Second turn
   try:
       result2 = chat.complete("Follow-up question", history=history, max_tokens=100, max_continues=3)
       history.add_user("Follow-up question")
       history.append_result(result2)
   except ChatIncompleteResponseError as e:
       print(f"Still incomplete after {e.continue_count} continues")
       print(f"Received: {len(e.final_result.text)} chars")

Key Features of complete():
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Automatically continues if ``finish_reason == "length"``
* Supports multiple continues (``max_continues`` parameter, default: 5)
* Raises ``ChatIncompleteResponseError`` if still truncated (if ``ensure_complete=True``)
* History parameter is **optional** (creates internally if None)
* History is **immutable** - original never modified

Customizable Continue Strategy
-------------------------------

The ``complete()`` method now supports extensive customization options:

**Custom Continue Prompt (Function):**

.. code-block:: python

   from lexilux import Chat

   chat = Chat(...)

   # Custom continue prompt function
   def smart_prompt(count, max_count, current_text, original_prompt):
       if "JSON" in original_prompt:
           return "Please continue the JSON response, ensuring valid format."
       return f"Please continue (attempt {count}/{max_count})"

   result = chat.complete(
       "Write a long JSON",
       max_tokens=100,
       continue_prompt=smart_prompt,
   )

**Progress Tracking:**

.. code-block:: python

   def on_progress(count, max_count, current_result, all_results):
       print(f"🔄 Continuing {count}/{max_count}...")
       print(f"   Current length: {len(current_result.text)} chars")
       print(f"   Total parts: {len(all_results)}")

   result = chat.complete(
       "Write a long story",
       max_tokens=100,
       on_progress=on_progress,
   )

**Request Delay:**

.. code-block:: python

   # Fixed delay (1 second between continues)
   result = chat.complete(
       "Write JSON",
       max_tokens=100,
       continue_delay=1.0,
   )

   # Random delay (1-2 seconds)
   result = chat.complete(
       "Write JSON",
       max_tokens=100,
       continue_delay=(1.0, 2.0),
   )

**Error Handling:**

.. code-block:: python

   # Return partial result on error instead of raising
   result = chat.complete(
       "Write JSON",
       max_tokens=100,
       on_error="return_partial",  # Returns partial instead of raising
   )

   # Custom error callback
   def on_error_callback(error, partial_result):
       print(f"Error during continue: {error}")
       return {"action": "return_partial"}

   result = chat.complete(
       "Write JSON",
       max_tokens=100,
       on_error_callback=on_error_callback,
   )

Advanced Control: ChatContinue.continue_request()
---------------------------------------------------

For advanced use cases requiring full control:

**Basic Usage:**

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()

   # Initial request
   result = chat("Write a long story", history=history, max_tokens=50)
   # Manually update history
   history.add_user("Write a long story")
   history.append_result(result)
   
   if result.finish_reason == "length":
       # continue_request also doesn't modify original history
       full_result = ChatContinue.continue_request(
           chat,
           result,
           history=history,  # Required
           max_continues=3
       )
       # Update history with merged result
       history.append_result(full_result)
       print(full_result.text)  # Complete merged text

**Get All Intermediate Results:**

.. code-block:: python

   history = ChatHistory()
   result = chat("Story", history=history, max_tokens=50)
   history.add_user("Story")
   history.append_result(result)

   if result.finish_reason == "length":
       all_results = ChatContinue.continue_request(
           chat, result, history=history, auto_merge=False, max_continues=3
       )
       # all_results = [result, continue_result1, continue_result2, ...]
       for i, r in enumerate(all_results):
           print(f"Part {i+1}: {len(r.text)} chars")

**With Customization:**

.. code-block:: python

   def on_progress(count, max_count, current_result, all_results):
       print(f"Continue {count}/{max_count}")
       print(f"   Current length: {len(current_result.text)} chars")

   def custom_prompt(count, max_count, current_text, original_prompt):
       return f"Continue from: {current_text[-50:]}..."

   full_result = ChatContinue.continue_request(
       chat,
       result,
       history=history,
       continue_prompt=custom_prompt,
       on_progress=on_progress,
       continue_delay=0.5,
       max_continues=3,
   )

Key Parameters:
~~~~~~~~~~~~~~~

* ``history``: **Required**. ChatHistory instance (cloned internally, original unchanged)
* ``max_continues``: Maximum number of continuation attempts (default: 1)
* ``auto_merge``: If ``True``, automatically merge results (default: ``True``)
* ``add_continue_prompt``: Whether to add a user continue message (default: ``True``)
* ``continue_prompt``: User prompt for continuation (default: "continue"). Can be a string or
  a callable with signature: ``(count: int, max_count: int, current_text: str, original_prompt: str) -> str``
* ``on_progress``: Progress callback function with signature:
  ``(count: int, max_count: int, current_result: ChatResult, all_results: list[ChatResult]) -> None``
* ``continue_delay``: Delay between continues (float or tuple for random)
* ``on_error``: Error strategy ("raise" or "return_partial")
* ``on_error_callback``: Custom error callback function with signature:
  ``(error: Exception, partial_result: ChatResult) -> dict``.
  Should return ``{"action": "raise" | "return_partial" | "retry", "result": ChatResult}``

Return Types:
~~~~~~~~~~~~~

* If ``auto_merge=True``: Returns merged ``ChatResult``
* If ``auto_merge=False``: Returns list of ``ChatResult`` instances

Streaming Continue
------------------

Streaming versions provide real-time continuation:

complete_stream()
~~~~~~~~~~~~~~~~~

Stream complete response (handles truncation automatically):

**Single-turn (no history needed):**

.. code-block:: python

   from lexilux import Chat

   chat = Chat(...)

   # Automatically handles truncation and continues if needed
   iterator = chat.complete_stream(
       "Write a long JSON response",
       max_tokens=100,
       max_continues=3
   )
   
   for chunk in iterator:
       print(chunk.delta, end="", flush=True)
   
   # Result is guaranteed complete (or raises ChatIncompleteResponseError)
   result = iterator.result.to_chat_result()
   json_data = json.loads(result.text)

**Multi-turn (with history):**

.. code-block:: python

   from lexilux import Chat, ChatHistory

   chat = Chat(...)
   history = ChatHistory()

   # First turn
   iterator1 = chat.complete_stream("First question", history=history, max_tokens=100)
   for chunk in iterator1:
       print(chunk.delta, end="", flush=True)
   result1 = iterator1.result.to_chat_result()
   # Manually update history
   history.add_user("First question")
   history.append_result(result1)

   # Second turn
   iterator2 = chat.complete_stream("Follow-up", history=history, max_tokens=100)
   for chunk in iterator2:
       print(chunk.delta, end="", flush=True)
   result2 = iterator2.result.to_chat_result()
   history.add_user("Follow-up")
   history.append_result(result2)

**With Customization:**

.. code-block:: python

   def on_progress(count, max_count, current_result, all_results):
       print(f"\n🔄 Continuing {count}/{max_count}...")
       print(f"   Current length: {len(current_result.text)} chars")

   iterator = chat.complete_stream(
       "Write JSON",
       max_tokens=100,
       on_progress=on_progress,
       continue_delay=(1.0, 2.0),
   )
   
   for chunk in iterator:
       print(chunk.delta, end="", flush=True)

continue_request_stream()
~~~~~~~~~~~~~~~~~~~~~~~~~

Stream continuation chunks in real-time (for manual control):

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()

   # Initial request
   result = chat("Write a long story", history=history, max_tokens=50)
   # Manually update history
   history.add_user("Write a long story")
   history.append_result(result)
   
   if result.finish_reason == "length":
       # Stream continue chunks
       iterator = ChatContinue.continue_request_stream(
           chat, result, history=history, max_continues=2
       )
       
       for chunk in iterator:
           print(chunk.delta, end="", flush=True)
       
       # Get merged result
       full_result = iterator.result.to_chat_result()
       print(f"\nComplete: {len(full_result.text)} chars")
       # Update history with merged result
       history.append_result(full_result)

**With Customization:**

.. code-block:: python

   def on_progress(count, max_count, current_result, all_results):
       print(f"\n🔄 Continue {count}/{max_count}")
       print(f"   Current length: {len(current_result.text)} chars")

   iterator = ChatContinue.continue_request_stream(
       chat,
       result,
       history=history,
       continue_prompt=lambda c, m, t, p: f"Continue {c}/{m}",
       on_progress=on_progress,
       continue_delay=0.5,
       max_continues=3,
   )

Helper Method: needs_continue()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check if a result needs continuation:

.. code-block:: python

   from lexilux import ChatContinue

   result = chat("Write story", max_tokens=50)
   
   if ChatContinue.needs_continue(result):
       # result.finish_reason == "length"
       full_result = ChatContinue.continue_request(chat, result, history=history)

Result Merging
--------------

The ``merge_results()`` method combines multiple results:

.. code-block:: python

   from lexilux import ChatContinue

   history = ChatHistory()
   result1 = chat("Story part 1", history=history, max_tokens=50)
   history.add_user("Story part 1")
   history.append_result(result1)
   
   result2 = chat("Story part 2", history=history, max_tokens=50)
   history.add_user("Story part 2")
   history.append_result(result2)
   
   merged = ChatContinue.merge_results(result1, result2)
   # merged.text = result1.text + result2.text
   # merged.usage.total_tokens = result1.usage.total_tokens + result2.usage.total_tokens
   # merged.finish_reason = result2.finish_reason (from last result)

Common Patterns
---------------

Pattern 1: Ensure Complete Response (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``chat.complete()`` for scenarios requiring complete responses:

**Single-turn (simplest):**

.. code-block:: python

   # No history needed
   result = chat.complete("Extract data as JSON", max_tokens=100)
   json_data = json.loads(result.text)  # Guaranteed complete

**Multi-turn:**

.. code-block:: python

   history = ChatHistory()
   
   # JSON extraction
   result = chat.complete("Extract data as JSON", history=history, max_tokens=100)
   history.add_user("Extract data as JSON")
   history.append_result(result)
   json_data = json.loads(result.text)  # Guaranteed complete
   
   # Long-form content
   result2 = chat.complete("Write a comprehensive guide", history=history, max_tokens=200)
   history.add_user("Write a comprehensive guide")
   history.append_result(result2)

Pattern 2: Customizable Continue Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``chat.complete()`` with customization options:

.. code-block:: python

   def on_progress(count, max_count, current_result, all_results):
       print(f"🔄 Continuing {count}/{max_count}...")
       print(f"   Current length: {len(current_result.text)} chars")

   def smart_prompt(count, max_count, current_text, original_prompt):
       return f"Please continue (attempt {count}/{max_count})"

   # Single-turn conversation (no history needed)
   result = chat.complete(
       "Write JSON",
       max_tokens=100,
       continue_prompt=smart_prompt,
       on_progress=on_progress,
       continue_delay=(1.0, 2.0),
   )

Pattern 3: Advanced Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``ChatContinue.continue_request()`` for full control:

.. code-block:: python

   history = ChatHistory()
   
   result = chat("Story", history=history, max_tokens=50)
   history.add_user("Story")
   history.append_result(result)
   
   if result.finish_reason == "length":
       # Get all intermediate results
       all_results = ChatContinue.continue_request(
           chat, result, history=history, auto_merge=False, max_continues=3
       )
       for i, r in enumerate(all_results):
           print(f"Part {i+1}: {len(r.text)} chars")

Pattern 4: Streaming Continue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use streaming versions for real-time continuation:

**Using complete_stream() (recommended):**

.. code-block:: python

   # Single-turn (no history needed)
   iterator = chat.complete_stream("Long story", max_tokens=50)
   
   for chunk in iterator:
       print(chunk.delta, end="", flush=True)
   
   full_result = iterator.result.to_chat_result()
   print(f"\nComplete: {len(full_result.text)} chars")

**Using continue_request_stream() (manual control):**

.. code-block:: python

   history = ChatHistory()
   result = chat("Long story", history=history, max_tokens=50)
   history.add_user("Long story")
   history.append_result(result)
   
   if result.finish_reason == "length":
       iterator = ChatContinue.continue_request_stream(
           chat, result, history=history, max_continues=2
       )
       
       for chunk in iterator:
           print(chunk.delta, end="", flush=True)
       
       full_result = iterator.result.to_chat_result()
       history.append_result(full_result)

Error Handling
--------------

Handling Incomplete Responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using ``chat.complete()`` with ``ensure_complete=True`` (default),
``ChatIncompleteResponseError`` is raised if the response is still truncated
after ``max_continues``:

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError

   history = ChatHistory()
   
   try:
       result = chat.complete(
           "Very long response",
           history=history,
           max_tokens=30,
           max_continues=2
       )
       history.add_user("Very long response")
       history.append_result(result)
   except ChatIncompleteResponseError as e:
       print(f"Still incomplete after {e.continue_count} continues")
       print(f"Received: {len(e.final_result.text)} chars")
       # Use partial result if acceptable
       history.add_user("Very long response")
       history.append_result(e.final_result)

   # Or allow partial results
   result = chat.complete(
       "Very long response",
       history=history,
       max_tokens=30,
       max_continues=2,
       ensure_complete=False  # Returns partial result instead of raising
   )
   history.add_user("Very long response")
   history.append_result(result)
   if result.finish_reason == "length":
       print("Warning: Response was truncated")

Best Practices
--------------

1. **Use chat.complete() for Most Cases**: Simplest and most reliable

2. **Manually Update History**: Since history is immutable, always update after API calls:

   .. code-block:: python

      history = ChatHistory()
      result = chat.complete("Write JSON", history=history, max_tokens=100)
      # Don't forget!
      history.add_user("Write JSON")
      history.append_result(result)

3. **Set Appropriate max_continues**: Balance between completeness and API costs

4. **Handle ChatIncompleteResponseError**: Be prepared for cases where response
   is still incomplete after max_continues

5. **Monitor Token Usage**: Track total tokens across all continuations

6. **Consider Increasing max_tokens**: If you frequently need multiple continues,
   consider increasing ``max_tokens`` instead

7. **Use Helper Functions**: Create helpers to reduce boilerplate:

   .. code-block:: python

      def complete_and_update(chat, history, message, **kwargs):
          """Complete and automatically update history."""
          result = chat.complete(message, history=history, **kwargs)
          history.add_user(message)
          history.append_result(result)
          return result

      # Usage
      result = complete_and_update(chat, history, "Write JSON", max_tokens=100)

Examples
--------

Complete Workflow with complete()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Single-turn (no history):**

.. code-block:: python

   from lexilux import Chat
   import json

   chat = Chat(...)
   
   # Ensure complete JSON response
   result = chat.complete(
       "Extract user data as JSON",
       max_tokens=100,
       max_continues=3
   )
   
   # Guaranteed complete
   data = json.loads(result.text)

**Multi-turn (with history):**

.. code-block:: python

   from lexilux import Chat, ChatHistory
   import json

   chat = Chat(...)
   history = ChatHistory()
   
   # Ensure complete JSON response
   result = chat.complete(
       "Extract user data as JSON",
       history=history,
       max_tokens=100,
       max_continues=3
   )
   # Manually update history
   history.add_user("Extract user data as JSON")
   history.append_result(result)
   
   # Guaranteed complete
   data = json.loads(result.text)

Multiple Continues
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   history = ChatHistory()
   
   result = chat("Very long story", history=history, max_tokens=30)
   history.add_user("Very long story")
   history.append_result(result)
   
   if result.finish_reason == "length":
       # Automatically continues up to 3 times
       full_result = ChatContinue.continue_request(
           chat, result, history=history, max_continues=3
       )
       # Update history with merged result
       history.append_result(full_result)
       print(f"Complete story: {len(full_result.text)} chars")

Get All Intermediate Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   history = ChatHistory()
   
   result = chat("Story", history=history, max_tokens=50)
   history.add_user("Story")
   history.append_result(result)
   
   if result.finish_reason == "length":
       all_results = ChatContinue.continue_request(
           chat, result, history=history, auto_merge=False, max_continues=3
       )
       
       for i, r in enumerate(all_results):
           print(f"Part {i+1}: {len(r.text)} chars, tokens: {r.usage.total_tokens}")

Streaming Continue
~~~~~~~~~~~~~~~~~~

**Using complete_stream():**

.. code-block:: python

   # Single-turn (no history needed)
   iterator = chat.complete_stream("Long story", max_tokens=50, max_continues=2)
   
   for chunk in iterator:
       print(chunk.delta, end="", flush=True)
   
   full_result = iterator.result.to_chat_result()
   print(f"\nComplete: {len(full_result.text)} chars")

**Using continue_request_stream():**

.. code-block:: python

   history = ChatHistory()
   result = chat("Long story", history=history, max_tokens=50)
   history.add_user("Long story")
   history.append_result(result)
   
   if result.finish_reason == "length":
       iterator = ChatContinue.continue_request_stream(
           chat, result, history=history, max_continues=2
       )
       
       for chunk in iterator:
           print(chunk.delta, end="", flush=True)
       
       full_result = iterator.result.to_chat_result()
       history.append_result(full_result)
       print(f"\nComplete: {len(full_result.text)} chars")

See Also
--------

* :doc:`chat_history` - History management guide
* :doc:`api_reference/chat` - Full API reference
* :doc:`error_handling` - Error handling guide
