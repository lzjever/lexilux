Continue Generation (v2.0)
===========================

Lexilux provides functionality to continue generation when responses are cut off
due to token limits, allowing you to seamlessly extend incomplete responses.

Overview
--------

When a chat completion is stopped due to ``max_tokens`` limit (``finish_reason == "length"``),
you may want to continue the generation. Lexilux provides multiple ways to handle this:

1. **Chat.complete()** - Recommended for most cases, ensures complete response
2. **ChatContinue.continue_request()** - Advanced control with full flexibility
3. **Streaming versions** - ``complete_stream()`` and ``continue_request_stream()``

Key Features
------------

1. **Explicit History Management**: All methods require explicit history parameter (v2.0)
2. **Multiple Continues**: Automatically continue multiple times if needed
3. **Result Merging**: Automatically merge all continuation results
4. **Usage Aggregation**: Automatically combine token usage from multiple requests
5. **Streaming Support**: Stream continuation chunks in real-time

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

.. code-block:: python

   from lexilux import Chat, ChatHistory

   chat = Chat(...)
   history = ChatHistory()

   # Automatically handles truncation, returns complete result
   # History is optional for single-turn conversations
   result = chat.complete("Write a long JSON response", max_tokens=100)
   json_data = json.loads(result.text)  # Guaranteed complete

   # With error handling
   try:
       result = chat.complete("Very long response", history=history, max_tokens=50, max_continues=3)
   except ChatIncompleteResponseError as e:
       print(f"Still incomplete after {e.continue_count} continues")
       print(f"Received: {len(e.final_result.text)} chars")

Key Features of complete():
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Automatically continues if ``finish_reason == "length"``
* Supports multiple continues (``max_continues`` parameter)
* Raises ``ChatIncompleteResponseError`` if still truncated (if ``ensure_complete=True``)
* Requires explicit ``history`` parameter (v2.0)

Customizable Continue Strategy (v2.1)
--------------------------------------

The ``complete()`` method now supports extensive customization options:

.. code-block:: python

   from lexilux import Chat

   chat = Chat(...)

   # Custom continue prompt function
   def smart_prompt(count, max_count, current_text, original_prompt):
       return f"Please continue (attempt {count}/{max_count})"

   # Progress tracking
   def on_progress(count, max_count, current, all_results):
       print(f"🔄 Continuing {count}/{max_count}...")

   result = chat.complete(
       "Write a long JSON",
       max_tokens=100,
       continue_prompt=smart_prompt,
       on_progress=on_progress,
       continue_delay=(1.0, 2.0),  # Random delay 1-2 seconds
       on_error="return_partial",  # Return partial on error
   )

Advanced Control: ChatContinue.continue_request()
---------------------------------------------------

For advanced use cases requiring full control:

Enhanced API (v2.0)
~~~~~~~~~~~~~~~~~~~

The enhanced ``continue_request()`` requires explicit history and supports
multiple continues and automatic merging:

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()
   result = chat("Write a long story", history=history, max_tokens=50)
   
   if result.finish_reason == "length":
       # Explicit history required, multiple continues, automatic merging
       full_result = ChatContinue.continue_request(
           chat,
           result,
           history=history,  # Required in v2.0
           max_continues=3
       )
       print(full_result.text)  # Complete merged text

Key Parameters:
~~~~~~~~~~~~~~~

* ``history``: **Required**. ChatHistory instance (v2.0 requires explicit history)
* ``max_continues``: Maximum number of continuation attempts (default: 1)
* ``auto_merge``: If ``True``, automatically merge results (default: ``True``)
* ``add_continue_prompt``: Whether to add a user continue message (default: ``True``)
* ``continue_prompt``: User prompt for continuation (default: "continue")

Return Types:
~~~~~~~~~~~~~

* If ``auto_merge=True``: Returns merged ``ChatResult``
* If ``auto_merge=False``: Returns list of ``ChatResult`` instances

Examples:

.. code-block:: python

   history = ChatHistory()
   
   # Basic usage (explicit history, single continue, auto merge)
   result = chat("Story", history=history, max_tokens=50)
   if result.finish_reason == "length":
       full_result = ChatContinue.continue_request(chat, result, history=history)

   # Multiple continues
   result = chat("Very long story", history=history, max_tokens=30)
   if result.finish_reason == "length":
       full_result = ChatContinue.continue_request(
           chat, result, history=history, max_continues=3
       )

   # Get all intermediate results
   result = chat("Story", history=history, max_tokens=50)
   if result.finish_reason == "length":
       all_results = ChatContinue.continue_request(
           chat, result, history=history, auto_merge=False
       )
       # all_results = [result, continue_result1, continue_result2, ...]

Streaming Continue
------------------

Streaming versions provide real-time continuation:

continue_request_stream()
~~~~~~~~~~~~~~~~~~~~~~~~~

Stream continuation chunks in real-time:

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()
   result = chat("Write a long story", history=history, max_tokens=50)
   
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

continue_if_needed_stream()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stream continuation only if needed:

.. code-block:: python

   history = ChatHistory()
   result = chat("Long story", history=history, max_tokens=50)
   
   # Use complete_stream for automatic continuation
   iterator = chat.complete_stream("Long story", max_tokens=50)
   
   for chunk in iterator:
       print(chunk.delta, end="", flush=True)

complete_stream()
~~~~~~~~~~~~~~~~~

Stream complete response (handles truncation automatically):

.. code-block:: python

   history = ChatHistory()
   
   # Automatically handles truncation and continues if needed
   iterator = chat.complete_stream(
       "Write a long JSON response",
       history=history,
       max_tokens=100,
       max_continues=3
   )
   
   for chunk in iterator:
       print(chunk.delta, end="", flush=True)
   
   # Result is guaranteed complete (or raises ChatIncompleteResponseError)
   result = iterator.result.to_chat_result()
   json_data = json.loads(result.text)

Result Merging
--------------

The ``merge_results()`` method combines multiple results:

.. code-block:: python

   from lexilux import ChatContinue

   result1 = chat("Story part 1", history=history, max_tokens=50)
   result2 = chat("Story part 2", history=history, max_tokens=50)
   
   merged = ChatContinue.merge_results(result1, result2)
   # merged.text = result1.text + result2.text
   # merged.usage.total_tokens = result1.usage.total_tokens + result2.usage.total_tokens
   # merged.finish_reason = result2.finish_reason (from last result)

Common Patterns
---------------

Pattern 1: Ensure Complete Response (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``chat.complete()`` for scenarios requiring complete responses:

.. code-block:: python

   history = ChatHistory()
   
   # JSON extraction
   result = chat.complete("Extract data as JSON", history=history, max_tokens=100)
   json_data = json.loads(result.text)  # Guaranteed complete
   
   # Long-form content
   result = chat.complete("Write a comprehensive guide", history=history, max_tokens=200)

Pattern 2: Customizable Continue Strategy (v2.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``chat.complete()`` with customization options:

.. code-block:: python

   def on_progress(count, max_count, current, all_results):
       print(f"🔄 Continuing {count}/{max_count}...")

   # Single-turn conversation (no history needed)
   result = chat.complete(
       "Write JSON",
       max_tokens=100,
       on_progress=on_progress,
       continue_delay=(1.0, 2.0),
   )

Pattern 3: Advanced Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``ChatContinue.continue_request()`` for full control:

.. code-block:: python

   history = ChatHistory()
   
   result = chat("Story", history=history, max_tokens=50)
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

.. code-block:: python

   history = ChatHistory()
   result = chat("Long story", history=history, max_tokens=50)
   
   if result.finish_reason == "length":
       iterator = chat.complete_stream("Long story", max_tokens=50)
       
       for chunk in iterator:
           print(chunk.delta, end="", flush=True)
       
       full_result = iterator.result.to_chat_result()
       print(f"\nComplete: {len(full_result.text)} chars")

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
   except ChatIncompleteResponseError as e:
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

Best Practices
--------------

1. **Use chat.complete() for Most Cases**: Simplest and most reliable

2. **Pass History Explicitly**: Always create and pass history objects explicitly (v2.0)

3. **Set Appropriate max_continues**: Balance between completeness and API costs

4. **Handle ChatIncompleteResponseError**: Be prepared for cases where response
   is still incomplete after max_continues

5. **Monitor Token Usage**: Track total tokens across all continuations

6. **Consider Increasing max_tokens**: If you frequently need multiple continues,
   consider increasing ``max_tokens`` instead

7. **Use Same History Object**: For a conversation session, use the same history
   object across all calls:

   .. code-block:: python

      history = ChatHistory()
      result1 = chat("Question 1", history=history)
      result2 = chat("Question 2", history=history)
      # Both calls update the same history object

Examples
--------

Complete Workflow with complete()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
   
   # Guaranteed complete
   data = json.loads(result.text)

Multiple Continues
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   history = ChatHistory()
   
   result = chat("Very long story", history=history, max_tokens=30)
   if result.finish_reason == "length":
       # Automatically continues up to 3 times
       full_result = ChatContinue.continue_request(
           chat, result, history=history, max_continues=3
       )
       print(f"Complete story: {len(full_result.text)} chars")

Get All Intermediate Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   history = ChatHistory()
   
   result = chat("Story", history=history, max_tokens=50)
   if result.finish_reason == "length":
       all_results = ChatContinue.continue_request(
           chat, result, history=history, auto_merge=False, max_continues=3
       )
       
       for i, r in enumerate(all_results):
           print(f"Part {i+1}: {len(r.text)} chars, tokens: {r.usage.total_tokens}")

Streaming Continue
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   history = ChatHistory()
   result = chat("Long story", history=history, max_tokens=50)
   
   if result.finish_reason == "length":
       iterator = ChatContinue.continue_request_stream(
           chat, result, history=history, max_continues=2
       )
       
       for chunk in iterator:
           print(chunk.delta, end="", flush=True)
       
       full_result = iterator.result.to_chat_result()
       print(f"\nComplete: {len(full_result.text)} chars")

See Also
--------

* :doc:`chat_history` - History management guide
* :doc:`api_reference/chat` - Full API reference
* :doc:`error_handling` - Error handling guide
