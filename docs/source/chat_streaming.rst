Streaming with History Accumulation
====================================

Lexilux provides automatic text accumulation during streaming, allowing you to access
the complete accumulated text at any point during iteration and seamlessly integrate
with conversation history.

Overview
--------

When using streaming, Lexilux automatically accumulates the text as chunks arrive.
This allows you to:

* Access the current accumulated text at any time during streaming
* Convert streaming results to ``ChatResult`` for history management
* Update conversation history in real-time during streaming
* Handle interruptions gracefully

Key Concepts
------------

1. **StreamingResult**: Automatically accumulates text from chunks
2. **StreamingIterator**: Wraps the chunk iterator and provides accumulated result
3. **Real-time Access**: Get current accumulated text at any point
4. **Seamless Integration**: Convert to ``ChatResult`` for history management

Basic Usage
-----------

Automatic Accumulation
~~~~~~~~~~~~~~~~~~~~~~

The ``StreamingIterator`` automatically accumulates text as you iterate:

.. code-block:: python

   from lexilux import Chat
   from lexilux.chat import StreamingIterator

   chat = Chat(base_url="https://api.example.com/v1", api_key="key", model="gpt-4")

   # Get streaming iterator (when auto_history is implemented)
   iterator = chat.stream("Tell me a story")

   # Iterate chunks
   for chunk in iterator:
       print(chunk.delta, end="")
       # Access accumulated text at any time
       current_text = iterator.result.text
       print(f"\\n[Current length: {len(current_text)}]")

   # After streaming, get complete result
   complete_result = iterator.result.to_chat_result()
   print(f"\\nComplete text: {complete_result.text}")
   print(f"Finish reason: {complete_result.finish_reason}")

Accessing Accumulated Result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can access the accumulated result at any point:

.. code-block:: python

   iterator = chat.stream("Write a long story")

   # Before iteration
   assert iterator.result.text == ""

   # During iteration
   for chunk in iterator:
       # Result is updated automatically
       if len(iterator.result.text) > 100:
           print("Story is getting long...")
           print(f"Current: {iterator.result.text[:100]}...")

   # After iteration
   assert iterator.result.done is True
   assert len(iterator.result.text) > 0

Integration with History
------------------------

Convert to ChatResult
~~~~~~~~~~~~~~~~~~~~~

After streaming, convert the accumulated result to ``ChatResult`` for history management:

.. code-block:: python

   from lexilux.chat import ChatHistory

   iterator = chat.stream("Tell me a story")
   for chunk in iterator:
       print(chunk.delta, end="")

   # Convert to ChatResult
   result = iterator.result.to_chat_result()

   # Add to history
   history = ChatHistory()
   history.add_user("Tell me a story")
   history.append_result(result)

Real-time History Updates
~~~~~~~~~~~~~~~~~~~~~~~~~

You can update history in real-time during streaming:

.. code-block:: python

   history = ChatHistory()
   history.add_user("Tell me a story")

   iterator = chat.stream(history.get_messages())
   for chunk in iterator:
       print(chunk.delta, end="")
       # Update history with current accumulated text
       history.update_last_assistant(iterator.result.text)

   # After streaming, history already contains the complete response

.. note::
   This approach updates history in real-time, but be aware that it may
   cause multiple updates. For better performance, update only at the end
   or at intervals.

Handling Interruptions
----------------------

If streaming is interrupted, the accumulated result still contains what was received:

.. code-block:: python

   iterator = chat.stream("Write a long story")

   try:
       for chunk in iterator:
           print(chunk.delta, end="")
           # Simulate interruption
           if len(iterator.result.text) > 50:
               raise ConnectionError("Network interrupted")
   except ConnectionError as e:
       # Even though interrupted, we have partial result
       partial_text = iterator.result.text
       print(f"\\nInterrupted, but got: {partial_text}")
       # partial_text contains what was accumulated before interruption

   # Result reflects partial state
   assert iterator.result.done is False  # Not completed
   assert len(iterator.result.text) > 0   # But has partial text

Best Practices
--------------

1. **Check Completion**: Always check if streaming completed:

   .. code-block:: python

      iterator = chat.stream("Tell me a story")
      for chunk in iterator:
          print(chunk.delta, end="")

      if iterator.result.done:
          # Completed successfully
          result = iterator.result.to_chat_result()
          history.append_result(result)
      else:
          # Interrupted or incomplete
          print("Streaming was interrupted")

2. **Update History Efficiently**: Don't update history on every chunk:

   .. code-block:: python

      # Less efficient - updates on every chunk
      for chunk in iterator:
          history.update_last_assistant(iterator.result.text)

      # More efficient - update only at the end
      for chunk in iterator:
          print(chunk.delta, end="")
      if iterator.result.done:
          history.update_last_assistant(iterator.result.text)

3. **Handle Partial Results**: Be prepared for incomplete results:

   .. code-block:: python

      iterator = chat.stream("Write a story")
      try:
          for chunk in iterator:
              print(chunk.delta, end="")
      except Exception as e:
          # Handle error, but still use partial result
          if len(iterator.result.text) > 0:
              # Save partial result
              partial_result = iterator.result.to_chat_result()
              history.append_result(partial_result)

4. **Monitor Progress**: Use accumulated text to monitor progress:

   .. code-block:: python

      iterator = chat.stream("Write a long story")
      for chunk in iterator:
          print(chunk.delta, end="")
          # Monitor progress
          if len(iterator.result.text) % 100 == 0:
              print(f"\\n[Progress: {len(iterator.result.text)} chars]")

Common Pitfalls
---------------

1. **Assuming Completion**: Don't assume streaming completed just because
   the loop ended. Always check ``iterator.result.done``:

   .. code-block:: python

      # Wrong - may be incomplete
      iterator = chat.stream("Tell me a story")
      for chunk in iterator:
          pass
      result = iterator.result.to_chat_result()  # May be incomplete!

      # Correct - check completion
      iterator = chat.stream("Tell me a story")
      for chunk in iterator:
          pass
      if iterator.result.done:
          result = iterator.result.to_chat_result()

2. **Multiple Iterations**: Don't iterate the same iterator multiple times:

   .. code-block:: python

      iterator = chat.stream("Tell me a story")
      list(iterator)  # First iteration - consumes all chunks
      list(iterator)  # Second iteration - empty! No chunks left

3. **Result State During Iteration**: The result is updated during iteration,
   but ``done`` and ``finish_reason`` are only set when a chunk with ``done=True``
   arrives:

   .. code-block:: python

      iterator = chat.stream("Tell me a story")
      for chunk in iterator:
          # result.text is updated immediately
          # but result.done is False until done=True chunk arrives
          if iterator.result.done:
              # This only happens when done=True chunk is processed
              break

4. **Usage Statistics**: Usage statistics are only available in the final chunk
   (when ``done=True``). Don't rely on usage during intermediate chunks:

   .. code-block:: python

      iterator = chat.stream("Tell me a story")
      for chunk in iterator:
          # chunk.usage may be empty for intermediate chunks
          if chunk.done:
              # Now usage is complete
              print(f"Tokens: {chunk.usage.total_tokens}")

Examples
--------

Complete Streaming Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lexilux import Chat
   from lexilux.chat import ChatHistory, ChatHistoryFormatter

   chat = Chat(base_url="https://api.example.com/v1", api_key="key", model="gpt-4")

   history = ChatHistory(system="You are a storyteller")
   history.add_user("Tell me a story about Python")

   iterator = chat.stream(history.get_messages())

   # Stream and display
   for chunk in iterator:
       print(chunk.delta, end="", flush=True)

   # Check completion
   if iterator.result.done:
       # Add to history
       result = iterator.result.to_chat_result()
       history.append_result(result)

       # Export conversation
       ChatHistoryFormatter.save(history, "story.md")

Progress Monitoring
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   iterator = chat.stream("Write a long article")

   last_length = 0
   for chunk in iterator:
       print(chunk.delta, end="")
       current_length = len(iterator.result.text)

       # Report progress every 100 characters
       if current_length - last_length >= 100:
           print(f"\\n[Progress: {current_length} characters]", end="", flush=True)
           last_length = current_length

   print(f"\\n\\nComplete: {len(iterator.result.text)} characters")

Error Recovery
~~~~~~~~~~~~~~

.. code-block:: python

   iterator = chat.stream("Write a long story")

   try:
       for chunk in iterator:
           print(chunk.delta, end="")
   except Exception as e:
       print(f"\\nError: {e}")

       # Check if we got anything
       if len(iterator.result.text) > 0:
           print(f"\\nPartial result ({len(iterator.result.text)} chars):")
           print(iterator.result.text[:200] + "...")

           # Save partial result
           partial = iterator.result.to_chat_result()
           # Note: finish_reason will be None for incomplete results

