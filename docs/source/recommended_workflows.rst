Recommended Workflows
=====================

This guide provides recommended workflows for common use cases with Lexilux.
These patterns follow best practices and make your code simpler and more reliable.

.. important::
   **History Immutability**: All API methods create a clone of history internally and
   **never modify the original**. You must manually update your history after each API call.

Simple Conversation (Recommended)
----------------------------------

The simplest way to use Lexilux for basic conversations:

.. code-block:: python

   from lexilux import Chat, ChatHistory

   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       model="gpt-4",
   )
   
   # Create history explicitly
   history = ChatHistory()

   # Pass history explicitly - original is NOT modified
   result1 = chat("What is Python?", history=history)
   # Manually update history
   history.add_user("What is Python?")
   history.append_result(result1)
   
   result2 = chat("Tell me more", history=history)
   # Manually update again
   history.add_user("Tell me more")
   history.append_result(result2)

   # History contains complete conversation
   print(f"Total messages: {len(history.messages)}")  # 4 messages

   # Clear when starting new topic
   history.clear()

**Key Points**:
- Create and pass history objects explicitly
- **History is immutable** - manually update after each API call
- Use same history object for a conversation session

Helper Function Pattern
~~~~~~~~~~~~~~~~~~~~~~~

Create a helper function to reduce boilerplate:

.. code-block:: python

   def chat_with_history(chat, history, message, **kwargs):
       """Chat and automatically update history."""
       result = chat(message, history=history, **kwargs)
       history.add_user(message)
       history.append_result(result)
       return result

   # Usage
   history = ChatHistory()
   result1 = chat_with_history(chat, history, "What is Python?")
   result2 = chat_with_history(chat, history, "Tell me more")

Ensuring Complete Responses (Recommended)
------------------------------------------

When you need guaranteed complete responses (e.g., JSON extraction):

**Single-turn (no history needed):**

.. code-block:: python

   from lexilux import Chat
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import json

   chat = Chat(...)

   # No history needed for single-turn conversations
   try:
       result = chat.complete("Write a long JSON", max_tokens=100)
       json_data = json.loads(result.text)  # Guaranteed complete
   except ChatIncompleteResponseError as e:
       print(f"Still incomplete after continues: {e.continue_count}")
       json_data = json.loads(e.final_result.text)

**Multi-turn (with history):**

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import json

   chat = Chat(...)
   history = ChatHistory()

   try:
       result = chat.complete("Write a long JSON", history=history, max_tokens=100)
       # Manually update history
       history.add_user("Write a long JSON")
       history.append_result(result)
       json_data = json.loads(result.text)  # Guaranteed complete
   except ChatIncompleteResponseError as e:
       print(f"Still incomplete after continues: {e.continue_count}")
       history.add_user("Write a long JSON")
       history.append_result(e.final_result)
       json_data = json.loads(e.final_result.text)

**Key Points**:
- Use ``chat.complete()`` for guaranteed complete responses
- Automatically handles truncation
- History parameter is **optional** (creates internally if None)
- **History is immutable** - manually update after each call
- Raises ``ChatIncompleteResponseError`` if still incomplete (if ``ensure_complete=True``)

Streaming with Real-time Display
---------------------------------

Real-time output with manual history updates:

**Non-streaming:**

.. code-block:: python

   from lexilux import Chat, ChatHistory
   import requests

   chat = Chat(...)
   history = ChatHistory()

   try:
       result = chat("Long response", history=history)
       # Manually update history
       history.add_user("Long response")
       history.append_result(result)
       print(result.text)
   except requests.RequestException as e:
       print(f"Request failed: {e}")

**Streaming:**

.. code-block:: python

   from lexilux import Chat, ChatHistory
   import requests

   chat = Chat(...)
   history = ChatHistory()

   try:
       iterator = chat.stream("Long response", history=history)
       for chunk in iterator:
           print(chunk.delta, end="", flush=True)
           if chunk.done:
               print(f"\nFinish: {chunk.finish_reason}")
       
       # Get result and manually update history
       result = iterator.result.to_chat_result()
       history.add_user("Long response")
       history.append_result(result)
   except requests.RequestException as e:
       print(f"\nStream interrupted: {e}")
       # Get partial result if available
       try:
           result = iterator.result.to_chat_result()
           if result.text:
               history.add_user("Long response")
               history.append_result(result)
       except:
           pass  # No partial result available

**Key Points**:
- **History is immutable** - manually update after streaming completes
- Get result from `iterator.result.to_chat_result()`
- Handle interruptions gracefully
- Partial content can be preserved if you update history

Handling Errors Gracefully
--------------------------

Comprehensive error handling pattern:

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import requests
   import time

   chat = Chat(...)
   history = ChatHistory()

   def robust_chat(prompt, max_retries=3):
       """Robust chat with retry logic."""
       for attempt in range(max_retries):
           try:
               # Use complete() for guaranteed complete response
               result = chat.complete(prompt, history=history, max_tokens=200, max_continues=3)
               # Manually update history
               history.add_user(prompt)
               history.append_result(result)
               return result
           except ChatIncompleteResponseError as e:
               # Response still incomplete after continues
               print(f"Warning: Response incomplete after {e.continue_count} continues")
               history.add_user(prompt)
               history.append_result(e.final_result)
               return e.final_result  # Use partial result
           except requests.RequestException as e:
               # Network error - retry with exponential backoff
               if attempt < max_retries - 1:
                   wait_time = 2 ** attempt
                   print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                   time.sleep(wait_time)
                   continue
               raise  # Last attempt failed
       return None

   result = robust_chat("Your prompt here")

**Key Points**:
- Use ``chat.complete()`` for automatic continuation
- Handle ``ChatIncompleteResponseError`` for partial results
- Implement retry logic for network errors
- Use exponential backoff for retries
- **Always manually update history** after successful calls

Long-form Content Generation
----------------------------

Generating long content with automatic continuation:

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError

   chat = Chat(...)
   history = ChatHistory()

   def generate_long_content(prompt, target_length=None):
       """Generate long content with automatic continuation."""
       max_tokens = 500
       
       try:
           result = chat.complete(
               prompt,
               history=history,
               max_tokens=max_tokens,
               max_continues=5,  # Allow multiple continues
               ensure_complete=False  # Allow partial if needed
           )
           # Manually update history
           history.add_user(prompt)
           history.append_result(result)
           
           if target_length and len(result.text) < target_length:
               print(f"Warning: Generated {len(result.text)} chars, target was {target_length}")
           
           return result
       except ChatIncompleteResponseError as e:
           print(f"Generated {len(e.final_result.text)} chars before max continues")
           history.add_user(prompt)
           history.append_result(e.final_result)
           return e.final_result

   result = generate_long_content("Write a comprehensive guide to Python")

**Key Points**:
- Use ``chat.complete()`` with appropriate ``max_continues``
- Set ``ensure_complete=False`` if partial results are acceptable
- Monitor token usage across continues
- **Manually update history** after each call

Multi-turn Conversations
------------------------

Managing multi-turn conversations with context:

**Pattern 1: Non-streaming**

.. code-block:: python

   from lexilux import Chat, ChatHistory

   chat = Chat(...)
   history = ChatHistory()

   # Conversation with system message
   result1 = chat("Hello", history=history, system="You are a helpful Python tutor")
   history.add_user("Hello")
   history.append_result(result1)

   result2 = chat("What is a list?", history=history)
   history.add_user("What is a list?")
   history.append_result(result2)

   result3 = chat("How do I iterate over it?", history=history)
   history.add_user("How do I iterate over it?")
   history.append_result(result3)

   # History maintains context
   assert len(history.messages) == 6  # 3 user + 3 assistant

   # Continue conversation naturally
   result4 = chat("Give me an example", history=history)
   history.add_user("Give me an example")
   history.append_result(result4)

**Pattern 2: Mixed streaming and non-streaming**

.. code-block:: python

   from lexilux import Chat, ChatHistory

   chat = Chat(...)
   history = ChatHistory()

   # Turn 1: Non-streaming
   result1 = chat("Hello", history=history)
   history.add_user("Hello")
   history.append_result(result1)

   # Turn 2: Streaming
   iterator2 = chat.stream("How are you?", history=history)
   for chunk in iterator2:
       print(chunk.delta, end="", flush=True)
   result2 = iterator2.result.to_chat_result()
   history.add_user("How are you?")
   history.append_result(result2)

   # Turn 3: Complete (guaranteed complete)
   result3 = chat.complete("Write JSON", history=history, max_tokens=100)
   history.add_user("Write JSON")
   history.append_result(result3)

   # Turn 4: Complete streaming
   iterator4 = chat.complete_stream("Continue", history=history, max_tokens=100)
   for chunk in iterator4:
       print(chunk.delta, end="", flush=True)
   result4 = iterator4.result.to_chat_result()
   history.add_user("Continue")
   history.append_result(result4)

**Key Points**:
- System messages are preserved in history
- Context is maintained across turns
- Use new history object when switching topics
- **Always manually update history** after each API call
- Mix streaming and non-streaming as needed

JSON Extraction with Validation
-------------------------------

Extracting and validating JSON from responses:

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import json

   chat = Chat(...)
   history = ChatHistory()

   def extract_json(prompt, schema=None):
       """Extract JSON from response with validation."""
       try:
           # Single-turn (no history needed)
           result = chat.complete(
               f"{prompt}\n\nReturn the result as valid JSON.",
               max_tokens=500,
               max_continues=3
           )
           
           # Parse JSON
           try:
               data = json.loads(result.text)
           except json.JSONDecodeError as e:
               # Try to fix common issues
               # Remove markdown code blocks if present
               text = result.text.strip()
               if text.startswith("```"):
                   text = text.split("```")[1]
                   if text.startswith("json"):
                       text = text[4:]
                   text = text.strip()
               
               data = json.loads(text)
           
           # Validate schema if provided
           if schema:
               # Use jsonschema or similar for validation
               pass
           
           return data
       except ChatIncompleteResponseError as e:
           raise ValueError(f"Response incomplete, cannot extract JSON: {e.final_result.text}")
       except json.JSONDecodeError as e:
           raise ValueError(f"Invalid JSON in response: {e}")

   data = extract_json("List all users with their emails")

**Multi-turn JSON extraction:**

.. code-block:: python

   history = ChatHistory()
   
   # First question
   result1 = chat.complete("What is the schema?", history=history, max_tokens=200)
   history.add_user("What is the schema?")
   history.append_result(result1)
   
   # Second question (with context)
   result2 = chat.complete("Extract data as JSON", history=history, max_tokens=500)
   history.add_user("Extract data as JSON")
   history.append_result(result2)
   json_data = json.loads(result2.text)

**Key Points**:
- Use ``chat.complete()`` to ensure complete JSON
- Handle JSON parsing errors
- Consider response format (may include markdown code blocks)
- **Manually update history** for multi-turn conversations

Streaming with Progress Tracking
---------------------------------

Track progress during long streaming responses:

**Non-streaming with progress:**

.. code-block:: python

   from lexilux import Chat, ChatHistory
   import requests

   chat = Chat(...)
   history = ChatHistory()

   def chat_with_progress(prompt):
       """Chat with progress tracking."""
       result = chat(prompt, history=history)
       print(f"Received {len(result.text)} characters")
       history.add_user(prompt)
       history.append_result(result)
       return result

   result = chat_with_progress("Your prompt here")

**Streaming with progress:**

.. code-block:: python

   from lexilux import Chat, ChatHistory
   import requests

   chat = Chat(...)
   history = ChatHistory()

   def stream_with_progress(prompt):
       """Stream with progress tracking."""
       iterator = chat.stream(prompt, history=history)
       chunk_count = 0
       total_chars = 0
       
       try:
           for chunk in iterator:
               print(chunk.delta, end="", flush=True)
               chunk_count += 1
               total_chars += len(chunk.delta)
               
               # Progress update every 10 chunks
               if chunk_count % 10 == 0:
                   print(f"\n[Progress: {total_chars} chars, {chunk_count} chunks]", end="\r")
               
               if chunk.done:
                   print(f"\n[Complete: {total_chars} chars, finish_reason: {chunk.finish_reason}]")
                   break
           
           # Get result and update history
           result = iterator.result.to_chat_result()
           history.add_user(prompt)
           history.append_result(result)
       except requests.RequestException as e:
           print(f"\n[Interrupted: {total_chars} chars received]")
           # Try to get partial result
           try:
               result = iterator.result.to_chat_result()
               if result.text:
                   history.add_user(prompt)
                   history.append_result(result)
           except:
               pass
           raise

   stream_with_progress("Write a long story")

**Complete streaming with progress:**

.. code-block:: python

   def on_progress(count, max_count, current_result, all_results):
       print(f"\n🔄 Continuing {count}/{max_count}...")
       print(f"   Current: {len(current_result.text)} chars")
       print(f"   Total parts: {len(all_results)}")

   iterator = chat.complete_stream(
       "Write a long story",
       max_tokens=100,
       on_progress=on_progress,
   )
   
   for chunk in iterator:
       print(chunk.delta, end="", flush=True)
   
   result = iterator.result.to_chat_result()
   # No history to update for single-turn

**Key Points**:
- Track progress during streaming
- Handle interruptions gracefully
- Get result from `iterator.result.to_chat_result()`
- **Manually update history** after streaming completes

Error Recovery Patterns
-----------------------

Recovering from errors and interruptions:

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import requests
   import time

   chat = Chat(...)
   history = ChatHistory()

   def resilient_chat(prompt, max_retries=3):
       """Chat with automatic error recovery."""
       for attempt in range(max_retries):
           try:
               # Try to get complete response
               result = chat.complete(prompt, history=history, max_tokens=200, max_continues=3)
               # Manually update history
               history.add_user(prompt)
               history.append_result(result)
               return result
           except ChatIncompleteResponseError as e:
               # Response incomplete - use partial if acceptable
               if len(e.final_result.text) > 100:  # Minimum acceptable length
                   print(f"Using partial result ({len(e.final_result.text)} chars)")
                   history.add_user(prompt)
                   history.append_result(e.final_result)
                   return e.final_result
               # Too short, retry with higher max_tokens
               if attempt < max_retries - 1:
                   print(f"Retry {attempt + 1} with higher max_tokens...")
                   continue
               raise
           except requests.RequestException as e:
               # Network error - retry
               if attempt < max_retries - 1:
                   print(f"Network error, retry {attempt + 1}...")
                   time.sleep(2 ** attempt)
                   continue
               raise
       return None

   result = resilient_chat("Your prompt")

**Key Points**:
- Handle both ``ChatIncompleteResponseError`` and network errors
- Implement retry logic with different strategies
- Use partial results when acceptable
- **Always manually update history** after successful calls

Common Pitfalls to Avoid
------------------------

1. **Forgetting to Update History**:
   
   .. code-block:: python

      # Wrong - history not updated
      history = ChatHistory()
      result = chat("Hello", history=history)
      # history is still empty!
      
      # Correct - manually update
      history = ChatHistory()
      result = chat("Hello", history=history)
      history.add_user("Hello")
      history.append_result(result)

2. **Assuming History is Updated Automatically**:
   
   .. code-block:: python

      # Wrong assumption
      history = ChatHistory()
      result1 = chat("Hello", history=history)
      result2 = chat("How are you?", history=history)
      # result2 doesn't have context from result1 because history wasn't updated!

      # Correct
      history = ChatHistory()
      result1 = chat("Hello", history=history)
      history.add_user("Hello")
      history.append_result(result1)
      
      result2 = chat("How are you?", history=history)
      history.add_user("How are you?")
      history.append_result(result2)
      # Now result2 has context

3. **Not Handling ChatIncompleteResponseError**:
   
   .. code-block:: python

      # Wrong
      history = ChatHistory()
      result = chat.complete("Long response", history=history, max_tokens=30, max_continues=1)
      json.loads(result.text)  # May fail if still incomplete

      # Correct
      history = ChatHistory()
      try:
          result = chat.complete("Long response", history=history, max_tokens=30, max_continues=1)
          history.add_user("Long response")
          history.append_result(result)
          json.loads(result.text)
      except ChatIncompleteResponseError as e:
          # Handle partial result
          history.add_user("Long response")
          history.append_result(e.final_result)
          pass

4. **Not Updating History After Streaming**:
   
   .. code-block:: python

      # Wrong - history not updated
      history = ChatHistory()
      iterator = chat.stream("Long response", history=history)
      for chunk in iterator:
          print(chunk.delta)
      # history is still empty!

      # Correct
      history = ChatHistory()
      iterator = chat.stream("Long response", history=history)
      for chunk in iterator:
          print(chunk.delta)
      result = iterator.result.to_chat_result()
      history.add_user("Long response")
      history.append_result(result)

5. **Using Different History Objects for Same Conversation**:
   
   .. code-block:: python

      # Wrong - different history objects
      history1 = ChatHistory()
      result1 = chat("Hello", history=history1)
      history1.add_user("Hello")
      history1.append_result(result1)
      
      history2 = ChatHistory()
      result2 = chat("How are you?", history=history2)
      history2.add_user("How are you?")
      history2.append_result(result2)
      # history2 doesn't contain "Hello"

      # Correct - same history object
      history = ChatHistory()
      result1 = chat("Hello", history=history)
      history.add_user("Hello")
      history.append_result(result1)
      
      result2 = chat("How are you?", history=history)
      history.add_user("How are you?")
      history.append_result(result2)
      # history contains both turns

6. **Not Getting Result from Streaming Iterator**:
   
   .. code-block:: python

      # Wrong - no result obtained
      iterator = chat.stream("Hello", history=history)
      for chunk in iterator:
          print(chunk.delta)
      # How to get the result?

      # Correct - get result from iterator
      iterator = chat.stream("Hello", history=history)
      for chunk in iterator:
          print(chunk.delta)
      result = iterator.result.to_chat_result()
      history.add_user("Hello")
      history.append_result(result)

Complete Example: Multi-turn Conversation
-----------------------------------------

Complete example covering all interfaces:

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue
   from lexilux.chat.exceptions import ChatIncompleteResponseError

   chat = Chat(...)
   history = ChatHistory()

   # Turn 1: Non-streaming
   result1 = chat("Hello", history=history)
   history.add_user("Hello")
   history.append_result(result1)
   print(f"Turn 1: {result1.text}")

   # Turn 2: Streaming
   iterator2 = chat.stream("How are you?", history=history)
   print("Turn 2: ", end="")
   for chunk in iterator2:
       print(chunk.delta, end="", flush=True)
   print()
   result2 = iterator2.result.to_chat_result()
   history.add_user("How are you?")
   history.append_result(result2)

   # Turn 3: Complete (guaranteed complete)
   try:
       result3 = chat.complete("Write JSON", history=history, max_tokens=100)
       history.add_user("Write JSON")
       history.append_result(result3)
       print(f"Turn 3: {result3.text[:100]}...")
   except ChatIncompleteResponseError as e:
       print(f"Turn 3 incomplete: {e.continue_count} continues")
       history.add_user("Write JSON")
       history.append_result(e.final_result)

   # Turn 4: Complete streaming
   iterator4 = chat.complete_stream("Continue", history=history, max_tokens=100)
   print("Turn 4: ", end="")
   for chunk in iterator4:
       print(chunk.delta, end="", flush=True)
   print()
   result4 = iterator4.result.to_chat_result()
   history.add_user("Continue")
   history.append_result(result4)

   # Turn 5: Manual continue (if needed)
   result5 = chat("Tell me more", history=history, max_tokens=50)
   history.add_user("Tell me more")
   history.append_result(result5)
   
   if result5.finish_reason == "length":
       full_result = ChatContinue.continue_request(
           chat, result5, history=history, max_continues=2
       )
       history.append_result(full_result)
       print(f"Turn 5 (continued): {full_result.text[:100]}...")

   # History now contains all 5 turns
   print(f"\nTotal messages: {len(history.messages)}")
   print(f"Total rounds: {len(history.messages) // 2}")

See Also
--------

* :doc:`chat_history` - History management guide
* :doc:`chat_continue` - Continue generation guide
* :doc:`error_handling` - Error handling guide
* :doc:`quickstart` - Quick start guide
