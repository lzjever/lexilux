Recommended Workflows (v2.0)
============================

This guide provides recommended workflows for common use cases with Lexilux v2.0.
These patterns follow best practices and make your code simpler and more reliable.

Simple Conversation (Recommended)
----------------------------------

The simplest way to use Lexilux for basic conversations (v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory

   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       model="gpt-4",
   )
   
   # Create history explicitly
   history = ChatHistory()

   # Pass history explicitly - it's automatically updated
   result1 = chat("What is Python?", history=history)
   result2 = chat("Tell me more", history=history)

   # History contains complete conversation
   print(f"Total messages: {len(history.messages)}")

   # Clear when starting new topic
   history.clear()

**Key Points**:
- Create and pass history objects explicitly (v2.0)
- History is automatically updated when passed to chat methods
- Use same history object for a conversation session

Ensuring Complete Responses (Recommended)
------------------------------------------

When you need guaranteed complete responses (e.g., JSON extraction):

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import json

   chat = Chat(...)
   history = ChatHistory()

   # Method 1: Use complete() (simplest and recommended)
   try:
       result = chat.complete("Write a long JSON", history=history, max_tokens=100)
       json_data = json.loads(result.text)  # Guaranteed complete
   except ChatIncompleteResponseError as e:
       print(f"Still incomplete after continues: {e.continue_count}")
       # Handle partial result if acceptable
       json_data = json.loads(e.final_result.text)

   # Method 2: Conditional continue
   result = chat("Extract data as JSON", history=history, max_tokens=100)
   full_result = chat.complete("Write JSON", max_tokens=100)
   json_data = json.loads(full_result.text)

**Key Points**:
- Use ``chat.complete()`` for guaranteed complete responses
- Automatically handles truncation
- Raises ``ChatIncompleteResponseError`` if still incomplete (if ``ensure_complete=True``)
- Requires explicit ``history`` parameter (v2.0)

Streaming with Real-time Display
---------------------------------

Real-time output with automatic history updates (v2.0):

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
   except requests.RequestException as e:
       print(f"\nStream interrupted: {e}")
       # Partial content is preserved in history
       # Clean up if needed:
       if history.messages and history.messages[-1].get("role") == "assistant":
           history.remove_last()

**Key Points**:
- History updates in real-time during streaming (v2.0)
- Assistant message is added on first iteration
- Partial content is preserved on interruption
- Use ``history.remove_last()`` to clean up if needed

Handling Errors Gracefully
--------------------------

Comprehensive error handling pattern (v2.0):

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
               return result
           except ChatIncompleteResponseError as e:
               # Response still incomplete after continues
               print(f"Warning: Response incomplete after {e.continue_count} continues")
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
- Pass history explicitly (v2.0)

Long-form Content Generation
----------------------------

Generating long content with automatic continuation (v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError

   chat = Chat(...)
   history = ChatHistory()

   def generate_long_content(prompt, target_length=None):
       """Generate long content with automatic continuation."""
       # Start with reasonable max_tokens
       max_tokens = 500
       
       try:
           result = chat.complete(
               prompt,
               history=history,
               max_tokens=max_tokens,
               max_continues=5,  # Allow multiple continues
               ensure_complete=False  # Allow partial if needed
           )
           
           if target_length and len(result.text) < target_length:
               print(f"Warning: Generated {len(result.text)} chars, target was {target_length}")
           
           return result
       except ChatIncompleteResponseError as e:
           print(f"Generated {len(e.final_result.text)} chars before max continues")
           return e.final_result

   result = generate_long_content("Write a comprehensive guide to Python")

**Key Points**:
- Use ``chat.complete()`` with appropriate ``max_continues``
- Set ``ensure_complete=False`` if partial results are acceptable
- Monitor token usage across continues
- Pass history explicitly (v2.0)

Multi-turn Conversations
------------------------

Managing multi-turn conversations with context (v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory

   chat = Chat(...)
   history = ChatHistory()

   # Conversation with system message
   result1 = chat("Hello", history=history, system="You are a helpful Python tutor")
   result2 = chat("What is a list?", history=history)
   result3 = chat("How do I iterate over it?", history=history)

   # History maintains context
   assert history.system == "You are a helpful Python tutor"

   # Continue conversation naturally
   result4 = chat("Give me an example", history=history)

   # Start new topic with new history
   history2 = ChatHistory()
   result5 = chat("New topic", history=history2, system="You are a math tutor")

**Key Points**:
- System messages are preserved in history
- Context is maintained across turns
- Use new history object when switching topics
- Pass same history object for a conversation session

JSON Extraction with Validation
-------------------------------

Extracting and validating JSON from responses (v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import json

   chat = Chat(...)
   history = ChatHistory()

   def extract_json(prompt, schema=None):
       """Extract JSON from response with validation."""
       try:
           result = chat.complete(
               f"{prompt}\n\nReturn the result as valid JSON.",
               history=history,
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

**Key Points**:
- Use ``chat.complete()`` to ensure complete JSON
- Handle JSON parsing errors
- Consider response format (may include markdown code blocks)
- Pass history explicitly (v2.0)

Streaming with Progress Tracking
---------------------------------

Track progress during long streaming responses (v2.0):

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
       except requests.RequestException as e:
           print(f"\n[Interrupted: {total_chars} chars received]")
           # Clean up partial response
           if history.messages and history.messages[-1].get("role") == "assistant":
               history.remove_last()
           raise

   stream_with_progress("Write a long story")

**Key Points**:
- Track progress during streaming
- Handle interruptions gracefully
- Clean up partial responses if needed
- Pass history explicitly (v2.0)

Error Recovery Patterns
-----------------------

Recovering from errors and interruptions (v2.0):

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
               return result
           except ChatIncompleteResponseError as e:
               # Response incomplete - use partial if acceptable
               if len(e.final_result.text) > 100:  # Minimum acceptable length
                   print(f"Using partial result ({len(e.final_result.text)} chars)")
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
- Pass history explicitly (v2.0)

Common Pitfalls to Avoid
------------------------

1. **Forgetting to pass history** (v2.0):
   
   .. code-block:: python

      # Wrong - history=None, no history tracking
      result = chat.complete("JSON", max_tokens=100)  # Will fail: history required

      # Correct - pass history explicitly
      history = ChatHistory()
      result = chat.complete("JSON", history=history, max_tokens=100)  # Works

2. **Not handling ChatIncompleteResponseError**:
   
   .. code-block:: python

      # Wrong
      history = ChatHistory()
      result = chat.complete("Long response", history=history, max_tokens=30, max_continues=1)
      json.loads(result.text)  # May fail if still incomplete

      # Correct
      history = ChatHistory()
      try:
          result = chat.complete("Long response", history=history, max_tokens=30, max_continues=1)
          json.loads(result.text)
      except ChatIncompleteResponseError as e:
          # Handle partial result
          pass

3. **Not cleaning up partial streaming responses**:
   
   .. code-block:: python

      # Wrong
      history = ChatHistory()
      iterator = chat.stream("Long response", history=history)
      try:
          for chunk in iterator:
              print(chunk.delta)
      except Exception:
          pass  # Partial response left in history

      # Correct
      history = ChatHistory()
      iterator = chat.stream("Long response", history=history)
      try:
          for chunk in iterator:
              print(chunk.delta)
      except Exception:
          if history.messages and history.messages[-1].get("role") == "assistant":
              history.remove_last()  # Clean up

4. **Using different history objects for same conversation**:
   
   .. code-block:: python

      # Wrong - different history objects
      history1 = ChatHistory()
      result1 = chat("Hello", history=history1)
      
      history2 = ChatHistory()
      result2 = chat("How are you?", history=history2)
      # history2 doesn't contain "Hello"

      # Correct - same history object
      history = ChatHistory()
      result1 = chat("Hello", history=history)
      result2 = chat("How are you?", history=history)
      # history contains both turns

5. **Using old API when new API is simpler**:
   
   .. code-block:: python

      # Old way (still works but verbose)
      history = ChatHistory()
      result = chat("JSON", history=history, max_tokens=100)
      if result.finish_reason == "length":
          continue_result = ChatContinue.continue_request(chat, result, history=history)
          full_result = ChatContinue.merge_results(result, continue_result)
      else:
          full_result = result

      # New way (recommended)
      history = ChatHistory()
      full_result = chat.complete("JSON", history=history, max_tokens=100)

See Also
--------

* :doc:`chat_history` - History management guide (v2.0)
* :doc:`chat_continue` - Continue generation guide (v2.0)
* :doc:`error_handling` - Error handling guide
* :doc:`quickstart` - Quick start guide
