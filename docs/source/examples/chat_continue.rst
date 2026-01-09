Continue Generation Examples
============================

This section provides practical examples of using the continue generation functionality
with immutable history management.

.. important::
   **History Immutability**: All API methods create a clone of history internally and
   **never modify the original**. You must manually update your history after each API call.

Recommended: Using chat.complete()
-----------------------------------

The simplest and most recommended approach:

**Single-turn (no history needed):**

.. code-block:: python

   from lexilux import Chat
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import json

   chat = Chat(...)

   # No history needed for single-turn conversations
   try:
       result = chat.complete("Extract user data as JSON", max_tokens=100)
       json_data = json.loads(result.text)  # Guaranteed complete
   except ChatIncompleteResponseError as e:
       print(f"Still incomplete after {e.continue_count} continues")
       json_data = json.loads(e.final_result.text)

**Multi-turn (with history):**

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import json

   chat = Chat(...)
   history = ChatHistory()

   # Ensure complete JSON response
   try:
       result = chat.complete("Extract user data as JSON", history=history, max_tokens=100)
       # Manually update history
       history.add_user("Extract user data as JSON")
       history.append_result(result)
       json_data = json.loads(result.text)  # Guaranteed complete
   except ChatIncompleteResponseError as e:
       print(f"Still incomplete after {e.continue_count} continues")
       history.add_user("Extract user data as JSON")
       history.append_result(e.final_result)
       json_data = json.loads(e.final_result.text)

Enhanced ChatContinue API
--------------------------

Using the enhanced continue_request() with immutable history:

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
           chat, result, history=history, max_continues=3
       )
       # Update history with merged result
       history.append_result(full_result)
       print(f"Complete story: {len(full_result.text)} chars")

Get All Intermediate Results
-----------------------------

Get all parts separately:

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()
   
   result = chat("Story", history=history, max_tokens=50)
   # Manually update history
   history.add_user("Story")
   history.append_result(result)
   
   if result.finish_reason == "length":
       all_results = ChatContinue.continue_request(
           chat, result, history=history, auto_merge=False, max_continues=3
       )
       
       for i, r in enumerate(all_results):
           print(f"Part {i+1}: {len(r.text)} chars, tokens: {r.usage.total_tokens}")

Streaming Continue
-------------------

Stream continuation chunks in real-time:

**Using complete_stream() (recommended):**

.. code-block:: python

   from lexilux import Chat

   chat = Chat(...)

   # Single-turn (no history needed)
   iterator = chat.complete_stream("Write a long story", max_tokens=50, max_continues=2)
   
   for chunk in iterator:
       print(chunk.delta, end="", flush=True)
   
   full_result = iterator.result.to_chat_result()
   print(f"\nComplete: {len(full_result.text)} chars")

**Using continue_request_stream() (manual control):**

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
       # Update history with merged result
       history.append_result(full_result)
       print(f"\nComplete: {len(full_result.text)} chars")

Continue with Progress Tracking
-------------------------------

Track progress during continuation:

**Using complete() with progress callback:**

.. code-block:: python

   from lexilux import Chat

   chat = Chat(...)

   def on_progress(count, max_count, current_result, all_results):
       print(f"🔄 Continuing {count}/{max_count}...")
       print(f"   Current: {len(current_result.text)} chars")
       print(f"   Total parts: {len(all_results)}")

   # Single-turn (no history needed)
   result = chat.complete(
       "Write a detailed technical document",
       max_tokens=100,
       max_continues=3,
       on_progress=on_progress,
   )

**Using continue_request() with progress:**

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()

   result1 = chat("Write a detailed technical document", history=history, max_tokens=100)
   # Manually update history
   history.add_user("Write a detailed technical document")
   history.append_result(result1)
   print(f"Part 1: {len(result1.text)} chars, {result1.usage.total_tokens} tokens")

   if result1.finish_reason == "length":
       def on_progress(count, max_count, current_result, all_results):
           print(f"Part {count+1}: {len(current_result.text)} chars")

       all_results = ChatContinue.continue_request(
           chat, result1, history=history, auto_merge=False, max_continues=3,
           on_progress=on_progress,
       )
       
       for i, r in enumerate(all_results[1:], start=2):  # Skip first (already printed)
           print(f"Part {i}: {len(r.text)} chars, {r.usage.total_tokens} tokens")
       
       # Merge
       full_result = ChatContinue.merge_results(*all_results)
       history.append_result(full_result)
       print(f"Complete: {len(full_result.text)} chars, {full_result.usage.total_tokens} tokens")

Continue with Custom Parameters
-------------------------------

Pass additional parameters to continuation:

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()
   
   result1 = chat("Write a story", history=history, max_tokens=100, temperature=0.7)
   # Manually update history
   history.add_user("Write a story")
   history.append_result(result1)

   if result1.finish_reason == "length":
       # Continue with different parameters
       continue_result = ChatContinue.continue_request(
           chat,
           result1,
           history=history,
           temperature=0.8,  # Slightly more creative
           max_tokens=200,    # Longer continuation
           max_continues=2,
       )
       
       full_result = ChatContinue.merge_results(result1, continue_result)
       history.append_result(full_result)

Error Handling
--------------

Handle errors during continuation:

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import requests

   chat = Chat(...)
   history = ChatHistory()

   try:
       # Use complete() for automatic error handling
       result = chat.complete("Long content", history=history, max_tokens=100, max_continues=3)
       # Manually update history
       history.add_user("Long content")
       history.append_result(result)
   except ChatIncompleteResponseError as e:
       print(f"Response incomplete after {e.continue_count} continues")
       print(f"Received: {len(e.final_result.text)} chars")
       # Use partial result if acceptable
       history.add_user("Long content")
       history.append_result(e.final_result)
       result = e.final_result
   except requests.RequestException as e:
       print(f"Network error: {e}")
       result = None

Complete Workflow
-----------------

Complete workflow with continue (recommended pattern):

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatHistoryFormatter
   from lexilux.chat.exceptions import ChatIncompleteResponseError

   chat = Chat(...)
   history = ChatHistory()

   # Request long content
   prompt = "Write a comprehensive tutorial on Python"
   
   try:
       result = chat.complete(prompt, history=history, max_tokens=200, max_continues=3)
       # Manually update history
       history.add_user(prompt)
       history.append_result(result)
   except ChatIncompleteResponseError as e:
       print(f"Warning: Tutorial incomplete after {e.continue_count} continues")
       history.add_user(prompt)
       history.append_result(e.final_result)
       result = e.final_result

   # Save complete tutorial
   ChatHistoryFormatter.save(history, "python_tutorial.md")
   
   print(f"Tutorial saved: {len(result.text)} characters")
   print(f"Total tokens used: {result.usage.total_tokens}")

JSON Extraction Pattern
------------------------

Extract JSON with guaranteed completeness:

**Single-turn:**

.. code-block:: python

   from lexilux import Chat
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import json

   chat = Chat(...)

   def extract_json(prompt):
       """Extract JSON from response, ensuring completeness."""
       try:
           # No history needed for single-turn
           result = chat.complete(
               f"{prompt}\n\nReturn the result as valid JSON.",
               max_tokens=500,
               max_continues=3
           )
           
           # Parse JSON (may need to strip markdown code blocks)
           text = result.text.strip()
           if text.startswith("```"):
               text = text.split("```")[1]
               if text.startswith("json"):
                   text = text[4:]
               text = text.strip()
           
           return json.loads(text)
       except ChatIncompleteResponseError as e:
           raise ValueError(f"Cannot extract JSON from incomplete response")
       except json.JSONDecodeError as e:
           raise ValueError(f"Invalid JSON in response: {e}")

   data = extract_json("List all users with their emails and roles")

**Multi-turn:**

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import json

   chat = Chat(...)
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

Long-form Content Generation
-----------------------------

Generate long content with automatic continuation:

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError

   chat = Chat(...)
   history = ChatHistory()

   def generate_long_content(prompt, target_length=None):
       """Generate long content with automatic continuation."""
       try:
           result = chat.complete(
               prompt,
               history=history,
               max_tokens=500,
               max_continues=5,
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

Complete Example: All Interfaces
---------------------------------

Complete example covering all interfaces with history updates:

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue
   from lexilux.chat.exceptions import ChatIncompleteResponseError

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
   try:
       result3 = chat.complete("Write JSON", history=history, max_tokens=100)
       history.add_user("Write JSON")
       history.append_result(result3)
   except ChatIncompleteResponseError as e:
       history.add_user("Write JSON")
       history.append_result(e.final_result)

   # Turn 4: Complete streaming
   iterator4 = chat.complete_stream("Continue", history=history, max_tokens=100)
   for chunk in iterator4:
       print(chunk.delta, end="", flush=True)
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

   # History now contains all 5 turns
   print(f"\nTotal messages: {len(history.messages)}")
   print(f"Total rounds: {len(history.messages) // 2}")
