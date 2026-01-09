Continue Generation Examples (v2.0)
===================================

This section provides practical examples of using the continue generation functionality
with v2.0 explicit history management.

Recommended: Using chat.complete()
-----------------------------------

The simplest and most recommended approach (v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import json

   chat = Chat(...)
   history = ChatHistory()

   # Ensure complete JSON response
   try:
       result = chat.complete("Extract user data as JSON", history=history, max_tokens=100)
       json_data = json.loads(result.text)  # Guaranteed complete
   except ChatIncompleteResponseError as e:
       print(f"Still incomplete after {e.continue_count} continues")
       # Use partial result if acceptable
       json_data = json.loads(e.final_result.text)

Conditional Continue
--------------------

Continue only if needed (v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory

   chat = Chat(...)
   history = ChatHistory()

   result = chat("Long story", history=history, max_tokens=50)
   # Automatically continues if truncated, otherwise returns result unchanged
   full_result = chat.complete("Write JSON", max_tokens=100)

Enhanced ChatContinue API (v2.0)
---------------------------------

Using the enhanced continue_request() with explicit history:

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()
   result = chat("Write a long story", history=history, max_tokens=50)
   
   if result.finish_reason == "length":
       # Explicit history required, multiple continues, auto merge
       full_result = ChatContinue.continue_request(
           chat, result, history=history, max_continues=3
       )
       print(f"Complete story: {len(full_result.text)} chars")

Get All Intermediate Results
-----------------------------

Get all parts separately (v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()
   result = chat("Story", history=history, max_tokens=50)
   
   if result.finish_reason == "length":
       all_results = ChatContinue.continue_request(
           chat, result, history=history, auto_merge=False, max_continues=3
       )
       
       for i, r in enumerate(all_results):
           print(f"Part {i+1}: {len(r.text)} chars, tokens: {r.usage.total_tokens}")

Streaming Continue (v2.0)
-------------------------

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

Continue with Progress Tracking
-------------------------------

Track progress during continuation (v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()

   result1 = chat("Write a detailed technical document", history=history, max_tokens=100)
   print(f"Part 1: {len(result1.text)} chars, {result1.usage.total_tokens} tokens")

   if result1.finish_reason == "length":
       # Use enhanced API
       all_results = ChatContinue.continue_request(
           chat, result1, history=history, auto_merge=False, max_continues=3
       )
       
       for i, r in enumerate(all_results[1:], start=2):  # Skip first (already printed)
           print(f"Part {i}: {len(r.text)} chars, {r.usage.total_tokens} tokens")
       
       # Merge
       full_result = ChatContinue.merge_results(*all_results)
       print(f"Complete: {len(full_result.text)} chars, {full_result.usage.total_tokens} tokens")

Continue with Custom Parameters
-------------------------------

Pass additional parameters to continuation (v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()
   result1 = chat("Write a story", history=history, max_tokens=100, temperature=0.7)

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

Error Handling
--------------

Handle errors during continuation (v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import requests

   chat = Chat(...)
   history = ChatHistory()

   try:
       # Use complete() for automatic error handling
       result = chat.complete("Long content", history=history, max_tokens=100, max_continues=3)
   except ChatIncompleteResponseError as e:
       print(f"Response incomplete after {e.continue_count} continues")
       print(f"Received: {len(e.final_result.text)} chars")
       # Use partial result if acceptable
       result = e.final_result
   except requests.RequestException as e:
       print(f"Network error: {e}")
       result = None

Complete Workflow
-----------------

Complete workflow with continue (recommended pattern, v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatHistoryFormatter
   from lexilux.chat.exceptions import ChatIncompleteResponseError

   chat = Chat(...)
   history = ChatHistory()

   # Request long content
   prompt = "Write a comprehensive tutorial on Python"
   
   try:
       result = chat.complete(prompt, history=history, max_tokens=200, max_continues=3)
   except ChatIncompleteResponseError as e:
       print(f"Warning: Tutorial incomplete after {e.continue_count} continues")
       result = e.final_result

   # Save complete tutorial
   ChatHistoryFormatter.save(history, "python_tutorial.md")
   
   print(f"Tutorial saved: {len(result.text)} characters")
   print(f"Total tokens used: {result.usage.total_tokens}")

JSON Extraction Pattern
------------------------

Extract JSON with guaranteed completeness (v2.0):

.. code-block:: python

   from lexilux import Chat, ChatHistory
   from lexilux.chat.exceptions import ChatIncompleteResponseError
   import json

   chat = Chat(...)
   history = ChatHistory()

   def extract_json(prompt):
       """Extract JSON from response, ensuring completeness."""
       try:
           result = chat.complete(
               f"{prompt}\n\nReturn the result as valid JSON.",
               history=history,
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

Long-form Content Generation
----------------------------

Generate long content with automatic continuation (v2.0):

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
           
           if target_length and len(result.text) < target_length:
               print(f"Warning: Generated {len(result.text)} chars, target was {target_length}")
           
           return result
       except ChatIncompleteResponseError as e:
           print(f"Generated {len(e.final_result.text)} chars before max continues")
           return e.final_result

   result = generate_long_content("Write a comprehensive guide to Python")
