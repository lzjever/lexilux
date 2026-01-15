Quick Start
===========

This guide will help you get started with Lexilux in minutes.

Chat
----

Basic chat completion:

.. code-block:: python

   from lexilux import Chat

   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       model="gpt-4",
       proxies=None  # Optional: {"http": "http://proxy:port", "https": "https://proxy:port"}
   )

   result = chat("Hello, world!")
   print(result.text)
   print(result.usage.total_tokens)
   print(result.finish_reason)  # "stop", "length", "content_filter", or None

With parameters (direct arguments):

.. code-block:: python

   result = chat(
       "Tell me a joke",
       temperature=0.7,
       max_tokens=100,
       stop=".",
   )

Using ChatParams for structured configuration:

.. code-block:: python

   from lexilux import Chat, ChatParams

   # Create parameter configuration
   params = ChatParams(
       temperature=0.7,
       top_p=0.9,
       max_tokens=100,
       presence_penalty=0.2,
       frequency_penalty=0.1,
   )

   result = chat("Tell me a story", params=params)

   # You can also override params with direct arguments
   result = chat("Tell me a story", params=params, temperature=0.5)

**Advanced Configuration** (v2.2.0+):

.. code-block:: python

   # Enable automatic retry with exponential backoff
   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       model="gpt-4",
       max_retries=3,  # Automatically retry on transient failures
   )

   # Separate connection and read timeouts
   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       connect_timeout_s=5,  # Connection timeout
       read_timeout_s=30,     # Read timeout
   )

   # Configure connection pooling for high concurrency
   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       pool_connections=20,  # Increase for high concurrency
       pool_maxsize=20,
   )

.. note::
   Automatic retry is disabled by default. Enable it by setting ``max_retries > 0``.
   Only retryable errors (network issues, rate limits, server errors) are retried.

**Error Handling** (v2.2.0+):

.. code-block:: python

   from lexilux import Chat, AuthenticationError, RateLimitError, LexiluxError

   chat = Chat(base_url="https://api.example.com/v1", api_key="key")

   try:
       result = chat("Hello, world!")
   except AuthenticationError as e:
       print(f"Auth failed: {e.message}")
       print(f"Error code: {e.code}")  # "authentication_failed"
   except RateLimitError as e:
       print(f"Rate limited: {e.message}")
       print(f"Can retry: {e.retryable}")  # True
   except LexiluxError as e:
       print(f"Error: {e.code} - {e.message}")

.. seealso::
   :doc:`error_handling` - Comprehensive error handling guide

**Logging** (v2.2.0+):

.. code-block:: python

   import logging

   # Enable logging to see request timing and errors
   logging.basicConfig(level=logging.INFO)

   from lexilux import Chat
   chat = Chat(base_url="https://api.example.com/v1", api_key="key")
   result = chat("Hello")
   # Logs: "Request completed in 0.52s with status 200: https://..."

Streaming:

.. code-block:: python

   for chunk in chat.stream("Tell me a joke"):
       print(chunk.delta, end="")
       if chunk.done:
           print(f"\nUsage: {chunk.usage.total_tokens}")
           print(f"Finish reason: {chunk.finish_reason}")

Streaming with ChatParams:

.. code-block:: python

   params = ChatParams(temperature=0.5, max_tokens=50)
   for chunk in chat.stream("Write a short story", params=params):
       print(chunk.delta, end="")

Chat History Management
-----------------------

Lexilux provides explicit history management with **immutable history objects**.

.. important::
   **History Immutability**: All API methods create a clone of history internally and
   **never modify the original**. You must manually update your history after each API call.

**Explicit History Management with Manual Updates**:

.. code-block:: python

   from lexilux import Chat, ChatHistory

   chat = Chat(base_url="https://api.example.com/v1", api_key="key", model="gpt-4")

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
   
   # History now contains both turns
   print(f"Total messages: {len(history.messages)}")  # 4 messages

**History as MutableSequence**:

.. code-block:: python

   history = ChatHistory()
   history.add_user("Hello")
   history.add_assistant("Hi!")
   
   # Array-like operations
   assert len(history) == 2
   assert history[0]["role"] == "user"
   
   # Slicing (returns new ChatHistory)
   first_turn = history[:1]
   
   # Iteration
   for msg in history:
       print(f"{msg['role']}: {msg['content']}")

**History Serialization**:

.. code-block:: python

   # Save to JSON
   json_str = history.to_json(indent=2)
   with open("conversation.json", "w") as f:
       f.write(json_str)

   # Load from JSON
   with open("conversation.json", "r") as f:
       history = ChatHistory.from_json(f.read())

**History Formatting and Export**:

.. code-block:: python

   from lexilux.chat import ChatHistoryFormatter

   # Format as Markdown
   md = ChatHistoryFormatter.to_markdown(history)
   print(md)

   # Format as HTML (with themes)
   html = ChatHistoryFormatter.to_html(history, theme="dark")
   print(html)

   # Format as plain text (console-friendly)
   text = ChatHistoryFormatter.to_text(history, width=80)
   print(text)

   # Save to file (auto-detects format from extension)
   ChatHistoryFormatter.save(history, "conversation.md")
   ChatHistoryFormatter.save(history, "conversation.html", theme="minimal")
   ChatHistoryFormatter.save(history, "conversation.txt", width=100)

**Streaming with History**:

.. code-block:: python

   history = ChatHistory()
   
   # Pass history to stream - original history is NOT modified
   iterator = chat.stream("Tell me a story", history=history)
   for chunk in iterator:
       print(chunk.delta, end="")
       # Access accumulated text at any time
       current_text = iterator.result.text

   # Get result and manually update history
   result = iterator.result.to_chat_result()
   history.add_user("Tell me a story")
   history.append_result(result)
   assert len(history.messages) == 2  # user + assistant

.. note::
   For detailed guide on history management, see :doc:`chat_history`.

**Continue Generation** (when response is cut off):

Recommended approach - use ``chat.complete()``:

.. code-block:: python

   from lexilux import Chat, ChatHistory
   import json

   chat = Chat(...)
   history = ChatHistory()

   # Automatically handles truncation, returns complete result
   # History is optional for single-turn conversations
   result = chat.complete("Write a long JSON response", max_tokens=100)
   json_data = json.loads(result.text)  # Guaranteed complete

   # Multi-turn conversation (with history)
   history = ChatHistory()
   result = chat.complete("First question", history=history, max_tokens=100)
   # Manually update history (history is immutable)
   history.add_user("First question")
   history.append_result(result)

   result2 = chat.complete("Follow-up", history=history, max_tokens=100)
   history.add_user("Follow-up")
   history.append_result(result2)

Advanced control:

.. code-block:: python

   from lexilux import Chat, ChatHistory, ChatContinue

   chat = Chat(...)
   history = ChatHistory()
   result = chat("Story", history=history, max_tokens=50)
   # Manually update history
   history.add_user("Story")
   history.append_result(result)
   
   if result.finish_reason == "length":
       # continue_request also doesn't modify original history
       full_result = ChatContinue.continue_request(
           chat, result, history=history, max_continues=3
       )
       # Update history with merged result
       history.append_result(full_result)

.. note::
   For detailed guide on continue functionality, see :doc:`chat_continue`.

**Token Analysis** (comprehensive token statistics):

.. code-block:: python

   from lexilux import ChatHistory, Tokenizer, TokenAnalysis

   tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct")
   history = ChatHistory.from_messages("What is Python?")
   history.add_assistant("Python is a programming language...")

   # Comprehensive analysis
   analysis = history.analyze_tokens(tokenizer)
   print(f"Total: {analysis.total_tokens}")
   print(f"User: {analysis.user_tokens}, Assistant: {analysis.assistant_tokens}")
   
   # Per-round breakdown
   for idx, total, user, assistant in analysis.per_round:
       print(f"Round {idx}: {total} tokens")

.. note::
   For detailed guide on token analysis, see :doc:`token_analysis`.

Embedding
---------

Single text:

.. code-block:: python

   from lexilux import Embed

   embed = Embed(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       model="text-embedding-ada-002",
       proxies=None  # Optional: proxy configuration
   )

   result = embed("Hello, world!")
   vector = result.vectors  # List[float]

Batch:

.. code-block:: python

   result = embed(["text1", "text2"])
   vectors = result.vectors  # List[List[float]]

With parameters using EmbedParams:

.. code-block:: python

   from lexilux import Embed, EmbedParams

   # Configure embedding parameters
   params = EmbedParams(
       dimensions=512,  # For models that support it (e.g., text-embedding-3-*)
       encoding_format="float",  # or "base64"
   )

   result = embed("Hello, world!", params=params)
   vector = result.vectors

Rerank
------

.. code-block:: python

   from lexilux import Rerank

   rerank = Rerank(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       model="rerank-model",
       proxies=None  # Optional: proxy configuration
   )

   result = rerank("python http", ["urllib", "requests", "httpx"])
   ranked = result.results  # List[Tuple[int, float]]

Tokenizer
---------

.. note::
   The Tokenizer feature requires optional dependencies. Install with:
   ``pip install lexilux[tokenizer]`` or ``pip install lexilux[token]``

.. code-block:: python

   from lexilux import Tokenizer

   # Offline mode (use local cache only, fail if not found)
   tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct", offline=True)

   result = tokenizer("Hello, world!")
   print(result.usage.input_tokens)
   print(result.input_ids)

Next Steps
----------

* :doc:`api_reference/index` - Complete API reference
* :doc:`examples/index` - More examples

