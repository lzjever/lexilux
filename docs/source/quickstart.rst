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

Lexilux provides powerful conversation history management with automatic extraction,
serialization, and formatting capabilities.

**Automatic History Extraction** (Recommended):

.. code-block:: python

   from lexilux import Chat, ChatResult
   from lexilux.chat import ChatHistory

   chat = Chat(base_url="https://api.example.com/v1", api_key="key", model="gpt-4")

   # Extract history from any Chat call - no manual maintenance needed!
   result = chat("What is Python?")
   history = ChatHistory.from_chat_result("What is Python?", result)

   # Continue conversation
   result2 = chat(history.get_messages() + [{"role": "user", "content": "Tell me more"}])
   history = ChatHistory.from_chat_result(history.get_messages() + [...], result2)

**Manual History Construction** (Optional):

.. code-block:: python

   history = ChatHistory(system="You are a helpful assistant")
   history.add_user("What is Python?")
   result = chat(history.get_messages())
   history.append_result(result)

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

**Streaming with History Accumulation**:

.. code-block:: python

   from lexilux.chat import StreamingIterator

   # Wrap stream with StreamingIterator for automatic accumulation
   iterator = StreamingIterator(chat.stream("Tell me a story"))
   for chunk in iterator:
       print(chunk.delta, end="")
       # Access accumulated text at any time
       current_text = iterator.result.text

   # After streaming, convert to ChatResult and add to history
   result = iterator.result.to_chat_result()
   history.append_result(result)

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

   # Auto-offline mode (recommended)
   tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct", mode="auto_offline")

   result = tokenizer("Hello, world!")
   print(result.usage.input_tokens)
   print(result.input_ids)

Next Steps
----------

* :doc:`api_reference/index` - Complete API reference
* :doc:`examples/index` - More examples

