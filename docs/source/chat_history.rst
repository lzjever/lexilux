Chat History Management
========================

Lexilux provides comprehensive conversation history management with automatic extraction,
serialization, token counting, truncation, and multi-format export capabilities.

Overview
--------

The ``ChatHistory`` class eliminates the need for manual history maintenance by providing:

* **Automatic extraction** from any Chat call or message list
* **Serialization** to/from JSON for persistence
* **Token counting and truncation** for context window management
* **Round-based operations** for conversation management
* **Multi-format export** (Markdown, HTML, Text, JSON)

Key Features
------------

1. **Zero Maintenance**: Extract history from any Chat call automatically
2. **Flexible Input**: Supports all message formats (string, list of strings, list of dicts)
3. **Serialization**: Save and load conversations as JSON
4. **Token Management**: Count tokens and truncate by rounds to fit context windows
5. **Format Export**: Export to Markdown, HTML, Text, or JSON formats

Basic Usage
-----------

Automatic Extraction (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to use ChatHistory is to extract it from Chat calls:

.. code-block:: python

   from lexilux import Chat
   from lexilux.chat import ChatHistory

   chat = Chat(base_url="https://api.example.com/v1", api_key="key", model="gpt-4")

   # Extract history from a Chat call - no manual maintenance!
   result = chat("What is Python?")
   history = ChatHistory.from_chat_result("What is Python?", result)

   # Continue the conversation
   result2 = chat(history.get_messages() + [{"role": "user", "content": "Tell me more"}])
   history = ChatHistory.from_chat_result(
       history.get_messages() + [{"role": "user", "content": "Tell me more"}],
       result2
   )

   # Now history contains the complete conversation

.. note::
   This approach requires no manual history maintenance. Simply extract history
   from each Chat call, and the conversation is automatically tracked.

From Messages
~~~~~~~~~~~~~

You can also build history from message lists:

.. code-block:: python

   # From string
   history = ChatHistory.from_messages("Hello", system="You are helpful")

   # From list of strings
   history = ChatHistory.from_messages(["Hello", "How are you?"])

   # From list of dicts
   messages = [
       {"role": "system", "content": "You are helpful"},
       {"role": "user", "content": "Hello"},
   ]
   history = ChatHistory.from_messages(messages)

   # System message is automatically extracted if present
   assert history.system == "You are helpful"

Manual Construction
~~~~~~~~~~~~~~~~~~~

For more control, you can manually construct and manage history:

.. code-block:: python

   history = ChatHistory(system="You are a helpful assistant")

   # Add user message
   history.add_user("What is Python?")

   # Call API
   result = chat(history.get_messages())

   # Add assistant response
   history.append_result(result)

   # Continue conversation
   history.add_user("Tell me more")
   result2 = chat(history.get_messages())
   history.append_result(result2)

Serialization
-------------

Save and Load Conversations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

ChatHistory supports full serialization to/from JSON:

.. code-block:: python

   # Save to JSON
   json_str = history.to_json(indent=2)
   with open("conversation.json", "w") as f:
       f.write(json_str)

   # Or use to_dict for custom serialization
   data = history.to_dict()
   # data is a regular dict, can be processed as needed

   # Load from JSON
   with open("conversation.json", "r") as f:
       history = ChatHistory.from_json(f.read())

   # Or from dict
   history = ChatHistory.from_dict(data)

.. warning::
   When serializing, make sure to handle the system message correctly.
   The system message is stored separately from messages, so both need to be
   preserved during serialization.

Round Operations
----------------

Get Last N Rounds
~~~~~~~~~~~~~~~~~

Extract the most recent conversation rounds:

.. code-block:: python

   # Get last 2 rounds
   recent = history.get_last_n_rounds(2)
   # recent is a new ChatHistory instance with only the last 2 rounds

   # Use for context window management
   if history.count_tokens(tokenizer) > max_tokens:
       # Keep only recent rounds
       history = history.get_last_n_rounds(3)

Remove Last Round
~~~~~~~~~~~~~~~~~

Remove the most recent conversation round:

.. code-block:: python

   # Remove last round (user + assistant pair)
   history.remove_last_round()

   # Useful for undo operations or error recovery
   if result.finish_reason == "content_filter":
       history.remove_last_round()  # Remove the filtered response

.. note::
   If the last round is incomplete (only user message, no assistant),
   ``remove_last_round()`` will still remove it.

Token Management
----------------

Count Tokens
~~~~~~~~~~~~

Count tokens in the entire history:

.. code-block:: python

   from lexilux import Tokenizer

   tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct")
   total_tokens = history.count_tokens(tokenizer)
   print(f"Total tokens: {total_tokens}")

Count Tokens Per Round
~~~~~~~~~~~~~~~~~~~~~~

Count tokens for each conversation round:

.. code-block:: python

   round_tokens = history.count_tokens_per_round(tokenizer)
   # Returns: [(round_index, tokens), ...]
   for idx, tokens in round_tokens:
       print(f"Round {idx}: {tokens} tokens")

Truncate by Rounds
~~~~~~~~~~~~~~~~~~

Truncate history to fit within a token limit, keeping the most recent rounds:

.. code-block:: python

   # Truncate to fit within 4000 tokens, keeping system message
   truncated = history.truncate_by_rounds(
       tokenizer=tokenizer,
       max_tokens=4000,
       keep_system=True
   )

   # truncated is a new ChatHistory instance
   # Original history is not modified

.. important::
   ``truncate_by_rounds()`` returns a **new** ChatHistory instance.
   It does **not** modify the original history. Make sure to assign the result
   if you want to use the truncated version:

   .. code-block:: python

      # Wrong - original history unchanged
      history.truncate_by_rounds(tokenizer, max_tokens=4000)

      # Correct - use truncated version
      history = history.truncate_by_rounds(tokenizer, max_tokens=4000)

Best Practices
--------------

1. **Use Automatic Extraction**: Prefer ``ChatHistory.from_chat_result()`` over
   manual construction. It's simpler and less error-prone.

2. **Serialize Regularly**: Save important conversations to JSON for persistence:

   .. code-block:: python

      # After each important exchange
      with open(f"conversation_{timestamp}.json", "w") as f:
          f.write(history.to_json())

3. **Manage Context Windows**: Use token counting and truncation before long conversations:

   .. code-block:: python

      # Before making a new request
      if history.count_tokens(tokenizer) > max_context:
          history = history.truncate_by_rounds(tokenizer, max_tokens=max_context)

4. **Handle Incomplete Rounds**: Be aware that incomplete rounds (user message without
   assistant response) are preserved. Use ``remove_last_round()`` if needed.

Common Pitfalls
---------------

1. **Forgetting to Assign Truncated History**:
   ``truncate_by_rounds()`` returns a new instance. Don't forget to assign it:

   .. code-block:: python

      # Wrong
      history.truncate_by_rounds(tokenizer, max_tokens=4000)
      # history is unchanged!

      # Correct
      history = history.truncate_by_rounds(tokenizer, max_tokens=4000)

2. **Multiple System Messages**: If your messages contain multiple system messages,
   only the first one is extracted to ``history.system``. The rest remain in messages:

   .. code-block:: python

      messages = [
          {"role": "system", "content": "System 1"},
          {"role": "system", "content": "System 2"},  # This stays in messages
          {"role": "user", "content": "Hello"},
      ]
      history = ChatHistory.from_messages(messages)
      # history.system == "System 1"
      # history.messages[0] == {"role": "system", "content": "System 2"}

3. **Incomplete Rounds**: When removing or truncating, incomplete rounds (user without
   assistant) are treated as valid rounds. Check for completion if needed:

   .. code-block:: python

      # Check if last round is complete
      rounds = history._get_rounds()
      if rounds and len(rounds[-1]) == 1:  # Only user message
          # Incomplete round
          history.remove_last_round()

4. **Token Counting Performance**: Token counting can be slow for long histories.
   Consider caching results or only counting when necessary.

