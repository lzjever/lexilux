Chat History Formatting
======================

Lexilux provides comprehensive formatting and export capabilities for conversation history
in multiple formats: Markdown, HTML, plain text, and JSON.

Overview
--------

The ``ChatHistoryFormatter`` class provides static methods to format ``ChatHistory``
instances into various output formats, making it easy to:

* Export conversations for documentation
* Create readable conversation logs
* Generate HTML reports
* Save conversations in program-friendly formats

Supported Formats
-----------------

1. **Markdown** (``.md``, ``.markdown``) - For documentation and GitHub
2. **HTML** (``.html``, ``.htm``) - For web viewing with themes
3. **Plain Text** (``.txt``, ``.text``) - For console viewing
4. **JSON** (``.json``) - For programmatic processing

Markdown Formatting
-------------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from lexilux.chat import ChatHistory, ChatHistoryFormatter

   history = ChatHistory.from_chat_result("Hello", result)
   md = ChatHistoryFormatter.to_markdown(history)
   print(md)

Output includes:
* System message (highlighted)
* Round numbers
* User and Assistant messages clearly marked

Options
~~~~~~~

.. code-block:: python

   # Without round numbers
   md = ChatHistoryFormatter.to_markdown(history, show_round_numbers=False)

   # Without system highlighting
   md = ChatHistoryFormatter.to_markdown(history, highlight_system=False)

   # With timestamps (if available in metadata)
   md = ChatHistoryFormatter.to_markdown(history, show_timestamps=True)

Example Output
~~~~~~~~~~~~~~

.. code-block:: markdown

   ## System Message

   *You are a helpful assistant*

   ### Round 1

   **User:**

   What is Python?

   **Assistant:**

   Python is a programming language...

HTML Formatting
---------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   html = ChatHistoryFormatter.to_html(history)
   print(html)

The HTML output is a complete, self-contained HTML document with embedded CSS.

Themes
~~~~~~

Three built-in themes are available:

**Default Theme** (light, modern):

.. code-block:: python

   html = ChatHistoryFormatter.to_html(history, theme="default")

**Dark Theme** (dark background):

.. code-block:: python

   html = ChatHistoryFormatter.to_html(history, theme="dark")

**Minimal Theme** (clean, minimal styling):

.. code-block:: python

   html = ChatHistoryFormatter.to_html(history, theme="minimal")

Options
~~~~~~~

.. code-block:: python

   # Without round numbers
   html = ChatHistoryFormatter.to_html(history, show_round_numbers=False)

   # With timestamps
   html = ChatHistoryFormatter.to_html(history, show_timestamps=True)

HTML Output Features
~~~~~~~~~~~~~~~~~~~~

* Self-contained (no external CSS files needed)
* Responsive design
* Clear visual separation of rounds
* Color-coded user/assistant messages
* System message highlighting

Plain Text Formatting
---------------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   text = ChatHistoryFormatter.to_text(history)
   print(text)

Perfect for console viewing or simple logs.

Options
~~~~~~~

.. code-block:: python

   # Custom width for text wrapping
   text = ChatHistoryFormatter.to_text(history, width=100)

   # Without round numbers
   text = ChatHistoryFormatter.to_text(history, show_round_numbers=False)

Example Output
~~~~~~~~~~~~~~

.. code-block:: text

   ================================================================================
   SYSTEM MESSAGE
   ================================================================================
   You are a helpful assistant

   --------------------------------------------------------------------------------
   Round 1
   --------------------------------------------------------------------------------
   User:

   What is Python?

   Assistant:

   Python is a programming language...

JSON Formatting
---------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   json_str = ChatHistoryFormatter.to_json(history)
   print(json_str)

This is equivalent to ``history.to_json()`` but provided for consistency.

Options
~~~~~~~

.. code-block:: python

   # With indentation
   json_str = ChatHistoryFormatter.to_json(history, indent=2)

   # Compact format
   json_str = ChatHistoryFormatter.to_json(history, indent=None)

File Saving
-----------

Automatic Format Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``save()`` method automatically detects the format from the file extension:

.. code-block:: python

   # Automatically detects format from extension
   ChatHistoryFormatter.save(history, "conversation.md")      # Markdown
   ChatHistoryFormatter.save(history, "conversation.html")    # HTML
   ChatHistoryFormatter.save(history, "conversation.txt")     # Text
   ChatHistoryFormatter.save(history, "conversation.json")    # JSON

Explicit Format
~~~~~~~~~~~~~~~

You can also specify the format explicitly:

.. code-block:: python

   # Explicit format (useful for custom extensions)
   ChatHistoryFormatter.save(history, "conversation.log", format="text")

   # With options
   ChatHistoryFormatter.save(history, "conversation.html", format="html", theme="dark")
   ChatHistoryFormatter.save(history, "conversation.txt", format="text", width=100)

Supported Extensions
~~~~~~~~~~~~~~~~~~~~

* Markdown: ``.md``, ``.markdown``
* HTML: ``.html``, ``.htm``
* Text: ``.txt``, ``.text``
* JSON: ``.json``

If the extension is unknown, Markdown is used as the default.

Best Practices
--------------

1. **Use Appropriate Format**:
   * Markdown for documentation and GitHub
   * HTML for web viewing and reports
   * Text for console logs
   * JSON for programmatic processing

2. **Save Regularly**: Export important conversations:

   .. code-block:: python

      # After important exchanges
      ChatHistoryFormatter.save(history, f"conversation_{timestamp}.md")

3. **Choose Right Theme**: Use dark theme for presentations, minimal for printing:

   .. code-block:: python

      # For presentations
      ChatHistoryFormatter.save(history, "presentation.html", theme="dark")

      # For printing
      ChatHistoryFormatter.save(history, "print.html", theme="minimal")

4. **Text Width**: Adjust text width for your console:

   .. code-block:: python

      # For wide terminals
      ChatHistoryFormatter.save(history, "log.txt", width=120)

      # For narrow terminals
      ChatHistoryFormatter.save(history, "log.txt", width=80)

Common Pitfalls
---------------

1. **HTML File Size**: HTML files include embedded CSS, which can make them larger
   than Markdown. For very long conversations, consider using Markdown instead.

2. **Text Wrapping**: Text formatting wraps at the specified width, which may break
   long words or code blocks. Consider using Markdown or HTML for code-heavy conversations.

3. **JSON Formatting**: JSON output is for programmatic use. For human reading,
   use Markdown or HTML.

4. **File Overwriting**: ``save()`` will overwrite existing files without warning.
   Always check file existence if needed:

   .. code-block:: python

      from pathlib import Path
      filepath = Path("conversation.md")
      if filepath.exists():
          # Handle existing file
          pass
      ChatHistoryFormatter.save(history, str(filepath))

5. **Encoding**: All files are saved with UTF-8 encoding. Make sure your text editor
   supports UTF-8 for proper display of special characters.

Examples
--------

Complete Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lexilux import Chat
   from lexilux.chat import ChatHistory, ChatHistoryFormatter

   chat = Chat(base_url="https://api.example.com/v1", api_key="key", model="gpt-4")

   # Build conversation
   result1 = chat("What is Python?")
   history = ChatHistory.from_chat_result("What is Python?", result1)

   result2 = chat(history.get_messages() + [{"role": "user", "content": "Tell me more"}])
   history = ChatHistory.from_chat_result(
       history.get_messages() + [{"role": "user", "content": "Tell me more"}],
       result2
   )

   # Export in multiple formats
   ChatHistoryFormatter.save(history, "conversation.md")           # For docs
   ChatHistoryFormatter.save(history, "conversation.html", theme="dark")  # For viewing
   ChatHistoryFormatter.save(history, "conversation.json")          # For processing

Batch Export
~~~~~~~~~~~~

.. code-block:: python

   # Export multiple conversations
   conversations = [history1, history2, history3]

   for i, history in enumerate(conversations):
       ChatHistoryFormatter.save(
           history,
           f"conversation_{i+1}.md",
           show_round_numbers=True
       )

Custom Formatting
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get formatted string and process further
   md = ChatHistoryFormatter.to_markdown(history)
   # Add custom header
   custom_md = f"# Conversation Log\\n\\nDate: {timestamp}\\n\\n{md}"
   with open("custom.md", "w") as f:
       f.write(custom_md)

