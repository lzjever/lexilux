Chat Module
===========

The Chat module provides a comprehensive chat completion API with history management,
formatting, and streaming capabilities.

Core Classes
------------

Chat Client
~~~~~~~~~~~

.. autoclass:: lexilux.chat.client.Chat
   :members:
   :undoc-members:
   :show-inheritance:

Result Models
~~~~~~~~~~~~~

.. autoclass:: lexilux.chat.models.ChatResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lexilux.chat.models.ChatStreamChunk
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lexilux.chat.params.ChatParams
   :members:
   :undoc-members:
   :show-inheritance:

History Management
~~~~~~~~~~~~~~~~~

.. autoclass:: lexilux.chat.history.ChatHistory
   :members:
   :undoc-members:
   :show-inheritance:

Formatting
~~~~~~~~~~

.. autoclass:: lexilux.chat.formatters.ChatHistoryFormatter
   :members:
   :undoc-members:
   :show-inheritance:

Streaming
~~~~~~~~~

.. autoclass:: lexilux.chat.streaming.StreamingResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lexilux.chat.streaming.StreamingIterator
   :members:
   :undoc-members:
   :show-inheritance:

Type Aliases
~~~~~~~~~~~~

.. py:data:: lexilux.chat.models.Role
   :annotation: Literal["system", "user", "assistant", "tool"]

   Valid role types for chat messages.

.. py:data:: lexilux.chat.models.MessageLike
   :annotation: Union[str, dict[str, str]]

   A single message in various formats.

.. py:data:: lexilux.chat.models.MessagesLike
   :annotation: Union[str, Sequence[MessageLike]]

   Messages in various formats (string, list of strings, list of dicts).

See Also
--------

* :doc:`../chat_history` - Detailed guide on history management
* :doc:`../chat_formatting` - Guide on formatting and export
* :doc:`../chat_streaming` - Guide on streaming with accumulation

