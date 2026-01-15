Chat Module
===========

The Chat module provides a comprehensive chat completion API with history management,
formatting, streaming capabilities, function calling, and multimodal support.

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

.. autoclass:: lexilux.chat.models.ToolCall
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lexilux.chat.params.ChatParams
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

History Management
~~~~~~~~~~~~~~~~~~

.. autoclass:: lexilux.chat.history.ChatHistory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lexilux.chat.history.TokenAnalysis
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

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

Continue Functionality
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lexilux.chat.continue_.ChatContinue
   :members:
   :undoc-members:
   :show-inheritance:

Function Calling
~~~~~~~~~~~~~~~~

.. autoclass:: lexilux.chat.tools.FunctionTool
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lexilux.chat.tools.ToolChoice
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lexilux.chat.tool_helpers.ToolCallHelper
   :members:
   :undoc-members:
   :show-inheritance:

Tool Helpers
~~~~~~~~~~~~

.. autofunction:: lexilux.chat.tool_helpers.execute_tool_calls
.. autofunction:: lexilux.chat.tool_helpers.create_conversation_history

Content Blocks (Multimodal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:data:: lexilux.chat.content_blocks.ContentBlock
   :annotation: Union[TextContentBlock, ImageContentBlock]

   A content block for multimodal messages.

.. py:data:: lexilux.chat.content_blocks.TextContentBlock
   :annotation: TypedDict

   Text content block with type="text" and text field.

.. py:data:: lexilux.chat.content_blocks.ImageContentBlock
   :annotation: TypedDict

   Image content block with type="image_url" and image_url field.

.. py:data:: lexilux.chat.content_blocks.ImageUrlDetail
   :annotation: TypedDict

   Image URL detail configuration with url and optional detail field.

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: lexilux.chat.utils.normalize_messages

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
* :doc:`../chat_continue` - Guide on continuing generation
* :doc:`../token_analysis` - Guide on token analysis

