Rerank Module
=============

The Rerank module provides document reranking functionality with support for two provider modes:
OpenAI-compatible and DashScope.

The module uses a strategy pattern with separate handlers for each mode, ensuring a unified interface
while hiding provider-specific differences.

Supported Modes
---------------

- **openai**: OpenAI-compatible standard rerank API (e.g., Jina AI) (default)
- **dashscope**: Alibaba Cloud DashScope rerank API

All modes return the same ``RerankResult`` format, hiding provider differences from users.

Unified Result Format
---------------------

Regardless of the backend provider, all modes return a consistent ``RerankResult``:

- **Results**: List of tuples ``(index, score)`` or ``(index, score, document)``
- **Sorting**: Always sorted by score in descending order (higher is better)
- **Index Mapping**: Original document indices are preserved correctly
- **Usage**: Unified ``Usage`` object with token statistics

The internal transformation process handles:
- Request format adaptation (parameter name mapping, structure wrapping)
- Response parsing (extracting from provider-specific structures)
- Result normalization (sorting, filtering, formatting)

See :doc:`../rerank_modes_comparison` for detailed comparison of internal data formats.

.. automodule:: lexilux.rerank
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :doc:`../rerank_modes_comparison` - Comprehensive comparison of rerank modes

