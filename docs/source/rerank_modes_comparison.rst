Rerank Modes Comparison
========================

Lexilux supports two rerank modes: **OpenAI-compatible** and **DashScope**. This document provides a comprehensive comparison of their data formats, request/response structures, internal processing, and how they are unified into a standard ``RerankResult`` format.

Overview
--------

.. list-table:: Mode Comparison Summary
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - OpenAI-Compatible Mode
     - DashScope Mode
   * - **Mode Identifier**
     - ``mode="openai"`` (default)
     - ``mode="dashscope"``
   * - **API Endpoint**
     - ``POST {base_url}/rerank``
     - ``POST {base_url}/text-rerank/text-rerank``
   * - **Request Format**
     - Direct JSON object
     - Wrapped in ``input`` and ``parameters``
   * - **Response Format**
     - Direct JSON object
     - Wrapped in ``output``
   * - **Parameter Names**
     - ``top_n``, ``return_documents``
     - ``top_n``, ``return_documents`` (in ``parameters``)
   * - **Score Field Name**
     - ``relevance_score``
     - ``relevance_score``
   * - **Document Format**
     - Nested object: ``{"text": "..."}``
     - Nested object: ``{"text": "..."}``
   * - **Use Case**
     - Standard rerank APIs (Jina, Cohere, etc.)
     - Alibaba Cloud DashScope

Request Format Comparison
--------------------------

OpenAI-Compatible Mode
~~~~~~~~~~~~~~~~~~~~~~~

**Endpoint:**
.. code-block:: http

   POST {base_url}/rerank

**Request Headers:**
.. code-block:: http

   Content-Type: application/json
   Authorization: Bearer {api_key}

**Request Body:**

.. code-block:: json

   {
     "model": "jina-reranker-v3",
     "query": "python http library",
     "documents": [
       "urllib is a built-in Python library for HTTP requests",
       "requests is a popular third-party HTTP library for Python",
       "httpx is a modern async HTTP client for Python"
     ],
     "top_n": 3,
     "return_documents": true
   }


**Key Characteristics:**
- Direct JSON object structure at top level
- Uses ``documents`` array (not ``candidates``)
- Uses ``top_n`` parameter (not ``top_k``)
- Uses ``return_documents`` boolean (not ``include_docs``)
- All fields are at the top level

**Internal Processing:**
- Handler: ``OpenAICompatibleHandler``
- Request building: Maps ``top_k`` → ``top_n``, ``include_docs`` → ``return_documents``
- Endpoint detection: Checks if ``/rerank`` is in base_url, otherwise appends it

DashScope Mode
~~~~~~~~~~~~~~

**Endpoint:**
.. code-block:: http

   POST {base_url}/text-rerank/text-rerank

**Request Headers:**
.. code-block:: http

   Content-Type: application/json
   Authorization: Bearer {api_key}

**Request Body:**

.. code-block:: json

   {
     "model": "qwen3-rerank",
     "input": {
       "query": "python http library",
       "documents": [
         "urllib is a built-in Python library for HTTP requests",
         "requests is a popular third-party HTTP library for Python",
         "httpx is a modern async HTTP client for Python"
       ]
     },
     "parameters": {
       "top_n": 3,
       "return_documents": true
     }
   }


**Key Characteristics:**
- Query and documents wrapped in ``input`` object
- Additional parameters in ``parameters`` object
- Uses same parameter names as OpenAI (``top_n``, ``return_documents``)
- Endpoint is typically the full path (not appended)

**Internal Processing:**
- Handler: ``DashScopeHandler``
- Request building: Wraps query/docs in ``input``, puts options in ``parameters``
- Endpoint: Uses base_url as-is (full path expected)


Response Format Comparison
---------------------------

OpenAI-Compatible Mode Response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Response Structure:**

.. code-block:: json

   {
     "results": [
       {
         "index": 0,
         "relevance_score": 0.95,
         "document": {
           "text": "requests is a popular third-party HTTP library for Python"
         }
       },
       {
         "index": 1,
         "relevance_score": 0.85,
         "document": {
           "text": "httpx is a modern async HTTP client for Python"
         }
       },
       {
         "index": 2,
         "relevance_score": 0.70,
         "document": {
           "text": "urllib is a built-in Python library for HTTP requests"
         }
       }
     ],
     "usage": {
       "total_tokens": 150
     }
   }


**Key Characteristics:**
- Results array with objects containing ``index``, ``relevance_score``, and ``document``
- Document is nested object with ``text`` field
- Usage information at top level
- Results already sorted by relevance (descending)

**Internal Parsing:**
- Extracts ``index`` and ``relevance_score`` from each result
- Extracts document text from ``document.text`` or ``document.content``
- Handles both dict and string document formats

DashScope Mode Response
~~~~~~~~~~~~~~~~~~~~~~~

**Response Structure:**

.. code-block:: json

   {
     "output": {
       "results": [
         {
           "index": 0,
           "relevance_score": 0.95,
           "document": {
             "text": "requests is a popular third-party HTTP library for Python"
           }
         },
         {
           "index": 1,
           "relevance_score": 0.85,
           "document": {
             "text": "httpx is a modern async HTTP client for Python"
           }
         }
       ]
     },
     "usage": {
       "total_tokens": 150
     }
   }


**Key Characteristics:**
- Results wrapped in ``output`` object
- Same structure as OpenAI inside ``output.results``
- Usage information at top level
- Results format matches OpenAI standard

**Internal Parsing:**
- Extracts results from ``output.results``
- Same parsing logic as OpenAI mode (inherits from ``OpenAICompatibleHandler``)
- Handles document extraction identically


Unified Result Format
---------------------

Both modes are unified into a standard ``RerankResult`` format, hiding provider differences from users.

Standard RerankResult Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Result Structure:**

.. code-block:: python

   RerankResult(
       results=[
           (0, 0.95),           # (index, score) when include_docs=False
           (1, 0.85),
           (2, 0.70)
       ],
       usage=Usage(
           input_tokens=100,
           output_tokens=50,
           total_tokens=150
       ),
       raw={}  # Optional: full raw response if return_raw=True
   )


**With Documents:**

.. code-block:: python

   RerankResult(
       results=[
           (0, 0.95, "requests is a popular third-party HTTP library for Python"),
           (1, 0.85, "httpx is a modern async HTTP client for Python"),
           (2, 0.70, "urllib is a built-in Python library for HTTP requests")
       ],
       usage=Usage(...),
       raw={}
   )


**Key Characteristics:**
- Results are always tuples: ``(index, score)`` or ``(index, score, document)``
- Results are sorted by score in descending order (higher is better)
- Index refers to original document position in input array
- Score is always a float (handles both positive and negative)
- Documents are optional strings (only if ``include_docs=True``)

Unification Process
~~~~~~~~~~~~~~~~~~~

The unification process happens in three stages:

1. **Request Building** (``build_request``):
   - Maps unified parameters (``top_k``, ``include_docs``) to provider-specific names
   - Adapts request structure to provider format
   - Handles endpoint path differences

2. **Response Parsing** (``parse_response``):
   - Extracts results from provider-specific response structure
   - Normalizes field names (``score``/``relevance_score`` → ``score``)
   - Extracts document text from various formats
   - Maps results to original document indices
   - Returns intermediate format: ``List[Tuple[int, float, Optional[str]]]``

3. **Result Normalization** (``_normalize_results``):
   - Sorts results by score (descending)
   - Applies ``top_k`` filtering if specified
   - Formats results based on ``include_docs`` flag
   - Returns final unified format: ``List[Tuple[int, float]]`` or ``List[Tuple[int, float, str]]``

**Unification Flow Diagram:**

.. code-block:: text

   User Call
      ↓
   Rerank.__call__()
      ↓
   Handler.build_request()  →  Provider-specific request format
      ↓
   Handler.make_request()    →  HTTP POST to provider
      ↓
   Handler.parse_response()  →  Intermediate format: List[Tuple[int, float, Optional[str]]]
      ↓
   Handler._normalize_results()  →  Unified format: List[Tuple[int, float]] or List[Tuple[int, float, str]]
      ↓
   RerankResult  →  User receives consistent format

Internal Data Format Differences
---------------------------------

Request Data Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Unified Input (User API):**

.. code-block:: python

   rerank(
       query="python http library",
       docs=["urllib", "requests", "httpx"],
       top_k=2,
       include_docs=True
   )


**OpenAI-Compatible Transformation:**

.. code-block:: json

   {
     "model": "jina-reranker-v3",
     "query": "python http library",
     "documents": ["urllib", "requests", "httpx"],
     "top_n": 2,
     "return_documents": true
   }


**DashScope Transformation:**

.. code-block:: json

   {
     "model": "qwen3-rerank",
     "input": {
       "query": "python http library",
       "documents": ["urllib", "requests", "httpx"]
     },
     "parameters": {
       "top_n": 2,
       "return_documents": true
     }
   }



Response Data Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**OpenAI-Compatible Response:**

.. code-block:: json

   {
     "results": [
       {"index": 0, "relevance_score": 0.95, "document": {"text": "requests"}},
       {"index": 1, "relevance_score": 0.85, "document": {"text": "httpx"}}
     ],
     "usage": {"total_tokens": 150}
   }


**DashScope Response:**

.. code-block:: json

   {
     "output": {
       "results": [
         {"index": 0, "relevance_score": 0.95, "document": {"text": "requests"}},
         {"index": 1, "relevance_score": 0.85, "document": {"text": "httpx"}}
       ]
     },
     "usage": {"total_tokens": 150}
   }


**Unified Output (Both Modes):**

.. code-block:: python
.. code-block:: python

   RerankResult(
       results=[
           (0, 0.95, "requests"),
           (1, 0.85, "httpx")
       ],
       usage=Usage(total_tokens=150)
   )

Key Differences Summary
-----------------------

.. list-table:: Internal Format Differences
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - OpenAI-Compatible
     - DashScope
   * - **Request Wrapping**
     - Direct JSON object
     - Wrapped in ``input`` + ``parameters``
   * - **Parameter Names**
     - ``top_n``, ``return_documents``
     - ``top_n``, ``return_documents`` (in ``parameters``)
   * - **Document Field Name**
     - ``documents``
     - ``documents`` (in ``input``)
   * - **Response Wrapping**
     - Direct ``results`` array
     - Wrapped in ``output.results``
   * - **Score Field Name**
     - ``relevance_score``
     - ``relevance_score``
   * - **Document Format**
     - ``{"text": "..."}``
     - ``{"text": "..."}``
   * - **Result Format Flexibility**
     - Fixed structure
     - Fixed structure

Usage Examples
--------------

Both modes use the same unified API:

.. code-block:: python

   from lexilux import Rerank

   # OpenAI-compatible mode (Jina) - default
   rerank_openai = Rerank(
       base_url="https://api.jina.ai/v1",
       api_key="jina_xxx",
       model="jina-reranker-v3",
       mode="openai"
   )

   # DashScope mode
   rerank_dashscope = Rerank(
       base_url="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
       api_key="sk-xxx",
       model="qwen3-rerank",
       mode="dashscope"
   )

   # Both modes return the same format
   query = "python http library"
   docs = ["urllib", "requests", "httpx"]

   result_openai = rerank_openai(query, docs, top_k=2)
   result_dashscope = rerank_dashscope(query, docs, top_k=2)

   # Both results have the same structure
   assert isinstance(result_openai.results[0], tuple)
   assert isinstance(result_dashscope.results[0], tuple)

   # Both results are sorted by score (descending)
   assert result_openai.results[0][1] >= result_openai.results[1][1]
   assert result_dashscope.results[0][1] >= result_dashscope.results[1][1]

Benefits of Unification
-----------------------

1. **Consistent API**: Users don't need to learn different APIs for different providers
2. **Easy Provider Switching**: Change mode without changing business code
3. **Unified Result Format**: Same data structure regardless of backend
4. **Score Consistency**: All scores sorted descending (higher is better)
5. **Index Mapping**: Original document indices preserved correctly
6. **Error Handling**: Consistent error messages across all modes
7. **Usage Statistics**: Unified ``Usage`` object format

Migration Guide
---------------

**From OpenAI-Compatible to DashScope:**
- Change ``mode="openai"`` to ``mode="dashscope"``
- Update ``base_url`` to DashScope endpoint
- Update ``model`` to DashScope model name
- No code changes needed (same API)


**From Any Mode to Another:**
- Only change initialization parameters
- Business logic remains identical
- Result format is always the same
