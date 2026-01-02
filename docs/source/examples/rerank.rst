Rerank Example
===============

Document reranking example demonstrating various features of Lexilux Rerank API.

Lexilux supports two rerank modes:

1. **OpenAI-compatible mode** (``mode="openai"``): Standard rerank API format (default)
2. **DashScope mode** (``mode="dashscope"``): Alibaba Cloud DashScope rerank API

For a detailed comparison of both modes, see :doc:`../rerank_modes_comparison`.

Mode Selection
--------------

You can specify the mode when initializing the Rerank client:

.. code-block:: python

   # OpenAI-compatible mode
   rerank = Rerank(
       base_url="https://api.example.com/v1",
       api_key="your-api-key",
       model="rerank-model",
       mode="openai"  # Use OpenAI-compatible format
   )

   # DashScope mode
   rerank = Rerank(
       base_url="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
       api_key="your-api-key",
       model="qwen3-rerank",
       mode="dashscope"  # Use DashScope format
   )

You can also override the mode for individual calls:

.. code-block:: python

   # Use OpenAI mode for this call only
   result = rerank("query", docs, mode="openai")

Basic Usage
-----------

.. literalinclude:: ../../../examples/rerank_demo.py
   :language: python
   :linenos:
   :start-after: def demo_basic_rerank
   :end-before: def demo_top_k

Score Sorting Rules
--------------------

Lexilux automatically handles different score formats returned by rerank APIs:

**Positive Scores** (e.g., 0.95, 0.80, 0.70):
  - Higher score = Better relevance
  - Sorted in descending order: 0.95 > 0.80 > 0.70

**Negative Scores** (e.g., -3.0, -4.0, -5.0):
  - Less negative = Better relevance
  - Sorted in descending order: -3.0 > -4.0 > -5.0
  - Note: -3.0 is mathematically greater than -4.0

The library automatically detects which format is used and applies the correct sorting order.

Top-K Filtering
---------------

.. literalinclude:: ../../../examples/rerank_demo.py
   :language: python
   :linenos:
   :start-after: def demo_top_k
   :end-before: def demo_include_docs

Include Documents
-----------------

.. literalinclude:: ../../../examples/rerank_demo.py
   :language: python
   :linenos:
   :start-after: def demo_include_docs
   :end-before: def demo_score_sorting_rules

OpenAI-Compatible Mode
----------------------

When using ``mode="openai"``, Lexilux uses the standard OpenAI-compatible rerank API format:

**Request Format:**

- Endpoint: ``POST /rerank``
- Payload:

  .. code-block:: json
  
     {
       "model": "rerank-model",
       "query": "search query",
       "documents": ["doc1", "doc2", "doc3"],
       "top_n": 3,
       "return_documents": true
     }

**Response Format:**

- Expected response:

  .. code-block:: json
  
     {
       "results": [
         {
           "index": 0,
           "relevance_score": 0.95,
           "document": {"text": "doc1"}
         },
         {
           "index": 1,
           "relevance_score": 0.80,
           "document": {"text": "doc2"}
         }
       ],
       "usage": {"total_tokens": 100}
     }

**Key Differences:**
- Uses ``top_n`` instead of ``top_k``
- Uses ``return_documents`` instead of ``include_docs``
- Uses ``relevance_score`` instead of ``score``
- Document is wrapped in ``{"text": "..."}`` object

DashScope Mode
--------------

When using ``mode="dashscope"``, Lexilux uses the Alibaba Cloud DashScope rerank API format:

**Request Format:**

- Endpoint: ``POST /text-rerank/text-rerank``
- Payload:

  .. code-block:: json
  
     {
       "model": "qwen3-rerank",
       "input": {
         "query": "search query",
         "documents": ["doc1", "doc2", "doc3"]
       },
       "parameters": {
         "top_n": 3,
         "return_documents": true
       }
     }

**Response Format:**

- Expected response:

  .. code-block:: json
  
     {
       "output": {
         "results": [
           {
             "index": 0,
             "relevance_score": 0.95,
             "document": {"text": "doc1"}
           }
         ]
       },
       "usage": {"total_tokens": 100}
     }

**Key Features:**
- Query and documents wrapped in ``input`` object
- Additional parameters in ``parameters`` object
- Results wrapped in ``output.results``

Response Formats
----------------

Lexilux supports multiple response formats from rerank APIs:

1. **Dictionary format with results**:

   .. code-block:: json
   
      {
        "results": [
          {"index": 0, "score": 0.95},
          {"index": 1, "score": 0.80}
        ]
      }

2. **Dictionary format with data**:

   .. code-block:: json
   
      {
        "data": [
          {"index": 0, "score": 0.95},
          {"index": 1, "score": 0.80}
        ]
      }

3. **Direct list format with document text**:

   .. code-block:: json
   
      [
        ["doc1", 0.95],
        ["doc2", 0.80]
      ]

4. **Direct list format with index**:

   .. code-block:: json
   
      [
        [0, 0.95],
        [1, 0.80]
      ]


The library automatically detects and parses all these formats.

Extra Parameters
----------------

Some rerank APIs support additional parameters:

.. literalinclude:: ../../../examples/rerank_demo.py
   :language: python
   :linenos:
   :start-after: def demo_extra_parameters
   :end-before: def main
