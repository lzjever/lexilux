Async Support
=============

Lexilux provides full async support for concurrent API calls using httpx.

Basic Async Call
----------------

.. code-block:: python

   import asyncio
   from lexilux import Chat

   async def main():
       chat = Chat(base_url="...", api_key="...", model="gpt-4")
       result = await chat.a("Hello, async world!")
       print(result.text)

   asyncio.run(main())

Concurrent Requests
-------------------

.. code-block:: python

   import asyncio
   from lexilux import Chat

   async def main():
       chat = Chat(base_url="...", api_key="...", model="gpt-4")

       questions = ["What is Python?", "What is JavaScript?", "What is Rust?"]

       # Run all requests concurrently
       tasks = [chat.a(q) for q in questions]
       results = await asyncio.gather(*tasks)

       for result in results:
           print(result.text[:50] + "...")

   asyncio.run(main())

Async Streaming
---------------

.. code-block:: python

   import asyncio
   from lexilux import Chat

   async def main():
       chat = Chat(base_url="...", api_key="...", model="gpt-4")

       full_response = ""
       async for chunk in chat.astream("Tell me a joke"):
           if chunk.delta:
               full_response += chunk.delta
               print(chunk.delta, end="", flush=True)
           if chunk.done:
               print(f"\nTokens: {chunk.usage.total_tokens}")

   asyncio.run(main())

Async Embedding
---------------

.. code-block:: python

   import asyncio
   from lexilux import Embed

   async def main():
       embed = Embed(base_url="...", api_key="...", model="text-embedding-ada-002")

       texts = ["Hello", "World", "Async"]
       tasks = [embed.a(text) for text in texts]
       results = await asyncio.gather(*tasks)

       for result in results:
           print(f"Vector: {result.vectors[:3]}...")

   asyncio.run(main())

See Also
--------

* :doc:`examples/async_examples` - Full async examples
