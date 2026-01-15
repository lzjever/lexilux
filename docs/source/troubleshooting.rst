Troubleshooting Guide
=====================

This guide helps you diagnose and resolve common issues when using Lexilux.

.. contents::
   :local:
   :depth: 2

## Installation Issues

### Issue: "No module named 'lexilux'"

**Symptoms**:
.. code-block:: python

   >>> import lexilux
   ModuleNotFoundError: No module named 'lexilux'

**Causes**:
- Package not installed
- Installed in wrong environment
- Virtual environment not activated

**Solutions**:

1. **Install the package**:

.. code-block:: bash

   pip install lexilux
   # Or with uv:
   uv pip install lexilux

2. **Install in development mode**:

.. code-block:: bash

   # From project root
   pip install -e .
   # Or with uv:
   uv pip install -e .

3. **Verify installation**:

.. code-block:: bash

   python -c "import lexilux; print(lexilux.__version__)"

### Issue: Version conflicts

**Symptoms**:
- ``ImportError`` or ``ModuleNotFoundError`` for dependencies
- ``AttributeError`` when accessing module attributes

**Solutions**:

1. **Check installed versions**:

.. code-block:: bash

   pip list | grep lexilux
   pip list | grep requests

2. **Reinstall with correct dependencies**:

.. code-block:: bash

   pip install --force-reinstall lexilux

3. **Use a fresh virtual environment**:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate  # Windows
   pip install lexilux

## Connection Issues

### Issue: Connection timeout

**Symptoms**:
.. code-block:: python

   lexilux.TimeoutError: Request timeout: HTTPSConnectionPool...

**Causes**:
- Network connectivity problems
- Firewall blocking connections
- Server overloaded
- Timeout too short for the request

**Solutions**:

1. **Increase timeout**:

.. code-block:: python

   from lexilux import Chat
   
   # Old API (backward compatible)
   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       timeout_s=120.0,  # Increase timeout
   )
   
   # New API (separate connect/read timeouts)
   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       connect_timeout_s=10,
       read_timeout_s=120,
   )

2. **Check network connectivity**:

.. code-block:: bash

   # Test basic connectivity
   curl -I https://api.example.com
   
   # Test with Python
   python -c "import requests; print(requests.get('https://api.example.com').status_code)"

3. **Configure proxy** (if behind corporate firewall):

.. code-block:: python

   from lexilux import Chat
   
   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       proxies={
           "http": "http://proxy.example.com:8080",
           "https": "http://proxy.example.com:8080",
       },
   )
   
   # Or use environment variables
   # export HTTP_PROXY="http://proxy.example.com:8080"
   # export HTTPS_PROXY="http://proxy.example.com:8080"

4. **Enable retry logic**:

.. code-block:: python

   from lexilux import Chat
   
   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       max_retries=3,  # Automatically retry on transient failures
   )

### Issue: Connection refused

**Symptoms**:
.. code-block:: python

   lexilux.ConnectionError: Connection refused: ...

**Causes**:
- Server down
- Wrong URL
- Port blocked by firewall
- VPN required

**Solutions**:

1. **Verify base_url**:

.. code-block:: python

   from lexilux import Chat
   
   # Make sure URL is correct (no trailing slash)
   chat = Chat(
       base_url="https://api.openai.com/v1",  # Correct
       # base_url="https://api.openai.com/v1/",  # Wrong (trailing slash removed automatically)
       api_key="your-key",
   )

2. **Test with curl**:

.. code-block:: bash

   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer YOUR_KEY"

3. **Check if VPN is required**:

Some APIs require VPN connection. Connect to VPN and retry.

## Authentication Issues

### Issue: 401 Unauthorized

**Symptoms**:
.. code-block:: python

   lexilux.AuthenticationError: HTTP 401: Unauthorized

**Causes**:
- Invalid API key
- Expired API key
- Missing API key
- Wrong authentication format

**Solutions**:

1. **Verify API key**:

.. code-block:: python

   from lexilux import Chat
   
   # Make sure API key is correct
   chat = Chat(
       base_url="https://api.openai.com/v1",
       api_key="sk-...your-key-here...",  # Verify this is correct
   )

2. **Check API key format**:

Different providers use different formats:

.. code-block:: python

   # OpenAI: sk-...
   # Anthropic: sk-ant-...
   # Custom: Check provider documentation

3. **Regenerate API key**:

If key is expired or compromised, generate a new one from the provider's dashboard.

4. **Test API key**:

.. code-block:: bash

   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer YOUR_KEY"

### Issue: 403 Forbidden

**Symptoms**:
.. code-block:: python

   lexilux.APIError: HTTP 403: Forbidden

**Causes**:
- Insufficient permissions
- API key doesn't have access to this endpoint
- Rate limit exceeded

**Solutions**:

1. **Check API key permissions**:

Verify your API key has access to the required endpoints in the provider's dashboard.

2. **Check rate limits**:

.. code-block:: python

   from lexilux import Chat
   
   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       max_retries=3,  # Auto-retry on rate limit
   )
   
   # Handle rate limit explicitly
   from lexilux import RateLimitError
   
   try:
       result = chat("Hello, world!")
   except RateLimitError as e:
       print(f"Rate limited: {e}")
       print("Wait and retry, or upgrade your plan")

## Rate Limiting

### Issue: 429 Too Many Requests

**Symptoms**:
.. code-block:: python

   lexilux.RateLimitError: HTTP 429: Rate limit exceeded

**Solutions**:

1. **Enable automatic retry**:

.. code-block:: python

   from lexilux import Chat
   
   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       max_retries=3,  # Automatically retries with exponential backoff
   )

2. **Implement exponential backoff manually**:

.. code-block:: python

   import time
   from lexilux import Chat, RateLimitError
   
   chat = Chat(base_url="...", api_key="...")
   
   max_retries = 5
   for attempt in range(max_retries):
       try:
           result = chat("Hello, world!")
           break
       except RateLimitError as e:
           if attempt < max_retries - 1:
               wait_time = 2 ** attempt  # 1s, 2s, 4s, 8s...
               print(f"Rate limited. Waiting {wait_time}s...")
               time.sleep(wait_time)
           else:
               raise

3. **Reduce request frequency**:

.. code-block:: python

   import time
   
   for i in range(100):
       result = chat(f"Message {i}")
       time.sleep(0.1)  # Add delay between requests

4. **Upgrade your plan**:

If you consistently hit rate limits, consider upgrading your API plan.

## Streaming Issues

### Issue: Stream interrupted mid-response

**Symptoms**:
.. code-block:: python

   lexilux.ConnectionError: Connection lost during streaming

**Solutions**:

1. **Check network stability**:

Ensure you have a stable internet connection.

2. **Handle partial responses**:

.. code-block:: python

   from lexilux import Chat
   import requests
   
   chat = Chat(base_url="...", api_key="...")
   chunks = []
   
   try:
       for chunk in chat.stream("Write a long story"):
           print(chunk.delta, end="", flush=True)
           chunks.append(chunk)
   except requests.RequestException as e:
       print(f"\nStream interrupted: {e}")
       
       # Check if we got any content
       done_chunks = [c for c in chunks if c.done]
       if done_chunks:
           print("Got partial content before interruption")
       else:
           print("No content received")

3. **Use non-streaming for critical requests**:

.. code-block:: python

   # For critical requests where reliability is more important than real-time updates
   result = chat("Important message")  # Non-streaming

## Performance Issues

### Issue: Slow responses

**Symptoms**:
- Requests take a long time to complete
- High latency

**Solutions**:

1. **Enable connection pooling** (already enabled by default):

.. code-block:: python

   from lexilux import Chat
   
   chat = Chat(
       base_url="https://api.example.com/v1",
       api_key="your-key",
       pool_connections=20,  # Increase for high concurrency
       pool_maxsize=20,
   )

2. **Monitor performance with logging**:

.. code-block:: python

   import logging
   
   # Enable logging to see request timings
   logging.basicConfig(level=logging.INFO)
   
   from lexilux import Chat
   chat = Chat(base_url="...", api_key="...")
   result = chat("Hello")
   # Logs will show: "Request completed in X.XXs with status 200"

3. **Use streaming for faster perceived response**:

.. code-block:: python

   # Start displaying tokens immediately
   for chunk in chat.stream("Write a long response"):
       print(chunk.delta, end="", flush=True)

## Debugging

### Enable debug logging

.. code-block:: python

   import logging
   
   # Enable debug logging
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   
   from lexilux import Chat
   
   chat = Chat(base_url="...", api_key="...")
   result = chat("Hello")
   # Logs will show:
   # - Request URL
   # - Request timeout
   # - Response time
   # - Status code
   # - Any errors

### Inspect raw response

.. code-block:: python

   from lexilux import Chat
   
   chat = Chat(base_url="...", api_key="...")
   result = chat("Hello", return_raw=True)
   
   # Inspect raw API response
   print(result.raw)
   print(result.usage)

### Check exception details

.. code-block:: python

   from lexilux import Chat, LexiluxError
   
   chat = Chat(base_url="...", api_key="...")
   
   try:
       result = chat("Hello")
   except LexiluxError as e:
       print(f"Error: {e.message}")
       print(f"Error code: {e.code}")
       print(f"Retryable: {e.retryable}")
       
       if e.retryable:
           print("This error can be retried")
       else:
           print("This error cannot be retried")

## Getting Help

If you're still experiencing issues after trying these solutions:

1. **Check the documentation**: https://github.com/YOUR_USERNAME/lexilux/docs
2. **Search existing issues**: https://github.com/YOUR_USERNAME/lexilux/issues
3. **Create a new issue**: Include:
   - Python version
   - Lexilux version
   - Code snippet
   - Error message
   - Steps to reproduce
4. **Enable debug logging** and include logs in your issue

Common Errors Reference
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Common Errors and Solutions
   :widths: 30 30 40
   :header-rows: 1

   * - Error
     - Cause
     - Solution
   * - ``ModuleNotFoundError``
     - Package not installed
     - ``pip install lexilux``
   * - ``TimeoutError``
     - Request timeout
     - Increase ``timeout_s``
   * - ``ConnectionError``
     - Network failure
     - Check network, increase ``max_retries``
   * - ``AuthenticationError`` (401)
     - Invalid API key
     - Verify API key
   * - ``RateLimitError`` (429)
     - Too many requests
     - Implement backoff, reduce frequency
   * - ``NotFoundError`` (404)
     - Invalid endpoint
     - Check ``base_url``
   * - ``ValidationError`` (400)
     - Invalid input
     - Check request parameters
   * - ``ServerError`` (5xx)
     - Server error
     - Retry, check service status

