Installation
=============

Quick Install
-------------

Install Lexilux with pip (core dependencies only):

.. code-block:: bash

   pip install lexilux

This installs only the core dependencies (``requests>=2.28.0``). The Tokenizer feature requires additional optional dependencies.

With Tokenizer Support
----------------------

To use the Tokenizer feature, install with the optional dependencies. You can use either the full name or the short name:

**Using full name:**

.. code-block:: bash

   pip install lexilux[tokenizer]

**Using short name:**

.. code-block:: bash

   pip install lexilux[token]

Both commands install the same dependencies:
* ``transformers>=4.30.0``
* ``tokenizers>=0.13.0``

**Note:** If you try to use the ``Tokenizer`` class without installing these optional dependencies, you'll get a clear error message with installation instructions.

Using Requirements Files
-------------------------

You can also install using requirements files:

**Core dependencies only:**

.. code-block:: bash

   pip install -r requirements.txt

**With tokenizer support:**

.. code-block:: bash

   pip install -r requirements-tokenizer.txt

**Development dependencies:**

.. code-block:: bash

   pip install -r requirements-dev.txt

**Documentation dependencies:**

.. code-block:: bash

   pip install -r requirements-docs.txt

Development Install
-------------------

For development with all dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Or using Makefile:

.. code-block:: bash

   make dev-install

This installs development tools like pytest, ruff, and mypy.

If you want to run tokenizer tests with real transformers (not just mocks), you can also install tokenizer dependencies:

.. code-block:: bash

   pip install -e ".[dev,tokenizer]"

Or using requirements files:

.. code-block:: bash

   pip install -r requirements-dev.txt
   pip install -r requirements-tokenizer.txt

Multiple Optional Dependencies
-------------------------------

You can install multiple optional dependency groups at once:

.. code-block:: bash

   # Development with tokenizer
   pip install lexilux[dev,tokenizer]

   # Or using short name
   pip install lexilux[dev,token]

   # Documentation with tokenizer
   pip install lexilux[docs,tokenizer]

Requirements
------------

Core Dependencies
~~~~~~~~~~~~~~~~~~

* **Python**: 3.7 or higher
* **requests**: >=2.28.0

These are the only dependencies required for Chat, Embedding, and Rerank features.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* **transformers**: >=4.30.0 (required for Tokenizer)
* **tokenizers**: >=0.13.0 (required for Tokenizer)

These are only needed if you want to use the ``Tokenizer`` class for local tokenization.

Installation Verification
--------------------------

After installation, you can verify that Lexilux is installed correctly:

.. code-block:: python

   import lexilux
   print(lexilux.__version__)

   # Test core features (no optional dependencies needed)
   from lexilux import Chat, Embed, Rerank
   print("Core features available")

   # Test tokenizer (requires optional dependencies)
   try:
       from lexilux import Tokenizer
       print("Tokenizer available")
   except ImportError:
       print("Tokenizer not available (install with: pip install lexilux[tokenizer])")

Troubleshooting
---------------

**ImportError when using Tokenizer**

If you get an ``ImportError`` when trying to use ``Tokenizer``, it means the optional dependencies are not installed. Install them with:

.. code-block:: bash

   pip install lexilux[tokenizer]

Or:

.. code-block:: bash

   pip install lexilux[token]

**I don't need Tokenizer, can I avoid installing transformers?**

Yes! The core Lexilux package (Chat, Embedding, Rerank) only requires ``requests``. If you don't use ``Tokenizer``, you don't need to install ``transformers`` or ``tokenizers``. This keeps your environment lightweight.

**Can I install from source?**

Yes, you can install from the source repository:

.. code-block:: bash

   git clone https://github.com/lzjever/lexilux.git
   cd lexilux
   pip install -e .

   # With tokenizer support
   pip install -e ".[tokenizer]"

   # With development dependencies
   pip install -e ".[dev]"

Next Steps
----------

* :doc:`quickstart` - Get started in minutes
* :doc:`api_reference/index` - Complete API reference
* :doc:`examples/index` - Usage examples
