# Building Documentation

This directory contains the Sphinx documentation for Lexilux.

## Building

The Makefile automatically detects your environment:

- **If `uv` is available**: Uses `uv run sphinx-build` (recommended)
- **If `uv` is not available**: Falls back to `sphinx-build` (must be in PATH)

### Quick Start

```bash
# From project root
make docs

# Or directly from docs directory
cd docs
make html
```

### Manual Override

You can override the sphinx-build command via environment variable:

```bash
# Use a specific command
SPHINXBUILD=python -m sphinx.cmd.build make html

# Or use a different virtual environment
SPHINXBUILD=/path/to/venv/bin/sphinx-build make html
```

## Requirements

Install documentation dependencies:

```bash
# With uv (recommended)
uv sync --group docs --all-extras

# Or with pip
pip install -e ".[docs]"
```

## Output

Built documentation is in `build/html/`. Open `build/html/index.html` in a browser.

