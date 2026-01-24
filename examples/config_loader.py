#!/usr/bin/env python
"""
Configuration loader for examples.

Provides a unified way to load API endpoint configurations from JSON files.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def load_endpoints_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """
    Load endpoints configuration from JSON file.

    Args:
        config_path: Path to the configuration file. If None, tries default locations:
            1. tests/test_endpoints.json (default)
            2. examples/test_endpoints.json

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file is not found.
        json.JSONDecodeError: If configuration file is invalid JSON.
    """
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    else:
        # Try default locations
        default_paths = [
            Path(__file__).parent.parent / "tests" / "test_endpoints.json",
            Path(__file__).parent / "test_endpoints.json",
        ]

        config_file = None
        for path in default_paths:
            if path.exists():
                config_file = path
                break

        if config_file is None:
            raise FileNotFoundError(
                f"Configuration file not found. Tried:\n"
                f"  - {default_paths[0]}\n"
                f"  - {default_paths[1]}\n"
                f"Please create test_endpoints.json in one of these locations."
            )

    with open(config_file) as f:
        return json.load(f)


def get_chat_config(
    config: Optional[dict[str, Any]] = None, config_path: Optional[str] = None
) -> dict[str, Any]:
    """
    Get Chat API configuration from endpoints config.

    Args:
        config: Pre-loaded configuration dict. If None, loads from config_path.
        config_path: Path to configuration file. Only used if config is None.

    Returns:
        Dictionary with 'base_url', 'api_key', and 'model' keys.

    Raises:
        KeyError: If 'completion' key is not found in config.
    """
    if config is None:
        config = load_endpoints_config(config_path)

    if "completion" not in config:
        raise KeyError(
            "Configuration must contain 'completion' key for Chat API. "
            f"Available keys: {list(config.keys())}"
        )

    completion_config = config["completion"]
    return {
        "base_url": completion_config["api_base"],
        "api_key": completion_config["api_key"],
        "model": completion_config.get("model")
        or completion_config.get("source_model"),
    }


def get_embed_config(
    config: Optional[dict[str, Any]] = None, config_path: Optional[str] = None
) -> dict[str, Any]:
    """
    Get Embed API configuration from endpoints config.

    Args:
        config: Pre-loaded configuration dict. If None, loads from config_path.
        config_path: Path to configuration file. Only used if config is None.

    Returns:
        Dictionary with 'base_url', 'api_key', and 'model' keys.

    Raises:
        KeyError: If 'embedding' key is not found in config.
    """
    if config is None:
        config = load_endpoints_config(config_path)

    if "embedding" not in config:
        raise KeyError(
            "Configuration must contain 'embedding' key for Embed API. "
            f"Available keys: {list(config.keys())}"
        )

    embedding_config = config["embedding"]
    return {
        "base_url": embedding_config["api_base"],
        "api_key": embedding_config["api_key"],
        "model": embedding_config.get("model") or embedding_config.get("source_model"),
    }


def get_rerank_config(
    config: Optional[dict[str, Any]] = None, config_path: Optional[str] = None
) -> dict[str, Any]:
    """
    Get Rerank API configuration from endpoints config.

    Args:
        config: Pre-loaded configuration dict. If None, loads from config_path.
        config_path: Path to configuration file. Only used if config is None.

    Returns:
        Dictionary with 'base_url', 'api_key', 'model', and 'mode' keys.

    Raises:
        KeyError: If 'reranker' key is not found in config.
    """
    if config is None:
        config = load_endpoints_config(config_path)

    # Try 'reranker' first, then 'rerank_openai_jina', then 'rerank_dashscope'
    rerank_config = None
    for key in ["reranker", "rerank_openai_jina", "rerank_dashscope"]:
        if key in config:
            rerank_config = config[key]
            break

    if rerank_config is None:
        raise KeyError(
            "Configuration must contain one of 'reranker', 'rerank_openai_jina', or 'rerank_dashscope' keys. "
            f"Available keys: {list(config.keys())}"
        )

    return {
        "base_url": rerank_config["api_base"],
        "api_key": rerank_config["api_key"],
        "model": rerank_config.get("model") or rerank_config.get("source_model"),
        "mode": rerank_config.get("mode", "openai"),
    }


def parse_args():
    """
    Parse command line arguments for configuration file path.

    Returns:
        argparse.Namespace with 'config' attribute.
    """
    parser = argparse.ArgumentParser(
        description="Run example with custom endpoints configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/basic_chat.py
  python examples/basic_chat.py --config tests/test_endpoints.json
  python examples/basic_chat.py --config /path/to/custom_config.json
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to endpoints configuration JSON file (default: tests/test_endpoints.json)",
    )
    return parser.parse_args()
