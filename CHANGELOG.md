# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-01-03

### Changed
- Updated Python version support to 3.8-3.14
- Integrated CI workflow with uv for automated testing and building

## [0.2.0] - 2026-01-03

### Changed
- **BREAKING**: Removed chat-based rerank mode support. Rerank now only supports OpenAI-compatible and DashScope modes.
- Changed default rerank mode from `"chat"` to `"openai"`.
- Reorganized tests: all real API tests are now marked as `@pytest.mark.integration` and excluded from default test runs.

### Removed
- `ChatBasedHandler` class and chat-based rerank mode (`mode="chat"`).
- `chat_rerank_spec.rst` documentation (no longer needed as chat mode is removed).

### Improved
- Updated test endpoints to use `rerank_local_qwen3` and `embed_local_qwen3` for integration tests.
- Improved test organization following varlord's pattern for integration tests.
- Updated documentation to reflect rerank mode changes.

## [0.1.2] - 2025-12-29

### Added
- Scripts to automatically generate release notes
- Better github action workflow

## [0.1.1] - 2025-12-29

### Added
- Examples
- Comprehensive test suite
- Documentation

## [0.1.0] - 2025-12-28

### Added
- Initial release
- Chat API support with streaming
- Embedding API support
- Rerank API support
- Tokenizer support (optional dependency on transformers)
- Unified Usage and ResultBase classes
- Documentation

