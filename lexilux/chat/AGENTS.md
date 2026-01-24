# CHAT MODULE KNOWLEDGE BASE

**Generated:** 2026-01-24
**Commit:** Current
**Branch:** main

## OVERVIEW
Chat API module with streaming, history management, tool calling, and continuation support.

## STRUCTURE
```
lexilux/chat/       # Chat submodule (12 files, 5,944 lines) - PRIMARY COMPLEXITY HOTSPOT
├── client.py       # 1921 lines - Main Chat client (sync/async duplication)
├── history.py      # 1039 lines - ChatHistory, TokenAnalysis
├── continue_.py    # 984 lines - ChatContinue (auto-continue logic)
├── formatters.py   # 381 lines - ChatHistoryFormatter
├── models.py       # 322 lines - ChatResult, ChatStreamChunk, ToolCall
├── params.py       # 216 lines - ChatParams dataclass
├── tool_helpers.py # 239 lines - execute_tool_calls, ToolCallHelper
├── utils.py        # 181 lines - normalize_messages, parse_usage
├── streaming.py    # 170 lines - StreamingIterator patterns
├── tools.py        # 142 lines - Tool, FunctionTool, ToolChoice
├── content_blocks.py # 125 lines - ContentBlock, TextContentBlock, ImageContentBlock
└── exceptions.py   # 142 lines - Chat-specific exceptions
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Main Chat client | chat/client.py | 1921 lines, sync/async __call__/stream methods |
| Chat history | chat/history.py | 1039 lines, MutableSequence, TokenAnalysis dataclass |
| Auto-continue | chat/continue_.py | 984 lines, sync/async duplication, handle finish_reason=="length" |
| Streaming patterns | chat/streaming.py | StreamingResult, StreamingIterator accumulates text |
| Message normalization | chat/utils.py | normalize_messages handles str/list/dict inputs |
| Tool calling | chat/tools.py, tool_helpers.py | FunctionTool, execute_tool_calls, create_conversation_history |
| Result models | chat/models.py | ChatResult, ChatStreamChunk (usage is REQUIRED param) |
| Parameters | chat/params.py | ChatParams dataclass (temperature, max_tokens, tools, etc.) |
| History formatting | chat/formatters.py | ChatHistoryFormatter for serialization |

## ANTI-PATTERNS (CHAT MODULE)

### Legacy Files
- **DO NOT** use `../lexilux/chat.py` or `../lexilux/chat_params.py` (legacy duplicates)
- Use modular chat submodule: `from lexilux.chat import Chat, ChatParams`

### Bare Except Clauses
- **4 bare `except Exception:` clauses** in `chat/continue_.py` (lines 365, 594, 793, 930)
- All re-raise immediately but should be specific exceptions with logging

### Code Duplication
- **NEVER** add new sync/async method pairs without consolidating shared logic
- `chat/client.py` and `chat/continue_.py` both have full sync/async duplication
- Consider extracting shared logic into private methods or helper functions

### ChatStreamChunk Usage
- **usage parameter is REQUIRED**, not optional (unlike ChatResult where default is Usage())
- Always pass Usage() instance when creating ChatStreamChunk

### History Immutability
- When ChatHistory is passed to Chat methods, it's never modified internally
- A clone is created internally to preserve the original history object
