# CHAT MODULE KNOWLEDGE BASE

**Generated:** 2026-01-24
**Updated:** 2026-02-12
**Commit:** 9600b79
**Branch:** main

## OVERVIEW
Chat API - streaming, history, tools, conversations, continuation support.

## STRUCTURE
```
chat/                 # 17 files, ~5,900 lines - PRIMARY HOTSPOT
├── client.py         # 1174 lines - Main Chat (sync/async)
├── conversation.py   # 830 lines - Conversation (was continue_.py)
├── history.py        # 1039 lines - ChatHistory, TokenAnalysis
├── continuer.py      # 404 lines - ConversationContinuer (complete methods)
├── _complete.py      # 278 lines - Auto-continue (extracted)
├── _request.py       # 238 lines - Request handling (extracted)
├── formatters.py     # 381 lines - ChatHistoryFormatter
├── models.py         # 322 lines - ChatResult, ChatStreamChunk, ToolCall
├── params.py         # 216 lines - ChatParams dataclass
├── tool_helpers.py   # 241 lines - Tool helpers
├── utils.py          # 181 lines - normalize_messages, parse_usage
├── streaming.py      # 170 lines - StreamingIterator
├── tools.py          # 142 lines - Tool, FunctionTool, ToolChoice
├── content_blocks.py  # 125 lines - Content blocks
├── types.py          # 40 lines - Type aliases (JSONValue, MessageDict, etc.)
├── validation.py     # 200 lines - Input validation
└── exceptions.py     # 142 lines - Chat exceptions
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Main Chat | client.py | 1174 lines, sync/async __call__/stream |
| Chat history | history.py | 1039 lines, MutableSequence, TokenAnalysis |
| Conversation | conversation.py | 984 lines (renamed from continue_.py) |
| Auto-continue | _complete.py | 278 lines, finish_reason=="length" |
| Request handling | _request.py | 238 lines (extracted) |
| Streaming | streaming.py | StreamingIterator, accumulates text |
| Messages | utils.py | normalize_messages (str/list/dict inputs) |
| Tools | tools.py, tool_helpers.py | FunctionTool, execute_tool_calls, create_conversation_history |
| Results | models.py | ChatResult, ChatStreamChunk (usage REQUIRED) |
| Params | params.py | ChatParams (temperature, max_tokens, tools) |
| Formatting | formatters.py | ChatHistoryFormatter (serialization) |

## CONVENTIONS

### Sync/Async Pairs
- Pattern: `<method>` (sync) and `a<method>` (async)
- Example: `chat()`/`achat()`, `stream()`/`astream()`
- Both client.py and conversation.py have full sync/async versions

### Message Normalization
- Accepts: string, list, dict, ChatHistory
- Normalized to list[dict] internally
- Use `normalize_messages()` utility

### Streaming
- `StreamingIterator` (sync), `AsyncStreamingIterator` (async)
- Accumulate with `chunk.delta`
- `chunk.done` flag = completion

### Tool Calling
- Define: `FunctionTool` class
- Pass: `ChatParams(tools=[...])`
- Check: `result.has_tool_calls`
- Execute: `execute_tool_calls()` helper

## ANTI-PATTERNS

### Exception Handling (Updated 2026-02-12)
- **AVOID bare `except Exception:`** except for user-provided callbacks
- Remaining acceptable uses (3 instances):
  - `conversation.py:131` - User progress callback
  - `conversation.py:201` - User error callback
  - `tool_helpers.py:77` - User function execution
- All other exception handlers should use specific exception types:
  - `LexiluxError` for library-specific errors
  - `(TypeError, ValueError, AttributeError)` for validation errors
  - `(OSError, ValueError)` for filesystem/network errors

### Code Duplication
- **NEVER** add sync/async pairs without consolidating shared logic
- Extract to private methods or helpers

### ChatStreamChunk
- **usage parameter REQUIRED** (not optional like ChatResult)
- Always pass Usage() instance

### History Immutability
- ChatHistory never modified internally when passed
- Clone created to preserve original

### Old-Style Union
- **USE PEP 604**: `A | B` NOT `Union[A, B]`
- Instances: content_blocks.py:23,40; models.py:29; tools.py:57
- Migrate to `A | B`

## REFACTORING (9600b79)

### Module Rename & Extraction
- **continue_.py → conversation.py**: Renamed
- **_complete.py extracted**: 278 lines (auto-continue logic)
- **_request.py extracted**: 238 lines (request handling)
- **client.py reduced**: ~1900 → 1174 lines

### Backward Compatibility
- `ChatContinue` alias for `Conversation` class
- Legacy `lexilux/chat.py`, `lexilux/chat_params.py` REMOVED

## EXPORTS (from chat/__init__.py)

**Classes:** Chat, ChatResult, ChatStreamChunk, ChatParams, ChatHistory, ChatHistoryFormatter, Conversation (ChatContinue), StreamingResult, StreamingIterator, AsyncStreamingIterator, TokenAnalysis

**Tools:** Tool, FunctionTool, ToolChoice, ToolCall, ToolCallHelper, execute_tool_calls, create_conversation_history

**Multimodal:** ContentBlock, TextContentBlock, ImageContentBlock, ImageUrlDetail

**Types:** Role, MessageLike, MessagesLike, TokenAnalysis, JSONValue, JsonObject, MessageDict, ToolCallDict, UsageDict, ChatResponseChoice, ChatResponse, ContinuePromptCallable, ProgressCallback, ErrorCallback

**Utils:** normalize_messages, merge_histories, filter_by_role, search_content, get_statistics

**Exceptions:** ChatIncompleteResponseError, ChatStreamInterruptedError
