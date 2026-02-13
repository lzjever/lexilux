# CHAT MODULE KNOWLEDGE BASE

**Generated:** 2026-01-24
**Updated:** 2026-02-13
**Commit:** 6af2459
**Branch:** main

## OVERVIEW
Chat API - streaming, history, tools, conversations, continuation support.

## 核心概念：Chat vs ChatHistory

### 快速区分

| 类 | 角色 | 状态 | 主要用途 |
|---|------|------|----------|
| **Chat** | HTTP 客户端 | 无状态 | 发送请求、获取响应 |
| **ChatHistory** | 数据容器 | 有状态 | 管理对话历史 |
| **_ResponseContinuer** | 内部工具类 | 静态方法 | 处理响应截断（内部 API） |

**重要变更 (v2.7.1):**
- `Conversation` 已重命名为 `_ResponseContinuer`（内部 API）
- 用户应使用 `chat.complete()` 处理长响应
- `Conversation` 和 `ChatContinue` 保留为废弃别名（v3.0 移除）

### 详细说明

#### 1. Chat - API 客户端

```python
from lexilux import Chat

# Chat 是一个 HTTP 客户端，每次调用都是独立的
chat = Chat(base_url="...", api_key="...", model="gpt-4")

# 基本调用 - 无状态，不记住历史
result = chat("Hello")        # 返回 ChatResult
result = chat("What's my name?")  # 不知道你叫什么，因为没有历史
```

**Chat 的方法族：**

| 方法 | 功能 | 自动续写 | 历史管理 |
|------|------|----------|----------|
| `chat()` / `acall()` | 单次请求 | ❌ | 用户管理 |
| `stream()` / `astream()` | 流式请求 | ❌ | 用户管理 |
| `complete()` / `acomplete()` | 自动续写 | ✅ | 内部管理 |
| `complete_stream()` / `acomplete_stream()` | 流式+自动续写 | ✅ | 内部管理 |

#### 2. ChatHistory - 历史管理

```python
from lexilux import Chat, ChatHistory

# ChatHistory 管理对话历史
history = ChatHistory(system="You are helpful")

# 手动管理
result1 = chat("My name is Alice", history=history)
history.add_user("My name is Alice")
history.add_assistant(result1.text)

result2 = chat("What's my name?", history=history)
# 现在 AI 知道你叫 Alice 了
```

#### 3. _ResponseContinuer - 内部续写工具（用户通常不需要直接使用）

```python
# 用户应该用 chat.complete() 而不是直接使用 _ResponseContinuer
# chat.complete() 内部自动处理续写逻辑

# 简单方式 - 推荐
result = chat.complete("Write a long story", max_tokens=50)
```

### 使用场景指南

#### 场景 1：简单问答（无历史）
```python
chat = Chat(...)
result = chat("What is Python?")
# 一次性问答，不需要记住上下文
```

#### 场景 2：多轮对话（需要历史）
```python
# 方式 A：手动管理
history = ChatHistory(system="You are helpful")
while True:
    user_input = input("You: ")
    history.add_user(user_input)
    result = chat(history.get_messages())
    history.add_assistant(result.text)

# 方式 B：使用 ChatHistory.from_chat_result
history = ChatHistory.from_chat_result("Hello", chat("Hello"))
history.add_user("What's my name?")
result = chat(history.get_messages())
```

#### 场景 3：长文本生成（需要自动续写）
```python
# 推荐：使用 complete()
result = chat.complete(
    "Write a 1000-word essay about AI",
    max_tokens=500,  # 单次限制
    max_continues=10,  # 最多续写 10 次
)
```

### 类关系图

```
┌─────────────────────────────────────────────────────────────┐
│                         用户代码                              │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┴──────────────────┐
           ▼                                     ▼
    ┌─────────────┐                       ┌─────────────┐
    │    Chat     │                       │ ChatHistory │
    │  (客户端)   │                       │  (数据容器) │
    └─────────────┘                       └─────────────┘
           │                                     │
           │    ┌────────────────────────────────┘
           │    │
           ▼    ▼
    ┌───────────────────┐
    │  chat(messages,   │
    │       history=...)│
    └───────────────────┘
           │
           ▼
    ┌───────────────────┐
    │   ChatResult      │
    │   finish_reason   │
    └───────────────────┘
           │
           │ (if "length", chat.complete() handles auto-continue)
           ▼
    ┌───────────────────┐
    │  _ResponseContinuer│ (内部 API，用户通常不需要)
    └───────────────────┘
```

### 常见误区

1. **❌ 误区：Chat 会自动记住对话历史**
   ```python
   chat("My name is Alice")
   chat("What's my name?")  # AI 不知道你叫 Alice
   ```
   **✅ 正确：使用 ChatHistory 或手动传递历史**
   ```python
   history = ChatHistory()
   history.add_user("My name is Alice")
   result = chat(history.get_messages())
   history.add_assistant(result.text)
   history.add_user("What's my name?")
   result = chat(history.get_messages())  # 现在知道了
   ```

2. **❌ 误区：需要用 Conversation 类来处理续写**
   - `Conversation` 已废弃，请使用 `chat.complete()`
   - `chat.complete()` 内部自动处理所有续写逻辑

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
