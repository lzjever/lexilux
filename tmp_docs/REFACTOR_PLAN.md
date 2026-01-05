# Chat 模块重构方案

## 一、代码重组方案

### 当前问题
- `chat.py` 文件过大（764行），包含多个职责
- 代码组织不够清晰，难以维护和扩展

### 新的目录结构

```
lexilux/
├── __init__.py
├── chat/                          # Chat 模块目录
│   ├── __init__.py               # 导出主要类和函数
│   ├── client.py                 # Chat 客户端（核心 API 调用）
│   ├── models.py                 # ChatResult, ChatStreamChunk 等数据模型
│   ├── params.py                 # ChatParams 参数配置（从 chat_params.py 移入）
│   ├── history.py                # 对话历史管理（新增）
│   ├── formatters.py             # 格式化和导出功能（新增）
│   ├── utils.py                  # 工具函数（_normalize_messages 等）
│   └── continue.py               # Continue 功能（新增）
├── embed/
├── rerank/
└── ...
```

### 模块职责划分

1. **`client.py`**: Chat 客户端核心
   - `Chat` 类
   - `__call__` 和 `stream` 方法
   - HTTP 请求处理

2. **`models.py`**: 数据模型
   - `ChatResult`
   - `ChatStreamChunk`
   - 类型别名（Role, MessageLike, MessagesLike）

3. **`params.py`**: 参数配置
   - `ChatParams` dataclass
   - 参数验证和转换

4. **`utils.py`**: 工具函数
   - `_normalize_messages`
   - `_normalize_finish_reason`
   - `_parse_usage`

5. **`history.py`**: 对话历史管理（新增）
   - `ChatHistory` 类
   - 历史记录操作
   - 序列化/反序列化

6. **`formatters.py`**: 格式化导出（新增）
   - Markdown 格式化
   - HTML 格式化
   - 纯文本格式化
   - JSON 格式化

7. **`continue.py`**: Continue 功能（新增）
   - Continue 请求处理
   - 内容合并逻辑

## 二、新功能设计

### 1. 对话历史管理 (`history.py`)

#### 设计理念
- **自动提取**：可以从任何消息列表或 Chat 调用中自动构建
- **无需手动维护**：Chat 可以自动记录历史（可选）
- **灵活构建**：支持从多种来源构建历史

#### `ChatHistory` 类

```python
class ChatHistory:
    """对话历史管理器（自动提取，无需手动维护）"""
    
    def __init__(
        self, 
        messages: list[dict[str, str]] | None = None,
        system: str | None = None
    ):
        """
        初始化对话历史
        
        Args:
            messages: 消息列表（可选，可以从任何地方提取）
            system: 系统消息（可选）
        """
        self.system = system
        self.messages: list[dict[str, str]] = messages or []
        self.metadata: dict[str, Any] = {}  # 元数据（时间戳、模型等）
    
    # 自动构建方法（核心功能）
    @classmethod
    def from_messages(cls, messages: MessagesLike, system: str | None = None) -> ChatHistory:
        """从消息列表自动构建（支持所有 Chat 支持的格式）"""
    
    @classmethod
    def from_chat_result(cls, messages: MessagesLike, result: ChatResult) -> ChatHistory:
        """从 Chat 调用和结果自动构建完整历史"""
    
    @classmethod
    def from_dict(cls, data: dict) -> ChatHistory:
        """从字典反序列化"""
    
    @classmethod
    def from_json(cls, json_str: str) -> ChatHistory:
        """从 JSON 反序列化"""
    
    # 基本操作（可选，用于手动添加）
    def add_user(self, content: str) -> None
    def add_assistant(self, content: str) -> None
    def add_message(self, role: str, content: str) -> None
    def clear(self) -> None
    def get_messages(self, include_system: bool = True) -> list[dict[str, str]]
    
    # 序列化
    def to_dict(self) -> dict[str, Any]
    def to_json(self, **kwargs) -> str
    
    # Token 统计和截断
    def count_tokens(self, tokenizer: Tokenizer) -> int
    def count_tokens_per_round(self, tokenizer: Tokenizer) -> list[tuple[int, int]]
    def truncate_by_rounds(
        self, 
        tokenizer: Tokenizer, 
        max_tokens: int,
        keep_system: bool = True
    ) -> ChatHistory
    
    # 快捷操作
    def get_last_n_rounds(self, n: int) -> ChatHistory
    def remove_last_round(self) -> None
    def append_result(self, result: ChatResult) -> None
    def update_last_assistant(self, content: str) -> None  # 用于 continue 场景
```

### 2. Token 统计和按轮截断

#### 功能设计

```python
# 在 ChatHistory 中
def count_tokens_per_round(
    self, 
    tokenizer: Tokenizer
) -> list[tuple[int, int]]:  # [(round_index, tokens), ...]
    """统计每轮的 token 数"""
    
def truncate_by_rounds(
    self,
    tokenizer: Tokenizer,
    max_tokens: int,
    keep_system: bool = True
) -> ChatHistory:
    """
    按轮截断，保留最大 token 限制内的最近几轮
    
    Args:
        tokenizer: Tokenizer 实例
        max_tokens: 最大 token 数
        keep_system: 是否保留 system message
    
    Returns:
        新的 ChatHistory 实例（不修改原实例）
    """
```

### 3. 格式化导出 (`formatters.py`)

#### 功能设计

```python
class ChatHistoryFormatter:
    """对话历史格式化器"""
    
    @staticmethod
    def to_markdown(history: ChatHistory, **options) -> str:
        """
        格式化为 Markdown
        
        Options:
            - show_round_numbers: bool = True
            - show_timestamps: bool = False
            - highlight_system: bool = True
        """
    
    @staticmethod
    def to_html(history: ChatHistory, **options) -> str:
        """
        格式化为 HTML（美观、清晰）
        
        Options:
            - theme: str = "default"  # "default" | "dark" | "minimal"
            - show_round_numbers: bool = True
            - show_timestamps: bool = False
        """
    
    @staticmethod
    def to_text(history: ChatHistory, **options) -> str:
        """
        格式化为纯文本（控制台友好）
        
        Options:
            - show_round_numbers: bool = True
            - width: int = 80
        """
    
    @staticmethod
    def to_json(history: ChatHistory, **options) -> str:
        """格式化为 JSON（程序处理友好）"""
    
    @staticmethod
    def save(
        history: ChatHistory,
        filepath: str,
        format: str = "auto"  # "auto" | "markdown" | "html" | "text" | "json"
    ) -> None:
        """保存到文件（根据扩展名自动选择格式）"""
```

### 4. 历史操作快捷函数

#### 功能设计

```python
# 在 ChatHistory 中或作为独立函数

def merge_histories(*histories: ChatHistory) -> ChatHistory:
    """合并多个对话历史"""

def filter_by_role(history: ChatHistory, role: str) -> ChatHistory:
    """按角色过滤"""

def search_content(history: ChatHistory, pattern: str) -> list[dict]:
    """搜索内容"""

def get_statistics(history: ChatHistory) -> dict[str, Any]:
    """获取统计信息（轮数、总 token、平均长度等）"""
```

### 5. Streaming 历史管理（新增设计）

#### 设计理念
- **累积式 Result**：Streaming 过程中自动累积文本，可以当作普通 Result 使用
- **实时同步**：History 可以随时获取当前累积的内容，即使中断也能同步
- **零维护**：用户无需手动累积，result 自动维护

#### `StreamingResult` 类（新增）

```python
class StreamingResult:
    """
    Streaming 累积结果（可以当作 ChatResult 使用）
    
    在 streaming 过程中自动累积文本，每次迭代时内容自动更新。
    可以当作字符串使用，也可以获取完整的 ChatResult 信息。
    """
    
    def __init__(self):
        """初始化累积结果"""
        self._text: str = ""
        self._finish_reason: str | None = None
        self._usage: Usage = Usage()
        self._done: bool = False
    
    def update(self, chunk: ChatStreamChunk) -> None:
        """更新累积内容（内部调用）"""
        self._text += chunk.delta
        if chunk.done:
            self._done = True
            self._finish_reason = chunk.finish_reason
            if chunk.usage:
                self._usage = chunk.usage
    
    @property
    def text(self) -> str:
        """获取当前累积的文本（可以当作字符串使用）"""
        return self._text
    
    @property
    def finish_reason(self) -> str | None:
        """获取 finish_reason"""
        return self._finish_reason
    
    @property
    def usage(self) -> Usage:
        """获取 usage"""
        return self._usage
    
    @property
    def done(self) -> bool:
        """是否完成"""
        return self._done
    
    def to_chat_result(self) -> ChatResult:
        """转换为 ChatResult（用于 history）"""
        return ChatResult(
            text=self._text,
            finish_reason=self._finish_reason,
            usage=self._usage
        )
    
    def __str__(self) -> str:
        """当作字符串使用"""
        return self._text
    
    def __repr__(self) -> str:
        return f"StreamingResult(text={self._text!r}, done={self._done}, finish_reason={self._finish_reason!r})"
```

#### `StreamingIterator` 类（新增）

```python
class StreamingIterator:
    """
    Streaming 迭代器（包装原始迭代器，提供累积 result）
    
    每次迭代时自动更新累积的 result，用户可以随时获取当前状态。
    """
    
    def __init__(self, chunk_iterator: Iterator[ChatStreamChunk]):
        """初始化"""
        self._iterator = chunk_iterator
        self._result = StreamingResult()
    
    def __iter__(self) -> Iterator[ChatStreamChunk]:
        """迭代 chunks"""
        for chunk in self._iterator:
            self._result.update(chunk)  # 自动累积
            yield chunk
    
    @property
    def result(self) -> StreamingResult:
        """获取当前累积的 result（随时可访问）"""
        return self._result
```

#### Chat.stream() 改进

```python
class Chat:
    def stream(
        self,
        messages: MessagesLike,
        *,
        auto_history: bool | None = None,  # 继承自 __init__ 或覆盖
        **params
    ) -> StreamingIterator:
        """
        Make a streaming chat completion request.
        
        Returns:
            StreamingIterator: 迭代器，每次 yield ChatStreamChunk
            可以通过 iterator.result 获取当前累积的 StreamingResult
        
        Examples:
            >>> iterator = chat.stream("Hello", auto_history=True)
            >>> for chunk in iterator:
            ...     print(chunk.delta, end="")
            ...     # 随时可以获取当前累积的内容
            ...     current_text = iterator.result.text
            >>> # 完成后，history 中已经有完整内容
            >>> history = chat.get_history()
        """
        # ... 现有实现 ...
        
        # 创建累积迭代器
        chunk_iterator = self._stream_internal(...)
        streaming_iterator = StreamingIterator(chunk_iterator)
        
        # 如果 auto_history，自动更新历史
        if auto_history if auto_history is not None else self.auto_history:
            # 在迭代过程中自动更新历史
            # 通过包装迭代器实现
            return self._wrap_streaming_with_history(streaming_iterator, messages)
        
        return streaming_iterator
    
    def _wrap_streaming_with_history(
        self,
        iterator: StreamingIterator,
        messages: MessagesLike
    ) -> StreamingIterator:
        """包装迭代器，自动更新历史"""
        # 创建或更新历史
        if self._history is None:
            normalized = self._normalize_messages(messages)
            self._history = ChatHistory.from_messages(normalized)
        else:
            # 提取新的用户消息
            normalized = self._normalize_messages(messages)
            new_user_msgs = [m for m in normalized if m["role"] == "user"]
            for msg in new_user_msgs:
                self._history.add_user(msg["content"])
        
        # 包装迭代器，每次更新时同步更新历史
        class HistoryUpdatingIterator(StreamingIterator):
            def __iter__(self):
                for chunk in super().__iter__():
                    # 每次 chunk 到达时，更新历史中的最后一个 assistant 消息
                    if self._history.messages and self._history.messages[-1]["role"] == "assistant":
                        self._history.messages[-1]["content"] = iterator.result.text
                    else:
                        self._history.add_assistant(iterator.result.text)
                    yield chunk
        
        return HistoryUpdatingIterator(iterator._iterator)
```

#### 使用示例

```python
# 方式1：自动历史记录（推荐）
chat = Chat(..., auto_history=True)
iterator = chat.stream("Tell me a story")

for chunk in iterator:
    print(chunk.delta, end="")
    # 随时可以获取当前累积的内容
    current_text = iterator.result.text
    # 即使中断，history 中也有当前累积的内容
    history = chat.get_history()  # 包含当前累积的内容

# 方式2：手动提取
iterator = chat.stream("Tell me a story")
accumulated_text = ""
for chunk in iterator:
    print(chunk.delta, end="")
    accumulated_text = iterator.result.text  # 自动累积

# 完成后构建历史
result = iterator.result.to_chat_result()
history = ChatHistory.from_chat_result("Tell me a story", result)
```

### 6. Continue 功能 (`continue.py`)

#### 功能设计

```python
class ChatContinue:
    """Continue 功能处理器（用户负责判断是否需要 continue）"""
    
    @staticmethod
    def continue_request(
        chat: Chat,
        history: ChatHistory,
        last_result: ChatResult,
        *,
        add_continue_prompt: bool = True,
        continue_prompt: str = "continue",
        **params
    ) -> ChatResult:
        """
        继续生成请求
        
        Args:
            chat: Chat 客户端
            history: 对话历史（包含未完成的 assistant message）
            last_result: 上次的结果（finish_reason == "length"）
            add_continue_prompt: 是否添加用户 continue 指令 round
                - True: 添加一个用户消息（如 "continue"），然后继续生成
                - False: 直接发送原始 history，最后一个 round 是未完成的 assistant
            continue_prompt: 当 add_continue_prompt=True 时的用户提示词
            **params: 额外的请求参数（传递给 chat）
        
        Returns:
            新的结果（需要与 last_result 合并）
        
        Note:
            - 用户需要自己判断 finish_reason == "length"
            - 用户需要自己合并结果（或使用 merge_results）
        """
    
    @staticmethod
    def merge_results(*results: ChatResult) -> ChatResult:
        """
        合并多个结果
        
        Args:
            *results: 多个 ChatResult，按顺序合并
        
        Returns:
            合并后的完整结果
        """
```

#### 使用示例

```python
# 方式1：添加 continue prompt（默认）
result = chat("Write a long story", max_tokens=50)
history = ChatHistory.from_chat_result("Write a long story", result)

if result.finish_reason == "length":
    # 方式1：添加 "continue" 用户消息
    continue_result = ChatContinue.continue_request(
        chat, history, result,
        add_continue_prompt=True,
        continue_prompt="continue"
    )
    # 合并结果
    full_result = ChatContinue.merge_results(result, continue_result)

# 方式2：直接发送原始 history（最后一个 assistant 未完成）
if result.finish_reason == "length":
    # 方式2：直接继续，不添加用户消息
    continue_result = ChatContinue.continue_request(
        chat, history, result,
        add_continue_prompt=False  # 直接发送原始 history
    )
    # 合并结果
    full_result = ChatContinue.merge_results(result, continue_result)
```

## 三、API 设计原则

### 1. 向后兼容
- 现有的 `Chat` 类 API 保持不变
- 所有现有代码无需修改即可运行

### 2. 可选功能
- 历史管理功能通过 `ChatHistory` 类提供，不强制使用
- Continue 功能通过 `ChatContinue` 类提供，可选使用

### 3. 简单集成（最小化用户维护）

```python
# 方式1：完全自动（推荐，零维护）
chat = Chat(..., auto_history=True)
result1 = chat("Hello")
result2 = chat("Tell me more")
history = chat.get_history()  # 自动获取完整历史
ChatHistoryFormatter.save(history, "conversation.md")

# 方式2：从现有调用中提取（无需手动维护）
result = chat("Hello")
history = ChatHistory.from_chat_result("Hello", result)
# 继续对话
result2 = chat(history.get_messages() + [{"role": "user", "content": "More"}])
history = ChatHistory.from_chat_result(history.get_messages() + [...], result2)

# 方式3：从消息列表构建（灵活）
messages = [{"role": "user", "content": "Hello"}]
history = ChatHistory.from_messages(messages)
result = chat(history.get_messages())
history = ChatHistory.from_chat_result(history.get_messages(), result)

# Continue 使用（用户判断，API 处理）
if result.finish_reason == "length":
    # 方式1：添加 continue prompt
    continue_result = ChatContinue.continue_request(
        chat, history, result,
        add_continue_prompt=True
    )
    full_result = ChatContinue.merge_results(result, continue_result)
    
    # 方式2：直接继续（不添加用户消息）
    continue_result = ChatContinue.continue_request(
        chat, history, result,
        add_continue_prompt=False
    )
    full_result = ChatContinue.merge_results(result, continue_result)
```

### 4. 与 Chat 类的集成（可选增强，减少用户维护）

```python
class Chat:
    # 现有方法保持不变
    
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
        headers: dict[str, str] | None = None,
        proxies: dict[str, str] | None = None,
        auto_history: bool = False,  # 新增：是否自动记录历史
    ):
        # ... 现有初始化 ...
        self.auto_history = auto_history
        self._history: ChatHistory | None = None  # 自动记录的历史
    
    # 现有方法保持不变
    def __call__(self, messages: MessagesLike, **params) -> ChatResult:
        # ... 现有实现 ...
        result = ChatResult(...)
        
        # 自动记录历史（如果启用）
        if self.auto_history:
            if self._history is None:
                self._history = ChatHistory.from_messages(messages)
            else:
                # 提取新的用户消息并添加到历史
                new_messages = self._normalize_messages(messages)
                for msg in new_messages:
                    if msg["role"] == "user":
                        self._history.add_user(msg["content"])
                # 添加 assistant 回复
                self._history.append_result(result)
        
        return result
    
    def stream(
        self,
        messages: MessagesLike,
        *,
        auto_history: bool | None = None,  # 可以覆盖实例设置
        **params
    ) -> StreamingIterator:
        """
        Make a streaming chat completion request.
        
        Args:
            messages: Messages in various formats.
            auto_history: Override instance auto_history setting (optional).
            **params: Other parameters.
        
        Returns:
            StreamingIterator: 迭代器，每次 yield ChatStreamChunk
            可以通过 iterator.result 获取当前累积的 StreamingResult
        
        Examples:
            >>> iterator = chat.stream("Hello", auto_history=True)
            >>> for chunk in iterator:
            ...     print(chunk.delta, end="")
            ...     # 随时可以获取当前累积的内容
            ...     current_text = iterator.result.text
            >>> # 完成后，history 中已经有完整内容
            >>> history = chat.get_history()
        """
        # ... 现有实现 ...
        
        # 创建累积迭代器
        chunk_iterator = self._stream_internal(...)
        streaming_iterator = StreamingIterator(chunk_iterator)
        
        # 如果 auto_history，自动更新历史
        use_auto_history = auto_history if auto_history is not None else self.auto_history
        if use_auto_history:
            streaming_iterator = self._wrap_streaming_with_history(streaming_iterator, messages)
        
        return streaming_iterator
    
    def _wrap_streaming_with_history(
        self,
        iterator: StreamingIterator,
        messages: MessagesLike
    ) -> StreamingIterator:
        """包装迭代器，自动更新历史"""
        # 创建或更新历史
        normalized = self._normalize_messages(messages)
        if self._history is None:
            self._history = ChatHistory.from_messages(normalized)
        else:
            # 提取新的用户消息
            new_user_msgs = [m for m in normalized if m["role"] == "user"]
            for msg in new_user_msgs:
                self._history.add_user(msg["content"])
        
        # 添加 assistant 消息占位符
        self._history.add_assistant("")  # 初始为空，后续更新
        
        # 包装迭代器，每次更新时同步更新历史
        class HistoryUpdatingIterator:
            def __init__(self, base_iterator, history):
                self._base = base_iterator
                self._history = history
                self.result = base_iterator.result  # 暴露 result
            
            def __iter__(self):
                for chunk in self._base:
                    # 每次 chunk 到达时，更新历史中的最后一个 assistant 消息
                    if self._history.messages and self._history.messages[-1]["role"] == "assistant":
                        self._history.messages[-1]["content"] = self.result.text
                    yield chunk
        
        return HistoryUpdatingIterator(iterator, self._history)
    
    # 新增：获取自动记录的历史
    def get_history(self) -> ChatHistory | None:
        """获取自动记录的历史（如果 auto_history=True）"""
        return self._history
    
    # 新增：清除自动记录的历史
    def clear_history(self) -> None:
        """清除自动记录的历史"""
        self._history = None
    
    # 新增：使用历史进行对话（便捷方法）
    def chat_with_history(
        self,
        history: ChatHistory,
        **params
    ) -> ChatResult:
        """使用历史记录进行对话"""
        return self(history.get_messages(), **params)
    
    def stream_with_history(
        self,
        history: ChatHistory,
        **params
    ) -> StreamingIterator:
        """使用历史记录进行流式对话"""
        return self.stream(history.get_messages(), **params)
```

## 四、实现计划

### 阶段 1: 代码重组
1. 创建 `lexilux/chat/` 目录
2. 拆分现有代码到各个模块
3. 更新 `__init__.py` 保持向后兼容
4. 运行测试确保无破坏性变更

### 阶段 2: 历史管理功能
1. 实现 `ChatHistory` 类
2. 实现序列化/反序列化
3. 实现 token 统计和截断
4. 添加单元测试

### 阶段 3: 格式化功能
1. 实现各种格式化器
2. 实现文件保存功能
3. 添加示例和文档

### 阶段 4: Continue 功能
1. 实现 continue 请求逻辑
2. 实现结果合并
3. 添加错误处理

### 阶段 5: 快捷函数
1. 实现各种操作函数
2. 添加文档和示例

## 五、导出设计

### `lexilux/chat/__init__.py`

```python
# 向后兼容：从子模块导出
from lexilux.chat.client import Chat
from lexilux.chat.models import ChatResult, ChatStreamChunk
from lexilux.chat.params import ChatParams

# 新功能
from lexilux.chat.history import ChatHistory
from lexilux.chat.formatters import ChatHistoryFormatter
from lexilux.chat.continue_ import ChatContinue  # continue 是关键字，用 continue_

__all__ = [
    # 现有导出
    "Chat",
    "ChatResult", 
    "ChatStreamChunk",
    "ChatParams",
    # 新功能
    "ChatHistory",
    "ChatHistoryFormatter",
    "ChatContinue",
]
```

### `lexilux/__init__.py` 更新

```python
# 保持向后兼容
from lexilux.chat import Chat, ChatResult, ChatStreamChunk, ChatParams
# 或者
from lexilux.chat import *  # 如果 chat/__init__.py 已经导出

# 新功能可选导入
from lexilux.chat import ChatHistory, ChatHistoryFormatter, ChatContinue
```

## 六、使用示例

### 基本使用（向后兼容）

```python
from lexilux import Chat

chat = Chat(base_url="...", api_key="...", model="...")
result = chat("Hello")  # 完全兼容现有代码
```

### 使用历史管理（自动提取，零维护）

```python
from lexilux import Chat
from lexilux.chat import ChatHistory, ChatHistoryFormatter

# 方式1：自动记录（最简单）
chat = Chat(..., auto_history=True, system="You are helpful")
result1 = chat("What is Python?")
result2 = chat("Tell me more")
history = chat.get_history()  # 自动获取完整历史
ChatHistoryFormatter.save(history, "conversation.md")

# 方式2：Streaming 自动记录（实时同步）
chat = Chat(..., auto_history=True)
iterator = chat.stream("Tell me a story")
for chunk in iterator:
    print(chunk.delta, end="")
    # 随时可以获取当前累积的内容
    current_text = iterator.result.text
    # 即使中断，history 中也有当前累积的内容
    history = chat.get_history()  # 包含当前累积的内容（实时同步）

# 方式3：从调用中提取（无需手动维护）
chat = Chat(...)
result1 = chat("What is Python?")
history = ChatHistory.from_chat_result("What is Python?", result1)

result2 = chat(history.get_messages() + [{"role": "user", "content": "Tell me more"}])
history = ChatHistory.from_chat_result(
    history.get_messages() + [{"role": "user", "content": "Tell me more"}],
    result2
)

# 导出
ChatHistoryFormatter.save(history, "conversation.md")
print(ChatHistoryFormatter.to_html(history))
```

### 使用 Token 截断

```python
from lexilux import Tokenizer
from lexilux.chat import ChatHistory

history = ChatHistory(...)
# ... 添加很多消息 ...

tokenizer = Tokenizer("Qwen/Qwen2.5-7B-Instruct")
truncated = history.truncate_by_rounds(
    tokenizer=tokenizer,
    max_tokens=4000,
    keep_system=True
)
```

### 使用 Continue（用户判断，API 处理）

```python
from lexilux import Chat
from lexilux.chat import ChatHistory, ChatContinue

chat = Chat(..., auto_history=True)
result = chat("Write a long story", max_tokens=100)

if result.finish_reason == "length":
    history = chat.get_history()  # 自动获取历史
    
    # 方式1：添加 continue prompt（默认）
    continue_result = ChatContinue.continue_request(
        chat, history, result,
        add_continue_prompt=True,
        continue_prompt="continue"
    )
    # 合并结果
    full_result = ChatContinue.merge_results(result, continue_result)
    # 更新历史中的最后一个 assistant 消息
    history.update_last_assistant(full_result.text)
    
    # 方式2：直接继续（不添加用户消息）
    continue_result = ChatContinue.continue_request(
        chat, history, result,
        add_continue_prompt=False  # 直接发送原始 history
    )
    full_result = ChatContinue.merge_results(result, continue_result)
    history.update_last_assistant(full_result.text)
```

## 七、注意事项

1. **向后兼容性**: 所有现有代码必须继续工作
2. **可选依赖**: Tokenizer 功能是可选的，历史管理不强制依赖
3. **性能考虑**: Token 统计可能较慢，考虑缓存
4. **错误处理**: Continue 功能需要处理各种边界情况
5. **文档完善**: 新功能需要详细的文档和示例

## 八、设计改进总结

### 1. History 自动提取（无需手动维护）
- ✅ `ChatHistory.from_messages()`: 从任何消息列表自动构建
- ✅ `ChatHistory.from_chat_result()`: 从 Chat 调用和结果自动构建
- ✅ `Chat(auto_history=True)`: Chat 自动记录历史，用户只需 `chat.get_history()`
- ✅ 支持从 JSON/字典反序列化

### 2. Continue 功能优化
- ✅ 移除 `can_continue()`，用户自己判断 `finish_reason == "length"`
- ✅ 添加 `add_continue_prompt` 选项：
  - `True`: 添加用户 continue 指令 round（默认）
  - `False`: 直接发送原始 history（最后一个可能是未完成的 assistant）
- ✅ `continue_prompt` 参数可自定义提示词

### 3. 减少用户维护成本
- ✅ **自动历史记录**：`auto_history=True` 时完全自动
- ✅ **从调用提取**：`ChatHistory.from_chat_result()` 从任何调用中提取
- ✅ **无需手动添加消息**：历史自动构建，无需 `add_user()`/`add_assistant()`
- ✅ **Continue 自动处理**：只需判断 `finish_reason`，API 处理细节

### 4. API 简洁性
- ✅ 向后兼容：现有代码无需修改
- ✅ 可选功能：新功能都是可选的
- ✅ 渐进式采用：可以逐步使用新功能
- ✅ 清晰的职责分离：每个类职责单一

## 九、Streaming 历史管理设计说明

### 设计优势

1. **累积式 Result**：
   - `StreamingResult` 自动累积文本，无需用户手动拼接
   - 可以当作字符串使用（`str(result)` 或 `result.text`）
   - 可以转换为 `ChatResult`（`result.to_chat_result()`）

2. **实时同步**：
   - History 在 streaming 过程中实时更新
   - 即使中断，history 中也包含当前累积的内容
   - 用户无需手动维护累积状态

3. **零维护**：
   - `auto_history=True` 时完全自动
   - 用户只需迭代 chunks，history 自动同步
   - 可以随时获取当前状态（`iterator.result.text`）

4. **向后兼容**：
   - 现有的 `for chunk in chat.stream():` 用法完全兼容
   - 新功能通过 `iterator.result` 访问，不影响现有代码

### 实现细节

1. **StreamingIterator**：
   - 包装原始 chunk 迭代器
   - 每次 yield chunk 时自动更新累积 result
   - 暴露 `result` 属性供用户访问

2. **History 更新**：
   - 在 streaming 开始时添加 assistant 消息占位符（空字符串）
   - 每次 chunk 到达时更新占位符内容
   - 完成后包含完整内容

3. **中断处理**：
   - 如果 streaming 中断，history 中包含已累积的内容
   - `finish_reason` 为 `None`（表示未完成）
   - 用户可以继续使用或保存部分结果

## 十、待讨论的问题

1. **Token 统计的性能**:
   - 是否需要缓存 token 计数？
   - 是否需要异步 tokenizer？

2. **格式化选项**:
   - HTML 主题是否需要外部 CSS？
   - Markdown 是否需要支持代码高亮？

3. **Continue 多次调用**:
   - 是否需要支持链式 continue（continue 的结果再次 continue）？
   - 是否需要 `continue_until_complete()` 方法？

4. **StreamingResult 的线程安全**:
   - 如果多个线程同时访问 `iterator.result`，是否需要加锁？
   - 或者明确说明不支持多线程访问？

## 十、推荐的使用模式

### 最简单模式（零维护）
```python
chat = Chat(..., auto_history=True)
result = chat("Hello")
history = chat.get_history()  # 随时获取
```

### 灵活模式（从调用提取）
```python
result = chat("Hello")
history = ChatHistory.from_chat_result("Hello", result)
```

### Continue 模式
```python
if result.finish_reason == "length":
    continue_result = ChatContinue.continue_request(
        chat, history, result,
        add_continue_prompt=False  # 或 True
    )
    full_result = ChatContinue.merge_results(result, continue_result)
```

请确认这个改进后的方案是否符合您的需求。

