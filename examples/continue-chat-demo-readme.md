# Continue Chat Demo

This directory contains comprehensive examples demonstrating various continue chat scenarios.

## Files

1. **`continue_chat_demo.py`** - Comprehensive demo with 10 different scenarios
2. **`continue_chat_quick_test.py`** - Quick test script for basic functionality verification

## Scenarios Covered

### Scenario 1: Basic Continue (Single Interruption)
- Tests the most common use case: a single truncation followed by one continue
- Uses `ChatContinue.continue_request()` with default parameters

### Scenario 2: Multiple Continues (Chain of Interruptions)
- Tests scenarios where multiple continues are needed
- Uses `max_continues=3` to handle chains of truncations
- Simulates models with very restrictive output limits

### Scenario 3: Streaming Continue
- Tests continue functionality with streaming responses
- Uses `ChatContinue.continue_request_stream()` for real-time continuation
- Demonstrates how to handle streaming chunks during continuation

### Scenario 4: Manual Continue (Without Auto-Merge)
- Tests getting all intermediate results separately
- Uses `auto_merge=False` to receive a list of results
- Shows how to manually merge results if needed

### Scenario 5: Custom Continue Prompts
- Tests using custom prompts for continuation
- Demonstrates `continue_prompt` parameter
- Tests `add_continue_prompt=False` for direct continuation

### Scenario 6: Large Input + Small Output Limit
- Simulates production scenarios where:
  - Models have large input context windows
  - But small output token limits
  - Requires multiple continues to get full response
- Tests with `max_continues=5` to handle very restrictive limits

### Scenario 7: Concatenate Output to Input and Continue
- Tests a manual production pattern where:
  - Output is concatenated back to input as context
  - New chat request is made (not using continue mechanism)
  - Compares with library's automatic continue mechanism
- Useful for understanding when manual patterns might be needed

### Scenario 8: Error Handling and Edge Cases
- Tests error handling:
  - Attempting to continue non-truncated results
  - Attempting to continue without history
  - Edge cases with empty results

### Scenario 9: Using Chat.complete() Method
- Tests the recommended `Chat.complete()` method
- Demonstrates `ensure_complete=True` vs `ensure_complete=False`
- Shows automatic truncation handling

### Scenario 10: Using Chat.complete_stream() Method
- Tests streaming version of `complete()`
- Demonstrates streaming with automatic continuation
- Shows real-time chunk handling during continuation

## Running the Demo

### Quick Test (Recommended First)
```bash
cd lexilux
uv run python examples/continue_chat_quick_test.py --config tests/test_endpoints.json
```

### Full Demo
```bash
cd lexilux
uv run python examples/continue_chat_demo.py --config tests/test_endpoints.json
```

**Note:** The full demo makes many API calls and may take several minutes to complete. It's designed to thoroughly test all scenarios.

## Key Observations from Testing

### ✅ What Works Well

1. **Automatic History Management**: The library automatically manages history during continues, which is very convenient.

2. **Result Merging**: Automatic merging of results works correctly, combining text and usage statistics.

3. **Error Handling**: Proper validation (e.g., requiring `finish_reason == "length"`, requiring history) prevents misuse.

4. **Streaming Support**: Streaming continues work seamlessly, providing real-time chunks.

5. **Multiple Continues**: The `max_continues` parameter correctly handles chains of truncations.

### 💡 Suggestions for Improvement

1. **Continue Prompt Customization**: The current `continue_prompt` parameter is useful, but it might be helpful to have more control over how the continue message is formatted (e.g., system message vs user message).

2. **Continue Detection**: Consider adding a helper method to check if a result needs continuation:
   ```python
   if ChatContinue.needs_continue(result):
       full_result = ChatContinue.continue_request(...)
   ```

3. **Progress Tracking**: For long chains of continues, it might be useful to have progress callbacks or logging to show which continue attempt is being made.

4. **Token Budget Management**: When dealing with models with strict output limits, it might be helpful to have utilities for estimating how many continues will be needed based on desired output length.

5. **Continue Strategy Options**: Consider supporting different continuation strategies:
   - Current: Add "continue" prompt (works well)
   - Alternative: Direct continuation without prompt (already supported via `add_continue_prompt=False`)
   - Future: Context-aware continuation prompts

### 🐛 Potential Issues Found

1. **None Found**: The library behaves correctly in all tested scenarios. The API is well-designed and handles edge cases properly.

### 📝 Production Recommendations

1. **Use `Chat.complete()` for Critical Responses**: When you need guaranteed complete responses (e.g., JSON extraction), use `chat.complete()` with `ensure_complete=True`.

2. **Monitor Token Usage**: When using multiple continues, monitor the total token usage as it can accumulate quickly.

3. **Set Appropriate `max_continues`**: Based on your model's output limits and expected response length, set `max_continues` appropriately. Too few may leave responses incomplete; too many may waste tokens.

4. **Handle Truncation Gracefully**: Even with `max_continues`, responses might still be truncated. Always check `finish_reason` after continues.

5. **Consider Streaming for Long Responses**: For very long responses, use streaming to provide better user experience and allow early termination if needed.

## Testing Approach

The demo follows these principles:

1. **User Perspective**: Tests are written from a user's perspective, using only public APIs and documentation.

2. **Realistic Scenarios**: Uses small `max_tokens` values to simulate real production constraints (models with large input but small output limits).

3. **Comprehensive Coverage**: Tests various patterns including:
   - Automatic continue (library's mechanism)
   - Manual concatenation pattern (user workaround)
   - Error handling
   - Edge cases

4. **Strict Testing**: Tests don't "go easy" on the library - they challenge it with difficult scenarios to ensure robustness.

## Example Usage Patterns

### Pattern 1: Simple Continue (Recommended)
```python
from lexilux import Chat, ChatHistory, ChatContinue

chat = Chat(...)
history = ChatHistory()

result = chat("Long prompt", history=history, max_tokens=50)
if result.finish_reason == "length":
    full_result = ChatContinue.continue_request(chat, result, history=history)
```

### Pattern 2: Ensure Complete (Best for Critical Responses)
```python
from lexilux import Chat, ChatHistory

chat = Chat(...)
history = ChatHistory()

# Automatically handles continues
result = chat.complete("Long prompt", history=history, max_tokens=50, ensure_complete=True)
```

### Pattern 3: Manual Pattern (When Needed)
```python
# If you need to concatenate output to input manually
partial_output = result.text
new_prompt = f"{original_prompt}\n\nSo far: {partial_output}\n\nContinue..."
continue_result = chat(new_prompt, history=new_history, max_tokens=50)
merged = partial_output + continue_result.text
```

## Conclusion

The continue chat functionality is well-implemented and handles various production scenarios correctly. The API is intuitive and the automatic history management makes it easy to use. The demo demonstrates that the library works as expected across diverse use cases.
