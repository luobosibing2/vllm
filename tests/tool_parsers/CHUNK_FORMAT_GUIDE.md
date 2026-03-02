# 输入 Chunk 格式对比示例

## 场景对比

### 场景 1: 简单 Tool Call（完整）

**输入文件内容**:
```
[Chunk #1]
  Content: 'Hello'

[Chunk #2]
  Raw Text: '<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
</tool_call>'
```

**Parser 处理流程**:
```
Chunk #1:
  输入: 'Hello'
  输出: Content: 'Hello'
  状态: buffer='', in_tool_call=False

Chunk #2:
  输入: '<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>'
  输出: Tool Call #0: get_weather({"city":"Beijing"})
  状态: buffer='', in_tool_call=False, tool_id=0
```

---

### 场景 2: 增量 Tool Call（分散）

**输入文件内容**:
```
[Chunk #1]
  Raw Text: '<tool_call>'

[Chunk #2]
  Raw Text: 'get_weather'

[Chunk #3]
  Raw Text: '
<arg_key>city</arg_key>
'

[Chunk #4]
  Raw Text: '<arg_value>Beijing</arg_value>
</tool_call>'
```

**Parser 处理流程**:
```
Chunk #1:
  输入: '<tool_call>'
  输出: None (buffering)
  状态: buffer='', in_tool_call=True, tool_id=0

Chunk #2:
  输入: 'get_weather'
  输出: Tool Call #0: name='get_weather', arguments=''
  状态: buffer='', in_tool_call=True, current_tool_name='get_weather'

Chunk #3:
  输入: '\n<arg_key>city</arg_key>\n'
  输出: None (buffering, waiting for arg_value)
  状态: buffer='', pending_key='city'

Chunk #4:
  输入: '<arg_value>Beijing</arg_value>\n</tool_call>'
  输出: Tool Call #0: arguments='{"city":"Beijing"}'
  状态: buffer='', in_tool_call=False, tool_id=0
```

---

### 场景 3: 长字符串流式输出

**输入文件内容**:
```
[Chunk #1]
  Raw Text: '<tool_call>write_file
<arg_key>content</arg_key>
<arg_value>'

[Chunk #2]
  Raw Text: 'def hello():'

[Chunk #3]
  Raw Text: '
    print("Hello")'

[Chunk #4]
  Raw Text: '</arg_value>
</tool_call>'
```

**Parser 处理流程**:
```
Chunk #1:
  输入: '<tool_call>write_file\n<arg_key>content</arg_key>\n<arg_value>'
  输出: Tool Call #0: name='write_file', arguments='{"content":"'
  状态: streaming_string_value=True

Chunk #2:
  输入: 'def hello():'
  输出: Tool Call #0: arguments='def hello():'
  状态: streaming_string_value=True (继续流式输出)

Chunk #3:
  输入: '\n    print("Hello")'
  输出: Tool Call #0: arguments='\n    print(\\"Hello\\")'
  状态: streaming_string_value=True (JSON 转义)

Chunk #4:
  输入: '</arg_value>\n</tool_call>'
  输出: Tool Call #0: arguments='"}'
  状态: in_tool_call=False (完成)
```

**最终 arguments**: `{"content":"def hello():\n    print(\"Hello\")"}`

---

### 场景 4: 多个 Tool Call

**输入文件内容**:
```
[Chunk #1]
  Content: 'Checking both cities.'

[Chunk #2]
  Raw Text: '<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
</tool_call>'

[Chunk #3]
  Raw Text: '<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Shanghai</arg_value>
</tool_call>'
```

**Parser 处理流程**:
```
Chunk #1:
  输入: 'Checking both cities.'
  输出: Content: 'Checking both cities.'
  状态: tool_id=-1

Chunk #2:
  输入: '<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>'
  输出: Tool Call #0: get_weather({"city":"Beijing"})
  状态: tool_id=0, in_tool_call=False

Chunk #3:
  输入: '<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Shanghai</arg_value>\n</tool_call>'
  输出: Tool Call #1: get_weather({"city":"Shanghai"})
  状态: tool_id=1, in_tool_call=False
```

---

## 关键概念

### Buffer 机制
Parser 使用 buffer 来暂存不完整的标记：

```
输入: '<tool_'
Buffer: '<tool_' (等待更多输入)

输入: 'call>'
Buffer: '' (识别完整标记，清空 buffer)
```

### 状态转换

```
初始状态:
  in_tool_call = False
  tool_id = -1

遇到 <tool_call>:
  in_tool_call = True
  tool_id = 0

解析 tool name:
  current_tool_name = 'function_name'
  current_tool_name_sent = True

解析参数:
  pending_key = 'param_name'
  streaming_string_value = True (如果是 string 类型)

遇到 </tool_call>:
  in_tool_call = False
  完成当前 tool call
```

### String 类型流式处理

**非 String 类型** (number, boolean, object):
- 等待完整的 `</arg_value>` 标记
- 一次性输出完整参数

**String 类型**:
- 遇到 `<arg_value>` 后立即开始流式输出
- 每个 chunk 增量输出内容
- 自动进行 JSON 转义

---

## 实际运行示例

### 运行简单场景
```bash
$ python tests/tool_parsers/test_glm47_streaming_chunks.py tests/tool_parsers/example_simple.txt

正在加载输入文件: tests/tool_parsers/example_simple.txt
✓ 加载了 9 个 chunks
正在初始化 parser (model: zai-org/GLM-4.5)...
✓ Parser 初始化完成

============================================================
Chunk #1
============================================================

输入: 'I'

输出:
  Content: 'I'

Parser 状态:
  Buffer: ''
  In tool call: False
  Current tool ID: -1
  Streamed args: []

============================================================
Chunk #9
============================================================

输入: '<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>'

输出:
  Tool Call #0:
    Name: get_weather
    Arguments: {"city":"Beijing"}

Parser 状态:
  Buffer: ''
  In tool call: False
  Current tool ID: 0
  Streamed args: ['{"city":"Beijing"}']

处理了 9 个 chunks
```

### 运行增量场景
```bash
$ python tests/tool_parsers/test_glm47_streaming_chunks.py tests/tool_parsers/example_incremental.txt --start-chunk 5

============================================================
Chunk #5
============================================================

输入: '<tool_call>'

输出:
  None (buffering)

Parser 状态:
  Buffer: ''
  In tool call: True
  Current tool ID: 0
  Streamed args: ['']

============================================================
Chunk #6
============================================================

输入: 'get_weather'

输出:
  Tool Call #0:
    Name: get_weather
    Arguments:

Parser 状态:
  Buffer: '\n'
  In tool call: True
  Current tool ID: 0
  Streamed args: ['']

... (更多 chunks)
```

---

## 自定义测试用例

### 创建测试用例的步骤

1. **确定测试场景**:
   - 简单 tool call？
   - 增量 tool call？
   - 长字符串？
   - 多个 tool call？

2. **设计 chunk 分割**:
   - 在哪里分割？
   - 每个 chunk 包含多少内容？

3. **编写输入文件**:
   ```
   [Chunk #N]
     Content: '...' 或 Raw Text: '...'
   ```

4. **运行测试**:
   ```bash
   python tests/tool_parsers/test_glm47_streaming_chunks.py your_test.txt
   ```

5. **验证输出**:
   - 检查每个 chunk 的输出
   - 验证 parser 状态
   - 确认最终 JSON 有效

### 示例：测试边界情况

**测试部分标记**:
```
[Chunk #1]
  Raw Text: '<tool_ca'

[Chunk #2]
  Raw Text: 'll>func</tool_call>'
```

**测试空参数**:
```
[Chunk #1]
  Raw Text: '<tool_call>func
</tool_call>'
```

**测试特殊字符**:
```
[Chunk #1]
  Raw Text: '<tool_call>func
<arg_key>text</arg_key>
<arg_value>Hello "World" \n\t</arg_value>
</tool_call>'
```

---

## 总结

- **Content 字段**: 用于普通文本流式输出
- **Raw Text 字段**: 用于 GLM 格式的 tool call
- **增量处理**: Tool call 可以分散在多个 chunk 中
- **String 流式**: String 类型参数支持增量流式输出
- **状态维护**: Parser 正确维护内部状态
- **JSON 转义**: 自动处理特殊字符的 JSON 转义
