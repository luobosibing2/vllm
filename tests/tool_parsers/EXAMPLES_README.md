# 输入 Chunk 示例说明

本目录包含多个示例输入文件，展示不同的测试场景。

## 示例文件列表

### 1. example_simple.txt - 简单场景
**场景**: 单个 tool call，完整包含在一个 chunk 中

**特点**:
- 前面有流式文本内容（"I'll check the weather for you."）
- Tool call 完整出现在 chunk #9
- 适合测试基本的 tool call 解析

**运行**:
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py tests/tool_parsers/example_simple.txt
```

**预期输出**:
- Chunk #1-8: 输出文本内容
- Chunk #9: 输出 tool call (get_weather, city=Beijing)

---

### 2. example_incremental.txt - 增量场景
**场景**: Tool call 的各个部分分散在多个 chunk 中

**特点**:
- Tool call 标记、函数名、参数键、参数值都分散在不同 chunk
- 最细粒度的流式处理
- 测试 parser 的增量解析能力

**Chunk 分布**:
- Chunk #5: `<tool_call>`
- Chunk #6: `get_weather`
- Chunk #7: 换行符
- Chunk #8-10: `<arg_key>city</arg_key>`
- Chunk #11: 换行符
- Chunk #12-14: `<arg_value>Beijing</arg_value>`
- Chunk #15: 换行符
- Chunk #16: `</tool_call>`

**运行**:
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py tests/tool_parsers/example_incremental.txt
```

**预期输出**:
- Chunk #1-4: 输出文本内容
- Chunk #5-7: Parser 进入 tool call 状态，buffering
- Chunk #6-7: 输出 tool name
- Chunk #8-16: 逐步输出 arguments

---

### 3. example_long_string.txt - 长字符串流式场景
**场景**: String 类型参数的增量流式输出

**特点**:
- Tool call 包含长字符串参数（Python 代码）
- 字符串内容分散在多个 chunk 中（chunk #8-17）
- 测试 string 类型参数的流式处理

**参数内容**:
```python
def hello():
    print("Hello, World!")
    return True
```

**运行**:
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py tests/tool_parsers/example_long_string.txt
```

**预期输出**:
- Chunk #1-7: 输出文本内容
- Chunk #8: 开始 tool call，输出部分内容
- Chunk #9-16: 增量输出字符串内容（每个 chunk 输出一部分）
- Chunk #17: 完成 tool call

**关键验证点**:
- String 参数应该逐 chunk 流式输出，而不是等待完整的 `</arg_value>` 标记
- 每个 chunk 的输出应该是 JSON 转义后的内容

---

### 4. example_multiple_tools.txt - 多个 Tool Call 场景
**场景**: 连续的多个 tool call

**特点**:
- 包含两个连续的 tool call
- 测试 parser 处理多个 tool call 的能力
- 每个 tool call 完整包含在一个 chunk 中

**Tool Calls**:
1. `get_weather(city="Beijing")`
2. `get_weather(city="Shanghai")`

**运行**:
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py tests/tool_parsers/example_multiple_tools.txt
```

**预期输出**:
- Chunk #1-6: 输出文本内容
- Chunk #7: 输出第一个 tool call (tool #0, city=Beijing)
- Chunk #8: 输出第二个 tool call (tool #1, city=Shanghai)

---

## 输入格式说明

### 格式 1: Content 字段（文本内容）
```
[Chunk #N]
  Content: 'text content'
```
用于普通文本内容的流式输出。

### 格式 2: Raw Text 字段（GLM 原始格式）
```
[Chunk #N]
  Raw Text: '<tool_call>function_name
<arg_key>param</arg_key>
<arg_value>value</arg_value>
</tool_call>'
```
用于 tool call 的原始 GLM 格式。

### GLM Tool Call 格式
```
<tool_call>function_name
<arg_key>parameter_name</arg_key>
<arg_value>parameter_value</arg_value>
<arg_key>another_param</arg_key>
<arg_value>another_value</arg_value>
</tool_call>
```

## 测试命令示例

### 测试所有 chunks
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py tests/tool_parsers/example_simple.txt
```

### 测试特定范围
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py tests/tool_parsers/example_incremental.txt --start-chunk 5 --end-chunk 16
```

### 验证 JSON 有效性
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py tests/tool_parsers/example_long_string.txt --validate
```

### 指定模型
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py tests/tool_parsers/example_simple.txt --model zai-org/GLM-4.5
```

## 创建自定义输入文件

### 基本结构
```
[初始化信息] 模型: glm-4.7
[提示词] Your prompt here
[工具数量] N
------------------------------------------------------------

[Chunk #1]
  Content: 'text'

[Chunk #2]
  Raw Text: '<tool_call>...'
```

### 注意事项

1. **Chunk 编号**: 必须从 1 开始，连续递增
2. **Content vs Raw Text**:
   - Content: 用于普通文本
   - Raw Text: 用于 GLM 格式的 tool call
3. **换行符**: 在 Raw Text 中使用实际换行符，不要用 `\n`
4. **引号**: Content 字段的值用单引号包裹
5. **编码**: 文件必须使用 UTF-8 编码

### 测试不同场景

#### 场景 1: 空 chunk
```
[Chunk #N]
  Content: ''
```
Parser 应该跳过空 chunk。

#### 场景 2: 部分 tool call 标记
```
[Chunk #N]
  Raw Text: '<tool_'

[Chunk #N+1]
  Raw Text: 'call>'
```
Parser 应该正确 buffer 并识别完整标记。

#### 场景 3: 混合内容
```
[Chunk #N]
  Content: 'Some text'

[Chunk #N+1]
  Raw Text: '<tool_call>func</tool_call>'

[Chunk #N+2]
  Content: ' more text'
```
Parser 应该正确分离文本和 tool call。

## 验证要点

运行测试后，检查以下内容：

1. **文本输出**: Content chunks 应该正确输出文本
2. **Tool call 识别**: Parser 应该正确识别 `<tool_call>` 标记
3. **参数解析**: 参数应该正确解析为 JSON 格式
4. **状态维护**: Parser 状态（buffer, in_tool_call, tool_id）应该正确更新
5. **JSON 有效性**: 最终的 arguments 应该是有效的 JSON

## 常见问题

### Q: 为什么我的 tool call 没有被识别？
A: 检查 GLM 格式是否正确，特别是标记的拼写和闭合。

### Q: 为什么 string 参数没有流式输出？
A: 确保在 mock request 中定义了参数类型为 "string"。

### Q: 如何测试非 string 类型参数？
A: 修改 `_create_mock_request()` 中的 parameters 定义，添加其他类型（如 "number", "boolean", "object"）。

### Q: 如何测试更复杂的 tool call？
A: 创建包含多个参数、嵌套对象或数组的 tool call，参考 GLM 格式规范。
