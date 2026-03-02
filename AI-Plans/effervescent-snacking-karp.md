# 测试脚本计划：GLM-4.7 Tool Parser 流式处理测试

## 需求分析

创建测试脚本验证 GLM-4.7 tool parser 的流式处理能力。

**关键需求**：
1. ✅ 调用 `glm4_moe_tool_parser.py` 中的真实方法
2. ✅ 依赖 tokenizer（测试真实逻辑）
3. ✅ 支持 tool call 参数分散在多个 chunk 中
4. ✅ 输入是 GLM 原始格式（包含 `<tool_call>` 标记）

**输入格式**：
```
[Chunk #1]
  Raw Text: 'I'

[Chunk #2]
  Raw Text: "'ll check"

[Chunk #N]
  Raw Text: '<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n</tool_call>'
```

## 实现方案

### 文件位置
`tests/tool_parsers/test_glm47_streaming_chunks.py`

### 核心组件

#### 1. ChunkData 类
```python
@dataclass
class ChunkData:
    chunk_id: int
    raw_text: str
```

#### 2. StreamingSimulator 类
```python
class StreamingSimulator:
    def __init__(self, model_name: str = "zai-org/GLM-4.5"):
        self.tokenizer = get_tokenizer(tokenizer_name=model_name)
        self.parser = Glm47MoeModelToolParser(self.tokenizer)
        self.request = self._create_mock_request()
        self.accumulated_text = ""

    def reset_state(self):
        """重置 parser 所有内部状态"""
        self.parser._buffer = ""
        self.parser._in_tool_call = False
        self.parser.current_tool_name_sent = False
        self.parser._current_tool_name = None
        self.parser._pending_key = None
        self.parser._streaming_string_value = False
        self.parser.prev_tool_call_arr = []
        self.parser.current_tool_id = -1
        self.parser.streamed_args_for_tool = []
        self.parser._tool_call_ids = []
        self.parser._args_started = []
        self.parser._args_closed = []
        self.parser._seen_keys = []
        self.accumulated_text = ""

    def process_chunk(self, chunk: ChunkData):
        """处理单个 chunk，调用 parser"""
        delta_text = chunk.raw_text
        previous_text = self.accumulated_text
        current_text = self.accumulated_text + delta_text
        self.accumulated_text = current_text

        result = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=self.request,
        )
        return result
```

#### 3. 输入解析器
```python
def parse_input_file(filepath: str) -> list[ChunkData]:
    """解析 input.txt，提取每个 chunk 的原始文本"""
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    chunk_pattern = r'\[Chunk #(\d+)\](.*?)(?=\[Chunk #|\Z)'
    for match in re.finditer(chunk_pattern, content, re.DOTALL):
        chunk_num = int(match.group(1))
        chunk_text = match.group(2)

        # 提取原始文本
        raw_text = ""
        if content_match := re.search(r"Content:\s*'([^']*)'", chunk_text):
            raw_text = content_match.group(1)
        elif raw_match := re.search(r"Raw Text:\s*'([^']*)'", chunk_text):
            raw_text = raw_match.group(1)

        if raw_text:
            chunks.append(ChunkData(chunk_id=chunk_num, raw_text=raw_text))

    return chunks
```

#### 4. 主函数
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='输入文件路径')
    parser.add_argument('--model', default='zai-org/GLM-4.5')
    parser.add_argument('--start-chunk', type=int, default=1)
    parser.add_argument('--end-chunk', type=int)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    # 解析输入
    chunks = parse_input_file(args.input_file)
    print(f"✓ 加载了 {len(chunks)} 个 chunks")

    # 初始化
    simulator = StreamingSimulator(model_name=args.model)
    simulator.reset_state()

    # 处理每个 chunk
    for chunk in chunks:
        if chunk.chunk_id < args.start_chunk:
            continue
        if args.end_chunk and chunk.chunk_id > args.end_chunk:
            break

        result = simulator.process_chunk(chunk)
        display_result(chunk, result, simulator)

    # 最终验证
    if args.validate:
        validate_final_state(simulator)

def display_result(chunk, result, simulator):
    """显示输入输出和 parser 状态"""
    print(f"\n{'='*60}")
    print(f"Chunk #{chunk.chunk_id}")
    print(f"{'='*60}")

    print(f"\n输入: {repr(chunk.raw_text)}")

    print(f"\n输出:")
    if result is None:
        print("  None (buffering)")
    elif result.content:
        print(f"  Content: {repr(result.content)}")
    elif result.tool_calls:
        for tc in result.tool_calls:
            print(f"  Tool Call #{tc.index}:")
            if hasattr(tc, 'function'):
                func = tc.function
                if isinstance(func, dict):
                    print(f"    Name: {func.get('name')}")
                    print(f"    Arguments: {func.get('arguments')}")
                else:
                    print(f"    Name: {func.name}")
                    print(f"    Arguments: {func.arguments}")

    print(f"\nParser 状态:")
    print(f"  Buffer: {repr(simulator.parser._buffer[:50])}")
    print(f"  In tool call: {simulator.parser._in_tool_call}")
    print(f"  Current tool ID: {simulator.parser.current_tool_id}")
    print(f"  Streamed args: {simulator.parser.streamed_args_for_tool}")
```

### 命令行接口

```bash
# 基本使用
python tests/tool_parsers/test_glm47_streaming_chunks.py input.txt

# 指定范围
python tests/tool_parsers/test_glm47_streaming_chunks.py input.txt --start-chunk 64 --end-chunk 75

# 验证 JSON
python tests/tool_parsers/test_glm47_streaming_chunks.py input.txt --validate
```

## 关键文件

### 需要创建
- `tests/tool_parsers/test_glm47_streaming_chunks.py` - 主测试脚本

### 依赖文件
- `vllm/tool_parsers/glm47_moe_tool_parser.py` - GLM-4.7 parser
- `vllm/tool_parsers/glm4_moe_tool_parser.py` - 基类（extract_tool_calls_streaming）
- `vllm/tokenizers/__init__.py` - get_tokenizer
- `input.txt` - 用户提供的输入文件

## 验证方法

### 运行测试
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py input.txt --validate
```

### 预期输出
```
✓ 加载了 75 个 chunks

============================================================
Chunk #64
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
Chunk #74
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
```

### 验证要点
- ✅ 正确解析 input.txt
- ✅ Parser 正确处理增量 chunk
- ✅ 最终 JSON 有效
- ✅ Parser 状态正确维护

## 测试场景

1. **空 chunk**: 跳过
2. **内容流式**: 逐 token 输出
3. **Tool call 增量**: 分散在多个 chunk 的 tool call
4. **完整 tool call**: 单个 chunk 包含完整 tool call
5. **多个 tool call**: 连续的多个 tool call

## 实现注意事项

1. **Tokenizer**: 需要下载 GLM-4.5/4.7 tokenizer
2. **状态重置**: 测试前必须重置 parser 状态
3. **编码**: 使用 UTF-8 编码
4. **Token IDs**: 简化版本使用空列表
5. **Mock Request**: 定义 tools 参数以启用 tool parsing
