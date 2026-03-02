# 实现总结：GLM-4.7 Tool Parser 流式处理测试脚本

## 已完成的工作

### 1. 创建测试脚本
**文件**: `tests/tool_parsers/test_glm47_streaming_chunks.py`

实现了完整的测试脚本，包含以下核心组件：

#### ChunkData 类
- 数据类，表示单个流式 chunk
- 字段：`chunk_id` (int), `raw_text` (str)

#### StreamingSimulator 类
- 模拟流式处理的核心类
- 使用真实的 `Glm47MoeModelToolParser`
- 关键方法：
  - `__init__()`: 初始化 tokenizer 和 parser
  - `_create_mock_request()`: 创建 mock request 以启用 tool parsing
  - `reset_state()`: 重置 parser 所有内部状态
  - `process_chunk()`: 处理单个 chunk，调用真实的 `extract_tool_calls_streaming()`

#### parse_input_file() 函数
- 解析输入文件，提取每个 chunk 的原始文本
- 支持多种格式：
  - `Content: 'text'` - 文本内容
  - `Content: "text"` - 文本内容（双引号）
  - `Raw Text: 'text'` - GLM 原始格式
  - `Tool Calls:` - 自动转换为 GLM 原始格式

#### display_result() 函数
- 显示每个 chunk 的：
  - 输入（raw_text）
  - 输出（content 或 tool_calls）
  - Parser 状态（buffer, in_tool_call, current_tool_id, streamed_args）

#### validate_final_state() 函数
- 验证最终状态
- 检查 JSON 有效性
- 显示累积文本和 buffer 剩余

#### main() 函数
- 完整的命令行接口
- 支持参数：
  - `input_file`: 输入文件路径
  - `--model`: 模型名称（默认: zai-org/GLM-4.5）
  - `--start-chunk`: 起始 chunk 编号
  - `--end-chunk`: 结束 chunk 编号
  - `--validate`: 验证最终 JSON 有效性

### 2. 创建使用文档
**文件**: `tests/tool_parsers/README_streaming_test.md`

包含：
- 功能特性说明
- 环境要求
- 使用方法和示例
- 输入文件格式说明
- 输出示例
- 核心组件说明
- 测试场景
- 故障排除

## 关键特性

### ✅ 调用真实方法
- 使用 `Glm47MoeModelToolParser` 的真实实现
- 调用 `extract_tool_calls_streaming()` 方法

### ✅ 依赖 tokenizer
- 使用 `get_tokenizer()` 获取真实 tokenizer
- 测试真实的 tokenizer 逻辑

### ✅ 支持分散的 tool call
- 支持 tool call 参数分散在多个 chunk 中
- 正确处理增量流式输入

### ✅ 支持 GLM 原始格式
- 输入支持 `<tool_call>` 标记
- 自动转换其他格式为 GLM 原始格式

## 使用示例

### 基本使用
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py input.txt
```

### 指定范围
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py input.txt --start-chunk 64 --end-chunk 75
```

### 验证 JSON
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py input.txt --validate
```

## 测试场景覆盖

1. ✅ 空 chunk - 自动跳过
2. ✅ 内容流式 - 逐 token 输出
3. ✅ Tool call 增量 - 分散在多个 chunk 的 tool call
4. ✅ 完整 tool call - 单个 chunk 包含完整 tool call
5. ✅ 多个 tool call - 连续的多个 tool call

## 依赖文件

### 使用的文件
- `vllm/tool_parsers/glm47_moe_tool_parser.py` - GLM-4.7 parser
- `vllm/tool_parsers/glm4_moe_tool_parser.py` - 基类
- `vllm/tokenizers/__init__.py` - get_tokenizer
- `vllm/entrypoints/openai/chat_completion/protocol.py` - 协议定义

### 输入文件
- `input.txt` - 用户提供的输入文件

## 实现细节

### 状态重置
`reset_state()` 方法重置所有 parser 内部状态：
- `_buffer`
- `_in_tool_call`
- `current_tool_name_sent`
- `_current_tool_name`
- `_pending_key`
- `_streaming_string_value`
- `prev_tool_call_arr`
- `current_tool_id`
- `streamed_args_for_tool`
- `_tool_call_ids`
- `_args_started`
- `_args_closed`
- `_seen_keys`
- `accumulated_text`

### 流式处理
`process_chunk()` 方法：
1. 累积文本：`current_text = accumulated_text + delta_text`
2. 调用 parser：`extract_tool_calls_streaming()`
3. 传递参数：
   - `previous_text`: 之前的累积文本
   - `current_text`: 当前的累积文本
   - `delta_text`: 新增的文本
   - `previous_token_ids`: 空列表（简化）
   - `current_token_ids`: 空列表（简化）
   - `delta_token_ids`: 空列表（简化）
   - `request`: mock request

### 输入解析
支持多种格式，自动识别并提取：
1. Content 字段（单引号或双引号）
2. Raw Text 字段
3. Tool Calls 字段（自动转换为 GLM 格式）

### 输出显示
清晰显示：
- Chunk 编号
- 输入文本
- 输出内容或 tool calls
- Parser 内部状态

## 环境要求

需要安装：
- Python 3.8+
- torch
- transformers
- vllm（开发模式）

## 注意事项

1. **Tokenizer 下载**: 首次运行需要下载 GLM-4.5 tokenizer
2. **编码**: 使用 UTF-8 编码读取输入文件
3. **Token IDs**: 简化版本使用空列表
4. **Mock Request**: 定义 tools 参数以启用 tool parsing

## 验证方法

运行测试：
```bash
python tests/tool_parsers/test_glm47_streaming_chunks.py input.txt --validate
```

预期输出：
- 正确解析所有 chunks
- Parser 正确处理增量 chunk
- 最终 JSON 有效
- Parser 状态正确维护

## 文件清单

1. `tests/tool_parsers/test_glm47_streaming_chunks.py` - 主测试脚本（265 行）
2. `tests/tool_parsers/README_streaming_test.md` - 使用文档

## 实现状态

✅ 所有计划功能已实现
✅ 代码结构清晰
✅ 文档完整
✅ 符合 vLLM 代码规范
