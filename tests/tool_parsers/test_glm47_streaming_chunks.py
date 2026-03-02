# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
测试脚本：GLM-4.7 Tool Parser 流式处理测试

验证 GLM-4.7 tool parser 的流式处理能力，支持 tool call 参数分散在多个 chunk 中。
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

# 添加 vllm 到 Python 路径
script_dir = Path(__file__).resolve().parent
vllm_root = script_dir.parent.parent
sys.path.insert(0, str(vllm_root))

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tokenizers import get_tokenizer
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser


@dataclass
class ChunkData:
    """表示单个流式 chunk 的数据"""
    chunk_id: int
    raw_text: str


class StreamingSimulator:
    """模拟流式处理，调用真实的 parser 方法"""

    def __init__(self, model_name: str = "zai-org/GLM-4.5"):
        self.tokenizer = get_tokenizer(tokenizer_name=model_name)
        self.parser = Glm47MoeModelToolParser(self.tokenizer)
        self.request = self._create_mock_request()
        self.accumulated_text = ""

    def _create_mock_request(self) -> ChatCompletionRequest:
        """创建 mock request 以启用 tool parsing"""
        request = Mock(spec=ChatCompletionRequest)
        request.tools = [
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                    },
                ),
            ),
        ]
        request.tool_choice = "auto"
        return request

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


def parse_input_file(filepath: str) -> list[ChunkData]:
    """解析 input.txt，提取每个 chunk 的原始文本"""
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配每个 chunk 块
    chunk_pattern = r'\[Chunk #(\d+)\](.*?)(?=\[Chunk #|\Z)'
    for match in re.finditer(chunk_pattern, content, re.DOTALL):
        chunk_num = int(match.group(1))
        chunk_text = match.group(2)

        # 提取原始文本 - 支持多种格式
        raw_text = ""

        # 格式1: Content: 'text'
        if content_match := re.search(r"Content:\s*'([^']*)'", chunk_text):
            raw_text = content_match.group(1)
        # 格式2: Content: "text"
        elif content_match := re.search(r'Content:\s*"([^"]*)"', chunk_text):
            raw_text = content_match.group(1)
        # 格式3: Raw Text: 'text'
        elif raw_match := re.search(r"Raw Text:\s*'([^']*)'", chunk_text):
            raw_text = raw_match.group(1)
        # 格式4: Tool Calls (需要重构为 GLM 原始格式)
        elif "Tool Calls:" in chunk_text:
            # 提取 tool call 信息
            func_name_match = re.search(r"Function Name:\s*(\w+)", chunk_text)
            args_match = re.search(r"Arguments Delta:\s*'([^']*)'", chunk_text)

            if func_name_match and args_match:
                func_name = func_name_match.group(1)
                args_json = args_match.group(1)

                # 重构为 GLM 原始格式
                try:
                    args_dict = json.loads(args_json)
                    raw_text = f"<tool_call>{func_name}\n"
                    for key, value in args_dict.items():
                        raw_text += f"<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n"
                    raw_text += "</tool_call>"
                except json.JSONDecodeError:
                    pass

        if raw_text:
            chunks.append(ChunkData(chunk_id=chunk_num, raw_text=raw_text))

    return chunks


def display_result(chunk, result, simulator):
    """显示输入输出和 parser 状态"""
    print(f"\n{'='*60}")
    print(f"Chunk #{chunk.chunk_id}")
    print(f"{'='*60}")

    print(f"\n输入: {repr(chunk.raw_text)}")

    print(f"\n输出:")
    if result is None:
        print("  None (buffering)")
    elif hasattr(result, 'content') and result.content:
        print(f"  Content: {repr(result.content)}")
    elif hasattr(result, 'tool_calls') and result.tool_calls:
        for tc in result.tool_calls:
            print(f"  Tool Call #{tc.index}:")
            if hasattr(tc, 'function'):
                func = tc.function
                if isinstance(func, dict):
                    print(f"    Name: {func.get('name', 'N/A')}")
                    print(f"    Arguments: {func.get('arguments', 'N/A')}")
                else:
                    print(f"    Name: {getattr(func, 'name', 'N/A')}")
                    print(f"    Arguments: {getattr(func, 'arguments', 'N/A')}")

    print(f"\nParser 状态:")
    buffer_preview = simulator.parser._buffer[:50] if simulator.parser._buffer else ""
    print(f"  Buffer: {repr(buffer_preview)}")
    print(f"  In tool call: {simulator.parser._in_tool_call}")
    print(f"  Current tool ID: {simulator.parser.current_tool_id}")
    print(f"  Streamed args: {simulator.parser.streamed_args_for_tool}")


def validate_final_state(simulator):
    """验证最终状态"""
    print(f"\n{'='*60}")
    print("最终验证")
    print(f"{'='*60}")

    print(f"\n累积文本: {repr(simulator.accumulated_text[:200])}")
    print(f"Buffer 剩余: {repr(simulator.parser._buffer)}")
    print(f"Tool calls 数量: {len(simulator.parser.prev_tool_call_arr)}")

    # 验证 JSON 有效性
    for i, args_str in enumerate(simulator.parser.streamed_args_for_tool):
        if args_str:
            try:
                json.loads(args_str)
                print(f"✓ Tool #{i} arguments JSON 有效: {args_str}")
            except json.JSONDecodeError as e:
                print(f"✗ Tool #{i} arguments JSON 无效: {e}")
                print(f"  Content: {repr(args_str)}")


def main():
    parser = argparse.ArgumentParser(
        description="测试 GLM-4.7 tool parser 流式处理"
    )
    parser.add_argument('input_file', help='输入文件路径 (例如: input.txt)')
    parser.add_argument(
        '--model',
        default='zai-org/GLM-4.5',
        help='模型名称 (默认: zai-org/GLM-4.5)'
    )
    parser.add_argument(
        '--start-chunk',
        type=int,
        default=1,
        help='起始 chunk 编号 (默认: 1)'
    )
    parser.add_argument(
        '--end-chunk',
        type=int,
        help='结束 chunk 编号 (可选)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='验证最终 JSON 有效性'
    )
    args = parser.parse_args()

    # 解析输入
    print(f"正在加载输入文件: {args.input_file}")
    chunks = parse_input_file(args.input_file)
    print(f"✓ 加载了 {len(chunks)} 个 chunks")

    # 初始化
    print(f"正在初始化 parser (model: {args.model})...")
    simulator = StreamingSimulator(model_name=args.model)
    simulator.reset_state()
    print("✓ Parser 初始化完成")

    # 处理每个 chunk
    processed_count = 0
    for chunk in chunks:
        if chunk.chunk_id < args.start_chunk:
            continue
        if args.end_chunk and chunk.chunk_id > args.end_chunk:
            break

        result = simulator.process_chunk(chunk)
        display_result(chunk, result, simulator)
        processed_count += 1

    print(f"\n处理了 {processed_count} 个 chunks")

    # 最终验证
    if args.validate:
        validate_final_state(simulator)


if __name__ == "__main__":
    main()
