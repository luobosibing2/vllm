# disagg_proxy_demo.py 机制解析

## 概述

`disagg_proxy_demo.py` 是一个用于演示 **XpYd（分离式预填充和译码）** 架构的 HTTP 代理服务器。它将大语言模型推理拆分为两个独立的阶段：
- **Prefill 阶段**：处理输入 prompt 的并行计算，生成 KV 缓存
- **Decode 阶段**：逐 token 生成输出，复用 KV 缓存

> **重要说明**：这个代理**不负责 KV-cache 的存储和传输**。它只是一个 HTTP 请求路由器，底层的 KV-cache 传输由 vLLM 的 `MooncakeConnector` 和 Mooncake Transfer Engine 完成。

## 架构图

```
客户端请求 → Proxy (disagg_proxy_demo.py) → Prefill实例 → Decode实例 → 响应

内部 KV-cache 传输：
Prefill GPU 内存 ───[Mooncake Transfer Engine/RDMA]──→ Decode GPU 内存
```

## 核心组件

### 1. SchedulingPolicy（调度策略）

抽象基类，支持自定义调度策略。当前实现是 `RoundRobinSchedulingPolicy`（轮询调度）：

```python
class RoundRobinSchedulingPolicy(SchedulingPolicy):
    def schedule(self, cycler: itertools.cycle) -> str:
        return next(cycler)  # 轮询选择下一个实例
```

### 2. Proxy 类

主要功能：
- 管理多个 prefill/decode 实例
- 实现请求转发和流式响应
- 支持动态添加实例
- 自动故障移除

### 3. ProxyServer 类

FastAPI 服务器封装，提供 HTTP API 端点。

## 请求处理流程

以 `create_chat_completion` 为例：

### 阶段 1: Prefill（预填充）

```python
# 复制请求，只生成 1 个 token
kv_prepare_request = request.copy()
kv_prepare_request["max_tokens"] = 1

# 选择 prefill 实例
prefill_instance = self.schedule(self.prefill_cycler)

# 发送到 prefill 实例
async for _ in self.forward_request(
    f"http://{prefill_instance}/v1/chat/completions", kv_prepare_request
):
    continue  # 等待完成，不需要响应
```

**目的**：生成 KV 缓存并存储，供 decode 阶段使用

### 阶段 2: Decode（译码）

```python
# 选择 decode 实例
decode_instance = self.schedule(self.decode_cycler)

# 发送完整原始请求到 decode 实例
generator = self.forward_request(
    "http://" + decode_instance + "/v1/chat/completions", request
)

# 流式返回响应
response = StreamingResponse(content=generator)
return response
```

**流程**：decode 实例会从 Mooncake Transfer Engine 拉取 prefill 生成的 KV 缓存，然后进行 token 生成。

## API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/v1/completions` | POST | OpenAI 兼容的补全接口 |
| `/v1/chat/completions` | POST | OpenAI 兼容的对话补全接口 |
| `/status` | GET | 查看 prefill/decode 节点状态 |
| `/instances/add` | POST | 动态添加新实例（需 API Key 认证） |

## 实例管理

### 动态添加实例

```python
POST /instances/add
Headers: x-api-key: <ADMIN_API_KEY>
Body:
{
  "type": "prefill",  # 或 "decode"
  "instance": "host:port"
}
```

添加前会验证：
1. 实例格式是否正确（host:port）
2. 实例是否可访问（调用 `/v1/models` 端点）
3. 模型 ID 是否匹配

### 故障移除

当请求失败时，自动从轮询中移除实例：

```python
def remove_instance_endpoint(self, instance_type, instance):
    if instance_type == "decode" and instance in self.decode_instances:
        self.decode_instances.remove(instance)
        self.decode_cycler = itertools.cycle(self.decode_instances)
```

> **Bug 注意**：代码第 331 行有一个 bug，移除 prefill 实例时错误地检查了 `self.decode_instances`：
> ```python
> if instance_type == "prefill" and instance in self.decode_instances:  # 错误！
>     self.prefill_instances.remove(instance)
>     self.prefill_cycler = itertools.cycle(self.decode_instances)  # 又错了！
> ```

## 使用方法

### 启动代理

```bash
python3 examples/online_serving/disaggregated_serving/disagg_proxy_demo.py \
  --model $model_name \
  --prefill localhost:8100 localhost:8101 \
  --decode localhost:8200 localhost:8201 \
  --port 8000
```

### 环境变量

- `ADMIN_API_KEY`：用于 `/instances/add` 端点的认证
- `OPENAI_API_KEY`：转发请求时使用的认证密钥

## KV-cache 传输机制

### 代理的角色

`disagg_proxy_demo.py` **不处理 KV-cache**，它只是：
1. 发送请求到 prefill 实例（触发 KV-cache 计算）
2. 发送请求到 decode 实例（触发 KV-cache 拉取）

### 实际的 KV-cache 传输

真正的 KV-cache 传输由以下组件完成：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         disagg_proxy_demo.py                               │
│                    (只是一个简单的 HTTP 请求路由器)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP Request
                                    ▼
         ┌──────────────────────────────────────────────────────────┐
         │                    vLLM Prefill 实例                      │
         │  ┌──────────────────────────────────────────────────┐  │
         │  │  MooncakeConnector (Prefill/Producer 侧)         │  │
         │  │  1. 计算 kv-cache 并存储在 GPU 内存              │  │
         │  │  2. 调用 engine.batch_register_memory() 注册    │  │
         │  │  3. ZMQ 监听来自 Decode 的传输请求              │  │
         │  │  4. 使用 TransferEngine.batch_transfer_sync_write│  │
         │  │     直接传输到 Decode 的 GPU 内存 (RDMA/TCP)     │  │
         │  └──────────────────────────────────────────────────┘  │
         └──────────────────────────────────────────────────────────┘
                                    │
                                    │ Mooncake Transfer Engine
                                    │ (RDMA/Zero-Copy 直接内存传输)
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────────┐
         │                    vLLM Decode 实例                       │
         │  ┌──────────────────────────────────────────────────┐  │
         │  │  MooncakeConnector (Decode/Consumer 侧)        │  │
         │  │  1. 注册本地 kv-cache 内存池                     │  │
         │  │  2. 查询 bootstrap server 发现 prefill 实例       │  │
         │  │  3. 发送 MooncakeXferMetadata 请求            │  │
         │  │  4. 接收 prefill 传输过来的 kv-cache            │  │
         │  │  5. 使用接收到的 kv-cache 进行 decode            │  │
         │  └──────────────────────────────────────────────────┘  │
         └──────────────────────────────────────────────────────────┘
```

### MooncakeConnector 关键流程

#### Prefill 侧（Producer）

1. **注册内存** ([mooncake_connector.py:912](vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:912))
   ```python
   def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
       base_addr = cache.data_ptr()  # 获取 GPU 内存地址
       kv_data_ptrs.append(base_addr)
       # 注册到 TransferEngine
       ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
   ```

2. **发送数据** ([mooncake_connector.py:901](vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:901))
   ```python
   def _send_blocks(self, remote_session, src_ptrs, dst_ptrs, lengths):
       # 零拷贝传输
       ret_value = self.engine.batch_transfer_sync_write(
           remote_session, src_ptrs, dst_ptrs, lengths
       )
   ```

#### Decode 侧（Consumer）

1. **注册接收内存** ([mooncake_connector.py:949](vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:949))
   ```python
   self.kv_caches_base_addr = seen_base_addresses
   ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
   ```

2. **请求数据** ([mooncake_connector.py:1043](vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:1043))
   ```python
   async def receive_kv_from_single_worker(self, worker_addr, pull_metas):
       metadata = MooncakeXferMetadata(
           remote_hostname=self.hostname,
           remote_port=self.rpc_port,
           kv_caches_base_addr=self.kv_caches_base_addr,  # 我的接收内存地址
       )
       await sock.send(encoded_data)  # 发送传输请求
   ```

## 技术亮点

| 设计点 | 说明 |
|--------|------|
| **XpYd 分离** | Prefill 节点只需生成 1 个 token 用于 KV 缓存，无需完整生成 |
| **负载均衡** | 支持多个 prefill/decode 实例，自动轮询分配 |
| **动态扩容** | 运行时可添加新实例，无需重启 |
| **容错机制** | 实例请求失败时自动从轮询中移除 |
| **流式响应** | 保持 OpenAI API 兼容的流式输出 |
| **零拷贝传输** | 通过 Mooncake Transfer Engine 实现 GPU 间直接内存传输 |

## 适用场景

这种架构特别适合：

1. **长序列推理**：prefill 成本高，decode 较快，分离后可以独立优化
2. **高吞吐场景**：prefill 和 decode 分离优化，提高整体吞吐
3. **动态资源分配**：根据负载动态调整 prefill/decode 节点数量
4. **异构部署**：prefill 和 decode 可以使用不同硬件配置

## 已知限制

1. **临时性实现**：注释说明这个 demo 会在 PR 15343 的 PDController 支持 XpYd 后被移除
2. **Bug**：`remove_instance_endpoint` 方法中 prefill 实例移除逻辑有错误
3. **不支持 Pipeline Parallelism**：Mooncake Transfer Engine 暂不支持 pipeline parallelism

## 相关文件

- 代理实现：[vllm/examples/online_serving/disaggregated_serving/disagg_proxy_demo.py](../vllm/examples/online_serving/disaggregated_serving/disagg_proxy_demo.py)
- Mooncake Connector：[vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py](../vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py)
- KV Transfer 配置：[vllm/vllm/config/kv_transfer.py](../vllm/vllm/config/kv_transfer.py)

## 参考

- [vLLM Disaggregated Prefill 文档](https://docs.vllm.ai/en/latest/features/disagg_prefill.html)
- [Mooncake 项目](https://github.com/kvcache-ai/Mooncake)
- Mooncake Transfer Engine 支持的协议：RDMA、TCP、CXL、NVLink、EFA
