# vLLM Scheduler 和显存分析

> 基于 vLLM 代码库的详细分析，涵盖 scheduler 配置、显存峰值计算、KV Cache 分配等核心机制。

---

## 一、Scheduler 参数详解

### 核心并发与显存控制参数

#### 1. `max_num_batched_tokens` (默认: 2048)

**用途**：单次迭代中处理的 token 总数上限

**显存影响**：
- ✅ **直接影响激活显存峰值**（profiling 时使用这个数量执行 dummy forward pass）
- ✅ **决定 LoRA 静态缓冲区大小**（scheduler.py:191-193）
- ❌ **不直接决定 KV Cache 的 block 数量**

**并发影响**：
- 控制批次大小的硬性约束
- 决定了 LoRA 静态缓冲区的大小（影响 torch.compile 的图捕获）

#### 2. `max_num_seqs` (默认: 128)

**用途**：单次迭代中处理的序列（请求）数量上限

**显存影响**：
- ✅ **间接影响**（通过影响 metadata 和采样器 buffer）
- 影响 LoRA 激活时的额外显存

**并发影响**：
- 控制并发请求的上限
- 必须满足 `max_num_batched_tokens >= max_num_seqs`

**约束关系**：
```
max_num_batched_tokens >= max_num_seqs
max_num_batched_tokens >= max_model_len (当 enable_chunked_prefill=False 时)
```

---

### 分块预填充相关参数

#### 3. `enable_chunked_prefill` (默认: True)

**用途**：是否启用分块预填充，将长 prompt 分成多个块逐步处理

**显存影响**：
- ✅ **显著降低峰值显存**：长 prompt 不需要一次性加载全部 KV Cache
- 显存使用更加平滑，避免大请求导致的显存尖峰

**并发影响**：
- 允许小请求和大请求混合处理，提高整体吞吐量
- 关闭时需要 `max_num_batched_tokens >= max_model_len`

#### 4. `max_num_partial_prefills` (默认: 1)

**用途**：分块预填充时，可以并发处理的部分预填充序列数量

**显存影响**：
- ✅ **直接影响**：值越大，显存峰值越高（同时处理多个 prompt 的 chunk）

**并发影响**：
- 提高预填充阶段的并发度
- 必须 `enable_chunked_prefill=True` 才能设置 >1

#### 5. `max_long_partial_prefills` (默认: 1)

**用途**：超过 `long_prefill_token_threshold` 的长 prompt 的最大并发数

**显存影响**：
- ✅ **控制长请求对显存的抢占**：值越小，长 prompt 越少并发，显存压力越小

**并发影响**：
- 允许短 prompt "插队"到长 prompt 前面，改善短请求延迟
- 必须满足 `max_long_partial_prefills <= max_num_partial_prefills`

#### 6. `long_prefill_token_threshold` (默认: 0，实际为 max_model_len * 0.04)

**用途**：判定一个 prompt 是否为"长 prompt"的 token 阈值

**显存影响**：
- ✅ **间接影响**：通过分类控制显存分配策略

**并发影响**：
- 决定哪些请求受到 `max_long_partial_prefills` 限制
- 默认设置为最大模型长度的 4%（scheduler.py:231）

---

### KV Cache 管理参数

#### 7. `disable_hybrid_kv_cache_manager` (默认: None)

**用途**：是否禁用混合 KV Cache 管理器

**显存影响**：
- ✅ **False**（启用混合模式）：为不同类型的 attention 层（如 full attention 和 sliding window）分配不同大小的 KV Cache，**节省显存**
- ✅ **True**（禁用）：所有层分配相同大小，可能**浪费显存**

**并发影响**：
- 无直接影响
- 混合模式下可能支持更多并发请求（显存利用率更高）

#### 8. `kv_cache_memory_bytes` (默认: None)

**用途**：手动指定 KV Cache 显存大小（字节）

**显存影响**：
- ✅ **直接决定** available_memory 大小
- ⚠️ 设置后**不尊重** `gpu_memory_utilization` 配置

**并发影响**：
- 无直接影响
- 提供对 KV Cache 大小的精细控制

---

### 调度策略参数

#### 9. `policy` (默认: "fcfs")

**用途**：调度策略
- `"fcfs"`：先进先出
- `"priority"`：基于优先级调度（数值越小优先级越高）

**显存影响**：无直接影响

**并发影响**：
- `priority` 模式下高优先级请求可能抢占资源，影响低优先级请求延迟

#### 10. `async_scheduling` (默认: None)

**用途**：是否启用异步调度

**显存影响**：无直接影响

**并发影响**：
- ✅ 启用后减少 GPU 利用率空隙，提升延迟和吞吐量（gpu_worker.py:132-134）

#### 11. `stream_interval` (默认: 1)

**用途**：流式输出时的 token 间隔/缓冲大小

**显存影响**：
- ✅ 较大值减少 host 开销，可能略微增加吞吐量
- 无直接影响 GPU 显存

**并发影响**：主要影响延迟 vs 吞吐量的权衡

---

### 多模态相关参数

#### 12. `disable_chunked_mm_input` (默认: False)

**用途**：多模态模型时，是否禁用混合输入的分块

**显存影响**：
- 分块时显存更平滑，不分块时峰值更高

**并发影响**：
- 不分块可能阻塞其他请求，降低吞吐量

---

### 其他参数

#### 13. `runner_type` (默认: "generate")

**用途**：运行器类型（"generate"/"pooling"/"draft"）

#### 14. `scheduler_cls` (默认: None)

**用途**：自定义调度器类路径

#### 15. `max_num_encoder_input_tokens` (V1 使用，等于 max_num_batched_tokens)

**用途**：多模态编码器的计算预算

---

## 二、显存峰值计算

### 计算流程

```python
# gpu_worker.py:333-411
@torch.inference_mode()
def determine_available_memory(self) -> int:
    """Profiles the peak memory usage of the model..."""

    # Step 1: 显存 profiling
    with memory_profiling(
        self.init_snapshot,
        weights_memory=int(self.model_runner.model_memory_usage),
    ) as profile_result:
        self.model_runner.profile_run()  # 执行 dummy forward pass

    # Step 2: 计算可用显存
    self.available_kv_cache_memory_bytes = (
        self.requested_memory - profile_result.non_kv_cache_memory
    )
```

### Profiling 结果组成

```python
# gpu_worker.py:367-390
self.non_torch_memory = profile_result.non_torch_increase
self.peak_activation_memory = profile_result.torch_peak_increase
free_gpu_memory = profile_result.after_profile.free_memory

# 最终可用显存
self.available_kv_cache_memory_bytes = (
    self.requested_memory - profile_result.non_kv_cache_memory
)
```

### non_kv_cache_memory 包含

| 组件 | 说明 |
|--------|------|
| `weights_memory` | 模型权重显存 |
| `torch_peak_increase` | **峰值激活显存**（受 max_num_batched_tokens 影响） |
| `non_torch_increase` | 非 PyTorch 显存（NCCL 缓冲区、attention backend buffers） |
| `cuda_graph_memory_bytes` | CUDA Graph 显存（warmup 后） |

---

## 三、KV Cache Block 数量分配逻辑

### 核心公式

```python
# kv_cache_utils.py:830-845
def get_num_blocks(
    vllm_config: VllmConfig,
    num_layers: int,
    available_memory: int,  # 可用于 KV Cache 的显存（字节）
    page_size: int,  # 每个 block 的显存大小（字节）
) -> int:
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    return num_blocks
```

**公式**：
```
num_gpu_blocks = floor(available_memory_bytes / (page_size_bytes × num_layers))
```

### Page Size 计算（FullAttentionSpec）

```python
# kv_cache_interface.py:181-186
@property
def real_page_size_bytes(self) -> int:
    return (
        2                               # K 和 V 各占一份
        * self.block_size               # 每个块 token 数
        * self.num_kv_heads             # 注意力头数
        * self.head_size                 # 每个头维度
        * get_dtype_size(self.dtype)      # 数据类型大小 (fp16=2, bf16=2, fp8=1)
    )
```

**Page Size 公式**：
```
page_size_bytes = 2 × block_size × num_kv_heads × head_size × dtype_size
```

---

## 四、available_memory 计算流程

### 1. 初始请求显存计算

```python
# worker/utils.py:92-112
def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """Calculate the amount of memory required by vLLM."""
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )
    return requested_memory
```

**公式**：
```
requested_memory = ceil(gpu_total_memory × gpu_memory_utilization)
```

例如，24GB GPU，`gpu_memory_utilization=0.9`：
```
requested_memory = ceil(24 GB × 0.9) = 21.6 GB
```

---

### 2. 显存 Profiling 后的计算

```python
# gpu_worker.py:333-411
@torch.inference_mode()
def determine_available_memory(self) -> int:
    with memory_profiling(...) as profile_result:
        self.model_runner.profile_run()  # 执行 dummy forward pass

    # 最终可用显存
    self.available_kv_cache_memory_bytes = (
        self.requested_memory - profile_result.non_kv_cache_memory
    )
```

**核心公式**：
```
available_memory = requested_memory - non_kv_cache_memory
```

其中：
- `requested_memory` = `ceil(gpu_total × gpu_memory_utilization)`
- `non_kv_cache_memory` = `weights_memory + peak_activation_memory + non_torch_memory + cuda_graph_memory`

---

### 3. 完整计算流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. GPU Total Memory                                    │
│    例如: 24 GB                                          │
└──────────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. Requested Memory                                    │
│    = ceil(gpu_total × gpu_memory_utilization)              │
│    例如: ceil(24 × 0.9) = 21.6 GB               │
└──────────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. Memory Profiling                                    │
│    - 运行 dummy forward pass (max_num_batched_tokens)      │
│    - 测量峰值显存                                       │
└──────────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
         ┌──────────────────────┴────────────────────┐
         │  profiling 结果                │
         │                              │
         │  - weights_memory              │
         │  - peak_activation_memory      │
         │  - non_torch_memory            │
         │  - cuda_graph_memory           │
         │                              │
         ▼                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. Non KV Cache Memory                               │
│    = weights + peak_activation + non_torch + cuda_graph   │
└──────────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. Available Memory for KV Cache                      │
│    = requested_memory - non_kv_cache_memory               │
│    例如: 21.6 - (weights + activation + ...)         │
└─────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
              用于计算 num_gpu_blocks
```

---

## 五、参数对显存和并发的影响总结

### max_num_batched_tokens 的影响

| 影响维度 | 说明 |
|----------|------|
| ✅ KV Cache Block 计算 | ❌ **不参与** num_gpu_blocks 公式 |
| ✅ LoRA 静态缓冲区 | ✅ **决定**缓冲区大小 |
| ✅ 激活显存峰值 | ✅ **直接影响** profiling 时的测量结果 |
| ✅ 间接影响 available_memory | ✅ 通过影响 peak_activation_memory 间接影响 |

### 关键参数影响优先级

| 参数 | 显存影响 | 并发影响 | 调整方式 |
|------|----------|----------|----------|
| `gpu_memory_utilization` | **最大影响** | 影响 requested_memory | `--gpu-memory-utilization` |
| `max_num_batched_tokens` | **高影响** | 调度约束 | `--max-num-batched-tokens` |
| `max_num_seqs` | 中等影响 | 并发约束 | `--max-num-seqs` |
| `tensor_parallel_size` | **高影响** | 切分权重和激活 | `--tensor-parallel-size` |
| `pipeline_parallel_size` | **高影响** | 每卡只存部分层 | `--pipeline-parallel-size` |
| `data_parallel_size` | 不降低单卡显存 | 权重复制 | `--data-parallel-size` |
| `block_size` | 影响 page_size | 影响 block 大小 | 硬件决定 |
| `cache_dtype` (fp8/bf16) | 中等影响 | KV Cache 存储 | `--kv-cache-dtype` |
| `disable_hybrid_kv_cache_manager` | 中等影响 | 浪费/节省显存 | `--disable-hybrid-kv-cache-manager` |

---

## 六、计算示例

### 示例：Llama-2-7B 在 24GB GPU 上

| 参数 | 值 |
|------|-----|
| GPU 总显存 | 24 GB |
| `gpu_memory_utilization` | 0.9 |
| `max_num_batched_tokens` | 2048 |
| `num_kv_heads` | 32 |
| `head_size` | 128 |
| `dtype` | fp16 (2 bytes) |
| `num_layers` | 32 |
| `block_size` | 16 |

### 计算过程

```
# 1. Requested Memory
requested_memory = ceil(24 × 0.9) = 21.6 GB

# 2. Page Size
page_size_bytes = 2 × 16 × 32 × 128 × 2 = 262,144 bytes ≈ 256 KB

# 3. KV Cache Block 数量
num_gpu_blocks = floor(21.6 GB / (256 KB × 32))
              = floor(21.6 GB / 8.192 MB)
              = floor(21.6 GB / 8 MB)
              ≈ 2,637 blocks

# 4. 最大并发（粗略估算）
max_tokens = num_gpu_blocks × block_size
          = 2,637 × 16
          = 42,192 tokens
```

---

## 七、关键结论

1. **`max_num_batched_tokens` 对显存的双重影响**：
   - ✅ **直接影响激活显存峰值**（通过 profiling 测量）
   - ✅ **决定 LoRA 静态缓冲区大小**
   - ❌ **不直接参与** KV Cache block 数量计算公式
   - ✅ **间接影响**：通过影响 `available_memory` 最终影响 num_gpu_blocks

2. **KV Cache block 数量由**：
   - `available_memory`（由 `gpu_memory_utilization` 决定）
   - `page_size`（由模型架构和数据类型决定）
   - `num_layers`（模型配置）
   共同决定，计算公式：`floor(available_memory / (page_size × num_layers))`

3. **显存分配优先级**：
   ```
   gpu_memory_utilization > max_num_batched_tokens > 其他参数
   ```
   - `gpu_memory_utilization` 是控制 KV Cache 大小的最核心参数
   - `kv_cache_memory_bytes` 可以手动覆盖（不尊重 `gpu_memory_utilization`）

---

## 八、代码文件参考

| 文件 | 功能 |
|------|------|
| `vllm/config/scheduler.py` | Scheduler 配置定义 |
| `vllm/v1/worker/gpu_worker.py` | 显存 profiling 和 available_memory 计算 |
| `vllm/v1/core/kv_cache_utils.py` | KV Cache block 数量计算逻辑 |
| `vllm/v1/kv_cache_interface.py` | KV Cache spec 定义和 page_size 计算 |
| `vllm/v1/engine/core.py` | KV Cache 配置初始化 |
| `vllm/worker/utils.py` | requested_memory 计算 |
| `vllm/config/cache.py` | CacheConfig 定义 |
| `vllm/config/lora.py` | LoRA 配置 |

---

*分析日期：2026-02-16*
*分析基于 vLLM 代码库*
