# vLLM-Mooncake PD (Prefill-Decode) 分离技术总结

## 目录

1. [PD 分离基本概念](#pd-分离基本概念)
2. [两种模式对比](#两种模式对比)
3. [KV Cache 存储位置](#kv-cache-存储位置)
4. [CPU 内存开销分析](#cpu-内存开销分析)
5. [Mooncake Store 介绍](#mooncake-store-介绍)
6. [部署网络要求](#部署网络要求)

---

## PD 分离基本概念

### 什么是 PD 分离？

PD (Prefill-Decode) 分离是将大语言模型推理过程拆分为两个独立阶段：

- **P 节点 (Prefill Node)**: 负责处理 prompt，生成 KV Cache
- **D 节点 (Decode Node)**: 消费 KV Cache，生成输出 token

### 为什么需要 PD 分离？

1. **优化 TTFT (Time To First Token) 和 TPOT (Time Per Output Token)**
   - 可以灵活调整 P 和 D 节点的并行策略（dp、tp、ep）
   - 更好的系统性能调优能力

2. **优化 TPOT**
   - 避免 prefill 任务在解码过程中插入造成的延迟
   - 通过分块 prefill 任务，提供更可靠的 TPOT 控制

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│                     Global Proxy                        │
│                  (负载均衡 + 路由)                       │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
    ┌─────────┐                 ┌─────────┐
    │ P 节点   │  KV Cache  ←──  │ D 节点   │
    │(Prefill) │  Transfer  ──→  │(Decode) │
    └─────────┘                 └─────────┘
    (生成 KV)                  (消费 KV)
```

---

## 两种模式对比

### Pull 模式 (MooncakeConnector)

**核心机制**: D 节点主动拉取 P 节点的 KV Cache

#### 工作流程

1. 请求发送到 Proxy 的 `_handle_completions` 端点
2. Proxy 调用 `select_prefiller` 选择 P 节点，转发请求
3. P 节点完成 prefill 后，延迟释放 KV Cache
4. Proxy 调用 `select_decoder` 选择 D 节点，转发请求
5. D 节点预分配 KV Cache，调用 `kv_connector_no_forward` 拉取远程 KV Cache
6. D 节点通知 P 节点释放 KV Cache，开始解码

#### 配置参数

**P 节点**:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8010 \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_producer"
  }'
```

**D 节点**:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8020 \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer"
  }'
```

**Proxy**:
```bash
python examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
  --prefill http://192.168.0.2:8010 \
  --decode http://192.168.0.3:8020
```

#### 关键特性

- 不需要 metaserver
- D 节点发起 KV Cache 拉取
- 简单直接的点对点通信

---

### Push 模式 (MooncakeLayerwiseConnector)

**核心机制**: P 节点按层推送到 D 节点

#### 工作流程

1. 请求发送到 Proxy 的 `_handle_completions` 端点
2. Proxy 调用 `select_decoder` 选择 D 节点，转发请求，设置 metaserver 端点
3. D 节点标记请求为 `WAITING_FOR_REMOTE_KVS`，预分配 KV Cache
4. D 节点调用 `kv_connector_no_forward` 向 metaserver 发送请求，等待 KV Cache 传输
5. Proxy 的 metaserver 端点接收请求，调用 `select_prefiller` 选择 P 节点
6. P 节点逐层推送 KV Cache，完成后释放请求并通知 D 节点
7. D 节点开始解码并返回结果

#### 配置参数

**D 节点**:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8020 \
  --kv-transfer-config '{
    "kv_connector": "MooncakeLayerwiseConnector",
    "kv_role": "kv_consumer",
    "metaserver": "http://proxy-host:8000/metaserver",
    "pd_head_ratio": 2
  }'
```

**P 节点**:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8010 \
  --kv-transfer-config '{
    "kv_connector": "MooncakeLayerwiseConnector",
    "kv_role": "kv_producer"
  }'
```

#### 关键特性

- 需要 metaserver 协调 P 节点选择
- P 节点逐层推送 KV Cache
- 支持 `pd_head_ratio`，允许 P 节点 TP > D 节点 TP

---

### 两种模式对比表

| 特性 | Pull 模式 | Push 模式 |
|------|----------|----------|
| **传输方向** | D 节点拉取 | P 节点推送 |
| **Metaserver** | 不需要 | 必须 |
| **pd_head_ratio 支持** | 不支持 | 支持 |
| **CPU 额外内存** | 仅元数据缓存 (8-16MB) | 元数据缓存 + Resharding Buffer (pd_head_ratio > 1 时) |
| **适用场景** | 对称 TP 配置 | 非对称 TP 配置 (P_tp > D_tp) |
| **传输方式** | 一次性传输 | 逐层传输 |
| **复杂度** | 低 | 中 |

---

## KV Cache 存储位置

### 两种模式的共同点

1. **KV Cache 生成**: 在 P 节点上
2. **KV Cache 传输**: 通过 RDMA 从 P 节点传到 D 节点
3. **KV Cache 消费**: 在 D 节点上解码时使用

### KV Cache 生命周期

```
┌───────────────────────────────────────────────────────────────┐
│                        P 节点                                │
│  1. 处理 prompt 生成 KV Cache                                 │
│  2. 暂存 KV Cache (Pull模式: 延迟释放 | Push模式: 逐层推送)  │
│  3. 传输完成后释放 KV Cache                                   │
└───────────────────────────────────────────────────────────────┘
                           │ RDMA 传输
                           ▼
┌───────────────────────────────────────────────────────────────┐
│                        D 节点                                │
│  1. 预分配 KV Cache 空间 (WAITING_FOR_REMOTE_KVS 状态)       │
│  2. 接收并填充 KV Cache                                       │
│  3. 使用 KV Cache 进行解码                                    │
│  4. 解码完成后释放 KV Cache                                   │
└───────────────────────────────────────────────────────────────┘
```

### 关键代码位置

**Scheduler 中的等待状态** (`vllm/v1/core/sched/scheduler.py:724-772`):
```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,  # 预分配但暂不填充数据
)

if load_kv_async:
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
```

---

## CPU 内存开销分析

### 1. 元数据缓存 (所有模式都有)

**大小**: 8-16MB

**用途**: 缓存 KV Cache 的元数据信息

**实现**: SizedDict，最多缓存 16000 个条目

**示例**:
```python
# mooncake_layerwise_connector.py
self.metaserver_cache = SizedDict(maxsize=16000)  # 元数据缓存
```

---

### 2. Resharding Buffer (仅 Push 模式且 pd_head_ratio > 1 时需要)

**大小**: 2 × KV_cache_size

**用途**: 处理 P 节点 TP > D 节点 TP 时，需要对 KV Cache 进行重排序

**触发条件**: `pd_head_ratio > 1` (即 P 节点 TP 数量 > D 节点 TP 数量)

#### 关键代码 (`vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:879-904`):

```python
def create_kv_buffer(self, first_kv_cache):
    if self.pd_head_ratio > 1:
        alignment = 2 * 1024 * 1024  # 2MB 对齐
        self.k_buffer = torch.zeros(
            first_kv_cache.numel() + alignment,
            dtype=first_kv_cache.dtype,
            device=first_kv_cache.device
        )
        self.v_buffer = torch.zeros(
            first_kv_cache.numel() + alignment,
            dtype=first_kv_cache.dtype,
            device=first_kv_cache.device
        )
```

#### 内存计算示例

假设:
- 模型: Llama-2-7B
- KV Cache 大小: 10GB (per request)
- pd_head_ratio: 2

额外 CPU 内存需求:
- 元数据缓存: 16MB
- Resharding Buffer: 2 × 10GB = 20GB

---

### 内存开销总结表

| 场景 | Pull 模式 | Push 模式 (pd_head_ratio=1) | Push 模式 (pd_head_ratio>1) |
|------|----------|------------------------------|------------------------------|
| **元数据缓存** | 8-16MB | 8-16MB | 8-16MB |
| **Resharding Buffer** | 0 | 0 | 2 × KV_cache_size |
| **总计** | 8-16MB | 8-16MB | 2 × KV_cache_size + 8-16MB |

---

## Mooncake Store 介绍

### 什么是 Mooncake Store？

Mooncake Store 是一个独立的分布式 KV Cache 存储引擎，用于 LLM 推理场景。

**核心特性**:

1. **持久化**: 将 KV Cache 保存到持久化存储
2. **多副本**: 支持多副本冗余存储
3. **零拷贝传输**: 使用 RDMA 进行零拷贝数据传输
4. **分布式**: 支持多节点分布式部署

---

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Mooncake Store                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Master Node  │    │ Master Node  │    │ Master Node  │  │
│  │ (元数据管理) │    │ (元数据管理) │    │ (元数据管理) │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │           │
│         └───────────────────┴───────────────────┘           │
│                             │                               │
│         ┌───────────────────┼───────────────────┐           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ Client Node  │   │ Client Node  │   │ Client Node  │    │
│  │ (数据存储)   │   │ (数据存储)   │   │ (数据存储)   │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

### 核心组件

#### 1. Master Service (元数据服务)

- 管理 KV Cache 的元数据
- 提供 Get/Put/Remove 操作接口
- 支持多节点冗余

#### 2. Client (存储节点)

- 存储实际的 KV Cache 数据
- 提供数据访问接口
- 支持多副本

---

### 支持的操作

```python
# 获取 KV Cache
kv_cache = client.get(key)

# 存储 KV Cache
client.put(key, kv_cache)

# 删除 KV Cache
client.remove(key)
```

---

### 与 PD 分离的关系

**Mooncake Store 是可选的**:

- **不使用 Mooncake Store**: P 节点和 D 节点直接通过 RDMA 传输 KV Cache，不持久化
- **使用 Mooncake Store**:
  - P 节点可以将 KV Cache 存储到 Mooncake Store
  - D 节点从 Mooncake Store 拉取 KV Cache
  - 支持跨请求 KV Cache 复用

---

### 配置示例

**使用 etcd 作为元数据服务**:
```python
# mooncake-store-config.yaml
metadata_servers:
  - "http://etcd1:2379"
  - "http://etcd2:2379"
  - "http://etcd3:2379"

storage_backend: "mooncake"
replica_factor: 3
```

**使用 HTTP 作为元数据服务**:
```python
# mooncake-store-config.yaml
metadata_servers:
  - "http://metaserver1:8080"
  - "http://metaserver2:8080"

storage_backend: "mooncake"
```

---

## 部署网络要求

### 基本要求

✅ **不需要交换机接入**
✅ **基本网络连通性即可**

---

### 网络类型支持

#### 1. RDMA (推荐)

**优点**:
- 零拷贝传输
- 低延迟
- 高吞吐量

**要求**:
- 支持 RDMA 的网卡
- RDMA 网络环境 (RoCE 或 InfiniBand)

#### 2. TCP (默认 fallback)

**优点**:
- 无特殊硬件要求
- 部署简单

**缺点**:
- 需要数据拷贝
- 延迟相对较高

---

### 部署前置条件

1. **安装 Mooncake Transfer Engine**
   ```bash
   uv pip install mooncake-transfer-engine
   ```

2. **配置元数据服务** (使用 Mooncake Store 时)
   - etcd 集群 (推荐)
   - 或 HTTP 元数据服务器

3. **网络配置**
   - 确保各节点间网络互通
   - 配置正确的端口号 (默认 14579)
   - 防火墙规则允许相关端口

---

### 环境变量

```bash
# Mooncake bootstrap 端口 (仅 P 节点需要)
export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998

# 请求中断超时 (秒)
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480

# Python hash seed (多节点一致性)
export PYTHONHASHSEED=0
```

---

### 端口检查

建议在启动前检查端口冲突：

```bash
# 检查 RDMA 端口
netstat -tuln | grep 14579

# 检查 HTTP 端口
netstat -tuln | grep 8010
netstat -tuln | grep 8020
```

---

## 限制与注意事项

### 1. 硬件限制

- ❌ 不支持异构 P 和 D 节点 (如 P 节点用 A2，D 节点用 A3)
- ✅ 支持 A2 和 A3 硬件配置
- ✅ 支持对称 TP 配置
- ✅ 支持非对称 TP 配置 (P_tp > D_tp 且 P_tp % D_tp = 0)

### 2. 模型支持

- ✅ MLA (Multi-Head Latent Attention)
- ✅ GQA (Grouped Query Attention)

### 3. PD 比例限制

非对称 TP 配置下，仅支持以下情况:
- P 节点 TP > D 节点 TP
- P 节点 TP 数量是 D 节点 TP 的整数倍

---

## 参考文档

- [Mooncake 官方文档](https://kvcache-ai.github.io/Mooncake/)
- [vLLM Mooncake Connector 使用指南](../vllm/docs/features/mooncake_connector_usage.md)
- [Disaggregated-prefill 设计文档](../vllm-ascend/docs/source/developer_guide/feature_guide/disaggregated_prefill.md)
- [KV Pool 部署指南](../vllm-ascend/docs/source/user_guide/feature_guide/kv_pool.md)
- [Mooncake Store 设计文档](../Mooncake/docs/source/design/mooncake-store.md)

---

## 常见问题 (FAQ)

### Q1: Pull 模式和 Push 模式如何选择？

**Pull 模式**:
- 适用于对称 TP 配置
- 简单部署，不需要 metaserver
- 推荐刚开始使用 PD 分离的场景

**Push 模式**:
- 适用于非对称 TP 配置 (P_tp > D_tp)
- 需要额外的 CPU 内存用于 resharding
- 推荐需要灵活调整 P 和 D 节点规模的场景

---

### Q2: 是否必须使用 Mooncake Store？

**不是必须的**:
- 不使用 Mooncake Store: P 节点和 D 节点直接传输 KV Cache
- 使用 Mooncake Store: 支持持久化和跨请求复用

---

### Q3: 需要什么样的网络环境？

- 基本要求: 节点间网络互通即可
- 推荐配置: RDMA 网络 (RoCE 或 InfiniBand)
- 无需交换机接入

---

### Q4: 如何估算 CPU 内存需求？

Pull 模式:
```
额外内存 = 8-16MB (元数据缓存)
```

Push 模式 (pd_head_ratio > 1):
```
额外内存 = 2 × KV_cache_size + 8-16MB
```

---

### Q5: PD 分离能带来多少性能提升？

取决于具体场景:

1. **TTFT 优化**: 通过独立调整 P 节点并行策略
2. **TPOT 优化**: 避免 prefill 任务干扰解码
3. **资源利用率**: P 和 D 节点可以独立扩展

建议根据实际负载测试调整 P 和 D 节点的比例。

---

*文档版本: v1.0*
*最后更新: 2026-02-18*
