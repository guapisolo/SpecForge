# SpecForge 重新设计方案

> **状态**：草稿（已应用一轮 review）
> **最近更新**：2026-05-17

---

## 1. 背景

SpecForge 用来训练与 SGLang 服务侧对齐的投机解码草稿模型（EAGLE3、DFlash）。它在单集群 torchrun 工作流下运行良好，但积累了大量结构性债务，导致新增架构、新增训练模式、以及生产化工具难以接入。

TorchSpec（一个姊妹项目）通过 Ray + Mooncake 解耦解决了很多这类缺口，但代价是沉重的依赖和运维复杂度。本方案选择另一条路径：**保留 SpecForge 简洁的 torchrun 原生设计，但对内部结构进行重组，让新能力能够干净地组合接入。**

### 原样保留的部分

- `specforge/core/loss.py` — Triton `LogSoftmaxLoss` 与 `_compute_loss`。
- `specforge/optimizer.py` — `BF16Optimizer`（FP32 master weights + AdamW + grad clip）。
- `specforge/lr_scheduler.py` — `CosineAnnealingWarmupLR` 和 `TwoStageScheduler` 系列。
- `specforge/tracker.py` — `Tracker` ABC + `TRACKER_REGISTRY`（wandb/tensorboard/swanlab/mlflow）。
- `specforge/distributed.py` — `init_distributed`、device meshes、yunchang USP 集成。
- `specforge/core/eagle3_adapters.py` — `BackendAdapter` / `StepState` / `UspAdapter`。
- `specforge/modeling/target/sglang_backend/` 下的 SGLang target backend 代码。
- 现有的 30+ 个草稿模型 config。

### 抛弃的部分

- 把 `scripts/train_eagle3.py` 和 `scripts/train_dflash.py` 当作 god script 用（改成薄壳 shim）。
- `specforge/modeling/auto.py` 里硬编码的 `_model_mapping` / `_config_mapping` dict。
- 那个声称 `model_type: llama` 却用于 DeepSeek target 的 `configs/deepseek-v3-671b-eagle3.json`。
- argparse flags 与 per-arch JSON config 拆开的两套配置入口。
- `QwenVLOnlineEagle3Model`（VLM 应该由 target engine + 数据流水线处理，而不是再起一个单独的 model class）。

---

## 2. 当前状态差距分析

本节把设计扎根在当前代码库具体存在的问题上。两类差距：**(A) 缺失的能力** — 设计假设存在但代码里根本没有的部分；**(B) 结构性问题** — 代码存在但形状不对的部分。

### 2.1 缺失的能力


| 差距                                   | 当前代码树中的证据                                                                                                                                                                                             | 由谁修复                                                                       |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **MLA-aware EAGLE3 draft**           | 除测试数据外没有任何对 `DeepseekV3Config` / `Eagle3Deepseek`* 的引用。`configs/deepseek-v3-671b-eagle3.json` 实际上是个被错误标成 DeepSeek 的 *Llama* draft（`model_type: "llama"`、`architectures: ["LlamaForCausalLMEagle3"]`）。 | Phase 1 #2 — 从 TorchSpec port `deepseek_eagle.py`，通过 `@register_draft` 注册。 |
| **与 backbone 解耦的 DFlash**            | `DFlashDraftModel` 继承 `Qwen3PreTrainedModel`；直接使用 `Qwen3MLP`、`Qwen3DFlashAttention`、`Qwen3RMSNorm`（`specforge/modeling/draft/dflash.py:212`）。所有 DFlash config 都是 Qwen3-only。                          | Phase 5 #21 — 把 MLP/Norm/RoPE 参数化；新增 `dflash_llama.py`。                    |
| **远端 / 解耦 target**                   | `modeling/target/sglang_backend/` 只做进程内 SGLang（target 与 trainer 在同节点加载）。没有 HTTP client。671B target 跑 online 模式不可行。                                                                                    | Phase 4 #17 — `SGLangServerEngine` 走 HTTP。                                 |
| **训练中评测（正确实现）**                      | 没有 `EvalCache`、没有 `simulated_acc_len`、没有 per-position-accuracy 聚合、没有 best-checkpoint 追踪 — grep 是干净的。                                                                                                  | Phase 2 #9 — `Evaluator` + `EvalCache`。                                    |
| **Checkpoint 轮转 / best 追踪**          | `train_eagle3.py:552:def save_checkpoints(...)` 只是每 N 步写一份。没有 `max_checkpoints`、没有 `best_checkpointed_iteration.txt`、没有 `meta.json`。                                                                  | Phase 2 #8 — `CheckpointManager`。                                          |
| **真正会推进数据流的 resume**                 | 没有 `stream.seek()`。当前 `--resume` 会重新加载权重，但 dataloader 仍从 sample 0 开始 yield，悄悄在前缀上重复训练。                                                                                                                | §4.2 trainer 伪代码 + Phase 2 #5 — `HiddenStateStream.seek()`。                |
| **正确的 gradient accumulation**        | 代码库里完全找不到 `no_sync()` 调用。FSDP 在每个 micro-step 都做 all-reduce，让 `--accumulation-steps` 的意义荡然无存。                                                                                                          | Phase 2 #10 — 在非 sync 的 micro-step 调用 `model.no_sync()`。                   |
| **Draft 的 plugin registry**          | 硬编码的 `_model_mapping = {LlamaConfig: LlamaForCausalLMEagle3}`（`specforge/modeling/auto.py:35`）。代码里已经留有 TODO："should support lazy model mapping via registry"。                                         | Phase 1 #1 — `@register_draft` 装饰器。                                        |
| **带 MLA weight map 的 SGLang export** | 没有 `export/to_sglang.py`。MLA draft 权重（`q_a_proj`、`kv_a_proj_with_mqa`、`kv_b_proj`）需要显式重命名为 SGLang spec-decoder loader 期望的名字。                                                                          | Phase 3 #15 + `docs/export_weight_map_mla.md`。                             |
| **FSDP2 就绪度**                        | `apply_fsdp` 只支持 FSDP1，且 inline 写在 script 里。没有为 FSDP2 切换预留接缝。把 trainer 锁死在和 `torch.compile` 难以组合的模式上。                                                                                                 | §4.9 — 在 Phase 2 里加版本化的 `apply_fsdp` 接缝（FSDP2 实现延期到 Phase 6）。              |
| **Pydantic config + CLI**            | `specforge/args.py` 是 219 行的 argparse；架构走 JSON（`--draft-model-config`）；training flags 走 CLI。两个真值源，没有 validation。                                                                                      | Phase 3 #13/14 — 每次运行用一份经过 validation 的 YAML。                              |
| **VLM 统一**                           | VLM target 单独起 `QwenVLOnlineEagle3Model` class — 跟主抽象并列的另一套层次结构。                                                                                                                                      | Phase 4 #19 — 在 `TargetEngine` 里用 typed `MediaInputs`。                     |


### 2.2 结构性 / 工作流问题

这些代码是存在的，但形状不对：重复、缺抽象、随着每个新架构加入而复利化的维护债。


| 问题                                         | 具体证据                                                                                                                                                                                                | 由谁修复                                                                                  |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **God-script 重复**                          | `train_eagle3.py` = 1012 行 / 60 个 `add_argument`；`train_dflash.py` = 562 行 / 44 个 `add_argument`。同样的 argparse、同样的 distributed init、同样的 checkpoint save、同样的 logging — 复制粘贴了两份。两个顶层 script 共 ~1574 行。 | Phase 2 #6 — 单一 `Trainer`，用 strategy dispatch。压缩为一个约 70 行的 loop + strategy class。     |
| **modeling/ 下 target-model 重复**            | `modeling/target/eagle3_target_model.py`（873 行）和 `modeling/target/dflash_target_model.py`（315 行）是两套并列的层次结构，没有共享基类。                                                                                  | Phase 2 #4 — 单一 `TargetEngine` ABC；三个具体实现。                                            |
| **Online ≠ class、offline ≠ class**         | `core/eagle3.py` 是 "online" 的 wrapper（606 行）；offline 是另一条经过 `OfflineEagle3Dataset` + `TargetHead` 的路径。trainer 必须按模式分支。                                                                              | Phase 2 #5 — 两者都成为 `HiddenStateStream` 的实现。trainer 拿到的是同一个 iterator，与模式无关。            |
| **架构 dispatch 是 4 文件散弹枪**                  | 今天加一个新架构要动 `modeling/auto.py:35`、`modeling/auto.py:88`、`modeling/auto.py:134`，再加 `modeling/draft/__init__.py`。                                                                                      | Phase 1 #1 — 一个文件、一个装饰器。                                                              |
| **30 个手维护的 shell 脚本**                      | `examples/run_*_online.sh` × 30。默认值在它们之间漂移；加新特性要改 30 处。                                                                                                                                             | Phase 3 #14 — 由统一 schema 派生出 30 个 YAML 文件；shell 脚本坍缩成 `specforge train --config ...`。 |
| **没有数据 prefetch 重叠**                       | `core/eagle3.py` 在训练 step 里同步跑 target forward。Draft backward 被 target forward 阻塞。                                                                                                                   | Phase 2 #5 — `OnlineStream.prefetch_factor` 让生产者和消费者重叠。                               |
| **仓库里有误导性 config**                         | `configs/deepseek-v3-671b-eagle3.json` 给 DeepSeek target 声明了 `model_type: "llama"` — 如果当成 "MLA Eagle3 for V3" 读，就是悄悄的错误。                                                                            | §4.7 — 改名为 `..._llama_draft.json`，新增真正的 MLA config，加 deprecation warning。             |
| **重构没有数值等价 gate**                          | 没有任何机制能阻止 Phase 2 抽取 `Trainer` 时悄悄改变 loss 曲线。                                                                                                                                                       | §10.1 — 凡是触及 `core/`、`training/`、`models/drafts/` 的 PR，在固定 steps 上跑 `atol/rtol` gate。 |
| **没有 FSDP all-reduce 验证**                  | 没有 profiler 检查 accumulation 是否真的跳过了 sync。`--accumulation-steps` flag 存在，但 sync 并未被抑制（没有 `no_sync()`）— all-reduce 次数与不开 accumulation 相同。                                                             | §10.3 — 一次性 profiler 检查；每个 `optimizer.step()` 一次 all-reduce。                          |
| `**Eagle3DraftModel.backbone()` ABC 返回形状** | 返回 `Tensor`；KV cache 在 `past_key_values` 上 in-place 修改（HF `DynamicCache` 模式）。DFlash 遵守这点。一个简单粗暴的 MLA port 如果返回 `(out, k, v)`，就会破坏这个 ABC。                                                            | §4.7 — 显式规定 MLA "使用 `DynamicCache.update`，返回 `Tensor`"。                               |


### 2.3 杠杆最高的两块

如果这个季度只能交付两件事，就做这两件：

**Chunk 1 — Phase 1（Week 1-2）：`@register_draft` + MLA draft + Kimi-K2.5 TP/SP 冒烟测试。**
解锁真正的特性需求（DFlash + MLA Eagle3 on SGLang）。是增量式的 — 老的 `_model_mapping` dict 作为 fallback 仍然在。约 3 个新文件、`auto.py` 改 ~10 行。其中 TP/SP 冒烟测试（§4.7）是去风险的关键步骤，能把 Kimi-K2.5 从 "应该能跑" 变成可验证的交付物。

**Chunk 2 — Phase 2（Week 3-6）：`TargetEngine` + `HiddenStateStream` + `Trainer` 一起交付。**
这三者作为一个整体交付。只抽其中一个会出问题：要么 trainer 直接跟 target model 通信（Phase 4 又得拆掉），要么 stream 抽象没有 consumer（无法端到端测）。Phase 2 的验收 gate 是：legacy shell（已经走新内部实现）在数值等价容忍度内匹配旧 loss 曲线（§10.1）。这一关过了，Phase 4 就变成在稳定接口上 drop-in 接入 plugin（`SGLangServerEngine`、`RemoteStream`），不用再重新铺线。

其他所有事（export tool、CLI、VLM cleanup、FSDP2、WSD、Mooncake）都是会复利的债务清理 — 重要，但不阻塞当前目标。

---

## 3. 设计原则

1. **一个 trainer，多个 draft，多个 target。** 通过 plugin registry 做 config-driven dispatch。新增架构 = 新加一个文件，而不是新加一个 script。
2. **target 是接口，不是进程。** 同一个 trainer，target 可以是进程内 HF、独立的 SGLang server、或者远端集群。
3. **三种数据模式，一个抽象。** Offline-cached、online-local、online-remote — 都通过同一个 `HiddenStateStream` iterator 消费。但要尊重不对称性：online stream 拥有 GPU 资源和 backpressure；offline stream 是纯 reader。
4. **Strategy，而不是 fork。** EAGLE3（TTT unroll）和 DFlash（block-causal）是两个共享 trainer 的 `DraftTrainStrategy` 实现。
5. **保留能用的东西。** 不为了对称去重写经过实战检验的代码。

---

## 4. 目标架构

### 4.1 模块布局

```
specforge/
├── config/                          # Structured configuration
│   ├── schema.py                    # Pydantic models (Config, ModelConfig, DatasetConfig, ...)
│   ├── loader.py                    # YAML load + merge + CLI override + validation
│   └── draft_configs/               # Draft model JSONs (moved from top-level configs/)
│       ├── llama3_8b_eagle3.json
│       ├── qwen3_8b_eagle3.json
│       ├── qwen3_8b_eagle3_mla.json       # NEW
│       ├── kimi_k25_eagle3_mla.json       # NEW
│       ├── deepseek_v3_671b_eagle3.json   # NEW — real MLA config
│       └── ...
│
├── core/                            # Core algorithms (preserved)
│   ├── eagle3.py                    # Eagle3Model (renamed from OnlineEagle3Model)
│   ├── dflash.py                    # DFlashModel (renamed from OnlineDFlashModel)
│   ├── loss.py                      # Triton LogSoftmaxLoss (UNCHANGED)
│   └── adapters.py                  # BackendAdapter / UspAdapter (UNCHANGED)
│
├── models/
│   ├── drafts/                      # Plugin registry
│   │   ├── __init__.py              # DRAFT_REGISTRY + @register_draft decorator
│   │   ├── base.py                  # Eagle3DraftModel ABC (from modeling/draft/base.py)
│   │   ├── llama_eagle3.py          # LlamaForCausalLMEagle3 (existing)
│   │   ├── deepseek_eagle3.py       # Eagle3DeepseekV2ForCausalLM — MLA (NEW)
│   │   ├── dflash_qwen3.py          # DFlashDraftModel (existing)
│   │   └── auto.py                  # AutoEagle3DraftModel (registry-backed, no hardcoded dicts)
│   │
│   └── targets/                     # Target engine abstraction
│       ├── base.py                  # TargetEngine ABC + TargetOutput dataclass
│       ├── hf_engine.py             # In-process HF target (lazy import)
│       ├── sglang_engine.py         # In-process SGLang target (lazy import)
│       ├── sglang_server_engine.py  # SGLang-as-service over HTTP (NEW)
│       ├── custom_engine.py         # Custom TP backend (existing custom_backend/)
│       └── target_head.py           # TargetHead for offline logits (existing)
│
├── data/
│   ├── streams/                     # HiddenStateStream abstraction
│   │   ├── base.py                  # HiddenStateStream protocol
│   │   ├── online.py               # In-process target generates hidden states
│   │   ├── offline.py              # Pre-computed hidden states from disk
│   │   └── remote.py               # Target on separate SGLang server (NEW)
│   ├── template.py                  # TEMPLATE_REGISTRY (UNCHANGED)
│   ├── parse.py                     # Parsers + KimiK25Parser, MiniMaxParser (NEW)
│   ├── preprocessing.py             # build_eagle3_dataset etc. (UNCHANGED)
│   ├── collator.py                  # DataCollatorWithPadding (extracted from utils.py)
│   └── cache.py                     # Tokenization cache + eval cache (NEW)
│
├── training/
│   ├── trainer.py                   # Trainer: unified training loop
│   ├── strategies/
│   │   ├── __init__.py              # DraftTrainStrategy protocol
│   │   ├── eagle3_ttt.py            # Eagle3 TTT unroll + forward-KL
│   │   └── dflash_block.py          # DFlash block-causal + anchor sampling
│   ├── optimizer.py                 # BF16Optimizer (UNCHANGED)
│   ├── lr_scheduler.py              # CosineWarmup + WSD scheduler (NEW)
│   ├── checkpoint.py                # CheckpointManager (NEW)
│   ├── fsdp.py                      # apply_fsdp (FSDP1 SHARD_GRAD_OP + future FSDP2)
│   └── distributed.py               # init_distributed etc. (moved from specforge/distributed.py)
│
├── eval/                            # Evaluation system (NEW)
│   ├── evaluator.py                 # Evaluator: eval loop + metric aggregation
│   ├── cache.py                     # EvalCache: MD5-keyed disk cache
│   └── metrics.py                   # simulated_acc_len, avg_loss, avg_acc
│
├── export/                          # Model export tools (NEW)
│   ├── to_hf.py                     # FSDP checkpoint → HF format + vocab pruning
│   └── to_sglang.py                 # Checkpoint → SGLang spec-decoder layout
│
├── tracker.py                       # UNCHANGED
├── utils.py                         # UNCHANGED
│
├── cli.py                           # `specforge train|prepare|export|eval` (NEW)
└── __init__.py

scripts/
├── legacy/                          # Old scripts as thin shims
│   ├── train_eagle3.py
│   └── train_dflash.py
├── prepare_data.py                  # UNCHANGED
├── prepare_hidden_states.py         # UNCHANGED
└── regenerate_train_data.py         # UNCHANGED
```

### 4.2 关键抽象

#### Draft model registry

```python
# models/drafts/__init__.py
DRAFT_REGISTRY: dict[str, type] = {}

def register_draft(name: str):
    """Decorator. New arch = new file + @register_draft("deepseek_v3_eagle3")."""
    def wrapper(cls):
        DRAFT_REGISTRY[name] = cls
        return cls
    return wrapper

# models/drafts/llama_eagle3.py
@register_draft("llama_eagle3")
class LlamaForCausalLMEagle3(Eagle3DraftModel):
    ...

# models/drafts/deepseek_eagle3.py
@register_draft("deepseek_v3_eagle3")
class Eagle3DeepseekV2ForCausalLM(Eagle3DraftModel):
    config_class = DeepseekV3Config
    ...
```

`Eagle3DraftModel` ABC 完全原样保留。它的接口本身设计得不错：

```python
class Eagle3DraftModel(PreTrainedModel, ABC):
    def embed_input_ids(self, input_ids: Tensor) -> Tensor: ...
    def project_hidden_states(self, hidden_states: Tensor) -> Tensor: ...
    def backbone(self, input_embeds, hidden_states, cache_hidden,
                 attention_mask, position_ids, past_key_values=None,
                 use_cache=True) -> Tensor: ...
    def compute_logits(self, hidden_states: Tensor) -> Tensor: ...
    # Concrete: load_embedding, freeze_embedding, load_vocab_mapping, t2d/d2t
```

`AutoDraftModelConfig.from_file()` 改成先查 `DRAFT_REGISTRY`，再回退到 HF config type dispatch 以保持向后兼容。

#### Target engine

```python
# models/targets/base.py
@dataclass
class TargetOutput:
    aux_hidden_states: torch.Tensor  # [batch, seq, hidden * num_aux_layers] — raw concat (NOT projected)
    target_logits: torch.Tensor      # [batch, seq, vocab] — pre-softmax logits in draft vocab space
    loss_mask: torch.Tensor          # [batch, seq]
    input_ids: torch.Tensor          # [batch, seq]
    attention_mask: torch.Tensor     # [batch, seq]
    last_hidden_states: Optional[torch.Tensor] = None

class TargetEngine(ABC):
    @abstractmethod
    def generate_train_data(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        loss_mask: Tensor,
        media: Optional[MediaInputs] = None,   # typed VLM payload (pixel_values, image_grid_thw, ...)
    ) -> TargetOutput: ...

    @property
    @abstractmethod
    def aux_layer_ids(self) -> list[int]: ...

    def set_aux_hidden_states_layers(self, layers: list[int]) -> None: ...
```

命名：

- `target_logits`（不是 `target`）— 之前用 "target" 同时表示 "target model" 容易混淆。
- `aux_hidden_states` — 来自 target 的原始拼接多层 hidden states。投影（3·hidden → hidden）**留在 draft 模型内部**（`project_hidden_states`），这样 stream 和 engine 都不需要知道 draft 的 hidden_size。

这就是对现有 `Eagle3TargetModel` + `Eagle3TargetOutput` 的改名和小幅泛化。现有的 SGLang/HF/Custom 实现变成具体 engine，API 不变。VLM 通过 `**media_kwargs` 处理 — 不需要单独的 model class。

Lazy import 保证除非 `backend="sglang"`，否则永远不会 import SGLang：

```python
def get_target_engine(backend: str, **kwargs) -> TargetEngine:
    if backend == "sglang":
        from specforge.models.targets.sglang_engine import SGLangTargetEngine
        return SGLangTargetEngine(**kwargs)
    elif backend == "sglang_server":
        from specforge.models.targets.sglang_server_engine import SGLangServerEngine
        return SGLangServerEngine(**kwargs)
    elif backend == "hf":
        from specforge.models.targets.hf_engine import HFTargetEngine
        return HFTargetEngine(**kwargs)
    elif backend == "custom":
        from specforge.models.targets.custom_engine import CustomTargetEngine
        return CustomTargetEngine(**kwargs)
    raise ValueError(f"Unknown target backend: {backend}")
```

#### HiddenStateStream

```python
# data/streams/base.py
class HiddenStateStream(ABC):
    """Produces TrainBatch instances for the draft trainer."""

    def setup(self, draft_config) -> None:
        """Inform stream of draft requirements (aux layers, vocab mapping, etc.)."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[TrainBatch]: ...

    def seek(self, step: int) -> None:
        """Resume from step N. Required for checkpoint resume to behave correctly.

        Default behavior (provided by base class): advance the iterator by `step`
        batches. Implementations with random-access storage (e.g. OfflineStream)
        should override to seek directly without consuming.
        """

    def teardown(self) -> None:
        """Release GPU / network resources."""
        pass

@dataclass
class TrainBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_logits: torch.Tensor     # target logits on draft vocab (pre-softmax)
    loss_mask: torch.Tensor
    aux_hidden_states: torch.Tensor # raw concat of target aux layers — draft projects internally
    position_ids: Optional[torch.Tensor] = None
```

三种实现：


| Stream          | 数据源                    | GPU 成本               | 备注                                                                          |
| --------------- | ---------------------- | -------------------- | --------------------------------------------------------------------------- |
| `OnlineStream`  | 进程内 `TargetEngine`     | 高（target 在同一组 GPU 上） | 当前的 "online" 模式；支持 `prefetch_factor`，让 target inference 与 draft training 重叠 |
| `OfflineStream` | 预先计算好的 .pt 文件          | 无                    | 当前的 "offline" 模式；override `seek()` 实现 O(1) resume                           |
| `RemoteStream`  | 走 HTTP 的 SGLang server | 本地无                  | 新增 — target 在独立节点；支持 `prefetch_factor` 与请求批处理                               |


`OnlineStream` 持有 target engine 的生命周期，并负责 TP→DP 的 batch sharding（当前在 `train_eagle3.py:get_dp_data_shard_from_tp` 里手动做）。暴露 `prefetch_factor: int = 2`，让下一个 batch 的 target forward 与当前 batch 的 draft backward 重叠。

`OfflineStream` 包装现有的 `OfflineEagle3Dataset` + `TargetHead` 路径。override `seek()` 直接跳到 sample index，不用迭代中间文件。

`RemoteStream` 把 tokenized batch 发到 SGLang server，再把 hidden states 收回来。这覆盖了 "671B target 跑在专用 GPU 上" 的场景，不需要 Ray/Mooncake。也要暴露 `prefetch_factor`（不开 prefetch 时网络 RTT 会主导）和 `max_in_flight` 上限，让 back-pressure 显式。

#### DraftTrainStrategy

```python
# training/strategies/__init__.py
class DraftTrainStrategy(ABC):
    """Encapsulates one training algorithm's forward + loss computation."""

    @abstractmethod
    def forward_and_loss(
        self,
        draft_model: Eagle3DraftModel,
        batch: TrainBatch,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns (loss, metrics_dict). Target data is already in batch."""
        ...

    @abstractmethod
    def build_model(self, draft_model, **kwargs) -> nn.Module:
        """Wrap draft_model in the strategy-specific training wrapper."""
        ...

    def fsdp_wrap_policy(self) -> Optional[Callable]:
        """Optional: return an FSDP auto-wrap policy tailored to this strategy's wrapper.

        The trainer applies this if non-None; otherwise falls back to a default
        transformer-block wrap policy. Lets strategies declare wrapping intent
        instead of the trainer second-guessing.
        """
        return None
```

两种实现：

- `**Eagle3TTTStrategy**`：把 draft 包进 `Eagle3Model`（当前 `core/eagle3.py`），跑 TTT unroll，用 `LogSoftmaxLoss` 算 forward-KL loss。现有的 `Eagle3Model.forward()` 签名可直接映射。
- `**DFlashBlockStrategy**`：把 draft 包进 `DFlashModel`（当前 `core/dflash.py`），做 anchor sampling + block-causal CE。现有的 `OnlineDFlashModel.forward()` 可直接映射。

Strategy **不**触碰 target engine — batch 里已经有具体化的 tensor。这是一条硬边界。

#### Trainer

```python
# training/trainer.py
class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.strategy = self._build_strategy()
        self.checkpoint_mgr = CheckpointManager(config.training)
        self.evaluator = Evaluator(config.eval) if config.eval.enabled else None
        self.tracker = create_tracker(config.logging)

    def train(self):
        # 1. Distributed setup
        init_distributed(...)

        # 2. Build draft model from registry
        draft_config = AutoDraftModelConfig.from_file(self.config.model.draft_model_config)
        draft_model = DRAFT_REGISTRY[draft_config.architectures[0]].from_config(draft_config)

        # 3. Build strategy-specific wrapper
        model = self.strategy.build_model(draft_model, ...)

        # 4. Apply FSDP (strategy may override wrap policy)
        model = apply_fsdp(
            model,
            self.config.training,
            wrap_policy=self.strategy.fsdp_wrap_policy(),
        )

        # 5. Build optimizer
        optimizer = BF16Optimizer(model, ...)

        # 6. Build data stream
        stream = self._build_stream()
        stream.setup(draft_config)

        # 7. Resume if needed — note: stream.seek() is required for correctness.
        #    enumerate(stream, start=start_step) only changes the counter; without
        #    seek(), the stream still yields batch 0 first, silently re-training on it.
        start_step = 0
        if self.config.training.resume:
            start_step = self.checkpoint_mgr.load(model, optimizer)
            stream.seek(start_step)

        # 8. Training loop
        accum = self.config.training.accumulation_steps
        for step, batch in enumerate(stream, start=start_step):
            if step >= self.config.training.max_steps:
                break

            # Forward + loss (strategy-specific)
            loss, metrics = self.strategy.forward_and_loss(model, batch)
            scaled_loss = loss / accum

            # Skip gradient sync on all but the final micro-step of an accumulation
            # window. With FSDP this is the difference between one all-reduce per
            # micro-batch and one per optimizer step — non-trivial on large models.
            is_sync_step = ((step + 1) % accum == 0)
            sync_ctx = nullcontext() if is_sync_step else model.no_sync()
            with sync_ctx:
                scaled_loss.backward()

            if is_sync_step:
                optimizer.step()

            # Logging
            if (step + 1) % self.config.training.log_interval == 0:
                self.tracker.log(metrics, step=step)

            # Eval
            if self.evaluator and (step + 1) % self.config.eval.interval == 0:
                eval_metrics = self.evaluator.run(
                    lambda b: self.strategy.forward_and_loss(model, b),
                    self._eval_stream,
                )
                self.checkpoint_mgr.update_best(step, eval_metrics)
                self.tracker.log(eval_metrics, step=step)

            # Save
            if (step + 1) % self.config.training.save_interval == 0:
                self.checkpoint_mgr.save(step, model, optimizer)

        stream.teardown()
        self.tracker.close()
```

这是大约 70 行逻辑。其他一切都在接口后面。

伪代码明确指出两个非显然的正确性要点：

1. **Resume 需要 `stream.seek(start_step)`。** 否则 `enumerate(stream, start=N)` 只是重命名计数器 — iterator 仍会先 yield batch 0，导致 resume 会悄悄在前缀上重复训练。
2. **Gradient accumulation 需要在非 sync step 调 `model.no_sync()`。** 否则 FSDP 每个 micro-batch 都做 all-reduce，让 accumulation 失去意义。

### 4.3 Checkpoint manager

```python
# training/checkpoint.py
class CheckpointManager:
    def __init__(self, config: TrainingConfig):
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.max_checkpoints = config.max_checkpoints    # 0 = keep all
        self.best_score = -float("inf")

    def save(self, step: int, model, optimizer, meta: dict = None):
        """Save model + optimizer + LR state + RNG + meta.json."""
        step_dir = self.checkpoint_dir / f"iter_{step + 1:07d}"
        # torch.distributed.checkpoint.save for model, optimizer
        # rank-0: rng.pt, meta.json (step, timestamp, global_step, world_size)
        self._update_latest(step + 1)
        self._rotate()

    def load(self, model, optimizer=None, continual=False) -> int:
        """Load latest or specified checkpoint. Returns start_step."""
        step_id = self._read_latest()
        # torch.distributed.checkpoint.load
        # If continual: skip optimizer/LR/RNG, only restore weights + rebuild FP32 master
        return step_id

    def update_best(self, step: int, eval_metrics: dict):
        """Track best checkpoint by simulated_acc_len."""
        score = eval_metrics.get("simulated_acc_len", eval_metrics.get("avg_acc", 0))
        if score > self.best_score:
            self.best_score = score
            self._write_best(step + 1, eval_metrics)

    def _rotate(self):
        """Keep only max_checkpoints newest iter_* directories."""
        if self.max_checkpoints <= 0:
            return
        dirs = sorted(self.checkpoint_dir.glob("iter_*"))
        for d in dirs[:-self.max_checkpoints]:
            shutil.rmtree(d)

    def _update_latest(self, step_id: int):
        (self.checkpoint_dir / "latest_checkpointed_iteration.txt").write_text(str(step_id))

    def _write_best(self, step_id: int, metrics: dict):
        (self.checkpoint_dir / "best_checkpointed_iteration.txt").write_text(str(step_id))
        (self.checkpoint_dir / "best_meta.json").write_text(json.dumps(metrics, indent=2))
```

### 4.4 评测系统

```python
# eval/evaluator.py
class Evaluator:
    def __init__(self, config: EvalConfig):
        self.cache = EvalCache(config.cache_dir)
        self.micro_batch_size = config.micro_batch_size

    def run(self, forward_fn, eval_stream: HiddenStateStream) -> dict:
        """Run full eval pass, return aggregated metrics.

        Per-position accuracy must be averaged *across all batches first*, then fed
        into the geometric sum. Treating each batch's per-position vector as if it
        were positions makes simulated_acc_len batch-size-dependent.
        """
        total_loss_x_tokens = 0.0
        total_tokens = 0
        # Sum and count per draft position (TTT step), aggregated across the whole pass.
        per_pos_acc_sum: Optional[torch.Tensor] = None       # shape [ttt_length]
        per_pos_acc_count: Optional[torch.Tensor] = None     # shape [ttt_length]

        for batch in self._iter_micro_batches(eval_stream):
            with torch.no_grad():
                loss, metrics = forward_fn(batch)
            total_loss_x_tokens += metrics["loss"] * metrics["num_tokens"]
            total_tokens += metrics["num_tokens"]

            ppa = metrics.get("per_position_acc")        # [ttt_length], weighted by num_tokens
            ppc = metrics.get("per_position_count")      # [ttt_length], token counts per position
            if ppa is not None:
                if per_pos_acc_sum is None:
                    per_pos_acc_sum = torch.zeros_like(ppa)
                    per_pos_acc_count = torch.zeros_like(ppc)
                per_pos_acc_sum += ppa * ppc             # accumulate weighted sum
                per_pos_acc_count += ppc

        # Aggregate first, then geometric-sum.
        per_position_acc = (per_pos_acc_sum / per_pos_acc_count.clamp_min(1)).tolist()
        return {
            "eval/avg_loss": total_loss_x_tokens / max(total_tokens, 1),
            "eval/avg_acc": float(per_position_acc[0]) if per_position_acc else 0.0,
            "eval/simulated_acc_len": self._simulated_acc_len(per_position_acc),
        }

    @staticmethod
    def _simulated_acc_len(per_position_acc: list[float]) -> float:
        """E[accepted tokens] = acc_0 + acc_0*acc_1 + acc_0*acc_1*acc_2 + ...

        `per_position_acc` is the *aggregated* per-position accuracy across the full
        eval set, length = ttt_length. Not a list of per-batch vectors.
        """
        cumulative = 1.0
        total = 0.0
        for acc in per_position_acc:
            cumulative *= acc
            total += cumulative
        return total

# eval/cache.py
class EvalCache:
    """Disk cache for pre-computed eval hidden states.

    Cache key must cover everything that would change the cached tensors:
    eval data, target model identity & revision, tokenizer, chat template,
    aux layer ids, and sequence length. Missing any of these silently serves
    stale data after a target swap or template change.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

    def cache_key(
        self,
        eval_path: str,
        target_path: str,
        target_revision: str,
        tokenizer_path: str,
        chat_template: str,
        aux_layer_ids: list[int],
        max_seq_len: int,
    ) -> str:
        content = "|".join([
            eval_path,
            target_path,
            target_revision or "",
            tokenizer_path,
            chat_template,
            ",".join(map(str, aux_layer_ids)),
            str(max_seq_len),
        ])
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def try_load(self, key: str) -> Optional[list]:
        path = self.cache_dir / "eval_cache" / key
        if path.exists():
            return [torch.load(f) for f in sorted(path.glob("rank_*.pt"))]
        return None

    def save(self, key: str, rank: int, data: list):
        path = self.cache_dir / "eval_cache" / key
        path.mkdir(parents=True, exist_ok=True)
        torch.save(data, path / f"rank_{rank:04d}.pt")
```

注意：`data/cache.py`（tokenization cache）和 `eval/cache.py`（eval hidden-state cache）是有意分开的 — 它们 key 的内容不同，生命周期阶段也不同。不要合并。

### 4.5 Structured config

```python
# config/schema.py
from pydantic import BaseModel, Field
from typing import Optional, Literal

class ModelConfig(BaseModel):
    target_model_path: str
    draft_model_config: str               # path to JSON
    target_backend: Literal["sglang", "hf", "custom", "sglang_server"] = "sglang"
    trust_remote_code: bool = False
    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"
    is_vlm: bool = False

class DatasetConfig(BaseModel):
    train_data_path: str
    eval_data_path: str = ""
    train_hidden_states_path: str = ""    # non-empty = offline mode
    chat_template: str = "llama3"
    max_length: int = 2048
    train_only_last_turn: bool = False
    num_proc: int = 8

class TrainingConfig(BaseModel):
    strategy: Literal["eagle3", "dflash"] = "eagle3"
    num_epochs: int = 1
    max_steps: int = 10000
    batch_size: int = 1
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.015
    max_grad_norm: float = 0.5
    accumulation_steps: int = 1
    ttt_length: int = 7                   # Eagle3-specific
    block_size: int = 16                  # DFlash-specific
    num_anchors: int = 512                # DFlash-specific
    loss_decay_gamma: Optional[float] = None
    attention_backend: Literal["sdpa", "flex_attention", "fa", "usp"] = "sdpa"
    fsdp_strategy: Literal["NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD", "HYBRID_SHARD"] = "SHARD_GRAD_OP"
    fsdp_version: Literal[1, 2] = 1       # 2 = FSDP2 (PT 2.4+); see §4.9
    compile_model: bool = False
    tp_size: int = 1
    sp_ulysses_size: int = 1
    sp_ring_size: int = 1
    save_interval: int = 500
    log_interval: int = 10
    max_checkpoints: int = 5              # 0 = keep all
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    seed: int = 42

class EvalConfig(BaseModel):
    enabled: bool = False
    interval: int = 500
    micro_batch_size: int = 4
    cache_dir: str = "eval_cache"

class LRConfig(BaseModel):
    decay_style: Literal["cosine", "WSD"] = "cosine"
    wsd_decay_steps: int = 0
    wsd_decay_style: Literal["linear", "cosine", "exponential"] = "cosine"

class LoggingConfig(BaseModel):
    report_to: Literal["wandb", "tensorboard", "swanlab", "mlflow", "none"] = "wandb"
    wandb_project: str = "specforge"
    wandb_run_name: str = ""

class SGLangConfig(BaseModel):
    attention_backend: str = "flashinfer"
    # ... other SGLang server args

class Config(BaseModel):
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    lr: LRConfig = Field(default_factory=LRConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    sglang: SGLangConfig = Field(default_factory=SGLangConfig)
    output_dir: str = "output"
    cache_dir: str = "cache"
```

MLA Eagle3 训练的 YAML 示例：

```yaml
model:
  target_model_path: Qwen/Qwen3-8B
  draft_model_config: specforge/config/draft_configs/qwen3_8b_eagle3_mla.json
  target_backend: sglang

dataset:
  train_data_path: data/train.jsonl
  eval_data_path: data/eval.jsonl
  chat_template: qwen3
  max_length: 4096

training:
  strategy: eagle3
  max_steps: 20000
  batch_size: 2
  learning_rate: 3e-4
  ttt_length: 7
  attention_backend: flex_attention
  accumulation_steps: 2
  tp_size: 4
  max_checkpoints: 3

eval:
  enabled: true
  interval: 1000

output_dir: runs/qwen3_8b_mla_eagle3
```

CLI override：`specforge train --config config.yaml --training.learning_rate=1e-4`

### 4.6 WSD learning rate scheduler

> **延期到 Phase 5**（原草稿是 Phase 2）。Cosine warmup 对常见的 draft 训练 run 已经够用；WSD 更适合 continued pretraining 与很长的 stable 阶段 run，这些不在当前 roadmap 上。

加到 `specforge/training/lr_scheduler.py`：

```python
class WSDScheduler(_LRScheduler):
    """Warmup → Stable → Decay schedule.

    Stable phase holds max LR until (total_steps - wsd_decay_steps).
    Decay phase applies linear/cosine/exponential decay to min_lr.
    """
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr=0.0,
                 wsd_decay_steps=0, wsd_decay_style="cosine", last_epoch=-1):
        ...
```

在 trainer 里通过 `LRConfig.decay_style == "WSD"` 接入。

### 4.7 MLA Eagle3 draft model

从 TorchSpec 的 `deepseek_eagle.py` port 过来。关键设计决定：

**TTT 上下文中的 KV cache**：在 Eagle3 TTT unroll 期间，cache 的是 expanded K/V（而不是 compressed latent）。这与 TorchSpec 的做法一致，避免要求 `core/eagle3.py` 里出现 MLA 专属的 cache 逻辑。MLA 的 compression 只在投影（forward）时发生，cache 里不做。

**Cache 集成**：使用 HF 的 `DynamicCache` 模式（通过 `past_key_values.update(k, v, layer_idx, cache_kwargs)` in-place 修改 `past_key_values`）— 与现有 `DFlashDraftModel` 和 `Eagle3DraftModel.backbone()` ABC 一致，后者返回 `Tensor`（hidden states），永远不返回 `(Tensor, k, v)` tuple。返回 tuple 会破坏基 ABC。

```python
# models/drafts/deepseek_eagle3.py
@register_draft("deepseek_v3_eagle3")
class Eagle3DeepseekV2ForCausalLM(Eagle3DraftModel):
    config_class = DeepseekV3Config

    class DeepSeekMLAAttention(nn.Module):
        """MLA attention with Q/KV LoRA, decoupled RoPE."""
        def __init__(self, config: DeepseekV3Config):
            # Q path: q_a_proj → RMSNorm → q_b_proj (if q_lora_rank)
            # KV path: kv_a_proj_with_mqa → split(kv_compressed, k_rope_raw)
            #        → kv_a_layernorm → kv_b_proj → split(k_nope, value)
            # RoPE: interleaved rotation on qk_rope_head_dim slice only
            # o_proj: num_heads * v_head_dim → hidden_size
            ...

        def forward(self, hidden_states, past_key_values=None,
                    attention_mask=None, position_ids=None, cache_position=None,
                    use_cache=False):
            # 1. Project Q (with optional LoRA)
            # 2. Project KV (compressed → expand via kv_b_proj)
            # 3. Split k_nope from kv_b_proj, k_rope_raw from kv_a_proj
            # 4. Apply interleaved RoPE to q_rope and k_rope slices
            # 5. Expand k_rope (MQA → MHA) across heads
            # 6. Concat [k_nope, k_rope] → full expanded key
            # 7. If use_cache and past_key_values is not None:
            #        k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
            # 8. Attention (SDPA or FlexAttention)
            # 9. Return attn_output  — k/v stored in the mutable DynamicCache
            ...
```

从 `DeepseekV3Config` 消费的 config 字段：

- `q_lora_rank` — Q 低秩维度（None 表示稠密 Q）
- `kv_lora_rank` — KV 压缩维度
- `qk_nope_head_dim` — 不带 RoPE 的 per-head key 维度
- `qk_rope_head_dim` — 带 RoPE 的 per-head key 维度
- `v_head_dim` — per-head value 维度（可能与 key 维度不同）
- `rope_scaling` — YaRN 参数

**Expanded TTT cache 的显存预算**。每 token 每个 draft 层：

```
key_bytes_per_token   = num_heads * (qk_nope_head_dim + qk_rope_head_dim) * dtype_size
value_bytes_per_token = num_heads * v_head_dim                            * dtype_size
```

代入 DeepSeek-V3 的数（128 heads、128 nope + 64 rope、128 v_dim、bf16 = 2B）：

- 每 token：128*(128+64)*2 = 49 152 B key + 128*128*2 = 32 768 B value ≈ 80 KB
- seq=4096、ttt=7、一个 draft 层、一个 sample：~2.3 GB
- draft 通常只有 1 层，所以这是随 `batch_size` 而不是随 `num_layers` 扩展

所以来自 TTT cache 的 per-GPU 峰值 ≈ `batch_per_gpu * 2.3 GB`。相应地限制 `batch_size`，或者如果这阻塞了长上下文训练，再回头考虑 compressed-cache 选项。

**MLA 的 TP/SP — 在 Phase 1 sign-off 前要验证**。MLA 的 per-head 维度是不对称的（`qk_nope`、`qk_rope`、`v_head_dim`）— Yunchang USP 和某些 TP 切片在某些地方假设 per-head 同质。在 Kimi-K2.5 上跑个小冒烟测试（`tp_size=2`、`sp_ulysses_size=2`）属于 Phase 1 关键路径，不是 Phase 4 的事。

**旧 `configs/deepseek-v3-671b-eagle3.json` 的迁移**。那个文件声明的是 `model_type: "llama"`，用一个 Llama 风格的 draft 去训 DeepSeek target — 它**不是** MLA draft，尽管文件名误导。Phase 1 计划：

- 移动到 `specforge/config/draft_configs/deepseek_v3_671b_eagle3_llama_draft.json`（保留对当前在用者的向后兼容）。
- 在 `specforge/config/draft_configs/deepseek_v3_671b_eagle3.json` 新增真正的 MLA config。
- 加载旧路径时发一次性 deprecation warning。

### 4.8 SGLang export tool

```python
# export/to_sglang.py
def export_to_sglang(checkpoint_dir: str, output_dir: str, target_model_path: str,
                     prune_vocab: bool = False, prune_dataset: str = None):
    """Convert FSDP training checkpoint to SGLang-loadable spec-decoder format.

    Handles:
    - FSDP state dict → flat state dict
    - Weight key renaming (training names → SGLang spec-decoder expected names)
    - MLA weight naming: q_a_proj, kv_a_proj_with_mqa, kv_b_proj etc.
    - Optional vocab pruning based on dataset token frequency
    - Config generation (draft config JSON + tokenizer copy)
    """
    ...
```

**Weight-name 兼容性是整个重写里风险最大的单点** — 出问题时是无声的（loader 对缺失 key 拿到零、或者拒绝加载）。在 Phase 3 开始前，为每个 draft 架构产出一份明确的两列映射表：


| Trainer key                                            | SGLang spec-decoder loader key                          |
| ------------------------------------------------------ | ------------------------------------------------------- |
| `model.layers.{i}.self_attn.q_a_proj.weight`           | *例如* `draft_model.layers.{i}.self_attn.q_a_proj.weight` |
| `model.layers.{i}.self_attn.q_a_layernorm.weight`      | ...                                                     |
| `model.layers.{i}.self_attn.q_b_proj.weight`           | ...                                                     |
| `model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight` | ...                                                     |
| `model.layers.{i}.self_attn.kv_a_layernorm.weight`     | ...                                                     |
| `model.layers.{i}.self_attn.kv_b_proj.weight`          | ...                                                     |
| `model.layers.{i}.self_attn.o_proj.weight`             | ...                                                     |
| `model.embed_tokens.weight`                            | ...                                                     |
| `lm_head.weight`                                       | ...                                                     |
| `t2d`、`d2t`（vocab mapping）                             | ...                                                     |


要填充 RHS 列需要去读 SGLang 当前的 spec-decoding draft loader（sgl-project/sglang 仓里当前是 `Eagle3`* 或 `LightseekSpec*` loader，看哪个是当前实现），并且这应该作为一份显式文档（`docs/export_weight_map_mla.md`），而不是隐式写在代码里。

### 4.9 FSDP 接缝（现在 FSDP1，FSDP2-ready）

本方案先在 FSDP1 上交付（`SHARD_GRAD_OP`），但 `apply_fsdp` 从第一天起就是稳定接缝，由 `TrainingConfig.fsdp_version` 控制：

```python
# training/fsdp.py
def apply_fsdp(model, training_config, wrap_policy=None):
    if training_config.fsdp_version == 1:
        return _apply_fsdp1(model, training_config, wrap_policy)
    elif training_config.fsdp_version == 2:
        return _apply_fsdp2(model, training_config, wrap_policy)
    raise ValueError(...)
```

理由：FSDP2（PT 2.4+）跟 `torch.compile` 和 per-parameter sharding（`fully_shard` 用在每个子模块上）组合得好得多。未来 12 个月计算友好的默认就是 FSDP2。把 trainer 接口钉死在 FSDP1 上，意味着 `compile_model: True` 成为常态时还要再重写一遍。`_apply_fsdp2` 的实现延期到 Phase 5/6 — 现在重要的是这个接缝。

---

## 5. 特性列表（按优先级）

**关于顺序的说明（来自 review）：** 原草稿把统一 `Trainer` 放在 Phase 2，把 `TargetEngine` / `HiddenStateStream` 抽象放在 Phase 4。这意味着 Phase 2 会基于*老的* `Eagle3TargetModel`/`DFlashTargetModel` 接口去构建 trainer，然后 Phase 4 再重新铺线。已重排顺序：接口（以及每个接口的一个进程内实现）与 trainer 一起在 Phase 2 落地；Phase 4 再向稳定接口加入新实现（`SGLangServerEngine`、`RemoteStream`），而不是重新铺线。

### Phase 1：MLA Draft + Registry（Week 1-2）


| #   | 特性                                              | 为什么                                                                                                        | 工作量 |
| --- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --- |
| 1   | Draft plugin registry（`@register_draft`）        | 今天每加一个架构都要碰 4 个文件。这是单一最大的扩展性瓶颈。                                                                            | S   |
| 2   | MLA Eagle3 draft（`Eagle3DeepseekV2ForCausalLM`） | DeepSeek-V3 / Kimi-K2 的缺口。从 TorchSpec port。包含通过 HF `DynamicCache` 处理 MLA-aware KV cache（不破坏 ABC）。          | M   |
| 3   | MLA draft 配置                                    | `qwen3_8b_eagle3_mla.json`、`kimi_k25_eagle3_mla.json`、`deepseek_v3_671b_eagle3.json`（真正的 MLA，不是冒充的 Llama）。 | S   |
| 3a  | **MLA + TP/SP 冒烟测试**                            | 在宣布 Phase 1 完成前，验证 Yunchang USP 和 TP 切片在 Kimi-K2.5 的非对称 MLA head dim 下能正确工作。                               | S   |


**交付物**：使用现有 `train_eagle3.py` 训练一个 Qwen3-8B / Kimi-K2.5 的 MLA Eagle3 draft。
现有 script 不变 — registry 与旧 `_model_mapping` dict 并存。

### Phase 2：接口 + Trainer + Eval + Checkpoints（Week 3-6）


| #   | 特性                                                              | 为什么                                                                                                                                                               | 工作量 |
| --- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| 4   | `TargetEngine` protocol + 进程内实现                                 | 先把接口定义出来，让 trainer 从第一天起就构建在它之上。把现有 `Eagle3TargetModel` / `DFlashTargetModel` 适配为最初的具体实现。                                                                         | M   |
| 5   | `HiddenStateStream` protocol + `OnlineStream` + `OfflineStream` | 同样的原因 — 在 trainer 代码写之前就要有稳定抽象。`RemoteStream` 延期到 Phase 4。                                                                                                        | M   |
| 6   | `Trainer` class + `DraftTrainStrategy` protocol                 | 合并 `train_eagle3.py` 和 `train_dflash.py`。终止 script-per-arch 蔓延。Trainer 只消费 `TrainBatch` — 不直接调用 target model。                                                     | M   |
| 7   | `Eagle3TTTStrategy` + `DFlashBlockStrategy`                     | 从现有 script 抽取成 strategy 实现。各自声明自己的 `fsdp_wrap_policy()`。                                                                                                          | M   |
| 8   | `CheckpointManager`                                             | 轮转（`max_checkpoints`）、best 追踪、`latest_checkpointed_iteration.txt`、`meta.json`。在 resume 时必须与 `stream.seek()` 集成。                                                   | S   |
| 9   | `Evaluator` + `EvalCache`                                       | 训练中的 online eval：`simulated_acc_len`、`avg_loss`、`avg_acc`。Per-position acc 在所有 batch 上聚合后再做 geometric sum。Cache key 覆盖 eval/target/tokenizer/template/aux/seqlen。 | M   |
| 10  | Gradient accumulation 正确性                                       | 在非 sync micro-step 调 `model.no_sync()`、显式 `zero_grad`、在 config 里加 `accumulation_steps`。用 step-time profiling 验证。                                                  | S   |
| 11  | `target_layer_ids` 作为一等契约                                       | EAGLE3 硬编码 3 个 aux 层；DFlash 用 list。泛化成每个 draft 都声明自己需要什么，stream 据此实例化。                                                                                            | S   |
| 12  | FSDP 接缝（FSDP1 默认，FSDP2-ready）                                   | `apply_fsdp` 按 `fsdp_version` dispatch。FSDP2 实现延期 — 但接缝现在就稳定，FSDP2 落地时不用再重写 trainer。                                                                              | S   |


**交付物**：`Trainer`（暂时还没有 CLI — legacy script 调用它）同时支持 Eagle3 和 DFlash，使用新接口与单一进程内 target/stream 实现。训练期间 log eval metrics。Best checkpoint 自动保存。对照重构前的 script 检查数值等价（§10）。

### Phase 3：Config + Export（Week 7-8）


| #   | 特性                                         | 为什么                                                                                         | 工作量 |
| --- | ------------------------------------------ | ------------------------------------------------------------------------------------------- | --- |
| 13  | Pydantic config schema                     | 用一份经过 validation 的 YAML 替换 argparse + JSON 双入口。                                             | S-M |
| 14  | CLI（`specforge train|prepare|export|eval`） | 单一入口。Legacy script 此时变成 shim，构造一个 `Config` 然后调用 `Trainer`。                                  | S   |
| 15  | SGLang export tool                         | Checkpoint → SGLang spec-decoder layout。**Weight-name map（§4.8）是一份显式文档制品，在这个 phase 开始前敲定。** | M   |
| 16  | HF export tool + vocab pruning             | FSDP checkpoint → HF 格式。可选按数据集频率做 vocab pruning。                                            | S-M |


**交付物**：一个工具完成 train → eval → export → serve 全流水线。Phase 1 训出的旧 MLA draft 能在 SGLang server 中成功加载。

### Phase 4：远端 Target + VLM + Parsers（Week 9-11）


| #   | 特性                   | 为什么                                                                                       | 工作量 |
| --- | -------------------- | ----------------------------------------------------------------------------------------- | --- |
| 17  | `SGLangServerEngine` | Target 作为 HTTP 服务。解锁针对 671B target 在独立 GPU 上的 online 训练。Phase-2 `TargetEngine` 接口的新实现。    | M-H |
| 18  | `RemoteStream`       | 与 `SGLangServerEngine` 通信的 stream 后端。`prefetch_factor` + `max_in_flight` 做 back-pressure。 | M   |
| 19  | VLM 统一               | 删除 `QwenVLOnlineEagle3Model`。VLM 走 `TargetEngine` 中 typed 的 `MediaInputs` + 数据流水线。        | M   |
| 20  | 新增 parser            | 从 TorchSpec port `KimiK25Parser`、`MiniMaxParser`。                                         | S   |


**交付物**：用独立 SGLang server 集群上的 target 训练 DeepSeek-V3 的 Eagle3。

### Phase 5：打磨（Week 12-13）


| #   | 特性                                  | 为什么                                                                 | 工作量 |
| --- | ----------------------------------- | ------------------------------------------------------------------- | --- |
| 21  | 与 backbone 解耦的 DFlash               | 把 `DFlashDraftModel` 分解，让 MLP/Norm/RoPE 参数化。当前硬编码到 Qwen3。           | S-M |
| 22  | `torch.compile` 支持                  | 可选的 `compile_model` flag，用于 draft model 编译。                         | S   |
| 23  | `defer_tokenization` + 动态 loss mask | 在 fetch 时 tokenize，而不是在预处理时。启用动态 loss mask。                         | M   |
| 24  | Tokenization 磁盘 cache               | 按 data path + template + max_length 做 key 缓存 tokenized 数据集。         | S   |
| 25  | WSD learning rate scheduler         | Warmup-Stable-Decay。对长 run / continued pretraining 有用；如果没有具体需求就先延期。 | S   |
| 26  | 移除 legacy `scripts/legacy/` shim    | 在两个 minor release 都带 `DeprecationWarning` 后。具体目标：SpecForge 0.X。     | S   |


### Phase 6：可选 / 未来


| #   | 特性                         | 为什么                                                          | 工作量 |
| --- | -------------------------- | ------------------------------------------------------------ | --- |
| 27  | FSDP2 实现                   | 在 Phase-2 接缝后面填充 `_apply_fsdp2`。如果 `torch.compile` 成为默认就必须有。 | M   |
| 28  | Mooncake streaming backend | 多节点解耦下的 GPU↔GPU tensor 传输。仅当 profiling 显示 HTTP/gRPC 是瓶颈时再做。  | H   |
| 29  | Train-with-decode 模式       | 训练期间周期性把 draft 权重同步进 SGLang 推理 engine。                       | H   |
| 30  | Ray orchestration          | 用 Ray placement group 分离 training 和 inference 的 GPU 池。       | H   |
| 31  | FA4 attention backend      | 给 draft attention 加 FlashAttention 4 支持。                     | S-M |


---

## 6. 值得标注的权衡

### Ray + Mooncake vs HTTP/gRPC

TorchSpec 从 Ray + Mooncake 里获益良多，但它们是依赖很重、运维复杂度很高的组件。对 SpecForge，先用 HTTP/gRPC 到 SGLang server 做跨节点训练 — 这覆盖大多数场景（target 在专用 GPU、trainer 在其他 GPU），表面积只是一小块。仅当 profiling 显示传输是瓶颈时再加 Mooncake。

### Plugin registry vs HF AutoModel 模式

HF 的 "扔一个 config JSON 进来" 是不错的便利。两个都保留：装饰器 registry 是主 dispatch，HF type dispatch 作为 fallback 用于声明了已知 `transformers` config class 的 config。`AutoDraftModelConfig.from_file()` 先按 architecture 名查 `DRAFT_REGISTRY`，回退到 config type mapping。

### Strategy 模式是一层间接

如果永远只有 EAGLE3 和 DFlash，那两个 script 就够了。这里赌的是 Medusa、MTP、或混合变种会想要同一份 trainer 基础设施 — 真是这样的话，strategy 会很快回本。如果不是，代价不过是多一层 dispatch，不损可读性。

### Pydantic vs OmegaConf

Pydantic 的 validation 和 serialization 更好。OmegaConf 的 YAML merge + CLI override（`--training.lr=1e-4`）几乎免费。可选项：

1. **Pydantic + 自定义 CLI overlay**（解析 `--key=value` 参数，从合并后的 dict `model_validate`） — 我们的选择。
2. OmegaConf + dataclass（TorchSpec 的做法）。
3. Pydantic + typer/click 做 CLI。

我们选 (1)，因为 SpecForge 已经在用 Pydantic（`ChatTemplate`），现阶段 validation 比 merge 的便利更重要。CLI overlay 是大约 30 行代码。

### MLA cache：compressed vs expanded

Eagle3 TTT unroll 期间 KV cache 的两种选择：

- **Compressed**：cache `kv_compressed`（低秩）+ `k_rope_raw`。省显存但要求 `core/eagle3.py` 里有 MLA 专属 cache 逻辑。
- **Expanded**：在投影之后 cache 完整 `(K, V)`。显存多一些，但 `core/eagle3.py` 与架构无关。

我们选 **expanded**（与 TorchSpec 一致）。TTT unroll 通常 5-7 步、序列很短 — compressed cache 的显存节约是边际的，而保持 `core/eagle3.py` 与架构无关更有价值。

---

## 7. 迁移路径

每个 phase 独立交付，能在不破坏现有用户的前提下被验证。

### Phase 1 计划

1. 新增 `models/drafts/__init__.py`，里面放 `DRAFT_REGISTRY` 和 `@register_draft`。
2. 把 `LlamaForCausalLMEagle3` 移到 `models/drafts/llama_eagle3.py`，用 `@register_draft("LlamaForCausalLMEagle3")` 装饰。
3. 从 TorchSpec port `deepseek_eagle.py` → `models/drafts/deepseek_eagle3.py`，用 `@register_draft("Eagle3DeepseekV2ForCausalLM")` 装饰。
4. 改 `AutoDraftModelConfig.from_file()`，让它先查 `DRAFT_REGISTRY`。
5. 改 `AutoEagle3DraftModel.from_config()`，让它先查 `DRAFT_REGISTRY`。
6. 新增 MLA draft 配置（`qwen3_8b_eagle3_mla.json`、`kimi_k25_eagle3_mla.json`）。
7. 老的 `_model_mapping` / `_config_mapping` 作为 fallback 保留 — 以后再移除。
8. 测试：用现有 `train_eagle3.py` 训练 MLA Eagle3 draft — script 零改动。

### Phase 2 计划

1. 在 `models/targets/base.py` 定义 `TargetEngine` protocol。
2. 把现有 `SGLangEagle3TargetModel` → `SGLangTargetEngine`（进程内）。
3. 把现有 `HFEagle3TargetModel` → `HFTargetEngine`。
4. 把现有 `CustomEagle3TargetModel` → `CustomTargetEngine`。
5. 定义 `HiddenStateStream` protocol + `OnlineStream` + `OfflineStream`。`seek()` 方法对 resume 正确性是必需的。`OnlineStream` 暴露 `prefetch_factor`。
6. 新建 `training/trainer.py`，里面是 `Trainer` class。Trainer 只消费 `TrainBatch`，永远不直接拿 target model。
7. 从 `train_eagle3.py` 的逻辑里抽出 `Eagle3TTTStrategy`。实现 `fsdp_wrap_policy()`。
8. 从 `train_dflash.py` 的逻辑里抽出 `DFlashBlockStrategy`。实现 `fsdp_wrap_policy()`。
9. 加入按 `fsdp_version` dispatch 的 `apply_fsdp` 接缝（暂时只有 FSDP1）。
10. 实现 `CheckpointManager` — `save` / `load` / `update_best` / `_rotate`。
11. 实现 `Evaluator`（per-position acc 跨 batch 聚合）+ `EvalCache`（完整 key 的 MD5）。
12. 把旧 script 移到 `scripts/legacy/`，写 shim 来实例化 `Trainer`。
13. **数值等价测试（§10）**：legacy shim 在固定 seed 下，第 100/500/1000 步的 loss 曲线必须在容忍度内匹配重构前的 script。Gate。

### Phase 3 计划

1. 在 `config/schema.py` 用 Pydantic（Literal-typed 枚举）定义。
2. 实现 `config/loader.py`（YAML load + CLI override）。
3. 实现 `cli.py`（`specforge train|prepare|export|eval`）。
4. 通过读当前 SGLang spec-decoding loader，敲定 SGLang weight-name map（`docs/export_weight_map_mla.md`）。#5 被它阻塞。
5. 实现 `export/to_sglang.py` 和 `export/to_hf.py`。
6. 用新 schema 把现有 JSON config 机械化迁移到 YAML。
7. 测试：`specforge train --config x.yaml` 端到端跑通。Phase 1 训出的 MLA draft 能在 SGLang server 中加载，并跑出非平凡的接受率。

### Phase 4 计划

1. 实现 `SGLangServerEngine`（连接 SGLang server 的 HTTP client）。复用 Phase-2 的 `TargetEngine` 接口 — trainer 不变。
2. 实现 `RemoteStream`（`prefetch_factor`、`max_in_flight`）。复用 Phase-2 的 `HiddenStateStream` 接口 — trainer 不变。
3. 在 `TargetEngine.generate_train_data` 加入 typed `MediaInputs`。
4. 删除 `QwenVLOnlineEagle3Model`；VLM 走 `MediaInputs` + 数据流水线。
5. 从 TorchSpec port `KimiK25Parser`、`MiniMaxParser`。
6. 测试：用独立节点上的 SGLang server 做 online 训练；结果在容忍度内与进程内训练（同一份 target 权重）一致。

---

## 8. 非目标（显式列出）

- **不引入 Ray 依赖。** SpecForge 保持 torchrun 原生。
- **Phase 1-5 不引入 Mooncake。** 走 HTTP/gRPC 到 SGLang 足以做解耦。
- **不支持 vLLM target backend。** SGLang 是主选；HF 是参考。通过 `TargetEngine` 接口加 vLLM 是可行的，但不在优先级里。
- **不做多 engine 负载均衡。** 一个训练 job 对应一个 target engine 实例。扩缩容走 SGLang server 自己的能力（比如多个 TP worker）。
- **Phase 1-5 不做 train-with-decode。** 有用但复杂；延到 Phase 6。

---

## 9. 成功标准


| Phase | 标准                                                                                                   |
| ----- | ---------------------------------------------------------------------------------------------------- |
| 1     | MLA Eagle3 draft 在 Qwen3-8B 上能训练并收敛到与 Llama draft 可比的 loss。Kimi-K2.5 的 TP/SP 冒烟测试通过。                 |
| 2     | Legacy shim（内部走新 `Trainer`）在数值等价容忍度（§10）内匹配重构前的 script。Eval 指标与手工 eval 一致。                           |
| 3     | `specforge train --config x.yaml` 端到端跑通。Phase 1 训出的 MLA draft 能在 SGLang 中加载，产出非平凡的接受率。               |
| 4     | 用独立 SGLang server 上的 671B target 做 online 训练；在使用相同 target 权重的情况下，per-step loss 在与进程内 baseline 的容忍度内。 |
| 5     | DFlash 跑在 Llama backbone；tokenization cache 让数据准备时间下降 >50%；legacy shim 已移除。                          |


---

## 10. 测试策略（gating，不是 nice-to-have）

每个修改训练路径的 phase，都必须对照前一个 commit 通过数值等价 gate。把这些当作 CI 要求，不是可选项 — 行为保持的重构是无声回归最常潜伏的地方。

### 10.1 数值等价 gate（Phase 2、4 关键）

在固定 seed 和固定 micro-batch 序列（3 个 batch × 4 个 sample 就够）下，新代码路径必须匹配旧的：


| 指标                                          | 容忍度                                      | 检查 step            |
| ------------------------------------------- | ---------------------------------------- | ------------------ |
| Per-step 训练 loss                            | `atol=1e-4, rtol=1e-4`（BF16 master copy） | 0、1、100、500、1000   |
| Per-position eval acc（长度为 `ttt_length` 的向量） | `atol=1e-3`                              | step 1000 末尾的 eval |
| `simulated_acc_len`                         | `atol=1e-3`                              | step 1000 末尾的 eval |
| 模型 state dict（选定 key）                       | `atol=1e-4`                              | step 100 之后        |


Gate 运行使用：Llama Eagle3 draft、Qwen3-8B target、offline 模式、sharegpt eval 切片。便宜到可以在每个触及 `core/`、`training/`、`models/drafts/` 的 PR 上跑。

### 10.2 冒烟测试（每个 phase）

- 每个 draft 架构一份 config（`llama_eagle3`、`deepseek_v3_eagle3`、`dflash_qwen3`），在 TP=1、TP=2、和 TP=2+SP=2 下训 20 步不崩。
- Checkpoint save + resume 产生与不 resume 一致的 loss 曲线（验证 `stream.seek()`）。
- Eval cache miss + hit 产生相同的指标（验证 cache key 集合）。

### 10.3 分布式正确性（Phase 1 + 2）

- 在 Kimi-K2.5 上跑 MLA + Yunchang USP，且 `qk_nope_head_dim != qk_rope_head_dim != v_head_dim`。这是 Phase 1 的风险项，必须在关键路径上，不是 Phase 4 的 wishlist。
- Gradient accumulation：确认每个 `optimizer.step()` 一次 `all_reduce`，而不是每个 `backward()` 一次。用 NCCL communicator log 或 `torch.profiler` 跑一次确认。

### 10.4 Export-loop 测试（Phase 3）

- 训练 MLA draft 100 步，export 到 SGLang，在 SGLang server 中加载，跑 32 个生成请求。接受率 > 0（即 loader 真的消费了权重，而不是零）。这能在集成边界上抓到 weight-name-map 的回归。

