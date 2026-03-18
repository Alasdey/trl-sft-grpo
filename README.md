# trl-sft-grpo — Event Relation Extraction with SFT & GRPO

Fine-tune language models on the [MAVEN-ERE](https://github.com/THU-KEG/MAVEN-ERE)
event-relation-extraction task using **Supervised Fine-Tuning (SFT)** followed by
**Group Relative Policy Optimization (GRPO)** — all powered by
[TRL](https://huggingface.co/docs/trl) with optional
[vLLM](https://github.com/vllm-project/vllm) generation acceleration.

---

## Table of Contents

- [Overview](#overview)
- [Task](#task)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [SFT Training](#sft-training)
- [GRPO Training](#grpo-training)
- [Reward Functions](#reward-functions)
- [vLLM Acceleration](#vllm-acceleration)
- [Tips & Known Issues](#tips--known-issues)

---

## Overview

```
Raw dataset
    │
    ▼
[train_sft.py]  ──►  SFT checkpoint  (learns the answer format via teacher-forcing)
    │
    ▼
[train_grpo.py] ──►  GRPO checkpoint (maximises F1 reward via RL)
```

The pipeline follows the standard post-training recipe used in models like
DeepSeek-R1: SFT first to warm up the model on chain-of-thought examples, then
GRPO to align the model's outputs to the task metric (relation F1).

---

## Task

The model must identify **temporal**, **causal**, **subevent**, and
**coreference** relations between events annotated in a document.

**Relations:**

| Category | Subtypes |
|---|---|
| Temporal | `BEFORE`, `OVERLAP`, `CONTAINS`, `SIMULTANEOUS`, `BEGINS-ON`, `ENDS-ON` |
| Causal | `CAUSE`, `PRECONDITION` |
| Structural | `subevent`, `coreference` |

**Output format:**

```
CAUSE: e1 e3; PRECONDITION: none; BEFORE: e2 e5; ...
```

Wrapped inside `<think>...</think><answer>...</answer>` tags.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourorg/sft-grpo.git
cd sft-grpo

# 2. Create a clean environment
python -m venv .venv && source .venv/bin/activate  # Linux/macOS
# python -m venv .venv && .venv\Scripts\activate   # Windows

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
pip install -r requirements.txt

# 5. (Optional) Install flash-attention for faster training
pip install flash-attn --no-build-isolation
```

> **Minimum GPU requirements**
> | Script | Recommended |
> |---|---|
> | SFT 4B (4-bit LoRA) | 1 × 24 GB GPU |
> | GRPO 4B (bf16 LoRA) | 2 × 40 GB GPU |
> | GRPO 4B + vLLM | 2 × 40 GB GPU |
> | GRPO 70B+ | 8 × 80 GB GPU |

---

## Project Structure

```
sft-grpo/
├── eval.py               # F1 metric & relation extraction helpers
├── train_sft.py          # Supervised fine-tuning (TRL SFTTrainer)
├── train_grpo.py         # GRPO reinforcement learning (TRL GRPOTrainer + vLLM)
├── requirements.txt
└── README.md
```

---

## Quick Start

### Minimal SFT run (single GPU, 4-bit)

```bash
python train_sft.py \
  --model_path Qwen/Qwen3-4B \
  --data_path Nofing/maven-ere-llm-sft-agg \
  --output_dir outputs/sft \
  --bf16 \
  --quantize \
  --data_max_sample 8 \
  --batch_size 8 \
  --output_name "Nofing/qwen3-4B-sft-ere-test"
```

### Minimal GRPO run (continues from SFT checkpoint)

```bash
python train_grpo.py \
  --model_path outputs/sft \
  --data_path Nofing/maven-ere-llm-sft-agg \
  --output_dir outputs/grpo \
  --bf16 \
  --output_name "Nofing/qwen3-4B-grpo-ere-test"
```

---

## SFT Training

`train_sft.py` fine-tunes a causal LM on chain-of-thought + answer pairs
filtered by a minimum accuracy threshold.

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `Qwen/Qwen3-4B` | HF model id or local path |
| `--data_path` | `Nofing/maven-ere-llm-sft-agg` | HF dataset id |
| `--data_max_sample` | `10000` | Max training samples |
| `--output_dir` | `outputs/sft` | Local save directory |
| `--output_name` | `Nofing/qwen3-4B-sft-ere` | Hub repo name |
| `--hf_token` | *(prompted)* | Hugging Face token |
| `--max_length` | `4096` | Max sequence length |
| `--accuracy_thresh` | `0.0` | Min F1 to keep a sample |
| `--num_train_epochs` | `1` | Epochs |
| `--max_steps` | `-1` | Override epochs if > 0 |
| `--batch_size` | `4` | Per-device batch size |
| `--gradient_accumulation_steps` | `4` | Grad accumulation |
| `--gradient_checkpointing` | `True` | Save VRAM |
| `--learning_rate` | `5e-5` | Peak LR |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--max_grad_norm` | `1.0` | Gradient clipping |
| `--warmup_steps` | `5` | LR warmup |
| `--lr_scheduler_type` | `cosine` | Scheduler |
| `--packing` | `True` | Sequence packing |
| `--fp16` | `False` | FP16 mode |
| `--bf16` | `True` | BF16 mode (recommended) |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--lora_dropout` | `0.1` | LoRA dropout |
| `--quantize` | `True` | 4-bit NF4 quantisation |
| `--logging_steps` | `1` | Log every N steps |
| `--save_steps` | `400` | Save every N steps |
| `--report_to` | `none` | `wandb`, `tensorboard`, or `none` |
| `--push_to_hub` | `True` | Push merged model to Hub |

### Example commands

```bash
# ── Single GPU, minimal VRAM (4-bit quantised) ──────────────────────────────
python train_sft.py \
  --model_path Qwen/Qwen3-4B \
  --data_path Nofing/maven-ere-llm-sft-agg \
  --output_dir outputs/sft \
  --output_name YourHFUser/qwen3-4B-sft-ere \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_length 4096 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --bf16 \
  --quantize \
  --push_to_hub

# ── Multi-GPU with accelerate ────────────────────────────────────────────────
accelerate launch --num_processes 4 train_sft.py \
  --model_path Qwen/Qwen3-4B \
  --data_path Nofing/maven-ere-llm-sft-agg \
  --output_dir outputs/sft \
  --batch_size 8 \
  --gradient_accumulation_steps 2 \
  --bf16 \
  --no_quantize

# ── With WandB logging ───────────────────────────────────────────────────────
python train_sft.py \
  --model_path Qwen/Qwen3-4B \
  --data_path Nofing/maven-ere-llm-sft-agg \
  --output_dir outputs/sft \
  --report_to wandb \
  --bf16 \
  --quantize

# ── Quality filtering: keep only high-quality CoT samples ───────────────────
python train_sft.py \
  --model_path Qwen/Qwen3-4B \
  --data_path Nofing/maven-ere-llm-sft-agg \
  --output_dir outputs/sft-filtered \
  --accuracy_thresh 0.5 \
  --bf16 \
  --quantize
```

---

## GRPO Training

`train_grpo.py` uses `GRPOTrainer` to further align the SFT model using
RL with two reward signals: **relation F1** (primary) and **format compliance**
(auxiliary).

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `Qwen/Qwen3-4B` | SFT checkpoint or base model |
| `--data_path` | `Nofing/maven-ere-llm-sft-agg` | HF dataset id |
| `--data_max_sample` | `10000` | Max training samples |
| `--output_dir` | `outputs/grpo` | Local save directory |
| `--output_name` | `Nofing/qwen3-4B-grpo-ere` | Hub repo name |
| `--hf_token` | *(prompted)* | Hugging Face token |
| `--num_train_epochs` | `1` | Epochs |
| `--max_steps` | `-1` | Override epochs if > 0 |
| `--batch_size` | `2` | Per-device batch size |
| `--gradient_accumulation_steps` | `8` | Grad accumulation |
| `--gradient_checkpointing` | `True` | Save VRAM |
| `--learning_rate` | `1e-6` | Peak LR (lower than SFT) |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--max_grad_norm` | `0.1` | Gradient clipping |
| `--warmup_steps` | `10` | LR warmup |
| `--lr_scheduler_type` | `cosine` | Scheduler |
| `--num_generations` | `8` | GRPO group size $G$ |
| `--max_prompt_length` | `2048` | Truncate prompt to N tokens |
| `--max_completion_length` | `2048` | Max generated tokens |
| `--temperature` | `0.9` | Sampling temperature |
| `--top_p` | `0.95` | Nucleus sampling |
| `--beta` | `0.001` | KL penalty coefficient |
| `--no_std_norm` | `False` | Disable group std normalisation* |
| `--use_vllm` | `False` | Enable vLLM generation |
| `--vllm_server_host` | `0.0.0.0` | vLLM server host |
| `--vllm_server_port` | `8000` | vLLM server port |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--lora_dropout` | `0.05` | LoRA dropout |
| `--fp16` | `False` | FP16 mode |
| `--bf16` | `True` | BF16 mode |
| `--quantize` | `False` | 4-bit quantisation |
| `--logging_steps` | `1` | Log every N steps |
| `--save_steps` | `100` | Save every N steps |
| `--report_to` | `none` | `wandb`, `tensorboard`, or `none` |
| `--push_to_hub` | `False` | Push to Hub after training |

> *`--no_std_norm` disables group standard-deviation normalisation in the GRPO
> advantage computation. [Bereket & Leskovec (2025)](https://arxiv.org/abs/2508.11800)
> show this prevents overconfidence when rewards are stochastic. ERE F1 is
> deterministic, so the default (normalisation on) is appropriate here.

### Example commands

```bash
# ── Standard GRPO from SFT checkpoint (single node, 2 GPU) ──────────────────
accelerate launch --num_processes 2 train_grpo.py \
  --model_path outputs/sft \
  --data_path Nofing/maven-ere-llm-sft-agg \
  --output_dir outputs/grpo \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_generations 8 \
  --learning_rate 1e-6 \
  --beta 0.001 \
  --max_completion_length 2048 \
  --bf16

# ── GRPO + vLLM acceleration ─────────────────────────────────────────────────
# Step 1: launch a vLLM server in a separate terminal
python -m vllm.entrypoints.openai.api_server \
  --model outputs/sft \
  --port 8000 \
  --dtype bfloat16

# Step 2: run GRPO pointing at the server
python train_grpo.py \
  --model_path outputs/sft \
  --data_path Nofing/maven-ere-llm-sft-agg \
  --output_dir outputs/grpo-vllm \
  --use_vllm \
  --vllm_server_host 0.0.0.0 \
  --vllm_server_port 8000 \
  --batch_size 4 \
  --num_generations 8 \
  --bf16

# ── GRPO with larger group size for better variance reduction ────────────────
python train_grpo.py \
  --model_path outputs/sft \
  --data_path Nofing/maven-ere-llm-sft-agg \
  --output_dir outputs/grpo-g16 \
  --num_generations 16 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --bf16

# ── WandB logging ────────────────────────────────────────────────────────────
python train_grpo.py \
  --model_path outputs/sft \
  --data_path Nofing/maven-ere-llm-sft-agg \
  --output_dir outputs/grpo \
  --report_to wandb \
  --bf16

# ── Push final model to Hub ───────────────────────────────────────────────────
python train_grpo.py \
  --model_path outputs/sft \
  --data_path Nofing/maven-ere-llm-sft-agg \
  --output_dir outputs/grpo \
  --output_name YourHFUser/qwen3-4B-grpo-ere \
  --push_to_hub \
  --bf16
```

---

## Reward Functions

Two reward signals are combined additively by `GRPOTrainer`:

| Function | Range | Purpose |
|---|---|---|
| `reward_f1` | $[0, 1]$ | Token-level F1 between predicted and gold relation sets |
| `reward_format` | $\{0.0, 0.1\}$ | Bonus for well-formed `<think>`/`<answer>` tags |

The F1 reward is computed relation-type-by-relation-type and averaged,
rewarding both precision and recall across all 10 relation types.

---

## vLLM Acceleration

Enabling `--use_vllm` routes generation through a vLLM server, providing
significant throughput improvements when `num_generations` is large.
TRL's `GRPOTrainer` supports both an external vLLM server and a co-located
engine (TRL ≥ 0.16). See the [TRL docs](https://huggingface.co/docs/trl) for
the latest co-located vLLM setup.

```
Without vLLM:  generation ≈ 70–80% of total GRPO wall-clock time
With vLLM:     2–4× throughput improvement on generation stage
```

---

## Tips & Known Issues

- **OOM during GRPO**: reduce `--num_generations` or `--max_completion_length`
  first; then reduce `--batch_size`.
- **SFT dataset field mismatch**: the `SFTTrainer` uses `dataset_text_field="text"`.
  Ensure `format_sample` in `train_sft.py` returns a `"text"` key.
- **Quantize + vLLM**: 4-bit quantisation and vLLM are not compatible. Use
  `--no_quantize` when `--use_vllm` is set.
- **`pad_token` warning**: automatically handled by setting `pad_token = eos_token`.
- **GRPO overconfidence**: if your reward signal is stochastic, pass
  `--no_std_norm` to disable group standard-deviation normalisation
  ([arxiv:2508.11800](https://arxiv.org/abs/2508.11800)).