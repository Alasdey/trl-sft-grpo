"""
Supervised Fine-Tuning (SFT) script for ERE using TRL + vLLM-compatible setup.

Replaces the original unsloth-based train_supervised_unsloth.py with
a fully TRL-native implementation using SFTTrainer and PEFT LoRA.
"""

import os
import shutil
import logging
import argparse
import getpass
from datetime import datetime

import torch
from datasets import load_dataset, Dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from eval import accuracy_reward

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
COT_PROMPT = (
    "You are an assistant trained to process any text and extract event relations from it.\n"
    "Your task is to analyze user-provided text labeled relevant entities, to infer meaningful "
    "relationships between them.\n"
    "You need to formulate your reasoning process and encapsulate it in <think> </think> tag, "
    "and the answer between <answer> </answer>."
)

ERE_PROMPT = (
    "The current task is an event relation extraction task, which aims to identify temporal, "
    "causal, subevent and coreference relations among events in texts. "
    "The temporal relation between events refers to the chronological order in which they occur, "
    "involving six subtypes, namely, SIMULTANEOUS, ENDS-ON, BEGINS-ON, OVERLAP, CONTAINS, and BEFORE. "
    "The causal relation between events refers to the causality dependencies between the events, "
    "with two subtypes, CAUSE and PRECONDITION. "
    "A SUBEVENT relation denotes that an event is a part of another event. "
    "The COREFERENCE relation denotes that two event mentions are mentions of the same event. "
    "In the provided document, event mentions are annotated within angle brackets (<>). "
    "The desired outcome is a list of events in the document that have relations with the given event. "
    "The prescribed output format should follow this structure: "
    "'relation1: event1 event2; relation2: event3 event4'. "
    "The output 'relation: none' indicates that the given event lacks this particular type of "
    "relation with other events."
)

SYSTEM_PROMPT = COT_PROMPT + "\n\n" + ERE_PROMPT


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def save_code(dest_folder: str = "./trainer_output") -> None:
    """Copy this script into dest_folder with a timestamp suffix."""
    os.makedirs(dest_folder, exist_ok=True)
    current_file = os.path.abspath(__file__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(os.path.basename(current_file))
    dest_path = os.path.join(dest_folder, f"{name}_{timestamp}{ext}")
    shutil.copy2(current_file, dest_path)
    logger.info("Saved script snapshot to %s", dest_path)


def format_sample(item: dict, tokenizer) -> dict:
    """
    Convert a raw dataset row into a chat-templated 'text' field
    that SFTTrainer can consume directly via dataset_text_field.
    """
    user_text = item["extraction_instruction"][1]["content"]
    cot = item["chain_of_thought"]
    answer = item["answer"]

    # Keep only the <think>...</think> portion of the CoT then append the answer
    think_block = cot.split("</think>")[0] + "</think>"
    assistant_content = think_block + answer

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "text": tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def build_dataset(
    dataset_name: str,
    tokenizer,
    accuracy_thresh: float,
    data_max_sample: int,
) -> Dataset:
    """Load, filter by accuracy threshold, format, and subsample the dataset."""
    raw = load_dataset(dataset_name)["train"]
    logger.info("Loaded %d raw training samples.", len(raw))

    # Quality filter
    raw = raw.filter(
        lambda s: accuracy_reward(s["chain_of_thought"], s["answer"]) >= accuracy_thresh
    )
    logger.info(
        "After accuracy threshold (>= %.2f): %d samples remaining.",
        accuracy_thresh,
        len(raw),
    )

    # Subsample
    raw = raw.select(range(min(len(raw), data_max_sample)))
    logger.info("Using %d samples for training.", len(raw))

    # Apply chat template formatting
    formatted = Dataset.from_list(
        [format_sample(item, tokenizer) for item in raw]
    )
    logger.info("Dataset formatted. Example:\n%s", formatted[0]["text"][:300])
    return formatted


def build_bnb_config(args) -> BitsAndBytesConfig | None:
    """Return a BitsAndBytesConfig for 4-bit quantization, or None."""
    if not args.quantize:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(args) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args) -> None:
    logger.info("Configuration: %s", vars(args))
    save_code("trainer_output")
    save_code("logs")

    # HF login
    hf_token = args.hf_token or getpass.getpass("Enter your HF_TOKEN: ")
    os.environ["HF_TOKEN"] = hf_token
    login(token=hf_token)

    # Precision flags
    use_bf16 = args.bf16
    use_fp16 = args.fp16 and not use_bf16

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_dataset = build_dataset(
        args.data_path,
        tokenizer,
        args.accuracy_thresh,
        args.data_max_sample,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    bnb_config = build_bnb_config(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    lora_config = build_lora_config(args)

    # ------------------------------------------------------------------
    # SFT Training
    # ------------------------------------------------------------------
    sft_config = SFTConfig(
        # Data
        dataset_text_field="text",
        max_length=args.max_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=args.packing,
        # Batch / steps
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        # Scheduler / optimiser
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        optim="adamw_8bit",
        max_grad_norm=args.max_grad_norm,
        # Precision
        fp16=use_fp16,
        bf16=use_bf16,
        # Logging / saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    logger.info("Starting SFT training...")
    trainer_stats = trainer.train()
    logger.info("Training complete. Stats: %s", trainer_stats)

    # ------------------------------------------------------------------
    # Save / push
    # ------------------------------------------------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model saved locally to %s", args.output_dir)

    if args.push_to_hub:
        trainer.model.push_to_hub(args.output_name)
        tokenizer.push_to_hub(args.output_name)
        logger.info("Model pushed to Hub as %s", args.output_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFT training for Event Relation Extraction (ERE) using TRL."
    )

    # Model / data
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--data_path", type=str, default="Nofing/maven-ere-llm-sft-agg")
    parser.add_argument("--data_max_sample", type=int, default=10_000)
    parser.add_argument("--output_dir", type=str, default="outputs/sft")
    parser.add_argument("--output_name", type=str, default="Nofing/qwen3-4B-sft-ere")
    parser.add_argument("--hf_token", type=str, default=None)

    # Training
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--accuracy_thresh", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--packing", action="store_true", default=True)
    parser.add_argument("--dataset_num_proc", type=int, default=4)

    # Precision
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Misc
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--report_to", type=str, default="none", help="'wandb', 'tensorboard', or 'none'")
    parser.add_argument("--push_to_hub", action="store_true", default=True)

    args = parser.parse_args()
    main(args)