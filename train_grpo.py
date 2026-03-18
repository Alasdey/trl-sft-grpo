"""
GRPO training script for ERE using TRL + vLLM.
Converted from notebook grpo.ipynb — unsloth removed, trl-native throughout.
"""

import os
import re
import shutil
import logging
import argparse
import getpass
from datetime import datetime

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

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
SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.
The current task is an event temporal, causal, subevent and coreference relation extraction task, which aims to identify relations among events in texts. 
The relation between events refers to the chronological order in which they occur, the causality linking them, the coreferences in the text and subevent. 
This involves ten subtypes, namely "CAUSE", "PRECONDITION", "coreference", "subevent", "BEFORE", "BEGINS-ON", "CONTAINS", "ENDS-ON", "OVERLAP", and "SIMULTANEOUS".
In the provided document, event trigger words are annotated within angle brackets (<>). The desired outcome is a list of events in the document that have relations with the given event. 
The prescribed output format should follow this structure: 'relation1: event1, event2; relation2: event3, event4'. The output 'relation: none' indicates that the given event lacks this particular type of relation with other events.
Here is an unrelated example of correctly formated answer without the reasoning process:
<answer>CAUSE: none; PRECONDITION: e3; coreference: e7; subevent: none; BEFORE: e28 t16; BEGINS-ON: none; CONTAINS: e40; ENDS-ON: e20; OVERLAP: none; SIMULTANEOUS: none;</answer>
"""

MENTION_PROMPT = """Please identify the events in the document that have relations with the given event """

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def save_code(dest_folder: str = "./trainer_output") -> None:
    os.makedirs(dest_folder, exist_ok=True)
    current_file = os.path.abspath(__file__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(os.path.basename(current_file))
    dest_path = os.path.join(dest_folder, f"{name}_{timestamp}{ext}")
    shutil.copy2(current_file, dest_path)
    logger.info("Saved script snapshot to %s", dest_path)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def data_prep(data: dict) -> dict:
    data["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": MENTION_PROMPT + data["mention"] + "\n" + data["text"]},
    ]
    data["solution"] = [
        {"role": "assistant", "content": data["answer"]},
    ]
    return data


def build_dataset(dataset_name: str, skip: int, max_samples: int):
    dataset = load_dataset(dataset_name)["train"]
    logger.info("Loaded %d raw samples.", len(dataset))

    not_seen_yet = dataset.select(range(skip, len(dataset)))
    subset = not_seen_yet.filter(
        lambda example: len(example["solution"][0]["content"].split("none")) < 11
    )
    subset = subset.shuffle(seed=3407)
    subset = subset.select(range(min(len(subset), max_samples)))
    logger.info("Using %d samples after filtering.", len(subset))
    return subset


# ---------------------------------------------------------------------------
# Rewards  — kept exactly as in the original notebook
# ---------------------------------------------------------------------------
RELATIONS = [
    "CAUSE", "PRECONDITION", "coreference", "subevent",
    "BEFORE", "BEGINS-ON", "CONTAINS", "ENDS-ON", "OVERLAP", "SIMULTANEOUS",
]


def thinking_format_reward(completions, **kwargs):
    """Reward for correct <think>/<answer> structure."""
    pattern = [
        r"^<think>[\S\s]*?</think>\s*<answer>[\S\s]*?</answer>$",
        r"<think>[\S\s]*?</think>",
        r"<answer>[\S\s]*?</answer>",
        r"^<think>",
        r"</think>",
        r"</answer>",
        r"</answer>$",
        r"</think>\s*<answer>",
    ]
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        [re.search(p, content) is not None for p in pattern].count(True)
        for content in completion_contents
    ]
    return [0.1 * match / len(pattern) for match in matches]


def relation_mention_reward(completions, **kwargs):
    """Reward for presence of relation-type names in the output."""
    flag_pattern = r": "
    fps = [r + flag_pattern for r in RELATIONS]
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        [re.search(pattern, content) is not None for pattern in fps].count(True)
        for content in completion_contents
    ]
    return [0.1 * match / len(fps) for match in matches]


def relation_format_reward(completions, **kwargs):
    """Reward for correct relation-value format."""
    flag_pattern = r":((?:\s*?[et][0-9]{0,3})*?|\snone);"
    fps = [r + flag_pattern for r in RELATIONS]
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        [re.search(pattern, content) is not None for pattern in fps].count(True)
        for content in completion_contents
    ]
    return [0.1 * match / len(fps) for match in matches]


def compute_f1(pred_list, gold_list):
    npred = sum([len(set(pred)) for pred in pred_list])
    ngold = sum([len(set(gold)) for gold in gold_list])
    if npred == 0 and ngold == 0:
        return 0.05

    tp, fp, fn = 0, 0, 0
    for pred, gold in zip(pred_list, gold_list):
        tp += len(set(pred) & set(gold))
        fp += len(set(pred) - set(gold))
        fn += len(set(gold) - set(pred))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1


def rel_extract(text):
    """Extract best as possible the relations in the text."""
    flag_pattern = r":((?:\s*?[et][0-9]{1,3})*|\s*?none);?"
    fps = [r + flag_pattern for r in RELATIONS]
    res = []
    for pattern in fps:
        res.append([])
        rel_seg = re.findall(pattern, text)
        for segment in rel_seg:
            res[-1] = re.findall(r"[et][0-9]+", segment)
    return res


def accuracy_reward(completions, **kwargs):
    """Reward based on relation F1 vs ground truth solution."""
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold = rel_extract(solution[0]["content"])
        pred = rel_extract(content)
        rewards.append(compute_f1(pred, gold))
    return rewards


# ---------------------------------------------------------------------------
# Model / LoRA helpers
# ---------------------------------------------------------------------------
def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(lora_rank: int) -> LoraConfig:
    return LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
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
    logger.info("GRPO configuration: %s", vars(args))
    save_code("trainer_output")
    save_code("logs")

    # HF login
    hf_token = args.hf_token or getpass.getpass("Enter your HF_TOKEN: ")
    os.environ["HF_TOKEN"] = hf_token
    login(token=hf_token)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
        "pad_token and eos_token must not be the same. Check your tokenizer config."
    )
    logger.info(
        "pad_token='%s' (id=%d), padding_side='%s'",
        tokenizer.pad_token,
        tokenizer.pad_token_id,
        tokenizer.padding_side,
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = load_dataset(args.data_path)["train"]
    dataset = dataset.map(data_prep)

    not_seen_yet = dataset.select(range(args.dataset_skip, len(dataset)))
    subset = not_seen_yet.filter(
        lambda example: len(example["solution"][0]["content"].split("none")) < 11
    )
    subset = subset.shuffle(seed=3407)
    subset = subset.select(range(min(len(subset), args.data_max_sample)))
    logger.info("Training on %d samples.", len(subset))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        # quantization_config=build_bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ------------------------------------------------------------------
    # vLLM sampling params
    # ------------------------------------------------------------------
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    # ------------------------------------------------------------------
    # GRPO config
    # ------------------------------------------------------------------
    new_model_name = args.output_name or (args.model_path + "-grpo")
    logger.info("Output model name: %s", new_model_name)

    training_args = GRPOConfig(
        # vLLM
        use_vllm=args.use_vllm,
        # vllm_sampling_params=vllm_sampling_params,
        # Generation
        temperature=1.0,
        # max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        # Optimiser
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adamw_8bit",
        # Steps / batch
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        # Logging
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        # Hub
        hub_strategy="every_save",
        hub_token=hf_token,
        hub_model_id=new_model_name,
        push_to_hub=args.push_to_hub,
        seed=3407,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = GRPOTrainer(
        # model=model,
        model=args.model_path,
        processing_class=tokenizer,
        reward_funcs=[
            thinking_format_reward,
            relation_mention_reward,
            relation_format_reward,
            accuracy_reward,
        ],
        args=training_args,
        train_dataset=subset,
        peft_config=build_lora_config(args.lora_rank),
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    # ------------------------------------------------------------------
    # Timestamp log
    # ------------------------------------------------------------------
    with open("time_log.txt", "a") as f:
        f.write("GRPO end - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRPO training for ERE using TRL + vLLM."
    )

    # Model / data
    parser.add_argument("--model_path", type=str, default="Nofing/qwen3-4B-sft-full")
    parser.add_argument("--data_path", type=str, default="Nofing/maven-ere-llm")
    parser.add_argument("--data_max_sample", type=int, default=50_000)
    parser.add_argument("--dataset_skip", type=int, default=0,
                        help="Skip the first N samples (held out for SFT).")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Hub model id. Defaults to model_path + '-grpo'.")
    parser.add_argument("--hf_token", type=str, default=None)

    # Training
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=5)
    parser.add_argument("--max_prompt_length", type=int, default=2800)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)

    # vLLM
    parser.add_argument("--use_vllm", action="store_true", default=False)

    # Misc
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--push_to_hub", action="store_true", default=True)

    args = parser.parse_args()
    main(args)