# trl-sft-grpo
## Install
```
git clone https://github.com/Alasdey/trl-sft-grpo
cd trl-sft-grpo
uv sync
```

## Run
```
CUDA_VISIBLE_DEVICES=0  uv run train_sft.py   --model_path Qwen/Qwen3-0.5B   --output_dir outputs/sft   --quantize --bf16 --output_name "Nofing
/qwen3-0.5B-sft-ere"

uv run train_grpo.py   --model_path outputs/sft   --output_dir outputs/grpo  --output_name "Nofing/qwen3-0.8B-sft-grpo-
test" --model_path "Nofing/qwen3-0.6B-sft-ere" --data_max_sample 10
```

## Todo
Review some of the args that are broken (stuck on True)
Figure out max model size
Run full SFT
Run full GRPO