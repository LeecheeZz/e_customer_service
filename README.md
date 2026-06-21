# e_customer_service (SFT)

This repository contains a small refactor of the original `sft.py` demo into
a minimal package-style project. The layout mirrors internal structure used in
TRL-style projects: separate modules for data, modeling and training orchestration.

Quick start:

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run SFT training (example):

```bash
python sft.py --model_path /root/autodl-tmp/models/Qwen/Qwen3-8B-Base --train_file train_sft.jsonl --output_root output --run_name qlora_default --qlora
```

Outputs are organized by run:

```text
output/runs/<run_name>/
  sft/checkpoints/
  sft/final_adapter/
  sft/eval/
  sft/logs/
  dpo/checkpoints/
  dpo/final_adapter/
  dpo/eval/
  dpo/logs/
  artifacts/
```

3. Evaluate the SFT adapter:

```bash
python scripts/eval_generate.py --output-root output --run-name qlora_default --qlora
```

4. Train and evaluate DPO from the SFT adapter:

```bash
python scripts/train_dpo.py --output-root output --run-name qlora_default --qlora
python scripts/run_dpo_inference.py --output-root output --run-name qlora_default --qlora
```
