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

2. Run training (example):

```bash
python sft.py --model_path /path/to/model --train_file train_sft_lf.jsonl --output_dir output
```
