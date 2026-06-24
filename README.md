# e_customer_service

This repository contains SFT and DPO training scripts for the customer-service model workflow. Outputs are organized by experiment run under `output/runs/<run_name>/`.

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run SFT training:

```bash
python sft.py \
  --model-path /root/autodl-tmp/models/Qwen/Qwen3-8B-Instruct \
  --train-file train_sft.jsonl \
  --output-root output \
  --run-name qlora_default \
  --epochs 2 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --qlora
```

SFT saves to:

```text
output/runs/qlora_default/sft/checkpoints/
output/runs/qlora_default/sft/final_adapter/
```

3. Evaluate the SFT adapter:

```bash
python -m scripts.sft_eval \
  --output-root output \
  --run-name qlora_default \
  --val-file val_sft.jsonl \
  --qlora
```

SFT evaluation saves to:

```text
output/runs/qlora_default/sft/eval/sft_eval_outputs.jsonl
```

4. Run DPO training from the SFT adapter:

```bash
python -m scripts.dpo_train \
  --dpo-file dpo_pairs.jsonl \
  --output-root output \
  --run-name qlora_default \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --beta 0.3 \
  --qlora
```

By default, DPO reads the SFT adapter from:

```text
output/runs/qlora_default/sft/final_adapter/
```

DPO saves to:

```text
output/runs/qlora_default/dpo/checkpoints/
output/runs/qlora_default/dpo/final_adapter/
```

5. Evaluate the DPO adapter:

```bash
python -m scripts.dpo_eval \
  --output-root output \
  --run-name qlora_default \
  --val-file val_sft.jsonl \
  --qlora
```

DPO evaluation saves to:

```text
output/runs/qlora_default/dpo/eval/dpo_eval_outputs.jsonl
```


## vLLM Serving

Install vLLM in the deployment environment:

```bash
pip install -r requirements-vllm.txt
```

Serve the SFT LoRA adapter with the OpenAI-compatible vLLM server:

```bash
python -m scripts.vllm_serve \
  --base-model /root/autodl-tmp/models/Qwen/Qwen3-8B-Instruct \
  --output-root output \
  --run-name qlora_qkvo_r64 \
  --stage dpo \
  --lora-name customer-service \
  --max-lora-rank 64 \
  --trust-remote-code
  --max-model-len 4096 
  --gpu-memory-utilization 0.80
```

For a DPO adapter, change `--stage sft` to `--stage dpo`. The request model name is the LoRA name, so clients should use `customer-service`.

Smoke-test one request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "customer-service",
    "messages": [
      {"role": "user", "content": "我的快递显示已签收，但我没有收到货，该怎么办？"}
    ],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 256
  }'
```

Validate the service on the project validation set:

```bash
python -m scripts.vllm_validate \
  --base-url http://localhost:8000 \
  --model customer-service \
  --output-root output \
  --run-name qlora_qkvo_r64 \
  --stage dpo \
  --val-file val_sft.jsonl \
  --limit 100 \
  --out-file vllm_eval.jsonl
```

If you already generated a Transformers reference output, compare against it:

```bash
python -m scripts.vllm_validate \
  --base-url http://localhost:8000 \
  --model customer-service \
  --output-root output \
  --run-name qlora_default \
  --stage sft \
  --val-file val_sft.jsonl \
  --reference-file output/runs/qlora_default/sft/eval/sft_eval_outputs.jsonl \
  --limit 10
```

## Output Layout

```text
output/runs/<run_name>/
  config.json
  data_manifest.json
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

Use a different `--run-name` for each experiment you want to keep separate.
