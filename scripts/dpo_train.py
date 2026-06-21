#!/usr/bin/env python3
"""Train a model with Direct Preference Optimization (DPO)."""
import argparse
import json
import os
from typing import List

import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

from e_customer_service.data import format_messages
from e_customer_service.paths import build_run_paths, default_run_name, ensure_run_dirs, write_json


def read_pairs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def messages_to_text(messages: List[dict], tokenizer, *, add_generation_prompt: bool = False):
    return format_messages(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def build_dataset(dpo_file: str, tokenizer):
    examples = []
    for obj in read_pairs(dpo_file):
        prompt_msgs = obj.get("prompt") or obj.get("messages") or []
        chosen_msgs = obj.get("chosen") or obj.get("best") or []
        rejected_msgs = obj.get("rejected") or obj.get("worst") or []

        prompt = messages_to_text(prompt_msgs, tokenizer, add_generation_prompt=True)
        chosen = messages_to_text(chosen_msgs, tokenizer)
        rejected = messages_to_text(rejected_msgs, tokenizer)

        examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    return examples


def create_bnb_config(args):
    if not args.qlora:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="Train DPO from an SFT adapter")
    p.add_argument("--dpo-file", default="dpo_pairs.jsonl")
    p.add_argument("--output-root", default="output", help="Root directory for all experiment runs")
    p.add_argument("--run-name", default=None, help="Experiment run name under output/runs")
    p.add_argument(
        "--adapter-dir",
        default=None,
        help="SFT adapter dir; defaults to output/runs/<run-name>/sft/final_adapter",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Deprecated: use --output-root and --run-name instead; treated as a run name if set",
    )
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--bnb-4bit-quant-type", default="nf4")
    p.add_argument("--bnb-4bit-compute-dtype", default="bfloat16")
    p.add_argument("-dq", "--bnb-4bit-use-double-quant", action="store_true")
    args = p.parse_args(argv)

    run_name = args.run_name or (
        os.path.basename(os.path.normpath(args.output_dir)) if args.output_dir else default_run_name()
    )
    paths = build_run_paths(args.output_root, run_name)
    ensure_run_dirs(paths)

    adapter_dir = args.adapter_dir or str(paths["sft_final_adapter_dir"])

    print("Loading SFT adapter:", adapter_dir)
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        is_trainable=True,
        device_map=args.device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=create_bnb_config(args),
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    dataset = build_dataset(args.dpo_file, tokenizer)
    try:
        dataset = Dataset.from_list(dataset)
    except Exception as e:
        raise SystemExit(f"无法将生成的 list 转换为 datasets.Dataset，请安装 datasets 库。错误：{e}") from e

    try:
        from trl import DPOConfig, DPOTrainer
    except Exception as e:
        raise SystemExit(f"无法从 trl 导入 DPOTrainer/DPOConfig，请确认已安装支持 DPO 的 trl 版本。错误：{e}") from e

    dpo_args = DPOConfig(
        output_dir=os.path.abspath(paths["dpo_checkpoints_dir"]),
        logging_dir=os.path.abspath(paths["dpo_logs_dir"]),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        beta=args.beta,
        num_train_epochs=args.epochs,
        logging_steps=3,
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        save_total_limit=1,
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        report_to="none",
    )

    write_json(
        paths["dpo_dir"] / "config.json",
        {
            "stage": "dpo",
            "run_name": run_name,
            "args": vars(args),
            "adapter_dir": adapter_dir,
            "paths": paths,
        },
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    orig_log = getattr(trainer, "log", None)
    if orig_log is not None:

        def _log_wrapper(logs, *args, **kwargs):
            try:
                return orig_log(logs)
            except Exception:
                try:
                    return orig_log(*((logs,) + args), **kwargs)
                except Exception:
                    return None

        trainer.log = _log_wrapper

    print("Trainer created, starting DPO training...")
    trainer.train()

    save_dir = paths["dpo_final_adapter_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print("Saved DPO model to", save_dir)
    except Exception as e:
        print("保存模型失败：", e)


if __name__ == "__main__":
    main()
