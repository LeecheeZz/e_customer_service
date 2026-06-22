import argparse
import logging
import os
import random
from typing import List, Optional

import numpy as np
import torch
from transformers import BitsAndBytesConfig, set_seed

from .data import format_messages, load_jsonl, samples_to_dataset
from .modeling import load_model_and_tokenizer
from .paths import build_run_paths, default_run_name, ensure_run_dirs, write_json
from .trainer import create_peft_config, create_sft_config, run_training

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT training wrapper")
    p.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen/Qwen3-8B-Instruct")
    p.add_argument("--train-file", default="train_sft.jsonl")
    p.add_argument("--output-root", default="output", help="Root directory for all experiment runs")
    p.add_argument("--run-name", default=None, help="Experiment run name under output/runs")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Deprecated: use --output-root and --run-name instead; treated as a run name if set",
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--r", type=int, default=64, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="LoRA target modules, separated by spaces or commas",
    )
    p.add_argument("--no-bf16", action="store_false", dest="bf16", help="Use fp16 instead of bf16")
    p.add_argument(
        "--no-local-files-only",
        action="store_false",
        dest="local_files_only",
        help="Allow loading model/tokenizer files from remote sources",
    )
    p.set_defaults(bf16=True, local_files_only=True)
    p.add_argument("--qlora", action="store_true", help="Enable QLoRA 4-bit loading (bitsandbytes)")
    return p.parse_args(argv)


def setup_seed(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    if args.output_dir:
        return os.path.basename(os.path.normpath(args.output_dir))
    return default_run_name(qlora=args.qlora)


def normalize_target_modules(target_modules: Optional[List[str]]) -> Optional[List[str]]:
    if target_modules is None:
        return None

    modules = []
    for item in target_modules:
        modules.extend(module.strip() for module in item.split(",") if module.strip())
    return modules


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    setup_seed(args.seed)

    run_name = resolve_run_name(args)
    paths = build_run_paths(args.output_root, run_name)
    ensure_run_dirs(paths)

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=False,
        )
        model, tokenizer = load_model_and_tokenizer(
            args.model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            local_files_only=args.local_files_only,
            load_in_4bit=True,
            bnb_config=bnb_config,
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            args.model_path,
            torch_dtype=torch_dtype,
            local_files_only=args.local_files_only,
        )

    samples = load_jsonl(args.train_file)
    logger.info("Loaded samples: %d", len(samples))

    ds = samples_to_dataset(samples, tokenizer)

    try:
        if len(samples) > 0:
            first = samples[0]
            if "messages" in first:
                templated = format_messages(
                    tokenizer,
                    first["messages"],
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                logger.info("First sample templated (raw): %s", templated)
                logger.info("First sample templated (repr): %s", repr(templated))
                logger.info("Contains '<|im_end|>'?: %s", "<|im_end|>" in templated)
    except Exception as e:
        logger.warning("Failed to print templated first sample: %s", e)

    ds = ds.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    peft_config = create_peft_config(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=normalize_target_modules(args.target_modules),
    )
    training_args = create_sft_config(
        output_dir=os.path.abspath(paths["sft_checkpoints_dir"]),
        logging_dir=os.path.abspath(paths["sft_logs_dir"]),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    write_json(
        paths["sft_dir"] / "config.json",
        {
            "stage": "sft",
            "run_name": run_name,
            "args": vars(args),
            "paths": paths,
        },
    )
    write_json(
        paths["data_manifest_path"],
        {
            "train_file": os.path.abspath(args.train_file),
            "num_train_samples_before_split": len(samples),
            "split": {"eval_size": 0.05, "seed": args.seed},
        },
    )

    run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        training_args=training_args,
        output_dir=os.path.abspath(paths["sft_checkpoints_dir"]),
        save_dir=os.path.abspath(paths["sft_final_adapter_dir"]),
    )

    logger.info("SFT checkpoints saved to %s", paths["sft_checkpoints_dir"])
    logger.info("SFT final adapter saved to %s", paths["sft_final_adapter_dir"])


if __name__ == "__main__":
    main()
