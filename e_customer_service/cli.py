import argparse
import logging
import os
import random
from typing import List, Optional

import numpy as np
import torch
from transformers import set_seed
from transformers import BitsAndBytesConfig

from .data import load_jsonl, samples_to_dataset
from .modeling import load_model_and_tokenizer
from .trainer import create_peft_config, create_sft_config, run_training


logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT training wrapper")
    p.add_argument("--model_path", default='/root/autodl-tmp/models/Qwen/Qwen3-8B-Base')
    p.add_argument("--train_file", default="train_sft.jsonl")
    p.add_argument("--output_dir", default="output_sft_QLoRA")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-bf16", action="store_false", dest="bf16", help="Use fp16 instead of bf16")
    p.add_argument(
        "--no-local_files_only",
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

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    setup_seed(args.seed)

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # If user enabled QLoRA, reload model with 4-bit config (bitsandbytes)
    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False
        )

        # reload using 4-bit quantization (device_map auto)
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
            local_files_only=args.local_files_only
        )

    samples = load_jsonl(args.train_file)
    logger.info("Loaded samples: %d", len(samples))

    ds = samples_to_dataset(samples, tokenizer)

    # 打印第一条样本在应用 template 后的效果，便于排查训练时是否包含特殊分隔符或多语言噪声
    try:
        if len(samples) > 0:
            first = samples[0]
            if 'messages' in first:
                templated = tokenizer.apply_chat_template(first['messages'], tokenize=False, add_generation_prompt=False, enable_thinking=False)
                logger.info('First sample templated (raw): %s', templated)
                logger.info('First sample templated (repr): %s', repr(templated))
                logger.info("Contains '<|im_end|>'?: %s", '<|im_end|>' in templated)
                # 打印分词前若干 token 供排查
                try:
                    toks = tokenizer.tokenize(templated)
                    logger.info('First templated tokens (first 120): %s', toks[:120])
                except Exception as e:
                    logger.info('Failed to tokenize templated text: %s', e)
    except Exception as e:
        logger.warning('Failed to print templated first sample: %s', e)

    ds = ds.train_test_split(test_size=0.05, seed=args.seed)

    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    peft_config = create_peft_config()

    training_args = create_sft_config(
        output_dir=os.path.abspath(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        training_args=training_args,
        output_dir=os.path.abspath(args.output_dir),
    )


if __name__ == "__main__":
    main()
