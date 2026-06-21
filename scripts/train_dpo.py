#!/usr/bin/env python3
"""Train a model with Direct Preference Optimization (DPO) using a
JSONL file of preference pairs (prompt + chosen + rejected).

This script attempts to use `trl.DPOTrainer`. If your installed `trl`
version does not expose `DPOTrainer`, the script will exit with an
informative message.

Dataset format (one JSON object per line):
  {
    "prompt": [ {"role": "user", "content": "..."} ],
    "chosen": [ {"role": "assistant", "content": "..."} ],
    "rejected": [ {"role": "assistant", "content": "..."} ]
  }

Usage example:
  python scripts/train_dpo.py --dpo-file dpo_pairs.jsonl --model-path /path/to/base --output-dir output_dpo
"""
import argparse
import json
import os
import sys
from typing import List

import torch

from peft import AutoPeftModelForCausalLM
from transformers import BitsAndBytesConfig, AutoTokenizer
from datasets import Dataset

# ensure project root on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



def read_pairs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def messages_to_text(messages: List[dict], tokenizer):
    # If tokenizer provides chat template helper, prefer that
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    except Exception:
        pass
    # fallback: join assistant/user content by role
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        parts.append(f"<{role}>: {content}")
    return "\n".join(parts)


def build_dataset(dpo_file: str, tokenizer):
    examples = []
    for obj in read_pairs(dpo_file):
        prompt_msgs = obj.get("prompt") or obj.get("messages") or []
        chosen_msgs = obj.get("chosen") or obj.get("best") or []
        rejected_msgs = obj.get("rejected") or obj.get("worst") or []

        query = messages_to_text(prompt_msgs, tokenizer)
        chosen = messages_to_text(chosen_msgs, tokenizer)
        rejected = messages_to_text(rejected_msgs, tokenizer)

        examples.append({
            "query": query,
            "chosen": chosen,
            "rejected": rejected,
        })
    return examples


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dpo-file", default="dpo_pairs.jsonl")
    p.add_argument("--model-path", default="/media/ssd2/lyf/le/e_customer/models/Qwen/Qwen3-8B-Base")
    p.add_argument("--adapter-dir", default="output_qlora/lora", help="Path to LoRA adapter (directory with adapter_model.safetensors)")
    p.add_argument("--output-dir", default="output_qlora")
    p.add_argument("--beta", default=0.3)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--truncate-after-punct-before-bad", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--bnb-4bit-quant-type", default="nf4")
    p.add_argument("--bnb-4bit-compute-dtype", default="bfloat16")
    p.add_argument("-dq", "--bnb-4bit-use-double-quant", action="store_true")
    args = p.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    # prepare bnb config if requested
    bnb_cfg = None
    if args.qlora:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
        )

    # load model + tokenizer (trainable model)
    print("---------------Loading model-----------------")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_dir,      # output_sft_QLoRA/lora
        is_trainable=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        quantization_config=bnb_cfg,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_dir,
        trust_remote_code=True,
    )

    dataset = build_dataset(args.dpo_file, tokenizer)
    if isinstance(dataset, list):
        try:
            dataset = Dataset.from_list(dataset)
        except Exception as e:
            print("无法将生成的 list 转换为 datasets.Dataset，请安装 datasets 库。错误：", e)
            sys.exit(1)

    # try to import DPOTrainer from trl
    try:
        from trl import DPOTrainer, DPOConfig
    except Exception as e:
        print("无法从 trl 导入 DPOTrainer/DPOConfig，请确认已安装支持 DPO 的 trl 版本。错误：", e)
        sys.exit(1)

    # create peft (LoRA) config and DPO config
    # peft_config = create_peft_config()

    dpo_args = DPOConfig(
        output_dir=os.path.abspath(args.output_dir),
        # ========== 数据集与批次 ==========
        per_device_train_batch_size=args.batch_size,           # 显存不足可设为1或2，通过gradient_accumulation_steps增加有效批次
        # per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.truncate_after_punct_before_bad,           # 模拟全局批次大小 = 2*8 = 16
        # dataloader_num_workers=4,                # 数据加载线程数

        # ========== 学习率与优化 ==========
        learning_rate=args.learning_rate,                      # DPO建议1e-6到1e-5[reference:1]
        warmup_steps=10,                        # 预热10%的训练步数
        lr_scheduler_type="cosine",              # 余弦退火学习率调度器
        # optim="adamw_torch",                     # PyTorch的AdamW优化器[reference:2]
        weight_decay=0.01,                       # 权重衰减正则化

        # ========== DPO 核心参数 ==========
        beta=args.beta,                                # 控制与参考模型的偏差，0.1是平衡性能和稳定性的保守选择[reference:3]
        # loss_type="sigmoid",                     # 常用损失函数

        # ========== 序列长度设置 ==========
        # max_length=1024,                         # 支持的最大总长度
        # max_prompt_length=512,                   # prompt最大长度
        # max_completion_length=512,             # 回答最大长度，通常会自动从max_length和max_prompt_length计算

        # ========== 训练控制 ==========
        num_train_epochs=args.epochs,                      # 1000条数据，2-3轮足够
        logging_steps=3,                        # 每10步打印一次日志
        save_strategy="steps",                   # 按步数进行评估
        save_steps=100,                          # 每250步保存一次
        eval_steps=100,                          # 每250步评估一次
        save_total_limit=1,                      # 只保留最后2个检查点

        # ========== 其他设置 ==========
        fp16=False,                              # FP16混合精度
        bf16=True,                               # 30系及以后GPU可开启BF16
        gradient_checkpointing=False,            # 用计算换显存
        remove_unused_columns=False,             # 防止删除数据处理时需要的列
        report_to="none",                        # 不上传日志到云端，可设为'wandb'或'tensorboard'
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        # peft_config=peft_config,
    )

    # Some trl releases define `log(self, logs)` while `transformers.Trainer`
    # may call `self.log(logs, start_time)`. Patch instance `log` to accept
    # an extra positional argument to maintain compatibility.
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

    # save adapter or final model
    save_dir = os.path.join(args.output_dir, "dpo")
    os.makedirs(save_dir, exist_ok=True)
    try:
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Saved DPO model to", save_dir)
    except Exception as e:
        print("保存模型失败：", e)


if __name__ == "__main__":
    main()
