#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from glob import glob

import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig, pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from e_customer_service.data import format_messages
from e_customer_service.modeling import load_model_and_tokenizer
from e_customer_service.paths import build_run_paths, default_run_name, ensure_run_dirs


def truncate_after_punct_before_bad(s: str) -> str:
    allowed = re.compile(r"""[一-鿿A-Za-z0-9\s，。！？、；：,.!?;:()\[\]{}%+\-\/"“”‘’…—–·]""")
    for i, ch in enumerate(s):
        if not allowed.match(ch):
            for j in range(i - 1, -1, -1):
                if re.match(r"[。\.！？!？?,，、；：;:]", s[j]):
                    return s[: j + 1].strip()
            return s[: i].strip()
    return s.strip()


def find_latest_checkpoint(root_dir: str) -> str:
    pattern = os.path.join(root_dir, "checkpoint-*")
    candidates = [d for d in glob(pattern) if os.path.isdir(d)]
    if not candidates:
        if os.path.isdir(root_dir) and os.path.exists(os.path.join(root_dir, "adapter_model.safetensors")):
            return root_dir
        raise FileNotFoundError(f"No checkpoints found in {root_dir}")

    def num_of(path):
        base = os.path.basename(path)
        parts = base.split("-")
        try:
            return int(parts[-1])
        except Exception:
            return 0

    candidates.sort(key=num_of)
    return candidates[-1]


def load_sft_on_base(
    base_path: str,
    sft_checkpoint: str,
    qlora: bool = False,
    bnb_config: dict | None = None,
    torch_dtype=torch.bfloat16,
    local_files_only: bool = True,
):
    if qlora and bnb_config is not None:
        model, tokenizer = load_model_and_tokenizer(
            base_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            load_in_4bit=True,
            bnb_config=bnb_config,
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            base_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )

    try:
        if os.path.exists(os.path.join(sft_checkpoint, "adapter_model.safetensors")) or os.path.exists(
            os.path.join(sft_checkpoint, "adapter_config.json")
        ):
            model = PeftModel.from_pretrained(model, sft_checkpoint, device_map="auto")
    except Exception as e:
        print("加载 SFT Adapter 失败，继续使用基线模型：", e)
    return model, tokenizer


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output", help="Root directory for all experiment runs")
    parser.add_argument("--run-name", default=None, help="Experiment run name under output/runs")
    parser.add_argument(
        "--model_root",
        default=None,
        help="SFT adapter/checkpoint dir; defaults to output/runs/<run-name>/sft/final_adapter",
    )
    parser.add_argument("--base_model", default="/root/autodl-tmp/models/Qwen/Qwen3-8B-Base")
    parser.add_argument("--val_file", default="val_sft.jsonl", help="验证集路径（相对于 workspace 根或绝对路径）")
    parser.add_argument("--out_dir", default=None, help="输出目录，默认使用当前 run 的 sft/eval")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--qlora", action="store_true", help="Enable QLoRA 4-bit loading (bitsandbytes)")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", help="BitsAndBytes 4bit quant type")
    parser.add_argument("--bnb-4bit-compute-dtype", default="bfloat16", help="Compute dtype for 4-bit")
    parser.add_argument("-dq", "--bnb-4bit-use-double-quant", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or default_run_name()
    paths = build_run_paths(args.output_root, run_name)
    ensure_run_dirs(paths)

    model_root = args.model_root or str(paths["sft_final_adapter_dir"])
    try:
        latest_ckpt = find_latest_checkpoint(model_root)
        print("找到最新 checkpoint:", latest_ckpt)
    except Exception:
        latest_ckpt = model_root

    base_path = args.base_model or model_root
    print("加载基线模型路径：", base_path)

    bnb_cfg = None
    if args.qlora:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
        )

    model, tokenizer = load_sft_on_base(
        base_path,
        latest_ckpt,
        qlora=args.qlora,
        bnb_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True)

    val_path = args.val_file
    if not os.path.isabs(val_path):
        val_path = os.path.join(os.getcwd(), val_path)
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"val file not found: {val_path}")

    out_dir = args.out_dir or str(paths["sft_eval_dir"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sft_eval_outputs.jsonl")

    with open(out_path, "w", encoding="utf-8") as outf:
        for item in read_jsonl(val_path):
            try:
                messages = item.get("messages") or item.get("dialog") or None
                if messages:
                    prompt = format_messages(
                        tokenizer,
                        messages,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                else:
                    prompt = item.get("input") or item.get("text") or item.get("prompt") or json.dumps(
                        item,
                        ensure_ascii=False,
                    )
            except Exception:
                prompt = item.get("input") or item.get("text") or item.get("prompt") or json.dumps(
                    item,
                    ensure_ascii=False,
                )

            try:
                resp = gen(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    return_full_text=False,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                )
                if isinstance(resp, list) and len(resp) > 0:
                    out_text = resp[0].get("generated_text") or resp[0].get("text") or str(resp[0])
                elif isinstance(resp, dict):
                    out_text = resp.get("generated_text") or resp.get("text") or str(resp)
                else:
                    out_text = str(resp)
            except Exception as e:
                out_text = f"__GENERATE_ERROR__ {e}"

            if isinstance(prompt, str) and out_text.startswith(prompt):
                out_text = out_text[len(prompt):].lstrip()

            cleaned = truncate_after_punct_before_bad(out_text)
            out_item = dict(item)
            if "messages" in out_item and isinstance(out_item["messages"], list):
                assistant_msg = {"role": "assistant", "content": cleaned}
                out_item["messages"] = out_item["messages"] + [assistant_msg]
            else:
                out_item["assistant"] = cleaned

            outf.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    print("生成完成，保存到", out_path)


if __name__ == "__main__":
    main()
