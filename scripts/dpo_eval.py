#!/usr/bin/env python3
import argparse
import json
import os

import torch
from peft import AutoPeftModelForCausalLM
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline

from e_customer_service.data import format_messages
from e_customer_service.eval_utils import read_jsonl
from e_customer_service.paths import build_run_paths, default_run_name, ensure_run_dirs


def create_bnb_config(args):
    if not args.qlora:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
    )


def resolve_output_path(out_file: str, default_dir) -> str:
    if not os.path.isabs(out_file) and not os.path.dirname(out_file):
        return os.path.join(default_dir, out_file)
    return out_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a DPO adapter")
    parser.add_argument("--output-root", default="output", help="Root directory for all experiment runs")
    parser.add_argument("--run-name", default=None, help="Experiment run name under output/runs")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="DPO adapter dir; defaults to output/runs/<run-name>/dpo/final_adapter",
    )
    parser.add_argument("--val-file", default="val_sft.jsonl")
    parser.add_argument("--out-file", default="dpo_eval_outputs.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--qlora", action="store_true", help="Enable QLoRA 4-bit loading (bitsandbytes)")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", help="BitsAndBytes 4bit quant type")
    parser.add_argument("--bnb-4bit-compute-dtype", default="bfloat16", help="Compute dtype for 4-bit")
    parser.add_argument("-dq", "--bnb-4bit-use-double-quant", action="store_true")
    args = parser.parse_args(argv)

    run_name = args.run_name or default_run_name()
    paths = build_run_paths(args.output_root, run_name)
    ensure_run_dirs(paths)

    model_dir = args.model_dir or str(paths["dpo_final_adapter_dir"])
    out_path = resolve_output_path(args.out_file, paths["dpo_eval_dir"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        is_trainable=False,
        device_map=args.device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=create_bnb_config(args),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for obj in tqdm(list(read_jsonl(args.val_file)), desc="DPO Eval"):
            raw = obj.get("prompt") or obj.get("messages") or []
            prompt_model = format_messages(
                tokenizer,
                raw,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            try:
                resp = gen(
                    prompt_model,
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

            try:
                if isinstance(prompt_model, str) and out_text.startswith(prompt_model):
                    out_text = out_text[len(prompt_model):].lstrip()
            except Exception:
                pass

            out_item = dict(obj)
            if "messages" in out_item and isinstance(out_item["messages"], list):
                assistant_msg = {"role": "assistant", "content": out_text}
                out_item["messages"] = out_item["messages"] + [assistant_msg]
            else:
                out_item["assistant"] = out_text

            out_f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    print("DPO eval saved to", out_path)


if __name__ == "__main__":
    main()
