#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer

from e_customer_service.data import format_messages
from e_customer_service.eval_utils import read_jsonl


DEFAULT_BASE_MODEL = "/media/ssd2/lyf/le/models/Qwen/Qwen3-8B-Instruct"
DEFAULT_OUTPUT_ROOT = "/media/ssd2/lyf/le/models/Qwen"


def read_calibration_texts(path: str, tokenizer, limit: int, enable_thinking: bool) -> List[str]:
    texts = []
    for item in read_jsonl(path):
        messages = item.get("messages") or item.get("dialog") or item.get("prompt")
        if isinstance(messages, list):
            text = format_messages(
                tokenizer,
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        elif isinstance(messages, str):
            text = messages
        else:
            text = item.get("input") or item.get("text") or json.dumps(item, ensure_ascii=False)
        if text:
            texts.append(text)
        if limit > 0 and len(texts) >= limit:
            break
    if not texts:
        raise ValueError(f"no calibration texts found in {path}")
    return texts


def copy_tokenizer_files(model_dir: str, output_dir: str, tokenizer) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    for name in (
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        src = Path(model_dir) / name
        dst = Path(output_dir) / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def write_manifest(output_dir: str, payload: dict) -> None:
    path = Path(output_dir) / "quantization_manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def output_dir_for(args, suffix: str) -> str:
    if args.output_dir:
        return args.output_dir
    model_name = Path(args.base_model.rstrip("/")).name
    return str(Path(args.output_root) / f"{model_name}-{suffix}")


def quantize_awq(args, tokenizer, calib_texts: List[str]) -> str:
    try:
        from awq import AutoAWQForCausalLM
    except Exception as exc:
        raise SystemExit(
            "AutoAWQ import failed. This is usually a transformers/torch/torchvision "
            "compatibility issue. Use a separate quantization env and reinstall with:\n"
            "pip install --force-reinstall -r requirements-quantization.txt\n"
            f"Original error: {exc}"
        ) from exc

    output_dir = output_dir_for(args, "AWQ-INT4")
    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": 4,
        "version": args.awq_version,
    }
    print(f"Loading base model for AWQ: {args.base_model}")
    model = AutoAWQForCausalLM.from_pretrained(
        args.base_model,
        safetensors=True,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Quantizing AWQ INT4 with {len(calib_texts)} calibration samples")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_texts,
        max_calib_seq_len=args.max_calib_seq_len,
        n_parallel_calib_samples=args.n_parallel_calib_samples,
    )
    print(f"Saving AWQ model to {output_dir}")
    model.save_quantized(output_dir, safetensors=True)
    copy_tokenizer_files(args.base_model, output_dir, tokenizer)
    write_manifest(output_dir, {
        "method": "awq",
        "bits": 4,
        "group_size": args.group_size,
        "base_model": args.base_model,
        "calib_file": args.calib_file,
        "calib_samples": len(calib_texts),
        "vllm_args": ["--quantization", "awq", "--dtype", "float16"],
    })
    return output_dir


def quantize_int8_bnb(args, tokenizer) -> str:
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit("Missing transformers/bitsandbytes. Install with: pip install -r requirements-quantization.txt") from exc

    output_dir = output_dir_for(args, "BNB-INT8")
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    print(f"Loading base model in bitsandbytes INT8: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map=args.device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Saving INT8 model to {output_dir}")
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    except Exception as exc:
        raise SystemExit(
            "bitsandbytes INT8 was loaded, but this transformers/bitsandbytes combination "
            "could not save a standalone INT8 checkpoint. For the INT8 ablation, serve the "
            "original base model with: --quantization bitsandbytes --load-format bitsandbytes"
        ) from exc
    copy_tokenizer_files(args.base_model, output_dir, tokenizer)
    write_manifest(output_dir, {
        "method": "bitsandbytes_int8",
        "bits": 8,
        "base_model": args.base_model,
        "vllm_args": ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"],
        "note": "If vLLM cannot load this exported checkpoint, use the original model path with the same vLLM bitsandbytes args.",
    })
    return output_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Quantize a causal LM for vLLM ablation experiments")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", default=None, help="Use only with a single --method")
    parser.add_argument("--method", choices=["awq", "int8", "all"], required=True)
    parser.add_argument("--calib-file", default="val_sft.jsonl")
    parser.add_argument("--calib-samples", type=int, default=128, help="Use 0 for all samples")
    parser.add_argument("--max-calib-seq-len", type=int, default=2048)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--awq-version", default="GEMM", choices=["GEMM", "GEMV"])
    parser.add_argument("--n-parallel-calib-samples", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir and args.method == "all":
        raise SystemExit("--output-dir can only be used with one method, not --method all")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    methods = ["awq", "int8"] if args.method == "all" else [args.method]
    calib_texts = None
    outputs = {}
    for method in methods:
        if method == "awq" and calib_texts is None:
            calib_texts = read_calibration_texts(
                args.calib_file,
                tokenizer,
                args.calib_samples,
                args.enable_thinking,
            )
        if method == "awq":
            outputs[method] = quantize_awq(args, tokenizer, calib_texts or [])
        elif method == "int8":
            outputs[method] = quantize_int8_bnb(args, tokenizer)

    print("\nQuantized outputs:")
    for method, path in outputs.items():
        print(f"{method}: {path}")


if __name__ == "__main__":
    main()
