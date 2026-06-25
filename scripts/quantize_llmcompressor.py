#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e_customer_service.data import format_messages
from e_customer_service.eval_utils import read_jsonl


DEFAULT_BASE_MODEL = "/media/ssd2/lyf/le/models/Qwen/Qwen3-8B"
DEFAULT_OUTPUT_ROOT = "/media/ssd2/lyf/le/models/Qwen"

METHOD_SUFFIXES = {
    "awq": "AWQ-W4A16",
    "gptq": "GPTQ-W4A16",
    "smoothquant": "SmoothQuant-W8A8",
    "int8": "INT8-W8A8",
}


def read_calibration_texts(path: str, tokenizer, limit: int, enable_thinking: bool) -> List[str]:
    texts = []
    for item in read_jsonl(path):
        messages = item.get("messages") or item.get("dialog") or item.get("prompt")
        if isinstance(messages, list):
            text = format_messages(
                tokenizer,
                messages,
                add_generation_prompt=False,
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


def build_calibration_dataset(texts: List[str], tokenizer, max_seq_length: int):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit("Missing datasets. Install with: pip install -r requirements-quantization.txt") from exc

    ds = Dataset.from_list([{"text": text} for text in texts])

    def tokenize(sample: Dict[str, str]) -> Dict[str, Any]:
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    return ds.map(tokenize, remove_columns=ds.column_names, desc="Tokenize calibration data")


def output_dir_for(args, method: str) -> str:
    if args.output_dir:
        return args.output_dir
    model_name = Path(args.base_model.rstrip("/")).name
    return str(Path(args.output_root) / f"{model_name}-{METHOD_SUFFIXES[method]}")


def copy_extra_model_files(model_dir: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
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


def write_manifest(output_dir: str, payload: Dict[str, Any]) -> None:
    path = Path(output_dir) / "quantization_manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_model_and_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "device_map": args.device_map,
    }
    if args.dtype:
        model_kwargs["dtype"] = args.dtype

    print(f"Loading base model: {args.base_model}")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    return model, tokenizer


def build_recipe(method: str, args):
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier


    ignore = args.ignore or ["lm_head"]

    if method == "awq":
        return [
            AWQModifier(duo_scaling=args.awq_duo_scaling, n_grid=args.awq_n_grid),
            QuantizationModifier(
                targets=args.targets,
                scheme=args.awq_scheme,
                ignore=ignore,
            ),
        ]
    if method == "gptq":
        return GPTQModifier(targets=args.targets, scheme=args.gptq_scheme, ignore=ignore)
    if method == "smoothquant":
        return [
            SmoothQuantModifier(smoothing_strength=args.smoothing_strength),
            GPTQModifier(targets=args.targets, scheme=args.int8_scheme, ignore=ignore),
        ]
    if method == "int8":
        return GPTQModifier(targets=args.targets, scheme=args.int8_scheme, ignore=ignore)
    raise ValueError(f"unsupported method: {method}")


def quantize_one(method: str, args, calib_dataset, calib_count: int) -> str:
    from llmcompressor import oneshot

    output_dir = output_dir_for(args, method)
    model, tokenizer = load_model_and_tokenizer(args)
    recipe = build_recipe(method, args)

    print(f"Quantizing with {method}: {calib_count} samples, max_seq_length={args.max_seq_length}")
    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=calib_count,
        batch_size=args.batch_size,
        pipeline=args.pipeline,
    )

    print(f"Saving compressed model to {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    copy_extra_model_files(args.base_model, output_dir)

    manifest = {
        "tool": "llmcompressor",
        "method": method,
        "base_model": args.base_model,
        "output_dir": output_dir,
        "calib_file": args.calib_file,
        "calib_samples": calib_count,
        "max_seq_length": args.max_seq_length,
        "targets": args.targets,
        "ignore": args.ignore or ["lm_head"],
        "save_compressed": True,
        "vllm_args": ["--quantization", "compressed-tensors"],
    }
    if method == "awq":
        manifest.update({"scheme": args.awq_scheme, "algorithm": "AWQ + RTN"})
    elif method == "gptq":
        manifest.update({"scheme": args.gptq_scheme, "algorithm": "GPTQ"})
    elif method == "smoothquant":
        manifest.update({
            "scheme": args.int8_scheme,
            "algorithm": "SmoothQuant + GPTQ",
            "smoothing_strength": args.smoothing_strength,
        })
    elif method == "int8":
        manifest.update({"scheme": args.int8_scheme, "algorithm": "GPTQ"})
    write_manifest(output_dir, manifest)
    return output_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Quantize a causal LM with vLLM llm-compressor")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", default=None, help="Use only with a single --method")
    parser.add_argument(
        "--method",
        choices=["awq", "gptq", "smoothquant", "int8", "all"],
        required=True,
    )
    parser.add_argument("--calib-file", default="val_sft.jsonl")
    parser.add_argument("--calib-samples", type=int, default=128, help="Use 0 for all samples")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--pipeline",
        choices=["independent", "sequential", "basic"],
        default="independent",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--targets", nargs="+", default=["Linear"])
    parser.add_argument("--ignore", nargs="*", default=None)
    parser.add_argument("--awq-scheme", default="W4A16_ASYM")
    parser.add_argument("--gptq-scheme", default="W4A16")
    parser.add_argument("--int8-scheme", default="W8A8")
    parser.add_argument("--awq-duo-scaling", default="both", choices=["true", "false", "both"])
    parser.add_argument("--awq-n-grid", type=int, default=20)
    parser.add_argument("--smoothing-strength", type=float, default=0.8)
    return parser.parse_args(argv)


def normalize_args(args):
    if args.awq_duo_scaling == "true":
        args.awq_duo_scaling = True
    elif args.awq_duo_scaling == "false":
        args.awq_duo_scaling = False
    return args


def main(argv=None):
    args = normalize_args(parse_args(argv))

    from transformers import AutoTokenizer
    
    if args.output_dir and args.method == "all":
        raise SystemExit("--output-dir can only be used with one method, not --method all")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    texts = read_calibration_texts(
        args.calib_file,
        tokenizer,
        args.calib_samples,
        args.enable_thinking,
    )
    calib_dataset = build_calibration_dataset(texts, tokenizer, args.max_seq_length)

    methods = list(METHOD_SUFFIXES) if args.method == "all" else [args.method]
    outputs = {}
    for method in methods:
        outputs[method] = quantize_one(method, args, calib_dataset, len(texts))

    print("\nQuantized outputs:")
    for method, path in outputs.items():
        print(f"{method}: {path}")
        print(f"  vLLM: python -m scripts.vllm_serve --base-model {path} --use-quantization-manifest")


if __name__ == "__main__":
    main()

