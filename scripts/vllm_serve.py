#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys

from e_customer_service.paths import build_run_paths, default_run_name


def build_command(args):
    paths = build_run_paths(args.output_root, args.run_name or default_run_name())
    adapter_dir = args.adapter_dir or str(paths[f"{args.stage}_final_adapter_dir"])

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.base_model,
        "--served-model-name",
        args.base_served_model_name,
        "--enable-lora",
        "--lora-modules",
        f"{args.lora_name}={adapter_dir}",
        "--max-lora-rank",
        str(args.max_lora_rank),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.tensor_parallel_size is not None:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])
    if args.load_format:
        cmd.extend(["--load-format", args.load_format])
    if args.model_loader_extra_config:
        cmd.extend(["--model-loader-extra-config", args.model_loader_extra_config])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    return cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the customer-service LoRA adapter with vLLM")
    parser.add_argument("--base-model", default="/root/autodl-tmp/models/Qwen/Qwen3-8B-Instruct", help="Base model path used by the LoRA adapter")
    parser.add_argument("--output-root", default="output", help="Root directory for experiment runs")
    parser.add_argument("--run-name", default=None, help="Experiment run name under output/runs")
    parser.add_argument("--stage", choices=["sft", "dpo"], default="sft", help="Adapter stage to serve")
    parser.add_argument("--adapter-dir", default=None, help="Adapter dir; overrides --output-root/--run-name/--stage")
    parser.add_argument("--lora-name", default="customer-service", help="Model name used in API requests")
    parser.add_argument("--base-served-model-name", default="customer-service-base")
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--quantization", default=None, help="vLLM quantization backend, e.g. awq, gptq, bitsandbytes")
    parser.add_argument("--load-format", default=None, help="vLLM load format, e.g. auto, awq, gptq, bitsandbytes")
    parser.add_argument("--model-loader-extra-config", default=None, help="JSON string passed to vLLM model loader")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the vLLM command without starting it")
    args = parser.parse_args(argv)

    cmd = build_command(args)
    print("Running:")
    print(shlex.join(cmd))
    if args.dry_run:
        return

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
