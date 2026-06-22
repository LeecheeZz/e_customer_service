#!/usr/bin/env python3
import argparse
import json
import os
import time
import urllib.error
import urllib.request

from tqdm.auto import tqdm

from e_customer_service.eval_utils import read_jsonl, truncate_after_punct_before_bad
from e_customer_service.paths import build_run_paths, default_run_name, ensure_run_dirs


def resolve_output_path(out_file: str, default_dir) -> str:
    if not os.path.isabs(out_file) and not os.path.dirname(out_file):
        return os.path.join(default_dir, out_file)
    return out_file


def request_chat_completion(base_url: str, payload: dict, timeout: float) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_messages(item: dict):
    messages = item.get("messages") or item.get("prompt") or item.get("dialog")
    if isinstance(messages, list):
        return messages
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    text = item.get("input") or item.get("text")
    if isinstance(text, str):
        return [{"role": "user", "content": text}]
    raise ValueError("sample does not contain messages, prompt, dialog, input, or text")


def extract_generated_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") or {}
    return message.get("content") or first.get("text") or ""


def extract_reference_text(item: dict) -> str:
    if isinstance(item.get("assistant"), str):
        return item["assistant"]
    messages = item.get("messages") or []
    if isinstance(messages, list):
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return message.get("content", "")
    return ""


def normalize_text(value: str) -> str:
    return "".join(value.split())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate a vLLM OpenAI-compatible service on a JSONL set")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="customer-service", help="LoRA model name served by vLLM")
    parser.add_argument("--output-root", default="output", help="Root directory for experiment runs")
    parser.add_argument("--run-name", default=None, help="Experiment run name under output/runs")
    parser.add_argument("--stage", choices=["sft", "dpo"], default="sft")
    parser.add_argument("--val-file", default="val_sft.jsonl")
    parser.add_argument("--out-file", default="vllm_eval_outputs.jsonl")
    parser.add_argument("--reference-file", default=None, help="Optional JSONL output to compare against")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to validate; use 0 for all")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    run_name = args.run_name or default_run_name()
    paths = build_run_paths(args.output_root, run_name)
    ensure_run_dirs(paths)

    eval_dir = paths[f"{args.stage}_eval_dir"]
    out_path = resolve_output_path(args.out_file, eval_dir)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rows = list(read_jsonl(args.val_file))
    references = list(read_jsonl(args.reference_file)) if args.reference_file else []
    if args.limit > 0:
        rows = rows[: args.limit]
        references = references[: args.limit]

    ok_count = 0
    comparable_count = 0
    exact_match_count = 0
    normalized_match_count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for idx, item in enumerate(tqdm(rows, desc="vLLM Eval")):
            messages = extract_messages(item)
            payload = {
                "model": args.model,
                "messages": messages,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            }

            started = time.perf_counter()
            error = None
            response = None
            generated = ""
            try:
                response = request_chat_completion(args.base_url, payload, args.timeout)
                generated = truncate_after_punct_before_bad(extract_generated_text(response))
                ok_count += 1
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                error = str(e)

            out_item = dict(item)
            out_item.update({
                "vllm_model": args.model,
                "vllm_messages": messages,
                "assistant": generated,
                "latency_seconds": round(time.perf_counter() - started, 4),
            })
            if error is not None:
                out_item["error"] = error
            if response is not None:
                out_item["raw_response"] = response

            if idx < len(references):
                reference_text = extract_reference_text(references[idx])
                out_item["reference_assistant"] = reference_text
                out_item["reference_exact_match"] = generated == reference_text
                out_item["reference_normalized_match"] = (
                    normalize_text(generated) == normalize_text(reference_text)
                )
                comparable_count += 1
                exact_match_count += int(out_item["reference_exact_match"])
                normalized_match_count += int(out_item["reference_normalized_match"])

            out_f.write(json.dumps(out_item, ensure_ascii=False) + "\n")

            if idx < 3:
                print("\n--- sample", idx + 1, "---")
                print("user:", messages[-1].get("content", "") if messages else "")
                print("assistant:", generated or error)

    print(f"vLLM validation saved to {out_path}")
    print(f"Succeeded: {ok_count}/{len(rows)}")
    if comparable_count:
        print(f"Exact matches: {exact_match_count}/{comparable_count}")
        print(f"Normalized matches: {normalized_match_count}/{comparable_count}")


if __name__ == "__main__":
    main()
