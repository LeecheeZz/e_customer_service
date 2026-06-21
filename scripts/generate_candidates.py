#!/usr/bin/env python3
"""从训练/验证集抽样并生成可标注的候选回复文件。

用法示例：
  python scripts/generate_candidates.py --source val_sft.jsonl --out-dir annotation_candidates --sample-size 200 --candidates-per-prompt 4 --generate --base-model /root/autodl-tmp/models/Qwen/Qwen3-8B-Base

脚本输出：
  annotation_candidates/index.jsonl  -- 每行为一个 prompt 的元信息，包含 gens 文件路径
  annotation_candidates/gens/{prompt_idx}_gens.jsonl -- 每行一个生成候选 (gen_id, text, meta)
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List

import torch
from transformers import pipeline, BitsAndBytesConfig
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
from peft import PeftModel
from eval_generate import truncate_after_punct_before_bad
# optional progress bar (tqdm)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from e_customer_service.modeling import load_model_and_tokenizer


def read_jsonl(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def prepare_prompt(item: dict, tokenizer):
    try:
        messages = item.get('messages') or item.get('dialog') or None
        if messages and hasattr(tokenizer, 'apply_chat_template'):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            return item.get('input') or item.get('text') or item.get('prompt') or json.dumps(item, ensure_ascii=False)
    except Exception:
        return item.get('input') or item.get('text') or item.get('prompt') or json.dumps(item, ensure_ascii=False)


def load_generator(base_model: str, sft_checkpoint: str | None, qlora: bool, bnb_cfg, torch_dtype=torch.bfloat16):
    # load base model and optionally adapter
    if qlora and bnb_cfg is not None:
        model, tokenizer = load_model_and_tokenizer(base_model, device_map='auto', torch_dtype=torch_dtype, local_files_only=True, load_in_4bit=True, bnb_config=bnb_cfg)
    else:
        model, tokenizer = load_model_and_tokenizer(base_model, device_map='auto', torch_dtype=torch_dtype, local_files_only=True)

    # if sft adapter exists, attach it
    try:
        if sft_checkpoint and os.path.exists(os.path.join(sft_checkpoint, 'adapter_model.safetensors')):
            model = PeftModel.from_pretrained(model, sft_checkpoint, device_map='auto')
    except Exception as e:
        print('Warning: failed to load adapter:', e)
    # Avoid transformers warning when calling generation with both
    # `max_new_tokens` and a preset `max_length` (default 20). Clear
    # model/generation config max_length so only `max_new_tokens` is used.
    try:
        if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'max_length'):
            model.generation_config.max_length = None
    except Exception:
        pass
    try:
        if hasattr(model, 'config') and hasattr(model.config, 'max_length'):
            model.config.max_length = None
    except Exception:
        pass

    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, trust_remote_code=True)
    return gen, tokenizer


def generate_for_prompt(gen, prompt: str, n: int, max_new_tokens: int, temperature: float, top_p: float):
    outs = []
    for _ in tqdm(range(n), desc="gens", leave=False):
        try:
            r = gen(prompt, max_new_tokens=max_new_tokens, return_full_text=False, do_sample=True, temperature=temperature, top_p=top_p)
            if isinstance(r, list) and len(r) > 0:
                text = r[0].get('generated_text') or r[0].get('text') or str(r[0])
            elif isinstance(r, dict):
                text = r.get('generated_text') or text.get('text') or str(r)
            else:
                text = str(r)
        except Exception as e:
            text = f'__GENERATE_ERROR__ {e}'
        outs.append(truncate_after_punct_before_bad(text))
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='train_val_sft.jsonl', help='train_sft.jsonl or val_sft.jsonl (path relative to cwd or absolute)')
    p.add_argument('--out-dir', default='annotation_candidates')
    p.add_argument('--sample-size', type=int, default=1500)
    p.add_argument('--candidates-per-prompt', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--generate', action='store_true', help='Whether to run model generation for candidates')
    p.add_argument('--base-model', default='/root/autodl-tmp/models/Qwen/Qwen3-8B-Base', help='Base model path for generation')
    p.add_argument('--sft-checkpoint', default='output_sft_DQLoRA/checkpoint-208', help='SFT adapter checkpoint path to load on top of base model')
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.95)
    # QLoRA bitsandbytes options
    p.add_argument('--qlora', default=True)
    p.add_argument('--bnb-4bit-quant-type', default='nf4')
    p.add_argument('--bnb-4bit-compute-dtype', default='bfloat16')
    p.add_argument('--bnb-4bit-use-double-quant', default=True)

    args = p.parse_args()

    src = args.source
    if not os.path.isabs(src):
        src = os.path.join(os.getcwd(), src)
    if not os.path.exists(src):
        raise FileNotFoundError(f'source not found: {src}')

    items = read_jsonl(src)
    random.seed(args.seed)
    idxs = list(range(len(items)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.sample_size, len(idxs))]

    out_dir = Path(args.out_dir)
    gens_dir = out_dir / 'gens'
    gens_dir.mkdir(parents=True, exist_ok=True)

    gen = None
    tokenizer = None
    bnb_cfg = None
    if args.generate:
        if not args.base_model:
            raise ValueError('When --generate is set, --base-model must be provided')
        if args.qlora:
            # build BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
            )
        gen, tokenizer = load_generator(args.base_model, args.sft_checkpoint, args.qlora, bnb_cfg)

    index_path = out_dir / 'index.jsonl'
    with open(index_path, 'w', encoding='utf-8') as indexf:
        for rank, i in enumerate(tqdm(idxs, desc='Prompts', unit='prompt')):
            item = items[i]
            prompt = None
            if tokenizer is not None:
                prompt = prepare_prompt(item, tokenizer)
            else:
                # no tokenizer: try simple fallback
                prompt = item.get('input') or item.get('text') or item.get('prompt') or json.dumps(item, ensure_ascii=False)

            gens_path = gens_dir / f'{i}_gens.jsonl'
            # if generate, create candidates by model; otherwise leave empty placeholders
            with open(gens_path, 'w', encoding='utf-8') as gf:
                if args.generate and gen is not None:
                    texts = generate_for_prompt(gen, prompt, args.candidates_per_prompt, args.max_new_tokens, args.temperature, args.top_p)
                    for j, txt in enumerate(texts):
                        rec = {'gen_id': f'g{i}_{j}', 'text': txt, 'meta': {'prompt_idx': i, 'candidate_rank': j}}
                        gf.write(json.dumps(rec, ensure_ascii=False) + '\n')
                else:
                    # write empty template for manual annotation
                    for j in range(args.candidates_per_prompt):
                        rec = {'gen_id': f'g{i}_{j}', 'text': '', 'meta': {'prompt_idx': i, 'candidate_rank': j}}
                        gf.write(json.dumps(rec, ensure_ascii=False) + '\n')

            indexf.write(json.dumps({'prompt_idx': i, 'prompt': prompt, 'gens_path': str(gens_path)}, ensure_ascii=False) + '\n')

    print('Done. Candidates saved to', out_dir)


if __name__ == '__main__':
    main()
