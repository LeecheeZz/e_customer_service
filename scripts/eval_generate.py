#!/usr/bin/env python3
import argparse
import json
import os
import sys
import re
from glob import glob

import torch
from transformers import pipeline, BitsAndBytesConfig
from peft import PeftModel

# Ensure project root is on sys.path so `from e_customer_service...` works when
# the script is executed from anywhere.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from e_customer_service.modeling import load_model_and_tokenizer


def truncate_after_punct_before_bad(s: str) -> str:
    import re
    allowed = re.compile(r'[\u4e00-\u9fffA-Za-z0-9\s，。！？、；：,.!?;:()\[\]{}%+\-\\/\\"“”‘’…—–·]')
    for i, ch in enumerate(s):
        if not allowed.match(ch):
            for j in range(i - 1, -1, -1):
                if re.match(r'[。\.！？!？?,，、；：;:]', s[j]):
                    return s[: j + 1].strip()
            return s[: i].strip()
    return s.strip()


def find_latest_checkpoint(root_dir: str) -> str:
    # look for subdirs named checkpoint-<num>
    pattern = os.path.join(root_dir, 'checkpoint-*')
    candidates = [d for d in glob(pattern) if os.path.isdir(d)]
    if not candidates:
        # maybe root_dir itself is a checkpoint
        if os.path.isdir(root_dir) and os.path.exists(os.path.join(root_dir, 'adapter_model.safetensors')):
            return root_dir
        raise FileNotFoundError(f'No checkpoints found in {root_dir}')
    # sort by numeric suffix
    def num_of(path):
        base = os.path.basename(path)
        parts = base.split('-')
        try:
            return int(parts[-1])
        except Exception:
            return 0
    candidates.sort(key=num_of)
    return candidates[-1]


def load_sft_on_base(base_path: str, sft_checkpoint: str, qlora: bool = False, bnb_config: dict | None = None, torch_dtype=torch.bfloat16, local_files_only: bool = True):
    # load base model and tokenizer; support optional QLoRA (bitsandbytes 4-bit)
    if qlora and bnb_config is not None:
        model, tokenizer = load_model_and_tokenizer(
            base_path,
            device_map='auto',
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            load_in_4bit=True,
            bnb_config=bnb_config,
        )
    else:
        model, tokenizer = load_model_and_tokenizer(base_path, device_map='auto', torch_dtype=torch_dtype, local_files_only=local_files_only)
    # if sft_checkpoint contains adapter files, load via PeftModel
    try:
        if os.path.exists(os.path.join(sft_checkpoint, 'adapter_model.safetensors')) or os.path.exists(os.path.join(sft_checkpoint, 'adapter_config.json')):
            model = PeftModel.from_pretrained(model, sft_checkpoint, device_map='auto')
    except Exception as e:
        print('加载 SFT Adapter 失败，继续使用基线模型：', e)
    return model, tokenizer


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', default='output_sft_QLoRA')
    parser.add_argument('--base_model', default='/root/autodl-tmp/models/Qwen/Qwen3-8B-Base')
    parser.add_argument('--val_file', default='val_sft.jsonl', help='验证集路径（相对于 workspace 根或绝对路径）')
    parser.add_argument('--out_dir', default=None, help='输出目录，默认使用 model_root')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    # QLoRA / bitsandbytes options
    parser.add_argument('--qlora', action='store_true', help='Enable QLoRA 4-bit loading (bitsandbytes)')
    parser.add_argument('--bnb-4bit-quant-type', default='nf4', help='BitsAndBytes 4bit quant type (fp4 or nf4)')
    parser.add_argument('--bnb-4bit-compute-dtype', default='bfloat16', help='Compute dtype for 4-bit (e.g. bfloat16, float16)')
    parser.add_argument('-dq', '--bnb-4bit-use-double-quant', action='store_true', help='Enable double quantization for bnb 4-bit')
    args = parser.parse_args()

    model_root = args.model_root
    try:
        latest_ckpt = find_latest_checkpoint(model_root)
        print('找到最新 checkpoint:', latest_ckpt)
    except Exception:
        latest_ckpt = None

    base_path = args.base_model or model_root
    print('加载基线模型路径：', base_path)

    # prepare BitsAndBytesConfig if requested
    bnb_cfg = None
    if args.qlora:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
        )

    model, tokenizer = load_sft_on_base(base_path, latest_ckpt, qlora=args.qlora, bnb_config=bnb_cfg, torch_dtype=torch.bfloat16, local_files_only=True)
    # wrap pipeline; when model uses device_map='auto', don't pass device
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, trust_remote_code=True)

    # prepare val file path
    val_path = args.val_file
    if not os.path.isabs(val_path):
        val_path = os.path.join(os.getcwd(), val_path)
    if not os.path.exists(val_path):
        raise FileNotFoundError(f'val file not found: {val_path}')

    out_dir = args.out_dir or model_root
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.basename(os.path.normpath(model_root)) + '_eval.jsonl'
    out_path = os.path.join(out_dir, out_name)

    with open(out_path, 'w', encoding='utf-8') as outf:
        for item in read_jsonl(val_path):
            # expect item similar to val_sft.jsonl: contains messages or input fields
            # try to construct prompt via tokenizer.apply_chat_template if available
            try:
                messages = item.get('messages') or item.get('dialog') or None
                if messages and hasattr(tokenizer, 'apply_chat_template'):
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                else:
                    # fallback: use 'input' or 'text' field
                    prompt = item.get('input') or item.get('text') or item.get('prompt') or json.dumps(item, ensure_ascii=False)
            except Exception:
                prompt = item.get('input') or item.get('text') or item.get('prompt') or json.dumps(item, ensure_ascii=False)

            # generate
            try:
                resp = gen(prompt, max_new_tokens=args.max_new_tokens, return_full_text=False, do_sample=True, temperature=0.8, top_p=0.9)
                if isinstance(resp, list) and len(resp) > 0:
                    out_text = resp[0].get('generated_text') or resp[0].get('text') or str(resp[0])
                elif isinstance(resp, dict):
                    out_text = resp.get('generated_text') or resp.get('text') or str(resp)
                else:
                    out_text = str(resp)
            except Exception as e:
                out_text = f'__GENERATE_ERROR__ {e}'

            # if prompt is included in generated text, strip it
            if isinstance(prompt, str) and out_text.startswith(prompt):
                out_text = out_text[len(prompt):].lstrip()

            # clean and truncate using helper
            cleaned = truncate_after_punct_before_bad(out_text)

            # produce output record: copy original fields but replace role/user/etc with assistant answer
            out_item = dict(item)
            # place assistant reply into messages if original had messages
            if 'messages' in out_item and isinstance(out_item['messages'], list):
                # append assistant message
                assistant_msg = {'role': 'assistant', 'content': cleaned}
                out_item['messages'] = out_item['messages'] + [assistant_msg]
            else:
                # create assistant field
                out_item['assistant'] = cleaned

            outf.write(json.dumps(out_item, ensure_ascii=False) + '\n')

    print('生成完成，保存到', out_path)


if __name__ == '__main__':
    main()
