#!/usr/bin/env python3
import argparse
import json
import os
from tqdm.auto import tqdm
import sys
import torch
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from e_customer_service.modeling import load_model_and_tokenizer
from eval_generate import truncate_after_punct_before_bad, read_jsonl


def messages_to_text(messages, tokenizer=None, for_model: bool = False):
    """Convert structured `messages` to text.

    - if `for_model` is True and tokenizer provides `apply_chat_template`, return the template used for generation;
    - otherwise return a human-readable plain text version (no template tokens).
    """
    # if messages is already a string, return it as-is
    if isinstance(messages, str):
        return messages

    if for_model:
        try:
            if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            pass

    # human-readable fallback (no template tokens)
    parts = []
    for m in messages or []:
        role = m.get('role', '')
        content = m.get('content', '')
        parts.append(f"<{role}>: {content}")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='output_qlora/dpo')
    parser.add_argument('--val-file', default='val_sft.jsonl')
    parser.add_argument('--out-file', default='dpo_eval_outputs.jsonl')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--device-map', default='auto')
    # QLoRA / bitsandbytes options
    parser.add_argument('--qlora', action='store_true', help='Enable QLoRA 4-bit loading (bitsandbytes)')
    parser.add_argument('--bnb-4bit-quant-type', default='nf4', help='BitsAndBytes 4bit quant type (fp4 or nf4)')
    parser.add_argument('--bnb-4bit-compute-dtype', default='bfloat16', help='Compute dtype for 4-bit (e.g. bfloat16, float16)')
    parser.add_argument('-dq', '--bnb-4bit-use-double-quant', action='store_true', help='Enable double quantization for bnb 4-bit')
    args = parser.parse_args()
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file) or '.', exist_ok=True)

    # prepare BitsAndBytesConfig if requested
    bnb_cfg = None
    if args.qlora:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
        )

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_dir,      # output_sft_QLoRA/lora
        is_trainable=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        quantization_config=bnb_cfg, 
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )

    out_path = os.path.join(os.path.dirname(args.model_dir), args.out_file)
    
    with open(out_path, 'w', encoding='utf-8') as out_f:
        for obj in tqdm(list(read_jsonl(args.val_file)), desc='Eval'):
            # support both `prompt` (string) or `messages` (list)
            raw = obj.get('prompt') or obj.get('messages') or []
            prompt_model = messages_to_text(raw, tokenizer, for_model=True)

            # 使用 transformers pipeline 生成（模仿 eval_generate.py 的行为）
            gen = pipeline('text-generation', model=model, tokenizer=tokenizer, trust_remote_code=True)

            try:
                resp = gen(prompt_model, max_new_tokens=args.max_new_tokens, return_full_text=False, do_sample=True, temperature=0.8, top_p=0.9)
                if isinstance(resp, list) and len(resp) > 0:
                    out_text = resp[0].get('generated_text') or resp[0].get('text') or str(resp[0])
                elif isinstance(resp, dict):
                    out_text = resp.get('generated_text') or resp.get('text') or str(resp)
                else:
                    out_text = str(resp)
            except Exception as e:
                out_text = f'__GENERATE_ERROR__ {e}'

            # 如果 prompt 被包含在生成文本中，去掉它
            try:
                if isinstance(prompt_model, str) and out_text.startswith(prompt_model):
                    out_text = out_text[len(prompt_model):].lstrip()
            except Exception:
                pass

            cleaned = truncate_after_punct_before_bad(out_text)

            # 生成输出：保留原始字段，若原始含 messages 则追加 assistant 消息，否则新增 assistant 字段
            out_item = dict(obj)
            if 'messages' in out_item and isinstance(out_item['messages'], list):
                assistant_msg = {'role': 'assistant', 'content': cleaned}
                out_item['messages'] = out_item['messages'] + [assistant_msg]
            else:
                out_item['assistant'] = cleaned

            out_f.write(json.dumps(out_item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
