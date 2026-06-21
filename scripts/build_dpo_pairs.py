#!/usr/bin/env python3
"""从 annotation 文件和候选回复构建 DPO 训练数据 dpo_pairs.jsonl。

用法：
  python scripts/build_dpo_pairs.py --annotations annotations.jsonl --candidates-dir annotation_candidates --out dpo_pairs.jsonl
"""
import argparse
import json
import os
from pathlib import Path


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_index(index_path):
    mapping = {}
    for obj in read_jsonl(index_path):
        idx = obj.get('prompt_idx')
        mapping[idx] = obj
    return mapping


def load_gens_for_prompt(cands_dir, prompt_idx):
    path = Path(cands_dir) / 'gens' / f'{prompt_idx}_gens.jsonl'
    gens = {}
    if not path.exists():
        return gens
    for obj in read_jsonl(path):
        gid = obj.get('gen_id')
        gens[gid] = obj.get('text')
    return gens


def parse_gen_id(gen_id: str):
    # expected format g{prompt_idx}_{rank}
    if not gen_id or not gen_id.startswith('g'):
        return None
    try:
        rest = gen_id[1:]
        parts = rest.split('_')
        return int(parts[0])
    except Exception:
        return None


def build(args):
    annotations = list(read_jsonl(args.annotations))
    index_path = Path(args.candidates_dir) / 'index.jsonl'
    if not index_path.exists():
        raise FileNotFoundError(f'index not found: {index_path}')

    index = load_index(index_path)

    # optionally load original source items so we can use raw user content
    source_items = None
    if args.source:
        src_path = Path(args.source)
        if not src_path.exists():
            raise FileNotFoundError(f'source not found: {src_path}')
        source_items = list(read_jsonl(src_path))
        # build searchable plain-text list for matching: join user messages or use input/text/prompt
        source_texts = []
        for it in source_items:
            if 'messages' in it and isinstance(it['messages'], list):
                # join all user role contents
                parts = []
                for m in it['messages']:
                    if m.get('role') == 'user' and 'content' in m:
                        parts.append(m['content'])
                text = ' '.join(parts).strip() if parts else ''
            else:
                text = it.get('input') or it.get('text') or it.get('prompt') or ''
            source_texts.append(text)

    # cache gens loaded per prompt_idx
    gens_cache = {}

    out_path = Path(args.out)
    with open(out_path, 'w', encoding='utf-8') as outf:
        for ann in annotations:
            p_idx = ann.get('prompt_idx')
            prompt_entry = index.get(p_idx)
            if prompt_entry is None:
                # if missing, try to find by parsing gen id
                p_idx_parsed = parse_gen_id(ann.get('chosen_gen_id'))
                prompt_entry = index.get(p_idx_parsed)
            if prompt_entry is None:
                print(f'warning: prompt_idx {p_idx} not found in index, skipping')
                continue

            # prefer original source content (no apply_chat_template) if available
            prompt_text = None
            if source_items is not None:
                # try direct index first
                if p_idx is not None and 0 <= p_idx < len(source_items):
                    orig = source_items[p_idx]
                else:
                    orig = None

                # if direct index failed, try to find source by text matching
                if orig is None:
                    # attempt to match using index prompt string and source_texts
                    idx_prompt_str = str(prompt_entry.get('prompt') or '')
                    found = None
                    for si, stext in enumerate(source_texts):
                        if not stext:
                            continue
                        # check bidirectional containment for robustness
                        if stext in idx_prompt_str or idx_prompt_str in stext:
                            found = si
                            break
                    if found is not None:
                        orig = source_items[found]

                if orig is not None:
                    if 'messages' in orig and isinstance(orig['messages'], list):
                        prompt_list = orig['messages']
                    else:
                        user_text = orig.get('input') or orig.get('text') or orig.get('prompt') or json.dumps(orig, ensure_ascii=False)
                        prompt_list = [{'role': 'user', 'content': user_text}]
                    prompt_text = prompt_list
                else:
                    prompt_text = prompt_entry.get('prompt')
            else:
                # fallback: use the templated prompt string from index (single user content)
                prompt_text = prompt_entry.get('prompt')
            # ensure gens loaded for referenced gen ids
            chosen_id = ann.get('chosen_gen_id')
            rejected_id = ann.get('rejected_gen_id')

            chosen_prompt_idx = parse_gen_id(chosen_id) or p_idx
            rejected_prompt_idx = parse_gen_id(rejected_id) or p_idx

            if chosen_prompt_idx not in gens_cache:
                gens_cache[chosen_prompt_idx] = load_gens_for_prompt(args.candidates_dir, chosen_prompt_idx)
            if rejected_prompt_idx not in gens_cache:
                gens_cache[rejected_prompt_idx] = load_gens_for_prompt(args.candidates_dir, rejected_prompt_idx)

            chosen_text = gens_cache.get(chosen_prompt_idx, {}).get(chosen_id)
            rejected_text = gens_cache.get(rejected_prompt_idx, {}).get(rejected_id)

            if chosen_text is None or rejected_text is None:
                print(f'warning: missing gen text for chosen={chosen_id} or rejected={rejected_id}; skipping')
                continue

            # if prompt_text is already a list of messages, use it directly; otherwise wrap
            if isinstance(prompt_text, list):
                out_prompt = prompt_text
            else:
                out_prompt = [{'role': 'user', 'content': prompt_text}]

            out_obj = {
                'prompt': out_prompt,
                'chosen': [{'role': 'assistant', 'content': chosen_text}],
                'rejected': [{'role': 'assistant', 'content': rejected_text}],
            }
            outf.write(json.dumps(out_obj, ensure_ascii=False) + '\n')

    print('Wrote', out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--annotations', default='annotations.jsonl')
    p.add_argument('--candidates-dir', default='annotation_candidates')
    p.add_argument('--out', default='dpo_pairs.jsonl')
    p.add_argument('--source', default='train_val_sft.jsonl')
    args = p.parse_args()
    build(args)


if __name__ == '__main__':
    main()
