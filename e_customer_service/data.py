import json
from typing import Any, Dict, List

from datasets import Dataset


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSON Lines file into a list of dicts.

    Args:
        path: Path to a JSONL file.

    Returns:
        A list where each element is a parsed JSON object.
    """
    samples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def apply_template(tokenizer: Any, example: Dict) -> Dict:
    """Apply the model's chat template to an example.

    The function mirrors the behavior from the original demo: it calls
    `tokenizer.apply_chat_template` and strips empty thinking markers.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    
    return {"text": text}


def samples_to_dataset(samples: List[Dict], tokenizer: Any) -> Dataset:
    """Convert parsed samples to a `datasets.Dataset` and apply template.

    Args:
        samples: List of parsed JSON objects (from `load_jsonl`).
        tokenizer: Model tokenizer which exposes `apply_chat_template`.

    Returns:
        A `datasets.Dataset` with a single `text` field.
    """
    ds = Dataset.from_list(samples)
    ds = ds.map(
        lambda x: apply_template(tokenizer, x),
        remove_columns=ds.column_names,
        desc="Apply Chat Template",
    )
    return ds
