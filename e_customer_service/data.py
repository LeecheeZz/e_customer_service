import json
from typing import Any, Dict, List



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


def format_messages(
    tokenizer: Any,
    messages: Any,
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Format chat messages with the tokenizer's chat template.

    Keeps training and inference prompts consistent across SFT, DPO, and eval
    scripts. If a tokenizer has no chat template helper, fall back to a simple
    role/content transcript so non-chat tokenizers can still run.
    """
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        raise ValueError("messages must be a string or a list of role/content dicts")

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    parts = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        parts.append(f"<{role}>: {content}")
    return "\n".join(parts)


def apply_template(tokenizer: Any, example: Dict) -> Dict:
    """Apply the model's chat template to an example."""
    if "messages" not in example:
        raise ValueError("sample is missing required 'messages' field")

    text = format_messages(
        tokenizer,
        example["messages"],
        add_generation_prompt=False,
        enable_thinking=False,
    )

    return {"text": text}


def samples_to_dataset(samples: List[Dict], tokenizer: Any):
    """Convert parsed samples to a `datasets.Dataset` and apply template.

    Args:
        samples: List of parsed JSON objects (from `load_jsonl`).
        tokenizer: Model tokenizer which exposes `apply_chat_template`.

    Returns:
        A `datasets.Dataset` with a single `text` field.
    """
    from datasets import Dataset

    ds = Dataset.from_list(samples)
    ds = ds.map(
        lambda x: apply_template(tokenizer, x),
        remove_columns=ds.column_names,
        desc="Apply Chat Template",
    )
    return ds
