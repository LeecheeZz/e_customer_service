import json
import re


def truncate_after_punct_before_bad(s: str) -> str:
    allowed = re.compile(
        r"""[一-鿿A-Za-z0-9\s，。！？、；：,.!?;:()\[\]{}%+\-\/"“”‘’…—–·]"""
    )
    for i, ch in enumerate(s):
        if not allowed.match(ch):
            for j in range(i - 1, -1, -1):
                if re.match(r"[。\.！？!？?,，、；：;:]", s[j]):
                    return s[: j + 1].strip()
            return s[:i].strip()
    return s.strip()


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
