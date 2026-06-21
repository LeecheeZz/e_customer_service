import json
import re
from pathlib import Path
from typing import Any, Dict


RUNS_DIRNAME = "runs"


def slugify_run_name(name: str) -> str:
    """Return a filesystem-friendly run name."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    value = value.strip("._-")
    if not value:
        raise ValueError("run name must contain at least one alphanumeric character")
    return value


def default_run_name(qlora: bool = True) -> str:
    return "qlora_default" if qlora else "sft_default"


def build_run_paths(output_root: str = "output", run_name: str = "qlora_default") -> Dict[str, Path]:
    run = slugify_run_name(run_name)
    output_root_path = Path(output_root)
    run_dir = output_root_path / RUNS_DIRNAME / run
    return {
        "output_root": output_root_path,
        "run_dir": run_dir,
        "config_path": run_dir / "config.json",
        "data_manifest_path": run_dir / "data_manifest.json",
        "sft_dir": run_dir / "sft",
        "sft_checkpoints_dir": run_dir / "sft" / "checkpoints",
        "sft_final_adapter_dir": run_dir / "sft" / "final_adapter",
        "sft_eval_dir": run_dir / "sft" / "eval",
        "sft_logs_dir": run_dir / "sft" / "logs",
        "dpo_dir": run_dir / "dpo",
        "dpo_checkpoints_dir": run_dir / "dpo" / "checkpoints",
        "dpo_final_adapter_dir": run_dir / "dpo" / "final_adapter",
        "dpo_eval_dir": run_dir / "dpo" / "eval",
        "dpo_logs_dir": run_dir / "dpo" / "logs",
        "artifacts_dir": run_dir / "artifacts",
    }


def ensure_run_dirs(paths: Dict[str, Path]) -> None:
    for key, path in paths.items():
        if key.endswith("_dir") or key in {"output_root", "run_dir"}:
            path.mkdir(parents=True, exist_ok=True)


def path_to_str(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: path_to_str(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [path_to_str(v) for v in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(path_to_str(payload), ensure_ascii=False, indent=2)
    path.write_text(content + "\n", encoding="utf-8")
