import torch
from typing import Any, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: Any = torch.bfloat16,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    load_in_4bit: bool = False,
    bnb_config: Optional[Dict[str, Any]] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from a pretrained checkpoint.

    Args:
        model_path: Path or model identifier.
        device_map: Device mapping for `from_pretrained`.
        torch_dtype: Torch dtype to load the model with.
        trust_remote_code: Whether to trust remote code from hub.
        local_files_only: Only load from local files.

    Returns:
        Tuple of (model, tokenizer).
    """
    if load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, local_files_only=local_files_only
    )

    # ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
