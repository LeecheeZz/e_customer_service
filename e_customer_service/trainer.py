import os
import logging
from typing import Any, Optional

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


logger = logging.getLogger(__name__)


def create_peft_config(
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    target_modules=None,
) -> LoraConfig:
    """Create a `LoraConfig` with reasonable defaults used in the demo."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        target_modules=target_modules,
    )


def create_sft_config(output_dir: str, **kwargs) -> SFTConfig:
    """Create an `SFTConfig` used by `trl.SFTTrainer`.

    Additional kwargs override the defaults.
    """
    params = dict(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        fp16=False,
        bf16=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=5,
        logging_steps=5,
        report_to="none",
        dataloader_num_workers=4,
        packing=False,
    )

    params.update(kwargs)
    return SFTConfig(**params)


def run_training(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    peft_config: LoraConfig,
    training_args: SFTConfig,
    output_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> SFTTrainer:
    """Run SFT training and optionally save LoRA weights.

    Args:
        model: The model instance to train.
        tokenizer: Tokenizer used for preprocessing and saving.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        peft_config: LoRA configuration.
        training_args: SFT training arguments.
        output_dir: Base output directory (used to derive save_dir if not provided).
        save_dir: Explicit directory to save LoRA weights and tokenizer.

    Returns:
        The instantiated and trained `SFTTrainer`.
    """
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Trainable parameters:")
    trainer.model.print_trainable_parameters()

    logger.info("Starting training...")
    trainer.train()

    if save_dir is None and output_dir is not None:
        save_dir = os.path.join(output_dir, "lora")

    if save_dir is not None:
        logger.info("Saving LoRA weights to %s", save_dir)
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)

    return trainer
