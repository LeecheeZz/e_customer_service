#!/usr/bin/env python3
"""Plot training and evaluation loss from a HuggingFace/TRL trainer_state.json.

Usage example:
  python scripts/plot_losses.py --checkpoint_dir output_customer_service/checkpoint-366

The script will look for `trainer_state.json` inside the given checkpoint dir,
extract `log_history` entries with `loss` and `eval_loss`, draw a complete
loss curve (train + val) and save it as `loss_curve.png` in the same dir.
"""
import argparse
import json
import os
from typing import List, Dict

import matplotlib.pyplot as plt


def load_trainer_state(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("log_history", [])


def extract_losses(log_history: List[Dict]):
    train_x = []
    train_y = []
    val_x = []
    val_y = []

    for entry in log_history:
        # common keys: 'loss', 'eval_loss', 'step', 'epoch'
        step = entry.get("step")
        epoch = entry.get("epoch")

        x = epoch if epoch is not None else step

        if "loss" in entry:
            train_x.append(x)
            train_y.append(entry["loss"])

        if "eval_loss" in entry:
            val_x.append(x)
            val_y.append(entry["eval_loss"])

    return (train_x, train_y), (val_x, val_y)


def plot_and_save(train, val, out_path: str, title: str = "Training and Validation Loss"):
    (tx, ty) = train
    (vx, vy) = val

    plt.figure(figsize=(10, 6), dpi=150)

    if len(tx) > 0:
        plt.plot(tx, ty, label="train loss", color="#2383c7")
    if len(vx) > 0:
        plt.plot(vx, vy, label="val loss", color="#ff7f0e")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()

    # Tight layout + save
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def find_trainer_state(checkpoint_dir: str) -> str:
    candidate = os.path.join(checkpoint_dir, "trainer_state.json")
    if os.path.exists(candidate):
        return candidate
    # fallback: search files in dir
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith("trainer_state.json"):
            return os.path.join(checkpoint_dir, fname)
    raise FileNotFoundError(f"trainer_state.json not found in {checkpoint_dir}")


def main():
    p = argparse.ArgumentParser(description="Plot train/val loss from trainer_state.json")
    p.add_argument("-d", "--checkpoint_dir", required=True, help="Checkpoint directory containing trainer_state.json")
    p.add_argument("--out_name", default="loss_curve.png", help="Output image filename (saved inside checkpoint_dir)")
    p.add_argument("--title", default="Training and Validation Loss", help="Figure title")
    args = p.parse_args()

    checkpoint_dir = args.checkpoint_dir
    trainer_state_path = find_trainer_state(checkpoint_dir)

    log_history = load_trainer_state(trainer_state_path)

    train, val = extract_losses(log_history)

    out_path = os.path.join(checkpoint_dir, args.out_name)

    if len(train[0]) == 0 and len(val[0]) == 0:
        raise RuntimeError(f"No loss entries found in {trainer_state_path}")

    plot_and_save(train, val, out_path, title=args.title)

    print(f"Saved loss curve to: {out_path}")


if __name__ == "__main__":
    main()
