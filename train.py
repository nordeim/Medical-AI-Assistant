#!/usr/bin/env python3
"""
Minimal train.py smoke-run stub.

Behavior expected by run_repro.sh:
- CLI args: --model_name_or_path, --train_file, --validation_file, --output_dir,
            --max_steps, --per_device_train_batch_size, --seed, --logging_strategy, --logging_steps
- Writes a JSONL log file at <output_dir>/training_log.jsonl with one JSON line per logged step:
    {"step": <int>, "loss": <float>}
- Saves a compact adapter checkpoint under <output_dir>/adapter (state_dict file)
This is minimal and intended for smoke/deterministic runs; replace with your full training logic as needed.
"""

import argparse
import os
import json
import math
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import load_dataset
import random
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--validation_file", type=str, required=False, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_strategy", type=str, default="steps")
    p.add_argument("--logging_steps", type=int, default=1)
    return p.parse_args()

def set_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # recommended deterministic flags (may slow performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batch, tokenizer, max_length=256):
    texts = [ex.get("text") or ex.get("input") or ex.get("prompt") or "" for ex in batch]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return enc

def main():
    args = parse_args()
    set_determinism(args.seed)

    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "adapter"
    os.makedirs(adapter_dir, exist_ok=True)

    # prepare logging file
    log_path = output_dir / "training_log.jsonl"
    # ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # load tokenizer and model (small model recommended for smoke runs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load dataset (expects JSONL with at least a "text" field)
    data_files = {"train": args.train_file}
    ds = load_dataset("json", data_files=data_files)
    train_ds = ds["train"]

    # DataLoader
    def map_example_to_text(example):
        # Allow a few common keys; reduce to single "text" field
        if "text" in example:
            return {"text": example["text"]}
        if "prompt" in example:
            return {"text": example["prompt"]}
        # If present as Q/A style, try to join fields
        if "question" in example and "answer" in example:
            return {"text": f"Q: {example['question']}\nA: {example['answer']}"}
        # fallback: string representation
        return {"text": str(example)}
    train_ds = train_ds.map(map_example_to_text)
    train_ds = train_ds.shuffle(seed=args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop: iterate until max_steps reached
    step = 0
    epoch = 0
    log_lines = []
    it = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            it = iter(train_loader)
            batch = next(it)
        # move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss_value = loss.detach().cpu().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += 1

        # Logging according to logging_steps
        if args.logging_strategy == "steps" and (step % args.logging_steps == 0):
            line = {"step": step, "loss": float(loss_value)}
            # append to log file
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            print(f"[step {step}] loss={loss_value:.6f}")

    # Save a compact "adapter" checkpoint (for smoke-run: save model state_dict's small subset)
    # For an actual LoRA/PEFT adapter, replace this save with adapter.save_pretrained(...)
    ckpt_path = adapter_dir / "adapter_state.pt"
    # Save only a small sampled subset to keep it compact in smoke runs: here save model's first linear weight if exists
    state = {}
    # select up to N parameters to save deterministically
    saved_keys = []
    for i, (k, v) in enumerate(model.state_dict().items()):
        state[k] = v.cpu()
        saved_keys.append(k)
        # keep it small for smoke-run; break after saving a few tensors
        if i >= 4:
            break
    torch.save(state, ckpt_path)

    # Finalize: write a small metadata file
    metadata = {
        "model_name_or_path": args.model_name_or_path,
        "max_steps": args.max_steps,
        "batch_size": args.per_device_train_batch_size,
        "seed": args.seed,
        "saved_keys": saved_keys,
    }
    with open(adapter_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete. Logs written to:", str(log_path))
    print("Adapter checkpoint saved to:", str(ckpt_path))

if __name__ == "__main__":
    main()
