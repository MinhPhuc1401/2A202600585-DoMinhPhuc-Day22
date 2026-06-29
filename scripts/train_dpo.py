#!/usr/bin/env python3
"""CLI wrapper for NB3 logic — trains a DPO adapter.

Usage:
    python scripts/train_dpo.py
    python scripts/train_dpo.py --beta 0.05 --output-dir adapters/dpo-b0.05
    python scripts/train_dpo.py --beta 0.5  --output-dir adapters/dpo-b0.50

Mirrors `notebooks/03_dpo_train.py`. Used by `make beta-sweep` for the rigor
add-on +6 (β-sweep mini-experiment).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sft-path", default=str(REPO / "adapters" / "sft-mini"))
    parser.add_argument("--pref-path", default=str(REPO / "data" / "pref" / "train.parquet"))
    parser.add_argument("--output-dir", default=str(REPO / "adapters" / "dpo"))
    args = parser.parse_args()

    tier = os.environ.get("COMPUTE_TIER", "T4").upper()
    if tier == "T4":
        base_model = "unsloth/Qwen2.5-3B-bnb-4bit"
        max_len, max_prompt = 512, 256
        batch, grad_accum = 1, 8
    else:
        base_model = "unsloth/Qwen2.5-7B-bnb-4bit"
        max_len, max_prompt = 1024, 512
        batch, grad_accum = 1, 4

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Tier:       {tier}")
    print(f"Base:       {base_model}")
    print(f"Beta / LR:  {args.beta} / {args.lr}")
    print(f"Output:     {output}")

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()
    from trl import DPOConfig, DPOTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=max_len, dtype=None, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import shutil
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Reset DPO output directory: {output}")

    model = PeftModel.from_pretrained(model, args.sft_path, is_trainable=True)
    model.train()

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )
    print(f"Model class: {model.__class__.__name__}")
    if hasattr(model, "active_adapters"):
        print(f"Active PEFT adapters: {model.active_adapters}")
    print(f"Trainable params for DPO: {trainable_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.4f}%")

    if trainable_params == 0:
        raise RuntimeError(
            "No trainable LoRA parameters found. "
            "Make sure the SFT adapter was loaded with is_trainable=True."
        )

    config = DPOConfig(
        output_dir=str(output.parent / f"{output.name}-checkpoints"),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=max_len,
        max_prompt_length=max_prompt,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=42,
        loss_type="sigmoid",
        report_to="none",
    )

    pref = Dataset.from_parquet(args.pref_path)
    trainer = DPOTrainer(
        model=model, ref_model=None, args=config,
        train_dataset=pref, processing_class=tokenizer,
    )
    train_result = trainer.train()

    trainer.model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    # Headline metrics
    import pandas as pd

    logs = pd.DataFrame(trainer.state.log_history)
    chosen_col = next((c for c in ["rewards/chosen", "train/rewards/chosen", "train_rewards/chosen"] if c in logs.columns), None)
    rejected_col = next((c for c in ["rewards/rejected", "train/rewards/rejected", "train_rewards/rejected"] if c in logs.columns), None)

    metrics = {
        "compute_tier": tier,
        "base_model": base_model,
        "beta": args.beta,
        "lr": args.lr,
        "epochs": args.epochs,
        "final_train_loss": float(train_result.training_loss),
        "end_chosen_reward": float(logs[chosen_col].iloc[-5:].mean()) if chosen_col else None,
        "end_rejected_reward": float(logs[rejected_col].iloc[-5:].mean()) if rejected_col else None,
    }
    if metrics["end_chosen_reward"] is not None and metrics["end_rejected_reward"] is not None:
        metrics["end_reward_gap"] = metrics["end_chosen_reward"] - metrics["end_rejected_reward"]

    (output / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nFinal loss:     {train_result.training_loss:.4f}")
    if "end_reward_gap" in metrics:
        print(f"End reward gap: {metrics['end_reward_gap']:+.3f}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
