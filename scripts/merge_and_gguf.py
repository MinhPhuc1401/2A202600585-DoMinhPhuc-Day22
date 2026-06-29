#!/usr/bin/env python3
"""CLI wrapper for NB5 logic — merge adapter + quantize to GGUF.

Usage:
    python scripts/merge_and_gguf.py
    python scripts/merge_and_gguf.py --quant q5_k_m
    python scripts/merge_and_gguf.py --quant q4_k_m --quant q5_k_m --quant q8_0

Mirrors `notebooks/05_merge_deploy_gguf.py` cells 1-3. Used if you want to add
extra GGUF tiers (the +3 'GGUF release published' rigor add-on).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-path", default=str(REPO / "adapters" / "sft-mini"))
    parser.add_argument("--dpo-path", default=str(REPO / "adapters" / "dpo"))
    parser.add_argument("--merged-output", default=str(REPO / "adapters" / "merged-fp16"))
    parser.add_argument("--gguf-output", default=str(REPO / "gguf"))
    parser.add_argument("--quant", action="append", default=None,
                        help="Quantization tier(s). Repeat for multiple. Default: q4_k_m")
    args = parser.parse_args()

    quants = args.quant or ["q4_k_m"]

    tier = os.environ.get("COMPUTE_TIER", "T4").upper()
    base = (
        "unsloth/Qwen2.5-3B-bnb-4bit" if tier == "T4"
        else "unsloth/Qwen2.5-7B-bnb-4bit"
    )
    max_len = 512 if tier == "T4" else 1024

    Path(args.merged_output).mkdir(parents=True, exist_ok=True)
    Path(args.gguf_output).mkdir(parents=True, exist_ok=True)

    print(f"Tier: {tier}  base: {base}  quants: {quants}")

    from peft import PeftModel
    from unsloth import FastLanguageModel
    import gc
    import torch

    # Step 1: load clean base in 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=max_len, dtype=None, load_in_4bit=True,   # Must be True so Unsloth enables LoRA merge patches
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dpo_path_obj = Path(args.dpo_path)
    if not dpo_path_obj.exists():
        raise FileNotFoundError(f"Final DPO adapter not found: {dpo_path_obj}")

    model = PeftModel.from_pretrained(model, str(dpo_path_obj), is_trainable=False)
    print(f"Loaded final SFT+DPO adapter from: {dpo_path_obj}")
    print(f"Model class: {model.__class__.__name__}")
    if hasattr(model, "active_adapters"):
        print(f"Active PEFT adapters: {model.active_adapters}")

    # Step 2: save merged FP16
    # Merge the final SFT+DPO adapter into the base weights
    # and save a Hugging Face FP16 model.
    model.save_pretrained_merged(
        args.merged_output, tokenizer, save_method="merged_16bit",
    )
    print(f"Saved merged FP16 model to: {args.merged_output}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: GGUF quantize each tier
    # Reload the merged model in full precision (load_in_4bit=False)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.merged_output,
        max_seq_length=max_len, dtype=None, load_in_4bit=False,
    )

    for q in quants:
        print(f"Quantizing to GGUF {q}...")
        model.save_pretrained_gguf(
            args.gguf_output, tokenizer, quantization_method=q,
        )

    print(f"\nGGUF files in {args.gguf_output}:")
    for p in sorted(Path(args.gguf_output).iterdir()):
        if p.suffix == ".gguf":
            print(f"  {p.name:50s}  {p.stat().st_size / 1e6:>8.1f} MB")


if __name__ == "__main__":
    main()
