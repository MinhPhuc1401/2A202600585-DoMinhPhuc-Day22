# Hardware Guide — Pick Your Tier

> **DPO không phải SFT.** DPO load *cả policy và reference model* trong memory cùng lúc → ~2× SFT VRAM. Một 7B QLoRA SFT vừa 10 GB; 7B QLoRA DPO cần ~18-20 GB. Đó là lý do hardware guide cho lab này khác Day 21.

Pick a tier, model size, and notebook variant that **actually runs** on your hardware.

## 1. VRAM math for DPO

DPO = policy (trainable) + reference (frozen) + activations + KV cache + optimizer state.

| Base model | Mode | Policy | Reference | Activations | Optim | Total VRAM | Where it fits |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B | LoRA + 4bit | 1.0 GB | 1.0 GB | 2 GB | 1 GB | **~5 GB** | Any laptop GPU, free Colab T4 |
| Qwen2.5-3B | LoRA + 4bit | 2.0 GB | 2.0 GB | 4 GB | 1.5 GB | **~10 GB** | RTX 3060 12GB, free Colab T4 (16GB) ✓ |
| Qwen2.5-7B | LoRA + 4bit | 4.5 GB | 4.5 GB | 6 GB | 3 GB | **~18 GB** | A100 40GB, L4 24GB, RTX 3090/4090 |
| Qwen2.5-14B | LoRA + 4bit | 9 GB | 9 GB | 8 GB | 4 GB | **~30 GB** | A100 40GB ✓, H100 |

> Activations + KV cache scale with `max_length × batch_size`. T4 tier uses `max_length=512, batch=1, grad_accum=8`. BigGPU uses `max_length=1024, batch=2, grad_accum=4`.

## 2. Tier picker

| Available compute | Recommended tier | Notebook variant |
|---|---|---|
| Free Colab T4 (16 GB) | **T4** | `colab/Lab22_DPO_T4.ipynb` (badge in README) |
| Kaggle T4×2 (2 × 16 GB) | T4 (use single GPU) | `colab/Lab22_DPO_T4.ipynb` |
| Colab Pro L4 (22.5 GB) | **BigGPU** | `colab/Lab22_DPO_BigGPU.ipynb` |
| Colab Pro A100 (40 GB) | **BigGPU** | `colab/Lab22_DPO_BigGPU.ipynb` |
| Laptop RTX 3060/4060 (≥ 12 GB) | T4 | `setup-laptop.sh` + `make pipeline` |
| Laptop RTX 3090/4090 (24 GB) | BigGPU | `setup-laptop.sh` + `COMPUTE_TIER=BIGGPU make pipeline` |
| Cloud H100 / 8×A100 | BigGPU + bonus | `requirements-biggpu.txt` + try `Qwen2.5-14B` |
| **No GPU** | — | DPO needs GPU. Use a free Colab T4. CPU DPO would take ~24 hours per epoch. |

## 3. Decision tree

```
Do you have a usable NVIDIA GPU?
├─ No → Free Colab T4 (Runtime → Change runtime type → T4 GPU). Default.
├─ Yes, ≥ 24 GB VRAM (3090/4090/A100/L4)
│   └─ COMPUTE_TIER=BIGGPU → Qwen2.5-7B faithful path, deck-aligned numbers
└─ Yes, 12-23 GB VRAM (3060/3070/4060/3080)
    └─ COMPUTE_TIER=T4 → Qwen2.5-3B, runs comfortably with margin
```

If `make smoke` OOMs on your tier, the next move is:

1. Reduce `max_length` (512 → 384 → 256)
2. Reduce `per_device_train_batch_size` (already 1 — can't go lower)
3. Increase `gradient_accumulation_steps` (8 → 16 → 32) — slower but same effective batch
4. Drop a tier (BigGPU 7B → T4 3B)

## 4. Disk space

- Model weights: 2 GB (3B) or 5 GB (7B) — auto-downloaded by Unsloth
- HF cache: ~10 GB total during a run (model + tokenizer + dataset)
- Adapters: ~50 MB each (sft-mini + dpo + optional orpo)
- GGUF Q4_K_M output: 1.5 GB (3B) or 4 GB (7B)
- **Plan for 25 GB free** before starting. Colab gives 100 GB so this is only a concern locally.

## 5. Network

- Hugging Face downloads can stall behind university firewalls. The setup script tries HF, then falls back to listed mirrors.
- For the optional API judge in NB4, you need outbound HTTPS to `api.openai.com` or `api.anthropic.com`. If blocked, NB4 falls back to manual rubric mode (no points lost — see `rubric.md`).
- vLLM (BigGPU only, optional) needs to download from PyPI (~700 MB). Most universities allow this.

## 6. Why no Docker?

Day 19's lite-vs-Docker split made sense for retrieval (Qdrant runs as a server). For DPO, Docker adds setup pain without removing the *real* constraint, which is GPU access. So we drop Docker and split on compute tier instead. If you want production-style serving (vLLM container), see notebook 05's BigGPU-only optional cell — it's informational, not graded.

## 7. Why not laptop CPU?

DPO on CPU = ~24 hours per epoch for a 1.5B model. The graded core lab targets ≤ 60 min total. CPU is a non-starter. If you have no GPU access at all, free Colab T4 is the answer — sign-in with any Google account and you get 12 hours runtime. Plenty for one full pipeline run.

## 8. Apple Silicon (M1/M2/M3/M4)?

MPS backend on Apple Silicon does not yet support bitsandbytes 4-bit quantization, which Unsloth + DPO depend on for the reference model. Status as of 2026-05: **not viable** as the lab's primary path. Use free Colab T4 instead — it's faster than M-series Mac for this workload and works out of the box.

If you want an Apple-specific stretch experiment, see the bonus challenge file's "MLX-DPO" provocation (write your own DPO loop in MLX-LM — that's a 1-week stretch project, not a 4-hour lab).
