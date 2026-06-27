#!/usr/bin/env bash
# Colab one-line setup. Drops the venv layer (Colab is already Python-isolated)
# and skips the CUDA probe (Colab notebooks already display GPU info in their
# UI). Just installs deps + auto-tier-detects + converts notebooks.
#
# Usage in Colab cell:
#     !git clone https://github.com/<user>/Day22-Track3-DPO-Alignment-Lab.git
#     %cd Day22-Track3-DPO-Alignment-Lab
#     !bash setup-colab.sh

set -euo pipefail

echo "[colab] Day 22 lab — Colab setup"
echo "[colab] Stack: unsloth + trl + peft + bitsandbytes + llama-cpp-python"
echo

# ── 1. Auto-detect tier from torch.cuda ─────────────────────────────────
TIER=$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("CPU")
else:
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print("BIGGPU" if gb >= 22 else "T4")
PY
)
echo "[colab] Detected tier: $TIER"

case "$TIER" in
  CPU)
    echo "[colab] No GPU detected. Runtime → Change runtime type → T4 GPU, then retry."
    exit 1
    ;;
  T4)
    echo "[colab] T4 (or similar 16 GB) tier — using Qwen2.5-3B"
    ;;
  BIGGPU)
    echo "[colab] BigGPU (A100 / L4) tier — using Qwen2.5-7B"
    ;;
esac

# ── 2. Install deps ─────────────────────────────────────────────────────
# Colab pre-installs torch + transformers; let pip resolve compatible versions.
# Unsloth's installer picks the right CUDA wheel.
pip install -q -r requirements.txt

if [ "$TIER" = "BIGGPU" ]; then
  echo "[colab] Installing BigGPU extras (vllm, flash-attn) — may take 3-5 min"
  pip install -q -r requirements-biggpu.txt || echo "[colab] WARNING: vllm/flash-attn install failed; vLLM cell in NB5 will skip"
fi

# ── 3. Convert Jupytext sources ─────────────────────────────────────────
jupytext --to notebook --update notebooks/*.py 2>/dev/null || jupytext --to notebook notebooks/*.py

# ── 4. .env scaffold (optional in Colab) ────────────────────────────────
[ -f .env ] || cp .env.example .env
# Patch COMPUTE_TIER in .env to match auto-detect
sed -i.bak "s/^COMPUTE_TIER=.*/COMPUTE_TIER=$TIER/" .env && rm .env.bak

# ── 5. Make output folders ──────────────────────────────────────────────
mkdir -p data/pref adapters/sft-mini adapters/dpo gguf

cat <<EOF

[colab] Done — tier = $TIER.

In Colab, you can now either:

    1. Open notebooks/01_sft_mini.py — Jupytext will convert on first edit
    2. Or run the stitched single-file Colab notebook:
         - T4 tier:     colab/Lab22_DPO_T4.ipynb
         - BigGPU tier: colab/Lab22_DPO_BigGPU.ipynb
    3. Or use the make targets:

         !make smoke           # quick verification
         !make pipeline        # full run (~45 min T4 / ~30 min A100)

Tip: read VIBE-CODING.md before starting NB1.

EOF
