#!/usr/bin/env bash
# Local laptop setup — venv + deps + CUDA probe + smoke test.
# Pre-req: NVIDIA GPU with ≥ 12 GB VRAM (T4 tier) or ≥ 24 GB (BigGPU tier).
# ~5 min on a clean machine; longer if first time installing PyTorch wheels.

set -euo pipefail

echo "[laptop] Day 22 lab setup — DPO/ORPO Alignment"
echo "[laptop] Stack: unsloth + trl + peft + bitsandbytes + llama-cpp-python"
echo

# ── 1. Python ───────────────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || { echo "[laptop] python3 not found. Install Python 3.10–3.12."; exit 1; }
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[laptop] Python $PY_VER detected"
case "$PY_VER" in
  3.10|3.11|3.12) PYREQ="$PY_VER" ;;
  *)
    if command -v uv >/dev/null 2>&1; then
      echo "[laptop] Python $PY_VER out of range (need 3.10-3.12) -- uv will fetch 3.12 for the venv"
      PYREQ="3.12"
    else
      echo "[laptop] ERROR: Python $PY_VER unsupported -- Unsloth needs 3.10-3.12."
      echo "[laptop]   Install uv (https://docs.astral.sh/uv/) and rerun (auto-fetches 3.12), or use pyenv."
      exit 1
    fi
    ;;
esac

# ── 2. venv ─────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  if command -v uv >/dev/null 2>&1; then
    echo "[laptop] Creating venv with uv (faster)"
    uv venv .venv --python "$PYREQ"
  else
    echo "[laptop] Creating venv with python -m venv"
    python3 -m venv .venv
  fi
fi
# shellcheck source=/dev/null
source .venv/bin/activate

# ── 3. Install deps ─────────────────────────────────────────────────────
if command -v uv >/dev/null 2>&1; then
  uv pip install -r requirements.txt
else
  pip install -q -U pip
  pip install -q -r requirements.txt
fi

# ── 4. CUDA / GPU probe ─────────────────────────────────────────────────
echo
python - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("[laptop] torch missing — pip install failed?")
    sys.exit(1)

if not torch.cuda.is_available():
    print("[laptop] WARNING: torch.cuda.is_available() == False")
    print("[laptop] DPO needs a CUDA GPU. Free Colab T4 is the recommended fallback.")
    print("[laptop] If you have a GPU and this still fails, your torch wheel likely doesn't match your CUDA.")
    sys.exit(0)

dev = torch.cuda.get_device_properties(0)
gb = dev.total_memory / (1024**3)
print(f"[laptop] GPU: {dev.name}")
print(f"[laptop] VRAM: {gb:.1f} GB")
print(f"[laptop] CUDA capability: {dev.major}.{dev.minor}")

if gb < 12:
    print("[laptop] WARNING: < 12 GB VRAM. DPO will OOM on 3B model.")
    print("[laptop]          Recommend free Colab T4 instead.")
elif gb < 24:
    print("[laptop] T4 tier (3B model) viable. Recommended COMPUTE_TIER=T4.")
else:
    print("[laptop] BigGPU tier (7B model) viable. Set COMPUTE_TIER=BIGGPU in .env.")
PY

# ── 5. Convert Jupytext sources to .ipynb ───────────────────────────────
echo
echo "[laptop] Converting Jupytext .py → .ipynb"
jupytext --to notebook --update notebooks/*.py 2>/dev/null || jupytext --to notebook notebooks/*.py

# ── 6. .env scaffold ────────────────────────────────────────────────────
[ -f .env ] || { cp .env.example .env; echo "[laptop] Created .env — edit to set COMPUTE_TIER and API keys"; }

# ── 7. Make folders that .gitkeep created exist ─────────────────────────
mkdir -p data/pref adapters/sft-mini adapters/dpo gguf

cat <<EOF

[laptop] Done. Activate the venv and start working:

    source .venv/bin/activate
    make smoke           # 2-step training run on each notebook to verify
    make pipeline        # full pipeline: sft → data → dpo → eval → deploy

Tip: read VIBE-CODING.md before starting NB1 — 5-10 min, tells you which
subtasks to delegate to AI and which to think through yourself.

EOF
