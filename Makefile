## Day 22 — DPO/ORPO Alignment lab.
## Tier-aware via COMPUTE_TIER (T4 default, BIGGPU optional).

VENV     := .venv
PY       := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
JUPYTEXT := $(VENV)/bin/jupytext
PYTEST   := $(VENV)/bin/pytest
JUPYTER  := $(VENV)/bin/jupyter

# If running on Colab there's no venv — fall back to system python.
ifeq ($(wildcard $(PY)),)
  PY := python
  PIP := pip
  JUPYTEXT := jupytext
  PYTEST := pytest
  JUPYTER := jupyter
endif

.DEFAULT_GOAL := help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nDay 22 DPO Lab targets:\n"} \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ─────────────────────────────────────────────────────────────
# Setup — auto-detect Colab vs laptop
# ─────────────────────────────────────────────────────────────

setup: ## Auto-detect Colab vs laptop, install deps + smoke check
	@if [ -d /content ]; then \
	  bash setup-colab.sh; \
	else \
	  bash setup-laptop.sh; \
	fi

smoke: ## 2-step training run on each notebook to verify imports + GPU
	@$(JUPYTEXT) --to notebook --update notebooks/*.py 2>/dev/null || true
	@$(PY) scripts/verify.py --smoke

# ─────────────────────────────────────────────────────────────
# Pipeline — core NB1-NB4 (NB5/NB6 optional via pipeline-full)
# ─────────────────────────────────────────────────────────────

sft: ## NB1 — build SFT-mini checkpoint (~10 min T4 / ~5 min A100)
	@$(JUPYTEXT) --to notebook --update notebooks/01_sft_mini.py
	@$(JUPYTER) nbconvert --to notebook --execute --inplace notebooks/01_sft_mini.ipynb

data: ## NB2 — preference data prep (~2 min)
	@$(JUPYTEXT) --to notebook --update notebooks/02_preference_data.py
	@$(JUPYTER) nbconvert --to notebook --execute --inplace notebooks/02_preference_data.ipynb

dpo: ## NB3 — full DPO training (~30 min T4 / ~20 min A100)
	@$(JUPYTEXT) --to notebook --update notebooks/03_dpo_train.py
	@$(JUPYTER) nbconvert --to notebook --execute --inplace notebooks/03_dpo_train.ipynb

eval: ## NB4 — side-by-side comparison + judge
	@$(JUPYTEXT) --to notebook --update notebooks/04_compare_and_eval.py
	@$(JUPYTER) nbconvert --to notebook --execute --inplace notebooks/04_compare_and_eval.ipynb

deploy: ## NB5 (OPTIONAL/bonus) — merge + GGUF + llama.cpp smoke
	@$(JUPYTEXT) --to notebook --update notebooks/05_merge_deploy_gguf.py
	@$(JUPYTER) nbconvert --to notebook --execute --inplace notebooks/05_merge_deploy_gguf.ipynb

bench: ## NB6 (OPTIONAL/bonus) — IFEval/GSM8K/MMLU + 4-bar plot (~30 min T4)
	@$(JUPYTEXT) --to notebook --update notebooks/06_benchmark.py
	@$(JUPYTER) nbconvert --to notebook --execute --inplace notebooks/06_benchmark.ipynb

pipeline: sft data dpo eval ## Run the 4 CORE notebooks (NB1-NB4, ~30 min T4)

pipeline-full: sft data dpo eval deploy bench ## Core + OPTIONAL NB5 (GGUF) + NB6 (benchmark)

# ─────────────────────────────────────────────────────────────
# Bonus rigor add-on (+6 pts)
# ─────────────────────────────────────────────────────────────

beta-sweep: ## Re-run NB3 with beta in {0.05, 0.1, 0.5}
	@$(PY) scripts/train_dpo.py --beta 0.05 --output-dir adapters/dpo-b0.05
	@$(PY) scripts/train_dpo.py --beta 0.1  --output-dir adapters/dpo-b0.10
	@$(PY) scripts/train_dpo.py --beta 0.5  --output-dir adapters/dpo-b0.50
	@$(PY) scripts/eval_judge.py --sweep-dir adapters --output submission/screenshots/bonus-beta-sweep.png

# ─────────────────────────────────────────────────────────────
# Verify + clean
# ─────────────────────────────────────────────────────────────

verify: ## Pre-submission gatekeeper — checks artifacts + REFLECTION edited
	@$(PY) scripts/verify.py

lab: ## Open Jupyter Lab (laptop only)
	@$(JUPYTEXT) --to notebook --update notebooks/*.py 2>/dev/null || true
	@$(JUPYTER) lab --notebook-dir=notebooks --ServerApp.token='' --no-browser

test: ## Run pytest (smoke tests only — no full training)
	@$(PYTEST) -q scripts/

clean: ## Wipe adapters/, data/pref/, gguf/, __pycache__
	rm -rf adapters/sft-mini adapters/dpo adapters/dpo-b* adapters/orpo \
	       data/pref/ gguf/ \
	       notebooks/*.ipynb notebooks/.ipynb_checkpoints \
	       __pycache__ scripts/__pycache__

clean-all: clean ## Wipe everything including venv + HF cache
	rm -rf $(VENV) ~/.cache/huggingface/hub

.PHONY: help setup smoke sft data dpo eval deploy bench pipeline pipeline-full beta-sweep verify lab test clean clean-all
