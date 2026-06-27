# Day 22 Lab — Grading Rubric (100 pts core + 20 bonus rigor add-ons)

Maps 1-to-1 with the slide deliverable (5 bullets) + repo conventions.
Track-3 Daily Lab weight = 30%.

The lab supports two tiers (T4 vs BigGPU). Both tiers produce identical output formats — each criterion accepts evidence from either tier. You are graded on the clarity of *your own* before/after, not absolute speed (an A100 student and a free-Colab T4 student can both score full marks).

Submit screenshots + notebook output for each criterion. See [`submission/screenshots/README.md`](submission/screenshots/README.md) for the screenshot list.

| # | Notebook | Criterion | Pts |
|---|---|---|---:|
| 1 | `01_sft_mini` | `adapters/sft-mini/adapter_config.json` exists with `lora_alpha: 32, r: 16` | 6 |
| 1 | `01_sft_mini` | SFT loss curve shows monotonic decrease over 1 epoch | 6 |
| 1 | `01_sft_mini` | At least 1 sample generation from SFT model printed in NB1 (sanity check) | 5 |
| 2 | `02_preference_data` | `data/pref/train.parquet` written with `prompt / chosen / rejected` columns | 6 |
| 2 | `02_preference_data` | 3 inspected examples printed; chosen ≠ rejected on each | 6 |
| 3 | `03_dpo_train` | `adapters/dpo/adapter_config.json` exists, distinct from sft-mini | 6 |
| 3 | `03_dpo_train` | Reward gap plot shows `chosen − rejected` increasing | 12 |
| 3 | `03_dpo_train` | Both `chosen` and `rejected` reward curves plotted separately + interpreted in REFLECTION | 10 |
| 4 | `04_compare_and_eval` | Side-by-side table with ≥ 8 prompts × 2 model outputs (SFT, SFT+DPO) | 8 |
| 4 | `04_compare_and_eval` | Win/loss/tie summary reported (manual or judge); 4 helpfulness + 4 safety mix | 7 |
| — | Core | Reproducible from clean `setup-laptop.sh` + `make pipeline` (NB1–NB4, or Colab Run-all) | 5 |
| — | Reflection | `submission/REFLECTION.md` core sections present, ≥ 150 words on §3 + §6 | 15 |
| — | Reflection | Section 3 (Reward curves) interprets *both* chosen and rejected trajectories (deck §3.4) | 5 |
| — | Verify | `make verify` exits 0 (core gatekeeper passes; NB5/NB6 not required) | 3 |
| **Subtotal** | | **Core (NB1–NB4)** | **100** |

## Optional rigor add-ons (+20 pts, listed but unranked)

These are *individually optional* — pick any combination, no minimum. Designed for honors students who finish core early. Not graded as pass/fail; instructor awards proportional to depth + clarity.

| Add-on | Pts | What it asks |
|---|---:|---|
| **NB5 — GGUF deploy** | +6 | Merge adapter, export `gguf/*.gguf` (< 5 GB Q4\_K\_M) + llama.cpp smoke shows coherent VN |
| **NB6 — benchmark** | +8 | IFEval/GSM8K/MMLU/AlpacaEval-lite on SFT vs SFT+DPO, 4-bar plot + REFLECTION §7 alignment-tax read |
| **β-sweep mini-experiment** | +6 | Run NB3 with β ∈ {0.05, 0.1, 0.5}; plot reward gap & win-rate vs β; ≥ 100-word interpretation |
| **HuggingFace Hub push** | +5 | Push DPO adapter to HF with model card. Submission Option B. |
| **GGUF release published** | +3 | Push the merged GGUF to HF with quantization variants (Q4_K_M + Q5_K_M minimum) |
| **MMLU full coverage** | +3 | Run NB6 with `LIMIT_MMLU=14000` (full); compare against the sampled-500 result |
| **Weights & Biases run link** | +2 | Add a public `wandb` link to your training run with all curves visible |
| **Cross-judge comparison** | +4 | Run NB4 + NB6 AlpacaEval-lite with both gpt-4o-mini AND claude-haiku, report disagreement rate |
| **Total** | **+37** | (capped at +20) |

The bonus rigor add-ons do **not** affect your core grade negatively; missing them is fine. They reward extra effort with proportional credit.

## Ungraded creative bonus

See [`BONUS-CHALLENGE.md`](BONUS-CHALLENGE.md) — completely separate, no points, no rubric. Sandbox to brainstorm + try ideas. A strong submission earns a written instructor review on *judgment*, not points.

## Submission Options A / B / C

(Same convention as Day 21 sibling lab, adapted for DPO artifacts.)

### Option A — Lightweight (default)
- GitHub repo (public) with executed notebooks (output cells preserved)
- `submission/screenshots/` (≥ 3 PNG/JPG: NB1 loss, NB3 reward curves, NB4 side-by-side)
- `submission/REFLECTION.md` (6 sections, ≥ 150 words on §3 + §6)
- `make verify` passes

### Option B — Professional (+5 bonus pts via "HuggingFace Hub push")
- All of Option A
- `adapters/dpo/` pushed to HF Hub: `huggingface-cli upload <user>/lab22-dpo-vn ./adapters/dpo`
- HF model card with: base, dataset, hyperparameters, evaluation results
- Repo `README.md` links to the HF model

### Option C — Code-only (no weights)
- All of Option A but skip pushing weights
- Useful for students who have hit Colab storage limits
- No bonus points; full core grade still possible

## Submission

**No PR. Submit a public GitHub URL into the VinUni LMS Day-22 box.**

1. Push your work to `<your-username>/Day22-Track3-DPO-Alignment-Lab` (forked or fresh repo — both fine), set repo **public**.
2. Include:
   - 5 executed notebooks (`.ipynb` with output cells preserved) OR a single executed `colab/Lab22_DPO_T4.ipynb` if you used the Colab path
   - `submission/screenshots/` — 6 required + 3 optional images
   - `submission/REFLECTION.md` — all 6 sections filled, your own numbers
   - **Optional:** `bonus/` folder for the ungraded creative challenge
3. Run `make verify` locally — it will list missing artifacts, exit non-zero until you fix them.
4. Paste the public repo URL into the LMS submission box.
5. **Keep the repo public until grades are released.** Private = 0.

## Late policy / regrade

Standard Track-3 policy applies. **Deadline:** 23:59 next day. **−10% per day late, 0 after 3 days late.** Regrade requests within 1 week of grade release.

## Why these criteria?

The criteria above map directly to the deck:
- §3.4 (DPO failure modes) → "interpret both chosen and rejected trajectories"
- §5.2 (TRL implementation) → adapter must use deck-specified hyperparameters
- §7.1 (Demo) → side-by-side comparison with ≥ 8 prompts mirrors the deck demo
- §7.2b (Tulu 3 stats) → REFLECTION encourages reporting your own equivalent numbers

If you can defend each criterion against the deck, you understand the lab. If you can't, re-read the deck before submitting.
