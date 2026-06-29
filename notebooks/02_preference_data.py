# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="2fdde66e"
# # NB2 — Preference Data
#
# **Stack:** `argilla/ultrafeedback-binarized-preferences-cleaned` + tokenizer apply_chat_template.
# Maps to deck §5.1 (preference data formats) + §5.4 (VN landscape — what exists vs not).
#
# > **Mục tiêu:** load preference dataset, format thành `{prompt, chosen, rejected}` với
# > chat template Qwen2.5, lưu Parquet vào `data/pref/`. Không train gì cả — đây là pure
# > data prep.
# >
# > Deck §5.4 lists VN preference data realities:
# > - **VinaLLaMA / PhoGPT / Vistral**: SFT-only, no published DPO data.
# > - **SeaLLM / Sailor2**: DPO-aligned, Sailor2 has `Sailor2-translated-ultrafeedback-vi`.
# > - **Native VN preference**: gap. **Bonus B** (xem `BONUS-CHALLENGE.md`) là cơ hội build.

# %% [markdown] id="7e9aeb5b"
# ## 0. Setup

# %% colab={"base_uri": "https://localhost:8080/"} id="178e5267" outputId="644952e3-36ee-4fa2-e7e2-784ab87ade08"
import os
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()

if COMPUTE_TIER == "T4":
    PREF_SLICE = 2000
    MAX_LEN = 512
    MAX_PROMPT_LEN = 256
else:
    PREF_SLICE = 5000
    MAX_LEN = 1024
    MAX_PROMPT_LEN = 512

PREF_DATASET = os.environ.get(
    "PREF_DATASET", "argilla/ultrafeedback-binarized-preferences-cleaned"
)

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
ADAPTER_DIR = REPO_ROOT / "adapters" / "sft-mini"
PREF_OUT = REPO_ROOT / "data" / "pref"
PREF_OUT.mkdir(parents=True, exist_ok=True)

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"PREF_DATASET:    {PREF_DATASET}  (slice: {PREF_SLICE})")
print(f"MAX_LEN:         {MAX_LEN}")
print(f"MAX_PROMPT_LEN:  {MAX_PROMPT_LEN}")
print(f"output:          {PREF_OUT}")

# %% [markdown] id="f0c25de3"
# ## 1. Load tokenizer (matches NB1 base model)

# %% colab={"base_uri": "https://localhost:8080/"} id="c90d1423" outputId="753049fa-6736-43b7-f5b0-5698f12cdb72"
from transformers import AutoTokenizer

assert ADAPTER_DIR.exists(), f"NB1 must run first — {ADAPTER_DIR} missing"
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
if getattr(tokenizer, "chat_template", None) is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer: {tokenizer.__class__.__name__}  vocab={tokenizer.vocab_size:,}")

# %% [markdown] id="507b7651"
# ## 2. Load UltraFeedback (English baseline)
#
# **Why English?** UltraFeedback was the canonical preference dataset of the deck
# demo (§7.1: "2k UltraFeedback pairs, 30 min A100, 3.2 → 4.1 helpfulness"). Using
# the same dataset = numbers comparable to deck.
#
# **Why not Vietnamese?** Native VN preference data is a gap (deck §5.4). Translated
# data (`Sailor2-translated-ultrafeedback-vi`) exists but is NLLB-MT-quality, not native.
# Bonus B has the full provocation.

# %% colab={"base_uri": "https://localhost:8080/"} id="6d9a7e6e" outputId="369eb794-d6f2-4655-9a13-ad5c140f4ab0"
from datasets import load_dataset

ds = load_dataset(PREF_DATASET, split=f"train[:{PREF_SLICE}]")
print(f"Loaded {len(ds)} pairs. Columns: {ds.column_names}")

# %% [markdown] id="e4bfdb46"
# ## 3. Format with chat template
#
# DPO Trainer expects `prompt / chosen / rejected` columns. Each must already
# include the chat template tokens — Trainer doesn't apply template internally.

# %% colab={"base_uri": "https://localhost:8080/"} id="6b8fbd16" outputId="61162b04-2127-4d89-dc70-93cc6515837c"
def format_pref(row):
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    prompt_msgs = [{"role": "user", "content": row["prompt"]}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    # `chosen` and `rejected` in this dataset are list-of-dicts with role/content.
    # Take just the assistant turn text (last message).
    chosen_text = row["chosen"][-1]["content"] if isinstance(row["chosen"], list) else row["chosen"]
    rejected_text = row["rejected"][-1]["content"] if isinstance(row["rejected"], list) else row["rejected"]
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


pref = ds.map(format_pref, remove_columns=ds.column_names)
print(f"Formatted: {len(pref)} pairs · cols: {pref.column_names}")

# %% [markdown] id="3c2f3697"
# ### 3a. Inspect 3 examples + token counts (deliverable: NB2 rubric §2)

# %% colab={"base_uri": "https://localhost:8080/"} id="4fdfa24d" outputId="78a4984c-a72a-4c94-bbde-02f7c48210f8"
import textwrap

for i in range(3):
    row = pref[i]
    n_prompt = len(tokenizer(row["prompt"]).input_ids)
    n_chosen = len(tokenizer(row["chosen"]).input_ids)
    n_rejected = len(tokenizer(row["rejected"]).input_ids)
    print(f"\n────── Example {i + 1} ──────")
    print(f"PROMPT ({n_prompt} tok):\n{textwrap.shorten(row['prompt'], 200)}")
    print(f"\nCHOSEN ({n_chosen} tok):\n{textwrap.shorten(row['chosen'], 250)}")
    print(f"\nREJECTED ({n_rejected} tok):\n{textwrap.shorten(row['rejected'], 250)}")
    assert row["chosen"] != row["rejected"], "chosen == rejected — dataset is corrupt!"

# %% [markdown] id="ce4afb8d"
# ### 3b. Length distribution check
#
# Pairs longer than `MAX_LEN` will be truncated by the trainer. If too many are
# clipped, DPO loses signal. Aim for ≥ 80% of pairs fitting.

# %% colab={"base_uri": "https://localhost:8080/"} id="fa5e2e06" outputId="8665daa8-af65-4335-d253-65bed9fdbb1f"
import numpy as np

prompt_lens = np.array([len(tokenizer(p).input_ids) for p in pref["prompt"]])
chosen_lens = np.array([len(tokenizer(c).input_ids) for c in pref["chosen"]])
rejected_lens = np.array([len(tokenizer(r).input_ids) for r in pref["rejected"]])

total_len = prompt_lens + np.maximum(chosen_lens, rejected_lens)
fit_pct = (total_len <= MAX_LEN).mean() * 100

print(f"Prompt:   median={np.median(prompt_lens):.0f}  P95={np.percentile(prompt_lens, 95):.0f}")
print(f"Chosen:   median={np.median(chosen_lens):.0f}  P95={np.percentile(chosen_lens, 95):.0f}")
print(f"Rejected: median={np.median(rejected_lens):.0f}  P95={np.percentile(rejected_lens, 95):.0f}")
print(f"\n{fit_pct:.1f}% of pairs fit in MAX_LEN={MAX_LEN}")
if fit_pct < 80:
    print("⚠  Less than 80% fit. Consider increasing MAX_LEN or filtering long pairs.")

# %% [markdown] id="5a0b3b60"
# ## 4. Save Parquet

# %% colab={"base_uri": "https://localhost:8080/", "height": 116, "referenced_widgets": ["3296b8543f584019b78b4724fa64d528", "897f423c88a84945b2f59adfa1abf64d", "06c2ef4c93864e2687f68b381e1ec653", "501bb464c02c4ab097aa6c0e09ea1c8d", "2827fc0b11714666b3debca9c9786d75", "1a30a25208f043559518e21ad3b616bf", "8b3f5e205c214f8cac6949277f49e887", "003d983334f64d47afdc4cc4763d943e", "a88de0e833024ba3b3bddf77ba900847", "1b234d3b964a4c51a0cbc90525a8293b", "1b0bb9f395fe43f38eb89d51e0855c54", "610ba062869045fd9d7d0ed52bdbb100", "b2e5557e828645c582f8dd4ecc8b478f", "177c499047a4402a9f266fd665f678d1", "df71e4fab7774bc58dbb5ead1f753ebf", "948358e1f8a549e0ad05b3be5d776286", "a07b8dc0d90d4f9aa53f51881d652271", "4c6320b4c0034190876148539643a526", "68b3f97928344d94add4cbd2070ce787", "351cf807b54948d8879798171656e265", "924dc29799574fd5a9737b6c85107fe2", "8ac76869fa6449c5a5a7664f417fcc8d"]} id="f997ea95" outputId="5ae94028-3ec5-46ae-92db-248623463a02"
pref.to_parquet(str(PREF_OUT / "train.parquet"))
print(f"Saved {len(pref)} pairs to {PREF_OUT / 'train.parquet'}")

# Also save a small eval slice (last 50 pairs) for NB4 use.
eval_slice = pref.select(range(len(pref) - 50, len(pref)))
eval_slice.to_parquet(str(PREF_OUT / "eval.parquet"))
print(f"Saved 50 eval pairs to {PREF_OUT / 'eval.parquet'}")

# %% [markdown] id="0a11189d"
# ## 5. Vibe-coding callout
#
# Bạn vừa load 2k cặp English UltraFeedback. Cho VN-aligned model thực sự bạn cần
# preference data tiếng Việt. Có 3 con đường (deck §5.3 — `BONUS-CHALLENGE.md`
# provocation #1 nếu muốn full):
#
# 1. **Translate**: chạy NLLB-3.3B trên 2k cặp này. Quality OK, không native.
# 2. **Generate native**: 200 prompts VN từ VMLU stems → 2 responses (Lab21-SFT vs
#    stronger model như Gemini Flash) → judge với GPT-4o → train DPO trên đó.
# 3. **Hybrid**: 1.8k UltraFeedback + 200 native VN. Best-of-both.
#
# Notebook 03 dùng English baseline (option 0) cho fairness với deck demo. Nếu
# bạn ambitious: thay `data/pref/train.parquet` ở NB3 bằng dataset của bạn — code
# sau đó không đổi.
#
# **Next:** NB3 — train DPO trainer với reward curves.
