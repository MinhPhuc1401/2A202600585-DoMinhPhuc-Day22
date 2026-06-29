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

# %% [markdown] id="ce480e26"
# # NB3 — DPO Training (the main event)
#
# **Stack:** TRL `DPOTrainer` + `DPOConfig(beta=0.1, lr=5e-7)` from deck §5.2.
# Maps to deck §3 (DPO derivation), §3.4 (failure modes — read closely!), §5.2 (TRL impl).
#
# > **Mục tiêu:** train DPO adapter on top of NB1 SFT-mini. Plot reward curves
# > (cả `chosen_rewards` và `rejected_rewards`). Save adapter to `adapters/dpo/`.
# >
# > Đây là **the** notebook quan trọng nhất của lab — 25/100 pts đến từ đây.
# > Đặc biệt là: **plot cả 2 curve riêng biệt**, không chỉ reward gap (deck §3.4).

# %% [markdown] id="294f159f"
# ## 0. Setup

# %% colab={"base_uri": "https://localhost:8080/"} id="15f93ec5" outputId="cecb1bd5-5dac-4a6a-dc34-5d3a1cc17044"
import os
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()

if COMPUTE_TIER == "T4":
    BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
    MAX_LEN = 512
    MAX_PROMPT_LEN = 256
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 8
else:
    BASE_MODEL = "unsloth/Qwen2.5-7B-bnb-4bit"
    MAX_LEN = 1024
    MAX_PROMPT_LEN = 512
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 4

# Hyperparameters from deck §5.2 lines 849–886
BETA = float(os.environ.get("DPO_BETA", "0.1"))
LR = float(os.environ.get("DPO_LR", "5e-7"))
EPOCHS = int(os.environ.get("DPO_EPOCHS", "1"))

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
DPO_OUT = REPO_ROOT / "adapters" / "dpo"
PREF_PATH = REPO_ROOT / "data" / "pref" / "train.parquet"

DPO_OUT.mkdir(parents=True, exist_ok=True)

assert SFT_PATH.exists(), f"NB1 must run first — {SFT_PATH} missing"
assert PREF_PATH.exists(), f"NB2 must run first — {PREF_PATH} missing"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"BASE_MODEL:      {BASE_MODEL}")
print(f"DPO hyperparams: beta={BETA}  lr={LR}  epochs={EPOCHS}")
print(f"max_length:      {MAX_LEN}  (prompt={MAX_PROMPT_LEN})")
print(f"effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"SFT input:       {SFT_PATH}")
print(f"output:          {DPO_OUT}")

# %% id="affdd8e5"
import torch

assert torch.cuda.is_available(), "DPO needs a CUDA GPU. See HARDWARE-GUIDE.md."

# %% [markdown] id="c58d7ee8"
# ## 1. Load policy + reference (the VRAM-doubling part)
#
# **Critical:** DPO needs the policy (trainable) AND a frozen reference (no grad).
# The reference is the SFT model at step 0; we load it twice. Unsloth's 4-bit base
# is shared across copies — only the LoRA adapter differs.

# %% colab={"base_uri": "https://localhost:8080/"} id="9788b1a9" outputId="b5633732-e9c4-4608-9e08-0c3d70c90bd4"
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from peft import PeftModel

# Policy — gets new DPO LoRA adapter on top of SFT LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
    attn_implementation="sdpa",
)

# Apply Qwen-2.5 chat template (base model tokenizers don't have it by default)
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)
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

# Load SFT adapter on top of base
model = PeftModel.from_pretrained(model, str(SFT_PATH), is_trainable=True)
print(f"Policy: {model.__class__.__name__} with SFT adapter loaded")

# %% colab={"base_uri": "https://localhost:8080/"} id="3a8bc06f" outputId="e541217e-11db-4d50-a286-670e5e992b90"
# Wrap policy with NEW LoRA adapter for DPO updates (don't merge SFT — keep stacked)
# Unsloth re-applies LoRA on top of the existing PeftModel.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)
print(f"Trainable params (DPO LoRA): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown] id="5ee89adf"
# > **Why no separate `ref_model=` argument?** Modern TRL (≥ 0.12) auto-detects
# > PEFT models and uses the *base model without the adapter* as the reference.
# > That's the same memory layout: 1 base + 2 adapter sets in VRAM. No deepcopy
# > needed.

# %% [markdown] id="887dfc04"
# ## 2. Build DPOConfig (deck §5.2 hyperparameters)

# %% colab={"base_uri": "https://localhost:8080/"} id="ee18188d" outputId="560ea7c2-b90d-442b-8f80-2d5412e304f5"
from trl import DPOConfig

dpo_config = DPOConfig(
    output_dir=str(DPO_OUT.parent / "dpo-checkpoints"),
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    beta=BETA,
    max_length=MAX_LEN,
    max_prompt_length=MAX_PROMPT_LEN,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_8bit",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    seed=42,
    loss_type="sigmoid",         # DPO standard (alternatives: ipo, hinge, kto)
    report_to="none",
)

print(f"DPOConfig: beta={dpo_config.beta}  lr={dpo_config.learning_rate}  loss_type={dpo_config.loss_type}")

# %% [markdown] id="98e4443c"
# ## 3. Load preference data

# %% colab={"base_uri": "https://localhost:8080/", "height": 84, "referenced_widgets": ["78a0c0f497d74d8f89faa67945765cdc", "0d0e664580fc4173bc7b19953cad116b", "bb8ae14d098d4113866193ea8a736220", "6fba337b525c4c78b799f4cfc361cb44", "f02b89afb9e04562b816b903e0f71ca5", "a8973a428a6d48c38c6c6e37e4deed22", "85e8c881c9da4bbc9e99b4dc10321c46", "e488926718ea4a6b8651db83d182adb6", "a92c57344aea42bc84e25ddbcac2ab05", "269fb539a3a6486a89740d60c425807b", "071cdc4329454cf6ba89bb00b57f6fa0"]} id="4f9d3304" outputId="f9c9e41b-cd4a-4a7c-c4dd-9028577d850b"
from datasets import Dataset

pref_ds = Dataset.from_parquet(str(PREF_PATH))
print(f"Loaded {len(pref_ds)} preference pairs from {PREF_PATH}")
print(f"Columns: {pref_ds.column_names}")

# %% colab={"base_uri": "https://localhost:8080/"} id="w5mh7_DTfgpi" outputId="09d0ae88-4718-4cb4-bf58-d86ffd95bf95"
# xformers already removed in setup cell — no-op here to avoid breaking the run
print("xformers already uninstalled in setup. Continuing...")


# %% [markdown] id="5e1358b8"
# ## 4. Train

# %% colab={"base_uri": "https://localhost:8080/", "height": 113, "referenced_widgets": ["fbbd057ae6f74d01876b6964f10df6ac", "da9e2d946df5438fa1635e8d24a2f461", "82a12d2dc3d64f6ca64dbb744caff6c7", "d768004d6a194e32a51e28fe1aecf76b", "da0f1aa2fedf4df3aa842e75914111e4", "b91cd132ce42416788a973e245c838b2", "5c5e9aee4d824d46b189d02b92ea1d55", "b787e9b3ef234ebea0bde7d222ef5e95", "5353a9aeed244cf896df2133a35a4acd", "1b951b1971ee41dab0a0442e830d6ad0", "c7b4308cd6b846b4b5d2f274d963cebb", "e70f5f335e1342fd964f5b006ba9e0d8", "8759ada4187a4ef688397003678f5f5e", "c245ea54f4ae4033b359940079eb2b4c", "04c61cc016114461b7b4db42281ac1b2", "145da67a369d4fd98a979c9c7e49cdc6", "0808d99421694227a50643d3ab7722a5", "1dc507e5e2574ac7a81ae9165874f59b", "2053e7e7fd9e49cf8487907b46c8a3c0", "c65a7b8582104dada440f276c8e9181f", "f3f5673600014aac8a80501a5d9bc3c0", "f745f3d8a57443a4bceb9ea0ef67de8f", "27ff3e206d0f4911a4269f73ae1b43bc", "428b41d673564cdcbab2c926fcb7f713", "7d68d648ed004ac483baa1e4e6e7bbc0", "f23655d71d024ed28bcf0ee22425ed12", "0939b85c9ee2476da0b73809000bafc5", "8edf2f05fc3e41c78a3bdee4a29549b5", "9dda3df7ed1f41eab6586c982e652198", "4b97ad314f5d495dbc0bb68fc1208981", "a8b56231f13d49599e2a4eed308afbfc", "0b567e565b4e43bfa59b5405a1eeb31e", "6361a386d40340c99168fc89412271d3"]} id="bbf9d1ac" outputId="cd123a4c-bad9-4c64-8334-0ef5be4cfc04"
# from unsloth import PatchDPOTrainer
# PatchDPOTrainer()
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=None,                # auto-derived from PEFT base
    args=dpo_config,
    train_dataset=pref_ds,
    processing_class=tokenizer,
)


# %% id="a99581d6" colab={"base_uri": "https://localhost:8080/", "height": 998} outputId="815ee280-7500-47f8-89e3-7e28ebe5091e"
train_result = trainer.train()
print(f"\nFinal DPO loss: {train_result.training_loss:.4f}")


# %% [markdown] id="5a7cc297"
# ## 5. Plot reward curves — THE diagnostic
#
# **Read deck §3.4 before interpreting these.** A growing reward gap can come from:
# - **(intended)** chosen reward going up + rejected staying flat
# - **(intended)** chosen rising slowly + rejected falling fast
# - **(likelihood displacement)** chosen reward going *down* + rejected falling faster
#
# The third case is what Razin et al. 2024 documented. It's not a bug, but it
# tells you the model is finding a way to widen the gap that doesn't necessarily
# improve actual chosen probability.

# %% id="92c49615" colab={"base_uri": "https://localhost:8080/", "height": 359} outputId="323c7e7c-fe29-44cd-a346-03466ab6cd43"
import matplotlib.pyplot as plt
import pandas as pd

logs = pd.DataFrame(trainer.state.log_history)

# Tự động tìm cột loss phù hợp (có hoặc không có tiền tố train/)
loss_col = next((c for c in ["loss", "train/loss", "train_loss"] if c in logs.columns), None)
if loss_col:
    logs = logs[logs[loss_col].notna()].copy()

# Tự động tìm cột rewards phù hợp
chosen_col = next((c for c in ["rewards/chosen", "train/rewards/chosen", "train_rewards/chosen"] if c in logs.columns), None)
rejected_col = next((c for c in ["rewards/rejected", "train/rewards/rejected", "train_rewards/rejected"] if c in logs.columns), None)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

if chosen_col and rejected_col:
    axes[0].plot(logs["step"], logs[chosen_col], label="chosen reward", color="#2e548a", linewidth=1.5)
    axes[0].plot(logs["step"], logs[rejected_col], label="rejected reward", color="#c83538", linewidth=1.5)
    axes[0].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Implicit reward (log π/π_ref)")
    axes[0].set_title("Chosen vs Rejected rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    gap = logs[chosen_col] - logs[rejected_col]
    axes[1].plot(logs["step"], gap, color="#1a3355", linewidth=1.8)
    axes[1].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Reward gap (chosen − rejected)")
    axes[1].set_title("Reward gap (the headline number)")
    axes[1].grid(True, alpha=0.3)
else:
    axes[0].text(0.5, 0.5, f"No reward columns in trainer.state.log_history.\nColumns found: {list(logs.columns)}",
                 ha="center", va="center", transform=axes[0].transAxes)
    axes[1].text(0.5, 0.5, "—", ha="center", va="center", transform=axes[1].transAxes)

fig.suptitle(f"DPO reward curves · {COMPUTE_TIER} · β={BETA} · lr={LR}", y=1.02)
fig.tight_layout()

screenshot_dir = REPO_ROOT / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(screenshot_dir / "03-dpo-reward-curves.png", dpi=120, bbox_inches="tight")
plt.show()


# %% [markdown] id="0b35619e"
# ### 5a. Failure-mode self-check
#
# Read this cell carefully — it tells you which kind of "reward gap up" you got.

# %% id="4c057f12" colab={"base_uri": "https://localhost:8080/"} outputId="6c22ff80-841e-4441-f370-e57b3c2f7d55"
if chosen_col and rejected_col and len(logs) >= 5:
    last_chosen = logs[chosen_col].iloc[-5:].mean()
    last_rejected = logs[rejected_col].iloc[-5:].mean()
    last_gap = last_chosen - last_rejected
    first_chosen = logs[chosen_col].iloc[:5].mean()

    chosen_delta = last_chosen - first_chosen

    print(f"END  chosen reward:    {last_chosen:+.3f}")
    print(f"END  rejected reward:  {last_rejected:+.3f}")
    print(f"END  reward gap:       {last_gap:+.3f}")
    print()

    if last_gap < 0:
        print("✗ FAILURE: reward gap went NEGATIVE. DPO did the opposite of what you wanted.")
        print("  Likely causes: data quality (chosen/rejected swapped?), beta too high, lr too low.")
    elif chosen_delta < -0.5 and last_gap > 0:
        print("⚠  LIKELIHOOD DISPLACEMENT (deck §3.4):")
        print(f"   Reward gap is positive ({last_gap:+.3f}) — good!")
        print(f"   But chosen reward FELL by {chosen_delta:+.3f} during training.")
        print("   The gap grew because rejected fell faster than chosen.")
        print("   Document this in REFLECTION § 3 — it's a teachable moment, not a bug.")
    elif chosen_delta > 0 and last_gap > 0:
        print("✓ INTENDED: chosen reward UP and gap positive. Classic DPO success.")
    else:
        print("?  AMBIGUOUS: weak chosen movement + positive gap. Try longer training or higher lr.")

# %% [markdown] id="dd5fb7ab"
# ## 6. Save adapter

# %% id="1faade78" colab={"base_uri": "https://localhost:8080/"} outputId="3a6467a2-d0c6-4693-a878-2615363c9c3e"
trainer.model.save_pretrained(str(DPO_OUT))
tokenizer.save_pretrained(str(DPO_OUT))
print(f"Saved DPO adapter to {DPO_OUT}")

# Save the headline metrics for verify.py + REFLECTION
import json

metrics = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "beta": BETA,
    "lr": LR,
    "epochs": EPOCHS,
    "final_train_loss": float(train_result.training_loss),
    "end_chosen_reward": float(last_chosen) if chosen_col else None,
    "end_rejected_reward": float(last_rejected) if rejected_col else None,
    "end_reward_gap": float(last_gap) if chosen_col and rejected_col else None,
}
(DPO_OUT / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Wrote metrics to {DPO_OUT / 'dpo_metrics.json'}")

# %% [markdown] id="5d981b98"
# ## 7. Vibe-coding callout
#
# Now's the time for the **β experiment** if you want the +6 rigor add-on.
#
# `make beta-sweep` runs this notebook 3 times with `DPO_BETA ∈ {0.05, 0.1, 0.5}`
# and saves to `adapters/dpo-b{0.05,0.1,0.5}/`. Plot the results yourself:
#
# ```python
# import json
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# results = []
# for d in sorted((REPO_ROOT / "adapters").glob("dpo-b*")):
#     m = json.loads((d / "dpo_metrics.json").read_text())
#     results.append((m["beta"], m["end_reward_gap"]))
# # plot β vs reward_gap
# ```
#
# **Think-hard zone:** what's the *expected* shape of the β-vs-reward-gap curve?
# Hypothesize before you look at the data. (Hint: deck §3.3.)
#
# **Next:** NB4 — qualitative side-by-side comparison.
