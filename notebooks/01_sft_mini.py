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

# %% [markdown] id="7db8de9c"
# # NB1 — SFT-mini: Build the Lab 21 SFT checkpoint inline
#
# **Stack:** Unsloth + LoRA r=16 + bitsandbytes 4-bit base + 1k VN Alpaca, 1 epoch.
# Maps to deck §1 (why SFT alone insufficient — motivates the upcoming DPO step) +
# deck §3 (DPO will need this SFT checkpoint as initial policy).
#
# > **Mục tiêu:** tạo 1 SFT adapter "đủ tốt" để DPO có gì align lên. Đây là
# > Lab 21 thu gọn — nếu bạn đã hoàn thành Lab 21 sibling repo
# > ([VinUni-AI20k/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA)),
# > bạn có thể SKIP notebook này và copy adapter cũ vào `adapters/sft-mini/`.
# >
# > Nếu chưa, notebook này build từ đầu trong ~10 phút trên T4 (15 phút trên Colab CPU runtime — *đừng làm vậy*).

# %% [markdown] id="c6e335ad"
# ## 0. Setup

# %% colab={"base_uri": "https://localhost:8080/"} id="af2ce656" outputId="f8663b14-76da-49fd-8b14-233f4a925918"
import os
from pathlib import Path

# Tier detection. Defaults to T4 if env not set.
COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
assert COMPUTE_TIER in ("T4", "BIGGPU"), f"Invalid COMPUTE_TIER: {COMPUTE_TIER}"

# Tier-specific hyperparameters
if COMPUTE_TIER == "T4":
    BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
    MAX_LEN = 512
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 8
else:  # BIGGPU
    BASE_MODEL = "unsloth/Qwen2.5-7B-bnb-4bit"
    MAX_LEN = 1024
    PER_DEVICE_BATCH = 2
    GRAD_ACCUM = 4

SFT_DATASET = os.environ.get("SFT_DATASET", "5CD-AI/Vietnamese-alpaca-gpt4-gg-translated")
SFT_SLICE = 1000
NUM_EPOCHS = 1

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
ADAPTER_OUT = REPO_ROOT / "adapters" / "sft-mini"
ADAPTER_OUT.mkdir(parents=True, exist_ok=True)

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"BASE_MODEL:      {BASE_MODEL}")
print(f"SFT_DATASET:     {SFT_DATASET}  (slice: {SFT_SLICE})")
print(f"max_seq_length:  {MAX_LEN}")
print(f"effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"output:          {ADAPTER_OUT}")

# %% colab={"base_uri": "https://localhost:8080/"} id="3c4311ed" outputId="7ed349e3-38da-4f28-c0c4-32779d6f17ca"
import torch

assert torch.cuda.is_available(), "DPO needs a CUDA GPU. See HARDWARE-GUIDE.md."
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name}  ({gpu.total_memory / 1e9:.1f} GB)")

# %% [markdown] id="92df64cf"
# ## 1. Load base model with Unsloth
#
# Unsloth bundles patched 4-bit kernels — that's how Qwen2.5-3B (or 7B) stays
# in T4 / A100 budget. The `FastLanguageModel.from_pretrained` call returns a
# 4-bit quantized base; `get_peft_model` attaches the LoRA adapter on top.

# %% colab={"base_uri": "https://localhost:8080/"} id="06640f23" outputId="1ec49c4c-edd2-4370-e321-2c44a3d54395"
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,                # auto: bf16 on Ampere+, fp16 on Turing
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

# Critical for batch training — Qwen tokenizers ship without pad token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer.pad_token = eos_token")

# %% colab={"base_uri": "https://localhost:8080/"} id="e289df9d" outputId="5b009137-631e-4093-a7ce-c99b969c6e10"
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,           # Unsloth supports dropout=0 for free speed
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # 30% VRAM savings
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown] id="5a259d5e"
# ## 2. Load + format VN Alpaca slice
#
# `5CD-AI/Vietnamese-alpaca-cleaned` is a 50k-row VN Alpaca translation. Lab 21
# uses 1k slice for the demo run; we match that exactly so reward gap is comparable.

# %% colab={"base_uri": "https://localhost:8080/"} id="ee08c96d" outputId="538e7b30-da1b-44d2-b922-0673fc1a939d"
from datasets import load_dataset

ds = load_dataset(SFT_DATASET, split=f"train[:{SFT_SLICE}]")
print(f"Loaded {len(ds)} rows. Columns: {ds.column_names}")
print(f"\nFirst row:\n{ds[0]}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 208, "referenced_widgets": ["e41bd41076ea48f5a3133474e3c183ed", "14537feb673646af94bdcdf4d2bdf7c1", "09780f5e11c54d3fa83688482214b8ba", "0f09726edb0c4e59880294a346bb62a7", "acdf82fa910d40928ad9abea7d24f952", "9d4a629e20644976bd16596f651e8027", "1220a7ff60a143b384e12a3ee3567fd2", "da98ecf907e146b3b3c09fdfb90249b8", "7fe01557ab2c470dadc981af4ff640c1", "d8d18c21e0fd423695c1d41378aa914b", "f8274b56632a4fc1ba1d12f7a2ca4734"]} id="bd51f4cf" outputId="108e9843-04d1-4061-e6d8-6af6c3505090"
# Alpaca → ChatML format (Qwen2.5's native template)
def format_alpaca_to_chat(row):
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    messages = []
    # Support both Vietnamese-alpaca-cleaned keys and Vietnamese-alpaca-gpt4-gg-translated keys
    instruction = row.get("instruction_vi") or row.get("instruction") or row.get("instruction_en")
    input_val = row.get("input_vi") or row.get("input") or row.get("input_en")
    output = row.get("output_vi") or row.get("output") or row.get("output_en")
    if instruction:
        prompt = instruction
        if input_val:
            prompt += "\n\n" + input_val
        messages.append({"role": "user", "content": prompt})
    if output:
        messages.append({"role": "assistant", "content": output})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


ds_formatted = ds.map(format_alpaca_to_chat, remove_columns=ds.column_names)
print(f"\nSample formatted text (first 500 chars):\n{ds_formatted[0]['text'][:500]}")


# %% [markdown] id="030f919f"
# ## 3. Train SFT-mini

# %% colab={"base_uri": "https://localhost:8080/", "height": 121, "referenced_widgets": ["c4409e8f6ccd471f83423dd8aff81a6f", "4ce4ad7cda88450b9dd2b6b81754f09b", "601db669b91d4e3ea18a9cff1fbd5d25", "e6111df2d3b24ab9916fbbb88279d113", "88675bd300084ef8bff092e1df5d0ac6", "eccb9ee214e846478630d2921a0ac71e", "77887cffde414aa6a622b830a0772659", "337675e6e1df4d62be6218c035201b75", "9252af3aabd64fb78c678faa258e1af4", "a5816a29699c4fcea6e9c63d02e0c4ba", "2191508aa2344ba6a8efd7ff57a43ac0"]} id="5984a1e9" outputId="2ed09b5c-a3cd-456d-b029-7706e53bddc8"
from trl import SFTTrainer, SFTConfig
import torch

sft_config = SFTConfig(
    output_dir=str(ADAPTER_OUT.parent / "sft-mini-checkpoints"),
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="no",        # Save only at the end via trainer.model.save_pretrained
    optim="adamw_8bit",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    seed=42,
    max_length=MAX_LEN,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=ds_formatted,
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 573} id="3218c9f0" outputId="abd99819-1d61-46f0-8fc2-cea8bd587024"
train_result = trainer.train()
print(f"\nFinal train loss: {train_result.training_loss:.4f}")

# %% [markdown] id="be45d9f5"
# ### 3a. Plot loss curve (deliverable: `02_sft_loss.png`)

# %% colab={"base_uri": "https://localhost:8080/", "height": 407} id="a6cf6e31" outputId="dd077d89-9552-4bdc-ddee-a35b727fd34e"
import matplotlib.pyplot as plt

losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
steps = [log["step"] for log in trainer.state.log_history if "loss" in log]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, losses, marker="o", markersize=3, linewidth=1.2)
ax.set_xlabel("Training step")
ax.set_ylabel("Loss")
ax.set_title(f"SFT-mini loss · {COMPUTE_TIER} · {BASE_MODEL.split('/')[-1]} · {SFT_SLICE} samples")
ax.grid(True, alpha=0.3)
fig.tight_layout()
screenshot_dir = REPO_ROOT / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(screenshot_dir / "02-sft-loss.png", dpi=120)
plt.show()

# %% [markdown] id="67a5d61f"
# ## 4. Save adapter + sanity-check generation

# %% colab={"base_uri": "https://localhost:8080/"} id="751143b8" outputId="f83e282e-d2a0-440e-87e7-12f61bbb9b27"
trainer.model.save_pretrained(str(ADAPTER_OUT))
tokenizer.save_pretrained(str(ADAPTER_OUT))
print(f"Saved SFT adapter to {ADAPTER_OUT}")

# %% colab={"base_uri": "https://localhost:8080/"} id="88a4b370" outputId="f622eaca-e94e-4554-d477-f541c99cccae"
# Sanity: generate 1 sample to confirm the adapter loaded correctly.
FastLanguageModel.for_inference(model)
prompt = "Giải thích ngắn gọn (3-4 câu) thuật toán quicksort hoạt động thế nào."
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to("cuda")
with torch.no_grad():
    out = model.generate(input_ids=inputs, max_new_tokens=200, do_sample=False)
generated = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"PROMPT: {prompt}\n")
print(f"SFT-mini response:\n{generated}")

# %% [markdown] id="5361ad0f"
# ## 5. Vibe-coding callout
#
# Bạn vừa tái tạo Lab 21 trong ~10 phút. Một câu hỏi để brainstorm:
#
# > **Thật ra, bạn cần *bao nhiêu* SFT để DPO có ý nghĩa?**
# >
# > Thử thay `SFT_SLICE = 1000` → `100`. Re-run NB1 → NB3 với SFT yếu hơn.
# > Quan sát: reward gap có còn tăng được không? Output coherent không?
# >
# > Đây là 1 design decision *think-hard zone* (xem VIBE-CODING.md): không có
# > đáp án sẵn trong deck. Hypothesize trước, train sau, viết kết quả vào
# > `submission/REFLECTION.md` § 6.
#
# **Next:** NB2 — load + format preference data.
