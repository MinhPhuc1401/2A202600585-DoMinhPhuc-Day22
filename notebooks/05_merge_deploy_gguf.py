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

# %% [markdown] id="254733bc"
# # NB5 — Merge + Deploy + GGUF
#
# **Stack:** Unsloth `merge_and_unload` + `save_pretrained_gguf(quantization='Q4_K_M')`
# + llama-cpp-python smoke test.
# Maps to deck §7.1 lab brief: "merge adapter, quantize GGUF, serve với vLLM".
#
# > **Mục tiêu:** export the SFT+DPO adapter as a deployable GGUF Q4_K_M file
# > (~1.5 GB on 3B / ~4 GB on 7B), then smoke-test it through llama-cpp-python.
# > Final cell shows the optional vLLM serving command (BigGPU only).

# %% [markdown] id="040f5373"
# ## 0. Setup

# %% id="3b8a150d" colab={"base_uri": "https://localhost:8080/"} outputId="f2413e5b-287e-4074-95ee-57c1d73453f4"
import os
import json
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = (
    "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4"
    else "unsloth/Qwen2.5-7B-bnb-4bit"
)
MAX_LEN = 512 if COMPUTE_TIER == "T4" else 1024

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
MERGED_PATH = REPO_ROOT / "adapters" / "merged-fp16"
GGUF_DIR = REPO_ROOT / "gguf"
MERGED_PATH.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)

assert DPO_PATH.exists(), "NB3 must run first"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"DPO adapter:     {DPO_PATH}")
print(f"merged output:   {MERGED_PATH}")
print(f"GGUF output:     {GGUF_DIR}")

# %% id="623c449a"
import torch

assert torch.cuda.is_available()

# %% [markdown] id="3833597a"
# ## 1. Load DPO model + merge adapter

# %% id="ef8a091d" colab={"base_uri": "https://localhost:8080/"} outputId="3c23b0f2-ae8e-4d41-ce56-f68ab4bbbcc2"
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from peft import PeftModel

# Always load a clean base model for final merge/export.
# Do not reuse the model object from DPO training.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,   # Must be True so Unsloth enables LoRA merge patches
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

if getattr(tokenizer, "chat_template", None) is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' "
        "+ message['content'] + '<|im_end|>\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{'<|im_start|>assistant\\n'}}"
        "{% endif %}"
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# DPO_PATH contains the original SFT adapter after continued DPO training.
# Therefore, apply only DPO_PATH to the clean base model.
model = PeftModel.from_pretrained(
    model,
    str(DPO_PATH),
    is_trainable=False,
)

print(f"Loaded final SFT+DPO adapter from: {DPO_PATH}")
print(f"Model class: {model.__class__.__name__}")

if hasattr(model, "active_adapters"):
    print(f"Active PEFT adapters: {model.active_adapters}")


# %% [markdown] id="264dcfc2"
# > **Note:** DPO training continues directly from the trainable SFT adapter.
# > Therefore, `DPO_PATH` contains the final SFT adapter after DPO updates.
# > For final export, load a clean base model and apply only `DPO_PATH`.

# %% [markdown] id="73db6320"
# ## 2. Save merged FP16 weights
#
# `save_pretrained_merged(method="merged_16bit")` produces a HuggingFace-format
# directory you can either upload to HF Hub directly OR feed into the GGUF
# converter in step 3.

# %% id="74f6d6eb" colab={"base_uri": "https://localhost:8080/"} outputId="31f5afa9-26ee-4923-9fe8-928847124a41"
if not DPO_PATH.exists():
    raise FileNotFoundError(f"Final DPO adapter not found: {DPO_PATH}")

print(f"Final model class: {model.__class__.__name__}")

if hasattr(model, "active_adapters"):
    print(f"Final active adapters: {model.active_adapters}")

# Merge the final SFT+DPO adapter into the base weights
# and save a Hugging Face FP16 model.
model.save_pretrained_merged(
    str(MERGED_PATH),
    tokenizer,
    save_method="merged_16bit",
)
print(f"Saved merged FP16 model to: {MERGED_PATH}")


# %% colab={"base_uri": "https://localhost:8080/"} id="6cbaef84" outputId="0a34e04c-86e9-46ff-8efa-189450fee0fb"
# [FIX v2] Chỉ clear cache, KHÔNG del model
# Model sau save_pretrained_merged vẫn dùng được cho GGUF export
import gc, torch

gc.collect()
torch.cuda.empty_cache()
mem_free = torch.cuda.mem_get_info()[0] / 1e9
print(f"Free VRAM: {mem_free:.1f} GB")
print(f"Model type: {type(model).__name__} — sẵn sàng cho GGUF export")

# %% [markdown] id="646907bd"
# ## 3. Quantize to GGUF Q4_K_M
#
# Q4_K_M is the sweet spot: ~4× compression vs FP16, minimal quality loss.
# Unsloth wraps llama.cpp's `quantize` binary — first run downloads + compiles
# llama.cpp (~3 min) then quantizes (~30 s).

# %% colab={"base_uri": "https://localhost:8080/"} id="db4fbd81" outputId="74927c0b-5644-403a-9a8d-c2b7429c937e"
# [FIX v2] KHÔNG cần reload — dùng model đang có trong memory
# model đã là merged FP16 sau save_pretrained_merged ở bước trên
print(f"Dùng model hiện có: {type(model).__name__}")

# %% id="564e2db2" colab={"base_uri": "https://localhost:8080/"} outputId="ec27ac14-afee-4b13-dba0-9e05aa4762cf"
import subprocess, sys
from pathlib import Path

def save_gguf_with_fallback(model, tokenizer, gguf_dir, quant="q4_k_m"):
    try:
        print("[Method 1] Unsloth save_pretrained_gguf ...")
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant)
        print(f"SUCCESS → {gguf_dir}")
        return True
    except Exception as e:
        print(f"[Method 1] FAILED: {e}")

    print("\n[Method 2] Fallback: manual llama.cpp pipeline ...")
    llama_cpp_dir = Path("/content/llama.cpp")

    if not llama_cpp_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1",
            "https://github.com/ggerganov/llama.cpp", str(llama_cpp_dir)], check=True)

    build_dir = llama_cpp_dir / "build"
    if not (build_dir / "bin" / "llama-quantize").exists():
        build_dir.mkdir(exist_ok=True)
        subprocess.run(["cmake", "-S", str(llama_cpp_dir), "-B", str(build_dir),
            "-DLLAMA_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", str(build_dir), "--config", "Release", "-j4"], check=True)

    # Lưu HF weights ra disk để convert
    hf_path = gguf_dir / "hf_model"
    if not hf_path.exists():
        print("  Saving HF weights ...")
        model.save_pretrained(str(hf_path))
        tokenizer.save_pretrained(str(hf_path))

    f16_gguf = gguf_dir / "model-f16.gguf"
    if not f16_gguf.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
            str(llama_cpp_dir / "requirements.txt")], check=True)
        subprocess.run([sys.executable,
            str(llama_cpp_dir / "convert_hf_to_gguf.py"),
            str(hf_path), "--outfile", str(f16_gguf), "--outtype", "f16"], check=True)

    out_gguf = gguf_dir / "model-Q4_K_M.gguf"
    subprocess.run([str(build_dir / "bin" / "llama-quantize"),
        str(f16_gguf), str(out_gguf), "Q4_K_M"], check=True)
    print(f"[Method 2] SUCCESS → {out_gguf}")
    return True

# Gọi trực tiếp trên model đang có — không reload
save_gguf_with_fallback(model, tokenizer, GGUF_DIR, quant="q4_k_m")

# %% [markdown] id="f37d19ed"
# ### 3a. Optional — additional quantization tiers (for the +3 rigor add-on)

# %% id="15176ac4"
# Uncomment if you want Q5_K_M + Q8_0 too (~2× total disk space).
# Each adds ~30s for an extra GGUF file.
#
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q5_k_m")
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q8_0")

# %% id="81694b0a"
import os
import gc
import torch

print("GGUF files:")
for p in sorted(GGUF_DIR.iterdir()):
    if p.suffix == ".gguf":
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:50s}  {size_mb:>8.1f} MB")

del model
gc.collect()
torch.cuda.empty_cache()
print("Model released from memory.")


# %% [markdown] id="ae8a451d"
# ## 4. Smoke test with llama-cpp-python

# %% id="22053504"
# [FIX] Load llama-cpp-python with graceful GPU → CPU fallback
from llama_cpp import Llama

gguf_files = (
    list(GGUF_DIR.glob("*Q4_K_M*.gguf"))
    + list(GGUF_DIR.glob("*q4_k_m*.gguf"))
    + list(GGUF_DIR.glob("*Q4_K_M*"))
    + list(GGUF_DIR.glob("model-Q4_K_M.gguf"))
)
assert gguf_files, f"No Q4_K_M GGUF found in {GGUF_DIR}. Check step 3."
gguf_path = gguf_files[0]
print(f"Loading: {gguf_path.name}  ({gguf_path.stat().st_size / 1e6:.0f} MB)")

# Try GPU offload; if llama-cpp was installed without CUDA support, fall back to CPU
try:
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=MAX_LEN,
        n_gpu_layers=-1,   # all layers on GPU
        verbose=False,
    )
    print("Loaded with GPU offload (n_gpu_layers=-1).")
except Exception as e:
    print(f"GPU load failed ({e}). Falling back to CPU ...")
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=MAX_LEN,
        n_gpu_layers=0,    # CPU only
        verbose=False,
    )
    print("Loaded on CPU (inference will be slower).")


# %% [markdown] id="8a4de1c9"
# ### 4a. Smoke prompt + response (deliverable: `06-gguf-smoke.png`)

# %% id="5c31bdab"
SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": SMOKE_PROMPT}],
    max_tokens=200,
    temperature=0.0,
)

print(f"PROMPT:\n  {SMOKE_PROMPT}\n")
print(f"RESPONSE (Q4_K_M GGUF, llama-cpp-python):\n  {response['choices'][0]['message']['content']}")
print(f"\nTokens used: {response['usage']}")

# %% [markdown] id="031e453e"
# ## 5. Optional — vLLM serving (BigGPU only)
#
# vLLM provides production-grade OpenAI-compatible serving. **Requires CUDA GPU
# with ≥ 16 GB VRAM** and `vllm` installed (see `requirements-biggpu.txt`).
# On T4 tier this cell will OOM. Skip on T4.
#
# Run in a SEPARATE terminal (NOT in the notebook — vLLM blocks until killed):
#
# ```bash
# pip install vllm                         # once
# vllm serve adapters/merged-fp16 \
#   --port 8000 \
#   --max-model-len 1024 \
#   --gpu-memory-utilization 0.9
# ```
#
# Then test:
#
# ```bash
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model": "merged-fp16", "messages": [{"role": "user", "content": "Hello"}]}'
# ```
#
# **Why not in the notebook?** vLLM's process model doesn't play nicely with
# Jupyter — it expects to own the GPU + a long-running HTTP server. Run it as
# a sidecar process. The deck mentions vLLM as the deploy target; for actual
# production you'd containerize this command. For the lab, llama-cpp-python in
# step 4 is the graded artifact.

# %% [markdown] id="5f6d5704"
# ## 6. Save deployment metadata

# %% id="d1081726"
deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "merged_path": str(MERGED_PATH),
    "gguf_path": str(gguf_path),
    "gguf_size_mb": round(gguf_path.stat().st_size / 1e6, 1),
    "quantization": "q4_k_m",
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_response": response["choices"][0]["message"]["content"],
}
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").parent.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").write_text(
    json.dumps(deploy_meta, ensure_ascii=False, indent=2)
)
print("Saved data/eval/deploy_meta.json")

# %% [markdown] id="65995a5e"
# ## 7. Submission checklist
#
# Bạn vừa hoàn thành core lab. Trước khi submit:
#
# 1. **Run** `make verify` — gatekeeper sẽ list missing artifacts.
# 2. **Take screenshots** vào `submission/screenshots/` (xem `submission/screenshots/README.md`).
# 3. **Fill** `submission/REFLECTION.md` — đặc biệt là § 3 (reward curves analysis,
#    cross-reference deck §3.4) và § 6 (single change that mattered most).
# 4. **(Optional)** Pick a rigor add-on từ rubric.md (β-sweep, HF push, GGUF
#    release, W&B link, cross-judge).
# 5. **(Optional)** Pick a `BONUS-CHALLENGE.md` provocation cho creative bonus.
#
# Push public repo + paste URL vào VinUni LMS Day-22 box.
#
# Câu hỏi cuối để brainstorm trước khi đóng laptop:
#
# > **The deck says:** "DPO + 30 min A100 + 2k UltraFeedback → 3.2 → 4.1 helpfulness."
# > **You measured:** _<your win-rate from NB4>_.
# > **Why might they differ?** Dataset (English vs VN), base model (Qwen2.5-3B vs
# > deck's unspecified base), judge bias, sample size (8 prompts vs deck's full eval).
# > Đó chính là § 6 trong REFLECTION — what 1 change would close the gap.
