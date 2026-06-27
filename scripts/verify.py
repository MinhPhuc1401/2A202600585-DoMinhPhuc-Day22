#!/usr/bin/env python3
"""Pre-submission sanity check + smoke mode.

Run from repo root: `make verify` (or `python scripts/verify.py`).
For a quick smoke run before training: `python scripts/verify.py --smoke`.

Exits 0 if every required artifact is present + REFLECTION.md edited beyond the
template. Exits non-zero with a checklist of what's missing — no files written.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TEMPLATE_MARKERS = [
    r"<Họ Tên>",
    r"<A20-K1 / A20-K2",
    r"<YYYY-MM-DD>",
    r"_Answer here\.\s*≥",
    r"_Answer here\._?\s*$",
    r"<e\.g\., Free Colab T4 16GB",
]


def check_file(path: Path, label: str, problems: list[str]) -> bool:
    if not path.exists():
        problems.append(f"MISSING  {label}: {path.relative_to(Path.cwd())}")
        return False
    if path.stat().st_size == 0:
        problems.append(f"EMPTY    {label}: {path.relative_to(Path.cwd())}")
        return False
    return True


def check_screenshots(folder: Path, min_count: int, problems: list[str]) -> int:
    if not folder.exists():
        problems.append("MISSING  submission/screenshots/ folder")
        return 0
    images = [p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if len(images) < min_count:
        problems.append(
            f"TOO FEW  submission/screenshots/: have {len(images)}, need at least {min_count}. "
            f"See submission/screenshots/README.md for the list."
        )
    return len(images)


def check_reflection_edited(path: Path, problems: list[str]) -> bool:
    if not path.exists():
        problems.append("MISSING  submission/REFLECTION.md")
        return False
    text = path.read_text(encoding="utf-8")
    leftover = []
    for pattern in TEMPLATE_MARKERS:
        flags = re.MULTILINE if pattern.startswith("^") else 0
        if re.search(pattern, text, flags):
            leftover.append(pattern)
    if len(leftover) >= 3:
        problems.append(
            f"UNEDITED submission/REFLECTION.md still has {len(leftover)} template placeholders. "
            f"Fill in your own numbers and answers."
        )
        return False
    return True


def check_dpo_metrics(repo: Path, problems: list[str]) -> bool:
    metrics_path = repo / "adapters" / "dpo" / "dpo_metrics.json"
    if not metrics_path.exists():
        problems.append("MISSING  adapters/dpo/dpo_metrics.json (NB3 didn't complete)")
        return False
    try:
        m = json.loads(metrics_path.read_text())
    except Exception as exc:
        problems.append(f"CORRUPT  adapters/dpo/dpo_metrics.json — {exc}")
        return False
    gap = m.get("end_reward_gap")
    if gap is None:
        problems.append("WARN     adapters/dpo/dpo_metrics.json has no end_reward_gap (TRL log columns missing?)")
        return True
    if gap <= 0:
        problems.append(
            f"WARN     end_reward_gap = {gap:+.3f} (≤ 0). DPO didn't separate chosen from rejected. "
            f"Check NB3 output. (Not a hard fail — explain in REFLECTION § 3 + § 5.)"
        )
    return True


def check_gguf(repo: Path, problems: list[str]) -> bool:
    gguf_dir = repo / "gguf"
    if not gguf_dir.exists():
        problems.append("MISSING  gguf/ directory (NB5 didn't run)")
        return False
    files = list(gguf_dir.glob("*.gguf"))
    if not files:
        problems.append("MISSING  gguf/*.gguf — NB5 quantization step didn't write a file")
        return False
    big = [p for p in files if p.stat().st_size > 5 * 1024**3]
    if big:
        problems.append(
            f"OVERSIZED  GGUF files > 5 GB: {[p.name for p in big]}. "
            f"Q4_K_M should be ≤ 5 GB even on 7B."
        )
    return True


def smoke_check(repo: Path) -> int:
    """Light-weight pre-training check: imports work, GPU visible, deck files present."""
    print("==> Smoke check (imports + GPU + deck files)\n")
    problems: list[str] = []

    # Imports
    try:
        import torch  # noqa: WPS433
        print(f"  ✓ torch              {torch.__version__}")
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            print(f"  ✓ CUDA               {dev.name} ({dev.total_memory / 1e9:.1f} GB)")
        else:
            problems.append("torch.cuda.is_available() == False -- DPO needs a CUDA/ROCm GPU. Use the Colab T4 path (see HARDWARE-GUIDE.md); this local smoke gate cannot pass on CPU/Mac.")
    except ImportError as exc:
        problems.append(f"torch import failed: {exc}")

    for mod in ["unsloth", "trl", "peft", "bitsandbytes", "datasets", "matplotlib"]:
        try:
            __import__(mod)
            print(f"  ✓ {mod}")
        except (ImportError, NotImplementedError, RuntimeError) as exc:
            # unsloth raises NotImplementedError (not ImportError) when no GPU is present.
            problems.append(f"{mod} import failed: {exc}")

    # Deck source (sibling file)
    deck = repo.parent / "day07-dpo-orpo-alignment-tu-sft-en-preference-learning.tex"
    if deck.exists():
        print(f"  ✓ deck source        {deck.name}")
    else:
        print(f"  ⚠ deck source not found at {deck} — fine if you cloned standalone")

    # Notebook source files
    nb_dir = repo / "notebooks"
    expected_nbs = [
        "01_sft_mini.py", "02_preference_data.py", "03_dpo_train.py",
        "04_compare_and_eval.py", "05_merge_deploy_gguf.py", "06_benchmark.py",
    ]
    for nb in expected_nbs:
        if (nb_dir / nb).exists():
            print(f"  ✓ {nb}")
        else:
            problems.append(f"missing notebook {nb_dir / nb}")

    # NB6 benchmark dependency check
    try:
        import lm_eval  # noqa: F401
        print(f"  ✓ lm_eval (NB6 benchmark suite)")
    except ImportError:
        problems.append("lm_eval missing — pip install -r requirements.txt (NB6 will fail)")

    print()
    if problems:
        print("✗ Smoke check FAILED:\n")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("✓ Smoke check passed. You can now run `make pipeline` (or open a notebook).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run pre-training smoke check (imports + GPU) instead of submission gatekeeper",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent

    if args.smoke:
        return smoke_check(repo)

    problems: list[str] = []
    print(f"==> Verifying submission readiness at {repo}\n")

    # Notebook source files
    for nb in ["01_sft_mini.py", "02_preference_data.py", "03_dpo_train.py",
               "04_compare_and_eval.py", "05_merge_deploy_gguf.py"]:
        check_file(repo / "notebooks" / nb, f"notebook {nb}", problems)

    # Adapter outputs
    check_file(repo / "adapters" / "sft-mini" / "adapter_config.json",
               "SFT-mini adapter config (NB1 output)", problems)
    check_file(repo / "adapters" / "dpo" / "adapter_config.json",
               "DPO adapter config (NB3 output)", problems)

    # DPO metrics
    check_dpo_metrics(repo, problems)

    # Preference data
    check_file(repo / "data" / "pref" / "train.parquet",
               "preference data parquet (NB2 output)", problems)

    # Eval outputs
    check_file(repo / "data" / "eval" / "side_by_side.jsonl",
               "side-by-side eval (NB4 output)", problems)
    check_file(repo / "data" / "eval" / "judge_results.json",
               "judge results (NB4 output)", problems)

    # OPTIONAL (bonus) — NB5 GGUF + NB6 benchmark: report, do NOT fail core
    optional = []
    if not list((repo / "gguf").glob("*.gguf")):
        optional.append("NB5 GGUF (gguf/*.gguf) not done")
    if not (repo / "data" / "eval" / "benchmark_results.json").exists():
        optional.append("NB6 benchmark (data/eval/benchmark_results.json) not done")

    # Submission artifacts (core)
    check_reflection_edited(repo / "submission" / "REFLECTION.md", problems)
    n_shots = check_screenshots(repo / "submission" / "screenshots", min_count=3, problems=problems)
    if n_shots:
        print(f"  ✓ submission/screenshots/ has {n_shots} image(s)")

    if optional:
        print("\nⓘ Optional (bonus) not done — fine for a core pass:")
        for line in optional:
            print(f"  - {line}")

    print()
    if not problems:
        print("✓ Core checks passed. Push your repo (public!) and paste the URL into LMS.")
        return 0

    print("✗ Submission not ready yet:\n")
    for line in problems:
        print(f"  - {line}")
    print(
        "\nFix the items above and rerun `make verify`. See rubric.md for full grading details."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
