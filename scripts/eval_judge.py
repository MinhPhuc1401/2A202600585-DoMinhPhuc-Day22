#!/usr/bin/env python3
"""CLI wrapper — run NB4 eval logic OR plot β-sweep results.

Two modes:
1. Standard eval (no flags): regenerate side-by-side eval from current SFT + DPO adapters.
2. β-sweep plot (`--sweep-dir`): collect dpo_metrics.json from adapters/dpo-b*/ and plot.

Usage:
    python scripts/eval_judge.py
    python scripts/eval_judge.py --sweep-dir adapters --output submission/screenshots/bonus-beta-sweep.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def plot_beta_sweep(sweep_dir: Path, output: Path):
    """Aggregate dpo_metrics.json from adapters/dpo-b* directories and plot."""
    import matplotlib.pyplot as plt

    rows = []
    for d in sorted(sweep_dir.glob("dpo-b*")):
        m_path = d / "dpo_metrics.json"
        if m_path.exists():
            m = json.loads(m_path.read_text())
            if m.get("end_reward_gap") is not None:
                rows.append({
                    "dir": d.name,
                    "beta": m.get("beta"),
                    "loss": m.get("final_train_loss"),
                    "gap": m.get("end_reward_gap"),
                    "chosen": m.get("end_chosen_reward"),
                    "rejected": m.get("end_rejected_reward"),
                })

    if not rows:
        print(f"No β-sweep results found under {sweep_dir}/dpo-b*/")
        print("Run `make beta-sweep` first.")
        return 1

    rows.sort(key=lambda r: r["beta"])
    betas = [r["beta"] for r in rows]
    gaps = [r["gap"] for r in rows]
    chosens = [r["chosen"] for r in rows]
    rejecteds = [r["rejected"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(betas, gaps, marker="o", color="#1a3355", linewidth=2)
    axes[0].set_xlabel("β (DPO regularization)")
    axes[0].set_ylabel("End reward gap (chosen − rejected)")
    axes[0].set_xscale("log")
    axes[0].set_title("Reward gap vs β")
    axes[0].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(betas, chosens, marker="o", color="#2e548a", label="chosen", linewidth=2)
    axes[1].plot(betas, rejecteds, marker="o", color="#c83538", label="rejected", linewidth=2)
    axes[1].set_xlabel("β")
    axes[1].set_ylabel("End mean reward")
    axes[1].set_xscale("log")
    axes[1].set_title("Chosen and rejected rewards vs β")
    axes[1].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"β-sweep ({len(rows)} runs)", y=1.02)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    print(f"Saved {output}")

    print("\nβ-sweep results:")
    for r in rows:
        print(f"  β={r['beta']:>6}   gap={r['gap']:+.3f}   chosen={r['chosen']:+.3f}   rejected={r['rejected']:+.3f}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir", default=None,
        help="Directory containing adapters/dpo-b* subdirs from `make beta-sweep`",
    )
    parser.add_argument(
        "--output", default=str(REPO / "submission" / "screenshots" / "bonus-beta-sweep.png"),
    )
    args = parser.parse_args()

    if args.sweep_dir:
        return plot_beta_sweep(Path(args.sweep_dir), Path(args.output))

    print("Standard eval CLI not yet implemented — run NB4 directly:")
    print("  jupyter nbconvert --to notebook --execute --inplace notebooks/04_compare_and_eval.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
