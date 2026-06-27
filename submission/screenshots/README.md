# Required screenshots

Drop the following PNG/JPG files into this folder before submitting. Filenames are *suggested*, not required — the grader reads `REFLECTION.md` to map screenshots to evidence.

## Minimum (6 shots)

1. **`01-setup-gpu.png`** — terminal output (or Colab cell output) showing `nvidia-smi` or `torch.cuda.get_device_name()` with VRAM. Establishes which tier you ran.
2. **`02-sft-loss.png`** — Notebook 01 final loss curve (matplotlib output) showing monotonic decrease over 1 epoch on the SFT-mini build.
3. **`03-dpo-reward-curves.png`** — Notebook 03 dual-curve plot: `chosen_rewards` and `rejected_rewards` plotted separately, plus their gap. This is THE diagnostic — if the only thing visible is "gap going up", you'll lose points (deck §3.4: chosen reward decreasing while gap grows is likelihood displacement, not winning).
4. **`04-side-by-side-table.png`** — Notebook 04 markdown table with ≥ 8 prompts × 2 model outputs (SFT vs SFT+DPO). Table must show category labels (helpfulness vs safety) and the judge's call (or your manual call).
5. **`05-judge-output.png`** *(or `05-manual-rubric.png`)* — If you used the API judge (gpt-4o-mini / claude-haiku), capture the judge's verbatim verdict for at least 3 of your 8 prompts. If you used manual rubric mode, capture your filled-in rubric instead.
6. **`06-gguf-smoke.png`** — Notebook 05 final cell: llama-cpp-python loading the merged GGUF and producing a coherent VN response to a smoke prompt. Must show the `Q4_K_M.gguf` filename in the load line + the actual generated tokens.
7. **`07-benchmark-comparison.png`** — Notebook 06 4-bar chart: SFT-only vs SFT+DPO scores across IFEval / GSM8K / MMLU / AlpacaEval-lite. Bars labeled with absolute scores; deltas annotated above each pair. This is THE quantitative summary of whether DPO worked — it should be the most-looked-at image in your submission.

## Optional — for the +20 bonus rigor add-ons (mentioned in `rubric.md`)

7. **`bonus-beta-sweep.png`** — chart of reward gap vs β over {0.05, 0.1, 0.5}. (+6 add-on)
8. **`bonus-vn-data-sample.png`** — if you completed the BONUS-CHALLENGE provocation #1 (VN preference set), screenshot of 3 native-VN preference pairs you generated.
9. **`bonus-creative-challenge.png`** — your choice. Whatever the most interesting visual from your `bonus/` folder is — collapse-curve from self-rewarding, win-rate matrix from DPO/ORPO/SimPO trinity, etc.

## Tips

- **Crop tight** — full-screen browser shots get rejected. The grader wants to see the data, not your wallpaper.
- **Dark or light terminal both fine** — just make sure text is readable.
- **For reward curves**: include both axes (steps + rewards) and a legend. Matplotlib's default works.
- **For the side-by-side table**: if it's longer than 1 screen, OK to take 2 screenshots labeled `04a-...` and `04b-...`.
- **API key handling**: if your judge cell shows the key in the screenshot — recrop! Never publish `sk-...` lines.
