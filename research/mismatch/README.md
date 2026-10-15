# Where does the training–inference mismatch come from?

A study on ratchet's RL stack (Qwen2.5-0.5B-Instruct, GSM8K, GRPO, 2× T4). In RL post-training
the engine that samples and the trainer that learns compute the probability of the same token
separately, and the importance ratio between them is assumed to be 1 at equal weights. Recent
work measures the gap as one number at the logits and corrects it in the loss (truncated or
calibrated importance sampling, FP16 instead of BF16). This study takes the gap apart instead:
ratchet controls both sides, so each source can be switched on alone and measured on the same
sampled tokens.

| Source | How it is isolated |
|---|---|
| weight rounding at sync | trainer with float32 master weights vs the same weights rounded to fp16 (as relay stores them) or bf16 |
| arithmetic precision | fp32 vs fp16 autocast, identical weights |
| engine kernels | what remains between relay and the closest trainer variant |

Each is measured at the released weights and after 1, 3, 10 and 30 real GRPO updates, together
with how many weights an update actually changes once rounded ("update visibility") and how many
master weights have left the low-precision grid. Then: which sources matter for training, and
which corrections recover them.

## Result (2× T4, 15 Oct 2026; log: [`results/kaggle-2026-10-15.txt`](results/kaggle-2026-10-15.txt))

Absolute per-token log-probability difference (nats) caused by each source alone, 64 fresh rollouts
per checkpoint, one run with ratchet's fix. "Off" is the share of tokens whose importance ratio
leaves 1 ± 0.2 (none for arithmetic or kernels at any checkpoint).

| updates | rounding fp16: mean / max / off | rounding bf16: mean / max / off | arithmetic: mean / max | engine kernels: mean / max | weights changed by the update, fp16 / bf16 |
|---|---|---|---|---|---|
| 0 | 0.000 / 0.0001 / 0 | 0 / 0 / 0 | 0.0013 / 0.081 | 0.0007 / 0.040 | – |
| 1 | 0.016 / 1.93 / 1.2% | 0.022 / 2.16 / 2.2% | 0.0012 / 0.073 | 0.0007 / 0.033 | 13.4% / 1.7% |
| 3 | 0.025 / 2.30 / 2.8% | 0.041 / 3.65 / 5.5% | 0.0012 / 0.136 | 0.0007 / 0.056 | 9.7% / 1.3% |
| 10 | 0.030 / 3.47 / 3.4% | 0.104 / 9.70 / 13.3% | 0.0014 / 0.064 | 0.0008 / 0.031 | 6.7% / 1.0% |
| 30 | 0.012 / 0.78 / 1.0% | 0.196 / 10.98 / 15.0% | 0.0013 / 0.069 | 0.0008 / 0.066 | 5.0% / 0.7% |

- **Rounding at weight sync is the gap.** On average it is 9–22× the effect of fp16 arithmetic and
  15–39× that of the engine's kernels, which never put a token outside the ratio bound. The engine
  is always closest to the trainer that uses its own fp16-rounded weights (ratchet's fix), within
  0.031–0.066 nats; without the fix (second run) the gap reached 1.8–4.6 nats by updates 1–10.
- **Most of an update is invisible to the engine.** Rounded to fp16, 13.4% of weight-matrix entries
  change in the first update and 5.0% by update 30; in bf16, 1.7% and 0.7%. (These count every
  entry, including any the update did not move.) 97–99.8% of master weights sit off either grid.
- **bf16 would be worse, and grows worse:** the gap from bf16 rounding reaches 11 nats with 15% of
  tokens outside the bound by update 30, while fp16 stays at a mean of 0.012–0.030. Why it grows is
  not established by one run.
- **The gap falls on low-probability tokens:** without the fix, 0.10–0.17 nats on tokens the engine
  gave < 0.1, against 0.001–0.002 above 0.9.

Limits: one model (Qwen2.5-0.5B-Instruct), one run, 64 rollouts per checkpoint; the bf16 column
rounds the trainer's weights to bf16 and is not a bf16 engine. Written up as §6 of the ratchet
technical report.

```
!curl -sL https://raw.githubusercontent.com/Shakhtar-Sankur/ratchet/main/research/mismatch/kaggle.sh | bash
```
