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

Status: experiment 1 (decomposition) is written and tested on the CPU; first T4 run pending.

```
!curl -sL https://raw.githubusercontent.com/Shakhtar-Sankur/ratchet/main/research/mismatch/kaggle.sh | bash
```
