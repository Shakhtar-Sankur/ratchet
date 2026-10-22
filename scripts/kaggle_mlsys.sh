#!/bin/bash
# Experiments for the training-inference mismatch paper, on Kaggle (Settings: Accelerator
# "GPU T4 x2", Internet on). It also runs on Colab or any machine with NVIDIA GPUs: it uses two
# GPUs when there are two, otherwise one (each step then takes about twice as long). The paper's
# runs all use 2 x T4, so runs on other hardware are for checking only. Every run is 100 colocated GRPO steps on GSM8K, then a greedy
# evaluation on all 1,319 test problems (the model before training is evaluated once).
#   RUN=smoke : every configuration for 2 steps on 64 test problems (~15 min); run this first.
#   RUN=A     : fix vs no fix, 3 seeds each (seed 0 of the fix is the published run), ~5.5 h
#   RUN=B     : truncated importance sampling (alone, and with the fix), and learning rates
#               3e-6 and 1e-5 with and without the fix, ~6 h
# Use "Save Version -> Save & Run All" for A and B, so a run survives a closed browser.
# Paste everything from "== ratchet mlsys" to the end back into the chat.
set -e
RUN=${RUN:-smoke}
WORK=/tmp; for d in /kaggle/working /content; do [ -d $d ] && WORK=$d && break; done
cd $WORK
rm -rf ratchet && git clone -q --recursive https://github.com/Shakhtar-Sankur/ratchet && cd ratchet
echo "== ratchet mlsys ($RUN): $(git log -1 --format='%h %s') | relay $(git -C relay log -1 --format=%h) | tandem $(git -C tandem log -1 --format=%h)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
NGPU=$(nvidia-smi -L | wc -l)
WORLD=${WORLD:-$([ "$NGPU" -ge 2 ] && echo 2 || echo 1)}
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
echo "== $NGPU GPU(s), sm_$ARCH, colocated on $WORLD"
[ "$WORLD" = 2 ] && [ "$ARCH" = 75 ] || echo "== not 2 x T4: use these runs to check the setup, not in the paper"
export PATH=/usr/local/cuda/bin:$PATH
scripts/build_relay.sh $ARCH 2>&1 | tail -1
pip install -q transformers datasets huggingface_hub 2>&1 | tail -1 || true

python - <<'PY'
import json, os
from huggingface_hub import snapshot_download
from datasets import load_dataset
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir="models/qwen",
                  allow_patterns=["*.json", "*.safetensors", "*.txt"])
os.makedirs("data", exist_ok=True)
for split in ("train", "test"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    with open(f"data/gsm8k_{split}.jsonl", "w") as f:
        for r in ds:
            f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
PY

export PYTHONPATH=$PWD
OUT=$WORK/runs_$RUN; mkdir -p $OUT
G="python -m ratchet.gsm8k"
COMMON="--model models/qwen --data data"
if [ "$RUN" = smoke ]; then STEPS=2; EVAL="--eval-limit 64"; else STEPS=100; EVAL=""; fi
SHOW() { grep --line-buffered -E '"phase": "eval"|"step": [0-9]*0,' || true; }

# name, then the flags that differ from the defaults (fix = trainer computes with the engine's
# fp16-rounded weights; nofix = trainer computes with its float32 master weights)
run() {
  local name=$1; shift
  echo "== $name: $*"
  $G colocated $COMMON --world $WORLD --steps $STEPS $EVAL --skip-eval-before --out $OUT/$name.jsonl "$@" 2>&1 | SHOW \
    || echo "== $name FAILED (continuing)"
}

if [ "$RUN" = smoke ] || [ "$RUN" = A ]; then
  echo "== base model (evaluated once)"
  $G eval $COMMON $EVAL --out $OUT/base.jsonl | SHOW
fi
if [ "$RUN" = smoke ]; then
  run fix_s1      --weight-dtype fp16 --seed 1
  run nofix_s0    --weight-dtype fp32 --seed 0
  run tis_nofix   --weight-dtype fp32 --tis-cap 2
  run lr1e-5_fix  --weight-dtype fp16 --lr 1e-5
fi
if [ "$RUN" = A ]; then
  run fix_s1      --weight-dtype fp16 --seed 1
  run fix_s2      --weight-dtype fp16 --seed 2
  run nofix_s0    --weight-dtype fp32 --seed 0
  run nofix_s1    --weight-dtype fp32 --seed 1
  run nofix_s2    --weight-dtype fp32 --seed 2
fi
if [ "$RUN" = B ]; then
  run tis_nofix   --weight-dtype fp32 --tis-cap 2
  run tis_fix     --weight-dtype fp16 --tis-cap 2
  run lr3e-6_fix   --weight-dtype fp16 --lr 3e-6
  run lr3e-6_nofix --weight-dtype fp32 --lr 3e-6
  run lr1e-5_fix   --weight-dtype fp16 --lr 1e-5
  run lr1e-5_nofix --weight-dtype fp32 --lr 1e-5
fi
echo "== summary"
python scripts/summarize_mlsys.py $OUT
echo "== done: copy from '== ratchet mlsys' to here and send it back"
