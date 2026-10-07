#!/bin/bash
# ratchet M5 on Kaggle (Settings: Accelerator "GPU T4 x2", Internet on).
#   RUN=smoke (default): every phase for a few steps on 64 test problems, ~20 minutes;
#                        finds problems before the long run.
#   RUN=full:            the measurements, ~2-3 hours; use "Save Version -> Save & Run All".
# Paste everything from "== ratchet M5" to the end back into the chat.
set -e
RUN=${RUN:-smoke}
cd /kaggle/working 2>/dev/null || cd /tmp
rm -rf ratchet && git clone -q --recursive https://github.com/Shakhtar-Sankur/ratchet && cd ratchet
echo "== ratchet M5 ($RUN): $(git log -1 --format='%h %s') | relay $(git -C relay log -1 --format=%h) | tandem $(git -C tandem log -1 --format=%h)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -1
scripts/build_relay.sh 75 2>&1 | tail -1
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
    print("gsm8k", split, len(ds))
PY

export PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs
G="python -m ratchet.gsm8k"
COMMON="--model models/qwen --data data"
if [ "$RUN" = full ]; then
  STEPS=100; EVAL=""; TSTEPS=20
else
  STEPS=3; EVAL="--eval-limit 64"; TSTEPS=3
fi

echo "== check: fp16 vs fp32 log-probabilities, weight sync, answer lengths (GPU 0)"
$G check $COMMON --out runs/check.jsonl
echo "== colocated: tandem DDP over both GPUs, each generates then trains ($STEPS steps)"
$G colocated $COMMON --steps $STEPS $EVAL --out runs/colocated.jsonl | tail -n 3
echo "== split, synchronous: GPU 1 generates, then GPU 0 trains ($TSTEPS steps, timing only)"
$G split $COMMON --train-device cuda:0 --relay-device 1 --steps $TSTEPS --skip-eval --out runs/split_sync.jsonl | tail -n 1
echo "== split, one step ahead + partial rollouts ($STEPS steps)"
$G split $COMMON --train-device cuda:0 --relay-device 1 --steps $STEPS --ahead --partial $EVAL --out runs/split_async.jsonl | tail -n 3
echo "== summary"
python scripts/summarize.py runs
echo "== done: copy from '== ratchet M5' to here and send it back"
