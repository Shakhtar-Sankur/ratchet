#!/bin/bash
# Mismatch decomposition on Kaggle (GPU T4 x2 or a single T4, Internet on): relay samples on
# GPU 0; the trainer runs on GPU 1 when there is one, else beside relay on GPU 0.
# ~45 minutes. Paste from "== mismatch" to "== done".
set -e
cd /kaggle/working 2>/dev/null || cd /tmp
rm -rf ratchet && git clone -q --recursive https://github.com/Shakhtar-Sankur/ratchet && cd ratchet
echo "== mismatch: $(git log -1 --format='%h %s')"
nvidia-smi --query-gpu=index,name --format=csv,noheader
export PATH=/usr/local/cuda/bin:$PATH
scripts/build_relay.sh 75 2>&1 | tail -1
pip install -q transformers datasets huggingface_hub 2>&1 | tail -1 || true
python - <<'PY'
import json, os
from huggingface_hub import snapshot_download
from datasets import load_dataset
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir="models/qwen", allow_patterns=["*.json", "*.safetensors", "*.txt"])
os.makedirs("data", exist_ok=True)
for split in ("train", "test"):
    with open(f"data/gsm8k_{split}.jsonl", "w") as f:
        for r in load_dataset("openai/gsm8k", "main", split=split):
            f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
print("ready")
PY
export PYTHONPATH=$PWD
mkdir -p runs
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
TRAIN=cuda:$([ "$NGPU" -ge 2 ] && echo 1 || echo 0)
echo "GPUs: $NGPU, relay on cuda:0, trainer on $TRAIN"
R="python research/mismatch/decompose.py --model models/qwen --data data --train-device $TRAIN --relay-device 0"
echo "== 1. trainer uses the engine's fp16-rounded weights (ratchet's fix): steps 0, 1, 3, 10, 30"
$R --train-weights fp16 --checkpoints 0,1,3,10,30 --out runs/fix.jsonl | python research/mismatch/show.py
echo "== 2. trainer uses its float32 master weights (the bug): steps 0, 1, 3, 10"
$R --train-weights fp32 --checkpoints 0,1,3,10 --out runs/bug.jsonl | python research/mismatch/show.py
echo "== done"
