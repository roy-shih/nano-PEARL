#!/bin/bash

# ============================================
# Nano-PEARL Online Serving 啟動腳本
# ============================================

# 1. 環境變數設定
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 根據實際 GPU 數量調整
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/mini-sglang/python"

# 2. 模型路徑設定（請修改為您的實際路徑）
# Prefer local model paths found on this machine (change if needed)
PYTHON="$(pwd)/.venv/bin/python"
if [ -x "$PYTHON" ]; then
  echo "Using Python from virtualenv: $PYTHON"
else
  PYTHON=python3
fi

DRAFT_MODEL="/home/ubuntu/models/meta-llama/Llama-3.3-8B-Instruct"    # Draft model (smaller)
TARGET_MODEL="/home/ubuntu/models/meta-llama/Llama-3.3-70B-Instruct"  # Target model (larger)

# 3. 效能參數設定
DRAFT_TP=1              # Draft Model 的 Tensor Parallelism 大小
TARGET_TP=4             # Target Model 的 Tensor Parallelism 大小
GAMMA=-1                # 投機步長（-1 = 自動調整）
MAX_BATCH=16384         # 最大批次 Token 數
MAX_REQS=256            # 最大並發請求數
PORT=8000               # API Server 埠號
MODEL_NAME="nano-pearl" # API 中顯示的模型名稱（可自定義）

# 4. 啟動服務
python3 -m nano_pearl.serve.launch \
    --model-path "${TARGET_MODEL}" \
    --draft-model-path "${DRAFT_MODEL}" \
    --draft-tp-size ${DRAFT_TP} \
    --target-tp-size ${TARGET_TP} \
    --gamma ${GAMMA} \
    --max-num-batched-tokens ${MAX_BATCH} \
    --max-running-requests ${MAX_REQS} \
    --port ${PORT} \
    --host 0.0.0.0 \
    --mem-fraction-static 0.9 \
    --served-model-name "${MODEL_NAME}"

# 服務啟動後，可透過以下方式測試：
# curl http://localhost:8000/v1/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "nano-pearl",
#     "prompt": "Once upon a time",
#     "max_tokens": 100,
#     "temperature": 0.7,
#     "stream": true
#   }'
