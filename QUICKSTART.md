# Nano-PEARL 快速啟動指南

## 📦 環境準備

```bash
# 1. 安裝依賴
pip install torch transformers accelerate
pip install fastapi uvicorn zmq

# 2. 設定 Python Path
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/mini-sglang/python"
```

## 🚀 單行啟動指令（基礎版）

```bash
# 請先修改模型路徑
python3 -m nano_pearl.serve.launch \
    --model-path meta-llama/Llama-3.1-70B \
    --draft-model-path meta-llama/Llama-3.2-3B \
    --draft-tp-size 1 \
    --target-tp-size 4 \
    --port 8000
```

## 🎯 完整啟動指令（生產環境）

```bash
# 推薦配置：6 張 GPU (Draft 1張 + Target 4張 + 1張備用)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python3 -m nano_pearl.serve.launch \
    --model-path /path/to/Llama-3.1-70B \
    --draft-model-path /path/to/Llama-3.2-3B \
    --draft-tp-size 1 \
    --target-tp-size 4 \
    --gamma -1 \
    --max-num-batched-tokens 16384 \
    --max-running-requests 256 \
    --port 8000 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.9
```

## 📝 參數說明

| 參數 | 說明 | 推薦值 |
|------|------|--------|
| `--model-path` | Target 模型路徑 | 必填 |
| `--draft-model-path` | Draft 模型路徑 | 必填 |
| `--draft-tp-size` | Draft TP 大小 | 1 |
| `--target-tp-size` | Target TP 大小 | 4-8 |
| `--gamma` | 投機步長 | -1 (自動) |
| `--max-num-batched-tokens` | 批次大小 | 16384 |
| `--max-running-requests` | 最大並發數 | 256 |
| `--port` | 服務埠號 | 8000 |

## 🧪 測試 API

### Completions API
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nano-pearl",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'
```

### Chat API
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nano-pearl",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "stream": false
  }'
```

## 🔧 常見問題

### Q1: GPU 記憶體不足
**解決方案**：降低 `--max-running-requests` 或 `--max-num-batched-tokens`

### Q2: 服務啟動失敗
**檢查項目**：
1. PYTHONPATH 是否正確設定
2. CUDA_VISIBLE_DEVICES 是否包含足夠的 GPU
3. Draft TP + Target TP 是否超過可用 GPU 數量

### Q3: 效能不如預期
**調優建議**：
1. 執行自動 Gamma 調整（`--gamma -1`）
2. 增加 Target TP 大小
3. 檢查 Acceptance Rate（日誌中的 `num_acc_tokens`）

## 📊 效能監控

服務啟動後會自動輸出：
- Draft/Target 模型載入狀態
- 自動 Gamma 測量結果
- KV Cache 容量
- 請求處理統計

## 🛑 停止服務

```bash
# Ctrl+C 或
pkill -f "nano_pearl.serve.launch"
```
