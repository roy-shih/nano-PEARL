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
| **模型配置** | | |
| `--model-path` | Target 模型路徑 | 必填 |
| `--draft-model-path` | Draft 模型路徑 | 必填 |
| `--target-tp-size` | Target TP 大小 | 4-8 |
| `--draft-tp-size` | Draft TP 大小 | 1 |
| **效能配置** | | |
| `--gamma` | 投機步長 | -1 (自動) |
| `--max-num-batched-tokens` | 批次大小 | 16384 |
| `--max-running-requests` | 最大並發數 | 256 |
| **Cache 配置** | | |
| `--use-radix-cache` | 使用 Radix Tree（預設） | 推薦 |
| `--no-use-radix-cache` | 使用 Hash-based Cache | 可選 |
| **服務配置** | | |
| `--host` | 監聽位址 | 127.0.0.1 |
| `--port` | 埠號 | 8000 |
| `--mem-fraction-static` | 記憶體佔用比例 | 0.9 |

> 💡 **Cache 模式選擇**：
> - **Radix Tree**（預設）：適合多輪對話與高並發場景，快取命中率高
> - **Hash-based**：簡單穩定，記憶體開銷低，適合單次請求
> - 詳細比較請參考 [CACHE_MODES.md](CACHE_MODES.md)

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
4. 根據場景選擇合適的 Cache 模式

### Q4: 該用哪種 Cache 模式？
**Radix Tree 模式**（推薦給）：
- 多輪對話應用（Chat API）
- 高並發環境（>50 並發）
- 共享 System Prompt 的場景

**Hash-based 模式**（推薦給）：
- 單次請求（Completions API）
- 低並發環境（<10 並發）
- 記憶體受限環境（<40GB）

## 📊 效能監控

服務啟動後會自動輸出：
- Draft/Target 模型載入狀態
- 自動 Gamma 測量結果
- KV Cache 容量與模式
- 請求處理統計

## 🛑 停止服務

```bash
# Ctrl+C 或
pkill -f "nano_pearl.serve.launch"
```
