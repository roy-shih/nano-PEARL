# KV Cache 模式切換說明

Nano-PEARL 現已支援兩種 KV Cache 管理模式，可透過參數控制：

## 🔧 兩種模式

### 1. Radix Tree 模式（預設，推薦）
- **優勢**：高效共享前綴快取，適合多輪對話與大量請求共享 System Prompt 的場景
- **實作**：使用 Radix Tree 資料結構追蹤 Token 序列，自動偵測與重用共享前綴
- **適用場景**：
  - 多輪對話（Chat API）
  - Few-shot Learning（共享範例）
  - 批量請求（相同 System Prompt）

### 2. Hash-based 模式（傳統實作）
- **優勢**：簡單直接，記憶體開銷低，適合單次請求或低並發場景
- **實作**：使用 XXHash 計算 Block 雜湊值進行前綴比對
- **適用場景**：
  - 單次請求（Completions API）
  - 低並發環境
  - 記憶體受限環境

## 🚀 使用方式

### Radix Tree 模式（預設）
```bash
python3 -m nano_pearl.serve.launch \
    --model-path /path/to/target/model \
    --draft-model-path /path/to/draft/model \
    --port 8000
    # 預設使用 Radix Tree
```

### Hash-based 模式
```bash
python3 -m nano_pearl.serve.launch \
    --model-path /path/to/target/model \
    --draft-model-path /path/to/draft/model \
    --port 8000 \
    --no-use-radix-cache  # 切換到 Hash-based 模式
```

## 📊 效能比較

| 場景 | Radix Tree | Hash-based |
|------|-----------|------------|
| 多輪對話（10輪） | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 單次請求 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 批量處理（共享前綴） | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 記憶體使用 | 中等 | 低 |
| 快取命中率 | 高 | 中 |

## 🔬 技術細節

### Radix Tree 內部機制
```python
# 配置時自動偵測共享前綴
input_ids = torch.tensor(seq.token_ids)
handle, cached_indices = self.radix_cache.match_prefix(input_ids)

# 鎖定並重用已快取的 Blocks
self.radix_cache.lock_handle(handle)
cached_block_ids = cached_indices[::block_size]
```

### Hash-based 內部機制
```python
# 計算雜湊值進行比對
h = xxhash.xxh64()
h.update(np.array(token_ids).tobytes())
block_id = self.hash_to_block_id.get(h.intdigest(), -1)

# 驗證 Token 序列完全匹配
if self.blocks[block_id].token_ids == token_ids:
    # 快取命中
```

## 💡 選擇建議

**使用 Radix Tree 當**：
- 部署 Chat 介面
- 預期高並發與長對話
- 記憶體充足（>80GB）

**使用 Hash-based 當**：
- 僅需 Completions API
- 低並發環境（<10 並發）
- 記憶體受限（<40GB）
- 偏好簡單穩定的實作

## 🧪 測試與驗證

兩種模式均通過相同的測試套件驗證：
- Rollback 正確性
- Cache 命中率
- 記憶體洩漏檢測
- 多輪對話一致性
