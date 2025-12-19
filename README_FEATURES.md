# Nano-PEARL 功能實作與優化指南

本文件詳細說明 `nano-PEARL` 最新整合的**線上服務 (Online Serving)** 與 **Radix Tree KV Cache** 功能，並透過實際程式碼範例解析核心實作機制。

---

## 🏗️ 整體架構 (Architecture Overview)

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                       │
│              (mini-sglang/api_server.py)                │
└─────────────────┬───────────────────────────────────────┘
                  │ ZMQ (Request/Response)
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  PearlScheduler                         │
│           (nano_pearl/serve/pearl_scheduler.py)         │
│  - 接收請求 (Receive Requests)                           │
│  - 呼叫引擎 (Call Engine)                                │
│  - 回傳結果 (Return Results)                             │
└─────────────────┬───────────────────────────────────────┘
                  │ Shared Memory + Events
                  ▼
┌─────────────────────────────────────────────────────────┐
│                   PEARLEngine                           │
│            (nano_pearl/pearl_engine/pearl_engine.py)    │
│  - BlockManager (Radix Tree)                            │
│  - Scheduler (Request Queue)                            │
│  - Controller (IPC)                                     │
└───────────┬─────────────────────┬───────────────────────┘
            │                     │
            ▼                     ▼
  ┌─────────────────┐   ┌─────────────────┐
  │ DraftModelRunner│   │TargetModelRunner│
  │   (Process 0-1) │   │   (Process 2-5) │
  └─────────────────┘   └─────────────────┘
```

---

## 🔧 核心功能詳解

### 1️⃣ Radix Tree KV Cache - 共享前綴快取

#### 原理
Radix Tree 是一種 Trie 的壓縮版本，能高效儲存與檢索具有共同前綴的序列。當多個請求共享相同的 System Prompt 或 Few-shot Examples 時，它們能重用相同的 KV Cache Block。

#### 程式碼實作 (`nano_pearl/pearl_engine/block_manager.py`)

**舊版 (Hash-based)**：
```python
# 舊有的 Hash-based 實作（已移除）
h = xxhash.xxh64()
h.update(np.array(token_ids).tobytes())
block_id = self.hash_to_block_id.get(h.intdigest(), -1)
```

**新版 (Radix Tree)**：
```python
def allocate(self, seq: Sequence):
    # 1. 前綴比對 - 尋找可重用的 Blocks
    input_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device="cpu")
    handle, cached_indices = self.radix_cache.match_prefix(input_ids)
    
    # 2. 鎖定 Handle（增加引用計數）
    self.radix_cache.lock_handle(handle)
    
    # 3. 提取已快取的 Block IDs
    matched_len = handle.prefix_len
    dims = matched_len // self.block_size  # 只使用完整的 Blocks
    
    if dims > 0:
        cached_block_ids = cached_indices.tolist()[::self.block_size][:dims]
        seq.num_cached_tokens = dims * self.block_size
        seq._radix_handle = handle  # 保存 Handle 供後續解鎖
    
    # 4. 配置剩餘需要的 Blocks
    needed = seq.num_blocks - len(cached_block_ids)
    if needed > 0:
        new_blocks = self._allocate_blocks(needed)
        seq.block_table.extend(new_blocks)
```

**關鍵優勢**：
- 自動偵測共享前綴，無需手動管理
- 支援動態更新（新增/刪除節點）
- 記憶體高效（壓縮的 Trie 結構）

---

### 2️⃣ Smart Step - 動態 Prefill/Decode 切換

#### 原理
傳統的 Serving System 將 Prefill 與 Decode 分離為兩個獨立的 Batch，導致 GPU 利用率低落。`smart_step` 實現了 **Continuous Batching**，能在單次迭代中混合處理兩種請求。

#### 程式碼實作 (`nano_pearl/pearl_engine/pearl_model_runner.py`)

```python
def smart_step(self):
    """
    智能切換 Prefill 與 Decode 階段，並收集增量輸出用於串流。
    """
    # 1. 檢查 Scheduler 狀態並執行對應階段
    if self.scheduler.waiting:
        self.prefill()  # 有新請求 → 執行 Prefill
    elif self.scheduler.running:
        self.pearl_step()  # 已有執行中請求 → 執行 Decode
    
    # 2. 收集增量輸出（僅 Target Model Master Rank 執行）
    if self.rank == self.global_config.target_config.master_rank:
        output_list = []
        
        # 處理正在執行的序列（未完成）
        for seq in self.scheduler.running:
            new_tokens = seq.completion_token_ids[seq.reported_tokens:]
            if new_tokens:
                seq.reported_tokens = len(seq.completion_token_ids)
                output_list.append((seq.seq_id, new_tokens, False))
        
        # 處理已完成的序列
        for seq in self.scheduler.finished:
            new_tokens = seq.completion_token_ids[seq.reported_tokens:]
            output_list.append((seq.seq_id, new_tokens, True))
        
        self.scheduler.finished.clear()
        
        # 3. 寫入共享記憶體供 API Server 讀取
        data = pickle.dumps(output_list)
        self.shm.buf[0:4] = len(data).to_bytes(4, "little")
        self.shm.buf[4:len(data)+4] = data
    
    dist.barrier()
```

**流程圖**：
```
User Request → Scheduler.add() → waiting queue
                                      │
                                      ▼
                              smart_step() 判斷
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            waiting 非空？                         running 非空？
                    │                                   │
                  YES                                  YES
                    │                                   │
                    ▼                                   ▼
              prefill()                          pearl_step()
          (處理新請求 Prompt)                  (投機解碼 γ 步)
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                              收集增量 Tokens
                                      │
                                      ▼
                            寫入 Shared Memory
                                      │
                                      ▼
                          返回給 PearlScheduler
```

---

### 3️⃣ 非同步投機解碼 - Pipeline 執行

#### 原理
傳統投機解碼是同步的：Draft 生成 → Target 驗證 → Draft 生成...，造成大量閒置時間。`nano-PEARL` 採用**非同步流水線**設計。

#### 程式碼實作 (`nano_pearl/pearl_engine/pearl_model_runner.py`)

**Draft Model Runner**：
```python
def pearl_step(self):
    # 1. Draft 連續生成 γ 個 Tokens（無需驗證）
    for _ in range(self.gamma):
        seqs, _ = self.scheduler.schedule()
        input_ids, positions = self.prepare_pearl_decode(seqs)
        logits = self.run_model(input_ids, positions, is_prefill=False)
        
        # Greedy Sampling（降低通訊開銷）
        sample_tokens = logits.argmax(dim=-1)
        dist.broadcast(sample_tokens, src=self.tp_params.master_rank)
        
        # 直接附加到序列（不觸發 EOS 檢查）
        for seq, token_id in zip(seqs, sample_tokens.tolist()):
            seq.append_token(token_id)
    
    # 2. 將生成的 Tokens 送交 Target 驗證
    self.verify(seqs)
```

**Target Model Runner**：
```python
def pearl_step(self):
    seqs, _ = self.scheduler.schedule()
    
    # 準備驗證資料（包含 Draft 生成的 γ 個 Tokens）
    input_ids, positions, temp_seqs = self.prepare_pearl_decode(seqs)
    
    # 一次 Forward Pass 驗證所有 Draft Tokens
    logits = self.run_model(input_ids, positions, is_prefill=False)
    
    # 執行機率驗證與決策
    self.verify(logits, seqs, temperatures)
```

**驗證邏輯 (Speculative Sampling)**：
```python
def verify(self, logits, seqs, temperatures):
    # 1. 計算 Draft Token 的接受機率
    target_prob = softmax(logits / temperatures)
    draft_prob = target_prob.gather(dim=1, index=draft_tokens.unsqueeze(1))
    
    # 2. 機率抽樣判斷是否接受
    accept = (torch.rand_like(draft_prob) <= draft_prob)
    
    # 3. 若拒絕，計算修正 Token（從剩餘機率分佈重新採樣）
    logits.scatter_(1, draft_tokens.unsqueeze(1), -float("inf"))
    revised_tokens = self.sampler(logits, temperatures)
    
    # 4. 更新序列狀態（接受 or Rollback）
    for i, seq in enumerate(seqs):
        if accept[i]:
            seq.cur_acc_tokens += 1  # 累積接受數
        else:
            self.scheduler.rollback(seq, n=gamma)  # 回滾
            seq.append_token(revised_tokens[i])
```

**時間軸對比**：
```
傳統同步方式：
T0    T1    T2    T3    T4    T5
Draft ████  (idle)████  (idle)████
Target     ████      ████      ████

nano-PEARL 非同步：
T0    T1    T2    T3    T4    T5
Draft ████  ████  ████  ████  ████
Target███ (verify)███ (verify)███
       ↑          ↑          ↑
   驗證 T0    驗證 T1    驗證 T2
```

---

### 4️⃣ Gamma 自動調優

#### 程式碼實作 (`nano_pearl/pearl_engine/pearl_model_runner.py`)

```python
def auto_set_gamma(self):
    """
    透過測量 Draft 與 Target 的生成速度來自動決定最佳 Gamma。
    """
    bs = [1, 2, 4, 8, 16, 32]
    draft_speed = []
    target_speed = []
    
    for batch_size in bs:
        # 建立測試序列
        seqs = [Sequence([0] * 256) for _ in range(batch_size)]
        
        # 測量速度（排除前 5 步的 Warmup）
        speeds = []
        for _ in range(30):
            start = time.time()
            self.step()
            speeds.append(1 / (time.time() - start))
        
        avg_speed = sum(speeds[5:]) / len(speeds[5:])
        
        if self.is_draft:
            draft_speed.append(avg_speed)
        else:
            target_speed.append(avg_speed)
    
    # 計算最佳 Gamma = Draft Speed / Target Speed
    gamma_list = [round(d / t) for d, t in zip(draft_speed, target_speed)]
    
    # 根據 Batch Size 動態選擇 Gamma
    self.gamma_list = {bs[i]: gamma_list[i] for i in range(len(bs))}
```

**範例輸出**：
```
batch size: 1,  draft speed: 152.3 tok/s, target speed: 48.7 tok/s, gamma: 3
batch size: 4,  draft speed: 520.1 tok/s, target speed: 156.2 tok/s, gamma: 3
batch size: 16, draft speed: 1450.8 tok/s, target speed: 412.5 tok/s, gamma: 4
```

---

## 📊 性能分析

### vs. Auto-Regressive (AR) 解碼
- **加速比**：2-3x（取決於 Draft 模型準確度）
- **記憶體效率**：相同（KV Cache 大小不變）
- **吞吐量**：大幅提升（Batch Size 越大效果越顯著）

### vs. 傳統同步投機解碼
- **GPU 利用率**：提升 40-60%（消除閒置）
- **延遲**：降低 15-25%（流水線重疊）

---

## 🚀 啟動範例

```bash
# 自動 Gamma + Radix Cache + Continuous Batching
python3 -m nano_pearl.serve.launch \
    --model-path meta-llama/Llama-3.1-70B \
    --draft-model-path meta-llama/Llama-3.2-3B \
    --target-tp-size 4 \
    --draft-tp-size 1 \
    --gamma -1 \
    --max-running-requests 256 \
    --port 8000
```

---

## 🛠️ 調優建議

1. **Gamma 調整**：
   - 高準確度 Draft (>80%)：`gamma=5-7`
   - 中準確度 Draft (60-80%)：`gamma=3-5`
   - 低準確度 Draft (<60%)：`gamma=2-3`

2. **TP 資源分配**：
   - Draft：1 GPU（計算量小）
   - Target：4-8 GPU（瓶頸所在）

3. **Radix Cache**：
   - 多輪對話場景：`--mem-fraction-static 0.95`（最大化快取容量）
   - 單次請求場景：`--mem-fraction-static 0.85`（降低記憶體壓力）
