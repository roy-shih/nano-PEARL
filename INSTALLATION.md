# nano-PEARL 安裝與運行指南

## 🔧 問題診斷：Module Not Found 與參數錯誤

### 問題 1：`ModuleNotFoundError: No module named 'minisgl'`

**原因**：`mini-sglang` 沒有被安裝為 Python package

**解決方案**：

```bash
# 在 nano-PEARL 根目錄下
cd mini-sglang
pip install -e .  # 以 editable 模式安裝

# 或者設置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/mini-sglang/python"
```

---

### 問題 2：`error: unrecognized arguments: --draft-model-path`

**原因**：錯誤地使用了 `mini-sglang` 的原始 `launch.py`，而非 `nano-pearl` 的擴展版本

**錯誤命令**：
```bash
# ❌ 錯誤：調用了 mini-sglang 的 launch.py
python -m mini-sglang.server.launch --draft-model-path ...
```

**正確命令**：
```bash
# ✅ 正確：調用 nano-pearl 的 launch.py
python -m nano_pearl.serve.launch \
    --model-path /path/to/target/model \
    --draft-model-path /path/to/draft/model \
    --draft-tp-size 1 \
    --target-tp-size 1 \
    --port 8000
```

---

## 📦 完整安裝步驟

### 步驟 1：安裝 nano-PEARL

```bash
cd /path/to/nano-PEARL
pip install -e .
```

### 步驟 2：安裝 mini-sglang（依賴）

```bash
cd mini-sglang
pip install -e .
```

### 步驟 3：驗證安裝

```bash
python -c "import nano_pearl; import minisgl; print('✅ Installation successful')"
```

### 步驟 4：運行服務

```bash
# 方法 1：直接運行模組（推薦）
python -m nano_pearl.serve.launch \
    --model-path /home/ubuntu/models/Qwen/Qwen3-72B \
    --draft-model-path /home/ubuntu/models/Qwen/Qwen3-1.7B \
    --draft-tp-size 1 \
    --target-tp-size 1 \
    --port 8000

# 方法 2：使用啟動腳本
# 先編輯 start_server.sh 設置模型路徑
./start_server.sh
```

---

## 🐛 常見錯誤與解決

### 錯誤 1：找不到 `nano_pearl` 模組

```bash
ModuleNotFoundError: No module named 'nano_pearl'
```

**解決**：
```bash
# 確認在正確目錄
cd /path/to/nano-PEARL
pip install -e .
```

---

### 錯誤 2：找不到 `minisgl` 模組

```bash
ModuleNotFoundError: No module named 'minisgl'
```

**解決**：
```bash
cd mini-sglang
pip install -e .

# 或設置環境變數
export PYTHONPATH="${PYTHONPATH}:$(pwd)/mini-sglang/python"
```

---

### 錯誤 3：參數不被識別

```bash
error: unrecognized arguments: --draft-model-path
```

**原因**：使用了錯誤的啟動命令

**檢查**：
```bash
# 查看實際執行的腳本
which python
python -m nano_pearl.serve.launch --help  # 應該顯示 PEARL 擴展參數
```

**解決**：確保使用 `python -m nano_pearl.serve.launch`

---

### 錯誤 4：Import 循環依賴

```bash
ImportError: cannot import name 'XXX' from partially initialized module
```

**解決**：確保所有 `__init__.py` 文件存在
```bash
# 檢查必要的 __init__.py
ls nano_pearl/__init__.py          # 應存在
ls nano_pearl/serve/__init__.py    # 應存在
ls nano_pearl/pearl_engine/__init__.py  # 應存在
```

---

## 📝 快速檢查清單

運行前請確認：

- [ ] `nano-PEARL` 已安裝：`pip list | grep nano-pearl`
- [ ] `mini-sglang` 已安裝：`pip list | grep minisgl`
- [ ] 使用正確的啟動命令：`python -m nano_pearl.serve.launch`
- [ ] 模型路徑正確且可訪問
- [ ] GPU 可用：`nvidia-smi`
- [ ] 足夠的 GPU 數量：`draft_tp_size + target_tp_size ≤ GPU 數量`

---

## 🔍 調試技巧

### 查看模組搜索路徑

```python
import sys
print('\n'.join(sys.path))
```

### 驗證模組導入

```python
# 測試 nano_pearl 導入
from nano_pearl.serve.args import parse_args
from nano_pearl.serve.launch import launch_server
print("✅ nano_pearl imports OK")

# 測試 minisgl 導入  
from minisgl.server.api_server import run_api_server
from minisgl.tokenizer import tokenize_worker
print("✅ minisgl imports OK")
```

### 查看可用參數

```bash
python -m nano_pearl.serve.launch --help
```

應顯示擴展參數：
- `--draft-model-path`
- `--draft-tp-size`
- `--target-tp-size`
- `--gamma`
- `--use-radix-cache / --no-use-radix-cache`

---

## 📚 相關文檔

- [QUICKSTART.md](../QUICKSTART.md) - 快速啟動指南
- [README_FEATURES.md](../README_FEATURES.md) - 功能說明
- [start_server.sh](../start_server.sh) - 啟動腳本範例
