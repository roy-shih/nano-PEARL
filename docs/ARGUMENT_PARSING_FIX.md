# 參數解析問題診斷與解決方案

## 🔍 問題根因分析

### 調用鏈
```
用戶執行：python -m nano_pearl.serve.launch --model-path X --draft-model-path Y --draft-tp-size 1
    ↓
nano_pearl/serve/launch.py:16  pearl_args, run_shell = parse_args(sys.argv[1:], run_shell)
    ↓  
nano_pearl/serve/args.py:81    server_args, run_shell = parse_server_args(args, run_shell)
    ↓
minisgl/server/args.py         parse_args(args) <- 這裡解析 minisgl 標準參數
    ↓
nano_pearl/serve/args.py:98    extra_parser.parse_known_args(args) <- 這裡解析PEARL擴展參數
    ↓
nano_pearl/serve/launch.py:76  run_api_server(pearl_args, start_subprocess, run_shell=run_shell)
    ↓
minisgl/server/api_server.py:384  run_api_server(config: ServerArgs, ...)
    ⚠️ 問題：如果這裡有某處重新使用 sys.argv，會報錯
```

### 可能的問題點

#### 問題 1：sys.argv 未清理
如果 `run_api_server` 或其內部調用的任何函數重新解析 `sys.argv[1:]`，會遇到無法識別 `--draft-model-path` 等參數。

#### 問題 2：參數傳遞不完整
`parse_known_args` 解析後可能有 `unknown` 參數殘留在某處。

## 🛠️ 解決方案

### 方案 1：清理 sys.argv（推薦）

在調用 `run_api_server` 前清理 `sys.argv`，只保留 minisgl 能識別的參數：

```python
# nano_pearl/serve/launch.py

def launch_server(run_shell: bool = False) -> None:
    # 1. 保存原始 argv
    original_argv = sys.argv.copy()
    
    # 2. Parse Args
    pearl_args, run_shell = parse_args(sys.argv[1:], run_shell)
    
    # ⭐ 關鍵修復：清理 sys.argv，移除 PEARL 專屬參數
    # 重新構造 sys.argv，只包含 minisgl 能識別的參數
    cleaned_argv = [sys.argv[0]]  # 保留程序名
    
    # 添加 minisgl 標準參數
    cleaned_argv.extend([
        '--model-path', pearl_args.model_path,
        '--host', pearl_args.server_host,
        '--port', str(pearl_args.server_port),
        '--tp-size', str(pearl_args.tp_size),
        # ... 其他 minisgl 標準參數
    ])
    
    sys.argv = cleaned_argv
    
    try:
        def start_subprocess() -> None:
            # ... （保持原樣）
            
        run_api_server(pearl_args, start_subprocess, run_shell=run_shell)
    finally:
        # 恢復原始 argv（如果需要）
        sys.argv = original_argv
```

### 方案 2：修改 parse_known_args 行為

確保所有未知參數都被正確處理：

```python
# nano_pearl/serve/args.py

known, unknown = extra_parser.parse_known_args(args)

# 如果有殘留的未知參數，記錄並忽略
if unknown:
    logger.warning(f"Ignoring unknown arguments: {unknown}")
```

### 方案 3：不使用 run_api_server（完全自定義）

如果 minisgl 的 `run_api_server` 有太多假設，可以複製並修改它：

```python
# nano_pearl/serve/pearl_api_server.py

def run_pearl_api_server(config: PearlServerArgs, start_backend, run_shell):
    """
    PEARL 專用的 API Server，不依賴 minisgl.server.api_server
    """
    # 實作自己的 API Server 啟動邏輯
    ...
```

## 🔧 立即可行的修復

### 修改 nano_pearl/serve/launch.py

