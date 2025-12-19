# Nano-PEARL Serving Setup Guide

This guide explains how to set up and run the Nano-PEARL online serving infrastructure.

## Overview

Nano-PEARL serving provides an OpenAI-compatible API server for parallel speculative decoding using draft and target models on separate GPU groups.

## Quick Start

### 1. Prerequisites

Ensure you have the following installed:
- Python 3.10+
- CUDA Toolkit (matching your driver version)
- PyTorch 2.4.0+
- Required dependencies (see `pyproject.toml`)

### 2. Environment Setup

The serving infrastructure requires `mini-sglang` to be in your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/mini-sglang/python"
```

### 3. Launch the Server

Edit `run.sh` to configure your model paths and parameters:

```bash
# Model paths (REQUIRED - update these to your actual model locations)
DRAFT_MODEL="/path/to/your/draft/model"    # e.g., Qwen3-1.7B
TARGET_MODEL="/path/to/your/target/model"  # e.g., Qwen3-32B

# Performance parameters
DRAFT_TP=1        # Tensor Parallelism size for draft model
TARGET_TP=1       # Tensor Parallelism size for target model
GAMMA=-1          # Speculative length (-1 = auto-adjust)
MAX_BATCH=16384   # Maximum batch tokens
MAX_REQS=256      # Maximum concurrent requests
PORT=8000         # API server port
```

Then launch:

```bash
bash run.sh
```

## Architecture

### Argument Parsing Flow

The serving infrastructure uses a two-stage argument parsing approach:

1. **PEARL-specific arguments** are parsed first:
   - `--draft-model-path`: Path to draft model
   - `--draft-tp-size`: Draft model tensor parallelism
   - `--target-tp-size`: Target model tensor parallelism
   - `--gamma`: Speculative decoding length
   - `--max-num-batched-tokens`: Max batch size
   - `--no-use-radix-cache`: Use naive cache instead of radix

2. **Remaining arguments** are passed to minisgl:
   - `--model-path`: Target model path
   - `--host`, `--port`: Server configuration
   - `--memory-ratio`: GPU memory fraction
   - `--max-running-requests`: Concurrent request limit
   - And other standard minisgl server arguments

3. **Automatic conversions**:
   - Total `--tensor-parallel-size` = `draft_tp_size + target_tp_size`
   - `--cache-type` = `radix` if `use_radix_cache=True`, else `naive`

### Components

- **nano_pearl/serve/launch.py**: Main entry point, orchestrates server startup
- **nano_pearl/serve/args.py**: Argument parsing and validation
- **nano_pearl/serve/pearl_scheduler.py**: PEARL-specific scheduling logic
- **mini-sglang/**: Underlying serving infrastructure (from Mini-SGLang)

## Testing the API

Once the server is running, test it with curl:

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

## GPU Configuration

Set visible GPUs using:

```bash
export CUDA_VISIBLE_DEVICES=0,1  # Use GPU 0 and 1
```

For multi-GPU setups, ensure:
- Draft model uses `draft_tp_size` GPUs
- Target model uses `target_tp_size` GPUs
- Total GPUs available ≥ `draft_tp_size + target_tp_size`

## Troubleshooting

### ArgumentError: conflicting option string
**Fixed in current version.** This was caused by duplicate argument definitions.

### Module 'minisgl' not found
Ensure PYTHONPATH includes `mini-sglang/python`:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/mini-sglang/python"
```

### CUDA out of memory
- Reduce `--max-num-batched-tokens`
- Reduce `--max-running-requests`
- Increase `--memory-ratio` (default: 0.9)
- Use smaller models or reduce TP size

## Advanced Configuration

### Custom Arguments

You can pass additional minisgl arguments through `run.sh`:

```bash
python3 -m nano_pearl.serve.launch \
    --model-path "${TARGET_MODEL}" \
    --draft-model-path "${DRAFT_MODEL}" \
    --draft-tp-size ${DRAFT_TP} \
    --target-tp-size ${TARGET_TP} \
    --dtype bfloat16 \              # Additional minisgl arg
    --attention-backend fi \        # Additional minisgl arg
    ...
```

### Development Mode

For debugging, enable shell mode:

```bash
python3 -m nano_pearl.serve.launch \
    --model-path "${TARGET_MODEL}" \
    --draft-model-path "${DRAFT_MODEL}" \
    --shell-mode
```

## References

- [Mini-SGLang Documentation](mini-sglang/README.md)
- [Nano-PEARL Paper](https://arxiv.org/abs/2408.11850)
- [Nano-PEARL Website](https://smart-lty.github.io/nano-PEARL/)
