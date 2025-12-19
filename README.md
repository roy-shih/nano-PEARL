<div align="center">

![nano-pearl](static/nano_pearl.gif)

[![Website](https://img.shields.io/badge/website-nano--PEARL-orange)](https://smart-lty.github.io/nano-PEARL/)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](#) 
[![ArXiv](https://img.shields.io/badge/arXiv-2408.11850-b31b1b)](https://arxiv.org/abs/2408.11850)
[![Conference](https://img.shields.io/badge/ICLR-2025-4B7BEC)](#)

<em><strong>Unleashing Batch Throughput: </strong><br>Reimagined Speculative Decoding with Parallelism and Adaptive Draft Length<br></em>

</div>

# 🚀 nano-PEARL

nano-PEARL is a single-node, multi-GPU parallel speculative decoding engine that decouples Draft and Target models onto separate device groups. With on-the-fly verification, prefix KV caching, CUDA Graphs, paged attention, flash attention, and tensor parallelism, it delivers exceptional throughput while maintaining output quality.

<font color="#ADD8E6">**Ready to jumpstart your research?** All advanced acceleration features are built-in—focus on your algorithm, not the engineering.</font>

> <font color="#90EE90">✨ No more integration worries from aggressive reviewers!</font> 

## 🎉 Latest News

🚧 **Coming Soon**: More updates and features are in development!

📚 **Paper Collection**: We are seeking for papers that follows parallel speculative decoding paradigm! We are going to prepare a paper collection of these papers for more comprehensive understanding.
- [2025.11] 🔥 We release some comparisons between EAGLE-3 and nano-PEARL! Check them at our [benchmark](https://smart-lty.github.io/nano-PEARL/benchmark.html) page.
- [2025.11] 🔥 We Support Non-2-Power TP size to enable better GPU utilization!
- [2025.11] 🔥 We release more benchmark results of nano-PEARL on NVIDIA L40S!
- [2025.11] 🔥 Our web page of [nano-PEARL](https://smart-lty.github.io/nano-PEARL/) is established!
- [2025.10] 🔥 We release the source code of nano-PEARL. Any PR is warmly welcomed!

## ✨ Key Features

- 🔄 **Draft-Target Disaggregation**: The draft model and the target model are loaded in separate devices, avoiding load-imbalance and resource competition.
- ⚡ **Parallel Inference**: Both the draft model and the target model run inference in parallel, fully exploiting the GPU utilization!
- 🎯 **Adaptive Draft Length**: 
  - When the alignment is good, the draft model could generate draft tokens without being interrupted by the target model.
  - When the alignment is poor, the target model could prevent the draft model from generating trash draft tokens.
- 🤖 **Auto-Set Hyper-parameters**: Automatically configure optimal parameters for your hardware setup.
- ⚙️ **Dynamic TP Size**: We support loading draft / target model with non-2-power TP size (3,6,7). To the best of our knowledge, we are **the first one** implementing this feature for LLM inference!
> ⚠️ Currently, this is an experimental feature.We implement dynamic TP by padding the parameters, which introduces additional computation and decrease the overall throughput. 
- 🚀 **High Performance**: Built on CUDA Graphs and tensor parallelism for maximum throughput.
- 💾 **Memory Efficient**: Prefix KV caching reduces memory usage while maintaining performance.


## 📦 Installation

Our nano-PEARL is built based on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), and the installation is almost same with nano-vllm (only need to additionally install `rich` for colorful log). 🎨

First create an environment with `python>=3.12`:
```shell
conda create -n nano-pearl python=3.12 -y
conda activate nano-pearl
```

Then, install packages with `uv` or `pip`:
**From source:**
```shell
uv pip install -e . # from current path
```

**From GitHub:**
```shell
pip install git+https://github.com/smart-lty/nano-PEARL.git # from github
```
<details><summary><b>Some hints for potential problem solving</b></summary>

> ⚠️ When you directly use pip for installation, you may encounter that build flash-attn needs torch installed. In this case, you should **install torch first**, and then re-run the installation command.

> ⚠️ If the installation of flash-attn is very slow, we strongly recommend you to download a whl file and **build flash attn from wheel**.
</details>

## 🌐 Online Serving (NEW!)

nano-PEARL now supports **OpenAI-compatible API Server** for production deployment! 🎉

```bash
# Start the server
python3 -m nano_pearl.serve.launch \
    --model-path /path/to/Llama-3.1-70B \
    --draft-model-path /path/to/Llama-3.2-3B \
    --draft-tp-size 1 \
    --target-tp-size 4 \
    --port 8000

# Test with curl
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 100, "stream": true}'
```

**Key Features**:
- ✅ **Continuous Batching**: Dynamic Prefill/Decode switching for maximum throughput
- ✅ **Streaming Support**: Real-time token streaming via SSE
- ✅ **Radix Tree KV Cache**: Efficient prefix sharing for multi-turn conversations
- ✅ **Hash-based Cache Option**: Use `--no-use-radix-cache` for simpler scenarios

📖 **Documentation**:
- [README_FEATURES.md](README_FEATURES.md) - Detailed technical implementation
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide with examples  
- [CACHE_MODES.md](CACHE_MODES.md) - Cache mode comparison & tuning

## 📋 TODOs & Roadmap

### ✅ Completed Features
- [x] **Dynamic TP Size**: Support dynamic TP size, including TP=6/7, hence the 8 GPUs can be fully used!
- [x] **Continuous Batching**: Support continuous batching and chunked prefill. ✅ (Online Serving)
- [x] **Online Serving**: OpenAI-compatible API Server with streaming ✅
- [x] **Radix Tree KV Cache**: Advanced prefix caching for multi-turn chat ✅

### 🚧 In Progress

#### 🎯 Algorithm Enhancements
- [ ] **Confidence-based Rollback**: 
  - Leverage draft model's perplexity/confidence scores for smarter rollback decisions
  - Score range: 1.0 (high confidence, trust draft) → 0.0 (low confidence, trust target)
  - Adaptive threshold tuning based on historical acceptance rate
  - Potential 10-15% acceptance rate improvement

- [ ] **Adaptive Gamma Scheduling**: 
  - Dynamic `gamma` tuning based on context size and model's performance
  - Auto-adjust per request based on measured acceptance rate
  - Context-aware gamma selection (longer context → smaller gamma)

#### ⚙️ System Optimization
- [ ] **Hyperparameter Auto-tuning**:
  - Integrate Hydra for unified configuration management
  - Optuna-based auto-tuning for:
    - Optimal `gamma` per model pair
    - `max_num_batched_tokens` tuning
    - KV cache size optimization
  - Multi-objective optimization (throughput + latency)

- [ ] **Draft Model as Standalone Service**:
  - Deploy draft model as independent µ-service
  - Support multiple target models sharing one draft service
  - Horizontal scaling for draft generation
  - Load balancing across draft instances

#### 🔧 Infrastructure & Hardware
- [ ] **TurboMind CUDA Kernel Integration**:
  - Low-level fused attention kernels
  - Optimized paged attention for speculative decoding
  - Expected 20-30% latency reduction

- [ ] **Triton Kernel Support**:
  - Multi-backend support (NVIDIA, AMD, Intel)
  - Custom fused kernels for PEARL-specific operations
  - Platform-agnostic deployment

- [ ] **Prefill-Decode (PD) Disaggregation**:
  - Separate prefill and decode onto different hardware
  - Prefill: High-memory GPU cluster
  - Decode: High-throughput GPU cluster
  - Better resource utilization for mixed workloads

#### 🧪 Advanced Features
- [ ] **Draft Model Temperature Control**: 
  - Support non-zero temperature for draft model
  - Diverse sampling strategies

- [ ] **Multi-modal Speculative Decoding**:
  - Image-based speculative decoding
  - Vision-Language Model (VLM) support
  - Extend PEARL to multi-modal scenarios

- [ ] **PEARL-2 (Self-Distillation)**:
  - Fine-tuning/distilling PEARL-specific draft model
  - Target-aware draft model training
  - Expected 30-40% acceptance rate improvement

### 💡 Research Directions
- [ ] **Speculative Decoding for Structured Output**:
  - JSON schema-guided generation
  - Grammar-constrained speculative decoding
  
- [ ] **Cross-Model Collaboration**:
  - Multiple draft models voting
  - Ensemble-based token selection

- [ ] **Hardware-specific Optimizations**:
  - Apple Silicon (Metal) support
  - AWS Inferentia/Trainium optimization
  - Google TPU integration

### 📊 Benchmarking & Evaluation
- [ ] **Comprehensive Benchmark Suite**:
  - Multi-domain evaluation (code, math, reasoning)
  - Latency percentile analysis (P50, P95, P99)
  - Memory profiling under high concurrency


## 🙏 Acknowledgements


This project has been influenced by many execellent projects in the LLM community, such as [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [PEARL](https://github.com/smart-lty/ParallelSpeculativeDecoding). The nano-PEARL logo is designed by Veo 3. 

## 📚 Citation
```bibtex
@inproceedings{
liu2025pearl,
title={{PEARL}: Parallel Speculative Decoding with Adaptive Draft Length},
author={Tianyu Liu and Yun Li and Qitan Lv and Kai Liu and Jianchen Zhu and Winston Hu and Xiao Sun},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=QOXrVMiHGK}
}

@misc{liu2025pearlparallelspeculativedecoding,
      title={PEARL: Parallel Speculative Decoding with Adaptive Draft Length}, 
      author={Tianyu Liu and Yun Li and Qitan Lv and Kai Liu and Jianchen Zhu and Winston Hu and Xiao Sun},
      year={2025},
      eprint={2408.11850},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.11850}, 
}
```
