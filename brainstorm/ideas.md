# Hardware Optimization Brainstorm

**Project:** llama.cpp fork (skynet)  
**Date:** 2026-10-26  
**Author:** Analysis for dual-hardware setup

---

## Executive Summary

Two distinct hardware targets requiring different optimization strategies:

| Target | Hardware | Best Model | Strategy |
|--------|----------|------------|----------|
| **Local** | Ryzen 7 2600 (6C/12T), 40 GB RAM, GTX 1050 Ti 4 GB | Qwen2.5-Coder-3B-Q4_K_M | Full GPU offload, quantized KV cache |
| **VPS** | AMD EPYC Zen 1 (6 vCPU), 11 GB RAM, **no GPU** | Qwen2.5-Coder-7B/14B-Q4_K_M | CPU-only, OpenMP, max context |

---

## Local Machine: Ryzen 2600 + GTX 1050 Ti (4 GB VRAM)

### Constraints
- **VRAM: 4 GB** — hard limit for model + KV cache
- **RAM: 40 GB** — abundant, use for mmap/mlock
- **GPU: Pascal (SM 6.1)** — no Tensor cores, no FP8, FlashAttn slower than default
- **CPU: Zen+ (AVX2, FMA)** — 6 cores / 12 threads

### Recommended Model: Qwen2.5-Coder-3B-Instruct-Q4_K_M
- **VRAM:** ~2.2 GB (model) + ~0.5 GB (KV @ 8K quantized) = **~2.7 GB** ✅
- **Quality:** Best-in-class for <4B coding models
- **Speed:** 25-40 tok/s decode

### Build Configuration
```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DCMAKE_C_FLAGS="-mavx2 -mfma -O3 -pipe" \
  -DCMAKE_CXX_FLAGS="-mavx2 -mfma -O3 -pipe" \
  -DCMAKE_BUILD_TYPE=Release
```

### Launch Script (`~/code.sh`)
```bash
#!/bin/bash
MODEL="$HOME/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf"

./llama-cli -m "$MODEL" \
  -ngl 99 \
  -fa \
  -t 10 \
  -c 8192 \
  --cache-type-k q4_k \
  --cache-type-v q4_k \
  --temp 0.2 \
  --min-p 0.05 \
  --repeat-penalty 1.1 \
  -cnv \
  -p "You are an expert programmer. Write clean, correct, well-documented code." \
  "$@"
```

### Server Mode (for IDE integration)
```bash
./llama-server -m "$MODEL" \
  -ngl 99 -fa -t 10 -c 8192 \
  --cache-type-k q4_k --cache-type-v q4_k \
  --port 8080 --parallel 2
```

### Model Download
```bash
wget -c https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  -O ~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
```

### Alternative: 7B Hybrid Mode (if quality > speed)
```bash
./llama-cli -m qwen2.5-coder-7b-q4_k_m.gguf \
  -ngl 15 --split-mode layer \
  -t 10 -c 4096 \
  --cache-type-k q4_k --cache-type-v q4_k
```
- ~15 layers on GPU, rest on CPU
- ~8-15 tok/s decode
- Better reasoning, slower

---

## VPS: AMD EPYC Zen 1 (6 vCPU), 11 GB RAM, No GPU

### Constraints
- **CPU-only** — 6 cores, no SMT (1 thread/core)
- **RAM: 11 GB** — fits 7B comfortably, 14B tight
- **Architecture:** Zen 1 (znver1) — AVX2, FMA, no AVX-512
- **Virtualized:** KVM, no GPU passthrough

### Recommended Models

| Model | Quant | RAM | Speed | Quality | Verdict |
|-------|-------|-----|-------|---------|---------|
| **Qwen2.5-Coder-7B** | Q4_K_M | ~5.5 GB | **15-25 tok/s** | ⭐⭐⭐⭐⭐ | **Primary choice** |
| Qwen2.5-Coder-14B | Q4_K_M | ~9.5 GB | 6-10 tok/s | ⭐⭐⭐⭐⭐ | Max quality, fits |
| DeepSeek-Coder-V2-Lite | Q4_K_M | ~2 GB | 30-50 tok/s | ⭐⭐⭐⭐ | Speed demon |
| CodeLlama-13B | Q4_K_M | ~8 GB | 8-12 tok/s | ⭐⭐⭐⭐ | Solid alternative |

### Build Configuration (Zen 1 Optimized)
```bash
cmake -B build \
  -DGGML_CUDA=OFF \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DCMAKE_C_FLAGS="-march=znver1 -mtune=znver1 -O3 -pipe" \
  -DCMAKE_CXX_FLAGS="-march=znver1 -mtune=znver1 -O3 -pipe" \
  -DCMAKE_BUILD_TYPE=Release
```

### Launch Configs

**Interactive CLI (7B):**
```bash
./llama-cli -m qwen2.5-coder-7b-q4_k_m.gguf \
  -t 6 \
  -ngl 0 \
  -c 8192 \
  --cache-type-k q4_k \
  --cache-type-v q4_k \
  --temp 0.2 \
  --min-p 0.05 \
  -cnv
```

**Server Mode:**
```bash
./llama-server -m qwen2.5-coder-7b-q4_k_m.gguf \
  -t 6 -ngl 0 -c 8192 \
  --cache-type-k q4_k --cache-type-v q4_k \
  --port 8080 --host 0.0.0.0 \
  --parallel 2 \
  --cont-batching
```

### System Tuning (Run Once on VPS)
```bash
# Reduce swap aggressiveness
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-llama.conf
echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.d/99-llama.conf
sudo sysctl --system

# Optional: Huge pages (~5% speedup)
sudo mkdir -p /hugepages
sudo mount -t hugetlbfs hugetlbfs /hugepages
echo 3000 | sudo tee /proc/sys/vm/nr_hugepages
# Add --huge-pages to llama-cli
```

### Model Download on VPS
```bash
apt-get install -y aria2
aria2c -x 16 -s 16 -d /models \
  https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf

# 14B option (9.5 GB RAM)
aria2c -x 16 -s 16 -d /models \
  https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/qwen2.5-coder-14b-instruct-q4_k_m.gguf
```

### Docker Deployment (Optional)
```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y build-essential cmake git aria2
WORKDIR /llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp . && \
    cmake -B build -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DCMAKE_C_FLAGS="-march=znver1 -O3" && \
    cmake --build build -j6
COPY models/ /models/
EXPOSE 8080
CMD ["./build/bin/llama-server", "-m", "/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf", "-t", "6", "-c", "8192", "--host", "0.0.0.0", "--port", "8080", "--parallel", "2"]
```

```bash
docker build -t llama-coder .
docker run -d -p 8080:8080 -v $(pwd)/models:/models --name coder llama-coder
```

---

## Comparison: Local vs VPS

| Aspect | Local (1050 Ti) | VPS (EPYC 6C) |
|--------|-----------------|---------------|
| **Best Model** | 3B (full GPU) | **7B-14B (CPU)** |
| **Decode Speed** | 25-40 tok/s | 15-25 tok/s |
| **Max Context** | 8K (VRAM limited) | **8K-32K (RAM rich)** |
| **Concurrency** | 1 | **2-3** |
| **Cost** | Free (owned) | $/month |
| **Privacy** | Full | Provider-dependent |
| **Latency** | ~0 ms (local) | Network RTT |

**VPS wins on:** Model intelligence, context length, concurrent requests  
**Local wins on:** Latency, privacy, zero marginal cost

---

## IDE Integration (Both Targets)

### VS Code / Continue.dev / Cline / Aider
```json
// Continue.dev config (~/.continue/config.json)
{
  "models": [
    {
      "title": "Local Qwen 3B",
      "provider": "openai",
      "model": "qwen2.5-coder-3b",
      "apiBase": "http://localhost:8080/v1",
      "apiKey": "local"
    },
    {
      "title": "VPS Qwen 7B",
      "provider": "openai",
      "model": "qwen2.5-coder-7b",
      "apiBase": "https://your-vps-ip:8080/v1",
      "apiKey": "local"
    }
  ]
}
```

### Aider
```bash
# Local
aider --model openai/qwen2.5-coder-3b --api-base http://localhost:8080/v1

# VPS
aider --model openai/qwen2.5-coder-7b --api-base https://your-vps-ip:8080/v1
```

---

## Future Optimization Ideas

### For Local (GTX 1050 Ti)
- [ ] Test **Q2_K quantization** on 7B for full GPU fit (~3.5 GB)
- [ ] Try **llama.cpp's new MMQ kernels** for Pascal (if backported)
- [ ] Experiment with **speculative decoding** using 1B draft model
- [ ] Profile **MoE cache** if using Mixtral/DeepSeek (unlikely on 4 GB)

### For VPS (CPU-only)
- [ ] Benchmark **7B vs 14B** on actual workload
- [ ] Test **Q3_K_M / Q5_K_M** quantization sweet spots
- [ ] Enable **continuous batching** for higher throughput
- [ ] Add **nginx + TLS** for secure remote access
- [ ] Set up **systemd service** for auto-restart
- [ ] Consider **GGML_SYCL** if VPS gets Intel GPU later

### Cross-cutting
- [ ] **Quantized KV cache** (already in config) — biggest memory win
- [ ] **Mmap + mlock** — prevents swap thrashing
- [ ] **Context shifting** — for long conversations
- [ ] **Grammar-guided generation** — for structured code output

---

## Quick Reference Cards

### Local Quick Start
```bash
cd ~/skynet
./build.sh              # Build with CUDA + AVX2
./download-model.sh     # Get Qwen2.5-Coder-3B
./code.sh               # Start coding
```

### VPS Quick Start
```bash
cd ~/skynet
./build-vps.sh          # Build with OpenMP + znver1
./download-model-vps.sh # Get Qwen2.5-Coder-7B
./serve-vps.sh          # Start server on :8080
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `build.sh` | Local build script |
| `build-vps.sh` | VPS build script |
| `code.sh` | Local interactive launcher |
| `serve-vps.sh` | VPS server launcher |
| `download-model.sh` | Local model fetcher |
| `download-model-vps.sh` | VPS model fetcher |
| `systemd/llama-coder.service` | VPS auto-start service |
| `nginx/llama-coder.conf` | Reverse proxy config |

---

*Generated from hardware analysis session. Update as benchmarks arrive.*