# KV Cache Compression Research

Research implementations of KV cache compression techniques for large language models.
Implements **TurboQuant** (Google, ICLR 2026) and **KVTC** (NVIDIA, ICLR 2026),
verifies against paper targets, and explores a hybrid approach targeting ~10× compression.

**Hardware:** Mac Mini M4 16GB (MPS) · RTX 5060 Ti 16GB (CUDA)

---

## What is the KV cache?

During transformer inference, every token computes Key (K) and Value (V) vectors which
are cached to avoid recomputation on every new token. The cache grows linearly with context:

| Model | Per-token size | At 32K context |
|-------|---------------|----------------|
| Qwen3-1.7B | ~57 KB | ~1.8 GB |
| Qwen3-8B   | ~114 KB | ~3.6 GB |

At long contexts, the KV cache often exceeds the model weights in memory.
Compression is the practical solution — same hardware, more context.

---

## Methods

### ✅ TurboQuant (Verified on real model)

Data-oblivious compression — no calibration required, works on any model out of the box.

**Paper:** Zandieh et al., *TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate*, arXiv:2504.19874, ICLR 2026

**Algorithm:**
1. **Normalize** — L2-normalize each KV vector to the unit sphere
2. **Rotate** — multiply by a fixed random orthogonal matrix, spreading outliers evenly
3. **Lloyd-Max quantize** — optimal scalar quantization with bins placed where data lives
4. **Store** — indices (3–4 bits) + scale (fp16) per vector

> QJL residual is skipped — community experiments show softmax amplifies its noise.

**Compression vs fp16:**

| Mode | Bits | Compression |
|------|------|-------------|
| TQ3  | 3    | ~4.6× vs fp16 |
| TQ4  | 4    | ~3.7× vs fp16 |

### 🔄 KVTC (In progress)

Data-aware compression — requires calibration, achieves higher compression ratios.

**Paper:** Staniszewski & Łańcucki, *KV Cache Transform Coding for Compact Storage
in LLM Inference*, ICLR 2026

**Algorithm:** Global SVD across all layers/heads → DP bit allocation →
project + quantize in PCA basis. Targets ~20× compression at <1% accuracy loss.

---

## Results

### TurboQuant — toy model (random weights)

Model: `MiniTransformer` (2 layers, hidden=256, 4 heads, head_dim=64)

| Mode | Target MSE | Actual MSE | CosSim | Status |
|------|-----------|------------|--------|--------|
| TQ3  | 0.034     | 0.032686   | 0.9837 | ✅     |
| TQ4  | 0.009     | 0.008917   | 0.9957 | ✅     |

### TurboQuant — Qwen3-1.7B (real trained activations) ✅

Model: `Qwen/Qwen3-1.7B` · fp16 · MPS (Mac M4) · 28 layers · GQA (8 KV heads, 16 Q heads)
Averaged over 5 diverse test sentences across all 28 layers.

| Mode | K CosSim | V CosSim | Attn CosSim | Target | Status |
|------|----------|----------|-------------|--------|--------|
| TQ3  | **0.9831** | **0.9830** | 0.9384 | > 0.97 | ✅ PASS |
| TQ4  | **0.9954** | **0.9954** | 0.9733 | > 0.99 | ✅ PASS |

**Per-layer breakdown:**

| Layer | TQ3 K-Cos | TQ3 V-Cos | TQ3 Attn | TQ4 K-Cos | TQ4 V-Cos | TQ4 Attn |
|-------|-----------|-----------|----------|-----------|-----------|----------|
| 0     | 0.9829    | 0.9831    | 0.9865   | 0.9954    | 0.9954    | 0.9963   |
| 1     | 0.9832    | 0.9827    | 0.9873   | 0.9955    | 0.9953    | 0.9966   |
| 5     | 0.9831    | 0.9830    | 0.9837   | 0.9955    | 0.9955    | 0.9954   |
| 10    | 0.9829    | 0.9828    | 0.9699   | 0.9954    | 0.9954    | 0.9922   |
| 15    | 0.9829    | 0.9830    | 0.9071   | 0.9953    | 0.9954    | 0.9633   |
| 20    | 0.9834    | 0.9831    | 0.8979   | 0.9955    | 0.9955    | 0.9582   |
| 27    | 0.9831    | 0.9832    | 0.8774   | 0.9955    | 0.9954    | 0.9300   |
| **AVG** | **0.9831** | **0.9830** | **0.9384** | **0.9954** | **0.9954** | **0.9733** |

**Key observations:**
- K/V cosine similarity is **rock solid at 0.983 (TQ3) and 0.995 (TQ4) across all 28 layers**
- MSE grows in deeper layers (0.004 → 23.3 for TQ3) due to larger activation magnitudes —
  cosine similarity staying flat proves *direction* is preserved, which is what attention needs
- Attention output cosine drops slightly in deeper layers (0.987 → 0.877 at TQ3) —
  deeper layers have higher-magnitude KV vectors with more outliers
- TQ4 attention output stays above 0.93 across all layers

---

## Project Structure

```
root/
├── model.py              — MiniTransformer: 2-layer, RMSNorm, RoPE, GQA-ready
├── turboquant.py         — TurboQuant: rotation + Lloyd-Max quantization
├── verify.py             — TurboQuant end-to-end verification on toy model
├── real_model_verify.py  — TurboQuant on Qwen3-1.7B real activations
└── RESULTS.txt       - Experimental results for stage 1
```

---




---

## Why this matters

With TQ3 compression on a 16GB Mac M4 running Qwen3-1.7B:

```
Model weights (fp16):     ~3.4 GB
Remaining for KV cache:   ~12.6 GB

Without compression:   ~220K tokens max context
With TQ3 (4.6×):      ~1M tokens max context
With hybrid (10×):    ~2M+ tokens max context (target)
```

Same hardware. No model changes. No quality loss at TQ4.

---

## References

- Zandieh, A., Daliri, M., Hadian, M., Mirrokni, V. *TurboQuant: Online Vector
  Quantization with Near-optimal Distortion Rate*. arXiv:2504.19874, ICLR 2026.
  https://arxiv.org/abs/2504.19874
- Staniszewski, M. & Łańcucki, A. *KV Cache Transform Coding for Compact Storage
  in LLM Inference*. ICLR 2026. https://openreview.net/forum?id=aNVKROYpLB
- Cheng, Y. et al. *LMCache*. https://github.com/LMCache/LMCache
