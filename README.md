# KV Cache Compression Research

Research implementations of KV cache compression techniques for large language models.

## Overview

This repository explores data-oblivious and data-dependent compression methods to reduce the memory footprint of KV caches in transformer inference.

## Implemented Methods

### ✅ TurboQuant (Verified)
Data-oblivious compression using random rotation + Lloyd-Max quantization.

- **Paper**: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (Zandieh et al., arXiv:2504.19874, 2025)
- **Compression**: 8-10.67x vs FP32
- **Quality**: >98% cosine similarity with 3-4 bits per dimension
- **Status**: Fully implemented and verified against paper targets

### KVTC next


## Project Structure

```
kv_research/
├── model.py         # MiniTransformer with RoPE and GQA
├── turboquant.py    # TurboQuant implementation
├── verify.py        # Verification script for TurboQuant
└── RESULTS.txt      # Experimental results
```


## Results

**TurboQuant Performance** (MiniTransformer, random weights):
- TQ3 (3 bits): MSE=0.0034, Cosine Sim=0.983, Attention Output Sim=0.991
- TQ4 (4 bits): MSE=0.0009, Cosine Sim=0.996, Attention Output Sim=0.997

See [RESULTS.txt](kv_research/RESULTS.txt) for detailed metrics.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- MPS/CUDA support (optional, CPU fallback available)

## References

- Zandieh, A., Daliri, M., Hadian, M., Mirrokni, V. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate". arXiv:2504.19874, 2025. https://arxiv.org/abs/2504.19874

