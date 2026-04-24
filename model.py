"""model.py — Minimal 2-layer transformer with explicit, inspectable KV cache.

Architecture mirrors Qwen/Llama:
    - RMSNorm (no bias, no mean-centering)
    - Rotary Position Embedding (RoPE)
    - GQA-ready multi-head attention (set N_KV_HEADS < N_HEADS for GQA)
    - SwiGLU feed-forward network
    - Pre-norm residual connections, weight-tied embeddings

Forward pass returns (logits, kv_cache_dict) so every KV tensor is
directly accessible for compression experiments.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model hyper-parameters ────────────────────────────────────────────────────
HIDDEN_DIM  = 256    # d_model — total model width
N_HEADS     = 4      # number of query heads
HEAD_DIM    = 64     # HIDDEN_DIM // N_HEADS
N_KV_HEADS  = 4      # key/value heads; set < N_HEADS to activate GQA
N_LAYERS    = 2      # number of transformer blocks
FFN_DIM     = 512    # SwiGLU inner dimension (≈ 2× hidden, halved by gating)
VOCAB_SIZE  = 4096   # small vocab, sufficient for research forward passes
MAX_SEQ_LEN = 512    # maximum sequence length for RoPE pre-computation

assert HIDDEN_DIM == N_HEADS * HEAD_DIM, "HIDDEN_DIM must equal N_HEADS * HEAD_DIM"
assert N_HEADS % N_KV_HEADS == 0,        "N_HEADS must be divisible by N_KV_HEADS"


# ── RMSNorm ───────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation — no bias, no mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))  # [dim] — learned scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:   [batch, seq, dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()  # [batch, seq, 1]
        return (x / rms) * self.weight                                   # [batch, seq, dim]


# ── Rotary Position Embedding (RoPE) ─────────────────────────────────────────
class RoPE(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2021).

    Pre-computes cos/sin tables up to MAX_SEQ_LEN at init.
    Applied to Q and K tensors inside Attention.forward().
    """

    def __init__(
        self,
        head_dim:    int,
        max_seq_len: int   = MAX_SEQ_LEN,
        base:        float = 10_000.0,
    ):
        super().__init__()
        # Inverse frequencies for each pair of dimensions: [head_dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)  # [head_dim/2]

        # Pre-compute full cos/sin embedding table
        positions = torch.arange(max_seq_len).float()       # [max_seq_len]
        freqs     = torch.outer(positions, inv_freq)        # [max_seq_len, head_dim/2]
        emb       = torch.cat([freqs, freqs], dim=-1)       # [max_seq_len, head_dim]
        self.register_buffer("cos_cache", emb.cos())        # [max_seq_len, head_dim]
        self.register_buffer("sin_cache", emb.sin())        # [max_seq_len, head_dim]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, seq, head_dim]
        h = x.shape[-1] // 2
        x1 = x[..., :h]                        # [batch, heads, seq, head_dim/2]
        x2 = x[..., h:]                        # [batch, heads, seq, head_dim/2]
        return torch.cat([-x2, x1], dim=-1)    # [batch, heads, seq, head_dim]

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: [batch, heads, seq, head_dim]
        seq = x.shape[2]
        cos = self.cos_cache[offset : offset + seq]          # [seq, head_dim]
        sin = self.sin_cache[offset : offset + seq]          # [seq, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)                  # [1, 1, seq, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)                  # [1, 1, seq, head_dim]
        return x * cos + self._rotate_half(x) * sin         # [batch, heads, seq, head_dim]


# ── Multi-head Self-Attention (GQA-ready) ────────────────────────────────────
class Attention(nn.Module):
    """
    Grouped-query attention (GQA).

    When N_KV_HEADS == N_HEADS this is standard MHA.
    KV tensors are returned explicitly so they can be inspected and compressed.
    """

    def __init__(
        self,
        hidden_dim:  int,
        n_heads:     int,
        n_kv_heads:  int,
        head_dim:    int,
    ):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.kv_groups  = n_heads // n_kv_heads   # repetition factor for GQA
        self.scale      = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, n_heads    * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads    * head_dim, hidden_dim, bias=False)

        self.rope = RoPE(head_dim)

    def forward(
        self,
        x:       torch.Tensor,                                           # [batch, seq, hidden_dim]
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        batch, seq, _ = x.shape

        # ── Project ──────────────────────────────────────────────────────────
        q = self.q_proj(x)  # [batch, seq, n_heads    * head_dim]
        k = self.k_proj(x)  # [batch, seq, n_kv_heads * head_dim]
        v = self.v_proj(x)  # [batch, seq, n_kv_heads * head_dim]

        # ── Reshape to head-per-row layout ───────────────────────────────────
        q = q.view(batch, seq, self.n_heads,    self.head_dim).transpose(1, 2)
        # q: [batch, n_heads,    seq, head_dim]
        k = k.view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # k: [batch, n_kv_heads, seq, head_dim]
        v = v.view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # v: [batch, n_kv_heads, seq, head_dim]

        # ── RoPE on Q and K ──────────────────────────────────────────────────
        offset = past_kv[0].shape[2] if past_kv is not None else 0
        q = self.rope(q, offset=offset)  # [batch, n_heads,    seq, head_dim]
        k = self.rope(k, offset=offset)  # [batch, n_kv_heads, seq, head_dim]

        # ── Append past context ──────────────────────────────────────────────
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # [batch, n_kv_heads, past+seq, head_dim]
            v = torch.cat([past_v, v], dim=2)  # [batch, n_kv_heads, past+seq, head_dim]

        # ── KV cache (explicit, named — ready for compression) ───────────────
        kv_cache: Tuple[torch.Tensor, torch.Tensor] = (k, v)
        # k: [batch, n_kv_heads, total_seq, head_dim]
        # v: [batch, n_kv_heads, total_seq, head_dim]

        # ── GQA: expand K/V heads to match Q heads ───────────────────────────
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)  # [batch, n_heads, total_seq, head_dim]
            v = v.repeat_interleave(self.kv_groups, dim=1)  # [batch, n_heads, total_seq, head_dim]

        total_seq = k.shape[2]

        # ── Scaled dot-product attention ─────────────────────────────────────
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn_weights: [batch, n_heads, seq, total_seq]

        # Causal mask: future tokens get -inf before softmax
        if seq > 1:
            # Upper-triangular mask: positions that should be masked
            mask = torch.triu(
                torch.full((seq, total_seq), float("-inf"), device=x.device),
                diagonal=total_seq - seq + 1,  # account for past context offset
            )  # [seq, total_seq]
            attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)
            # attn_weights: [batch, n_heads, seq, total_seq]

        attn_probs = F.softmax(attn_weights, dim=-1)  # [batch, n_heads, seq, total_seq]

        # ── Weighted aggregation ─────────────────────────────────────────────
        out = torch.matmul(attn_probs, v)             # [batch, n_heads, seq, head_dim]
        out = out.transpose(1, 2).contiguous()        # [batch, seq, n_heads, head_dim]
        out = out.view(batch, seq, -1)                # [batch, seq, n_heads * head_dim]
        out = self.o_proj(out)                        # [batch, seq, hidden_dim]

        return out, kv_cache


# ── SwiGLU Feed-Forward Network ───────────────────────────────────────────────
class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN (Shazeer, 2020) as used in Llama / Qwen.

    out = down_proj( silu(gate_proj(x)) * up_proj(x) )
    """

    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim,    hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, hidden_dim]
        gate = F.silu(self.gate_proj(x))   # [batch, seq, ffn_dim]
        up   = self.up_proj(x)             # [batch, seq, ffn_dim]
        return self.down_proj(gate * up)   # [batch, seq, hidden_dim]


# ── Transformer Block ─────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attention + FFN, both with residuals."""

    def __init__(self):
        super().__init__()
        self.attn_norm = RMSNorm(HIDDEN_DIM)
        self.attn      = Attention(HIDDEN_DIM, N_HEADS, N_KV_HEADS, HEAD_DIM)
        self.ffn_norm  = RMSNorm(HIDDEN_DIM)
        self.ffn       = SwiGLUFFN(HIDDEN_DIM, FFN_DIM)

    def forward(
        self,
        x:       torch.Tensor,                                           # [batch, seq, hidden_dim]
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # Attention sub-layer (pre-norm + residual)
        normed           = self.attn_norm(x)               # [batch, seq, hidden_dim]
        attn_out, kv     = self.attn(normed, past_kv)      # [batch, seq, hidden_dim]
        x                = x + attn_out                    # [batch, seq, hidden_dim]

        # FFN sub-layer (pre-norm + residual)
        x = x + self.ffn(self.ffn_norm(x))                 # [batch, seq, hidden_dim]

        return x, kv


# ── MiniTransformer ───────────────────────────────────────────────────────────
class MiniTransformer(nn.Module):
    """
    2-layer transformer for KV cache compression research.

    Design choices mirroring Qwen / Llama:
        - RMSNorm instead of LayerNorm
        - RoPE positional encoding
        - SwiGLU FFN
        - GQA-ready attention
        - Weight-tied input embeddings and LM head
        - Pre-norm residual connections throughout

    Returns
    -------
    logits        : [batch, seq, vocab_size]
    kv_cache_dict : {layer_idx: (k, v)}
                    k / v shapes: [batch, n_kv_heads, total_seq, head_dim]
    """

    def __init__(self):
        super().__init__()
        self.embed   = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.layers  = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.norm    = RMSNorm(HIDDEN_DIM)
        self.lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

        # Weight tying: embedding matrix shared with output projection
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Gaussian init with std=0.02 (GPT-2 convention)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids:     torch.Tensor,                                     # [batch, seq]
        past_kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Parameters
        ----------
        input_ids     : [batch, seq]  — integer token ids in [0, VOCAB_SIZE)
        past_kv_cache : optional dict {layer_idx: (k, v)} from a prior call
                        k / v: [batch, n_kv_heads, past_seq, head_dim]

        Returns
        -------
        logits         : [batch, seq, vocab_size]
        kv_cache_dict  : {layer_idx: (k, v)} for all N_LAYERS
                         k / v: [batch, n_kv_heads, total_seq, head_dim]
        """
        # Token lookup
        x = self.embed(input_ids)  # [batch, seq, hidden_dim]

        kv_cache_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        for idx, layer in enumerate(self.layers):
            past_kv      = past_kv_cache[idx] if (past_kv_cache is not None and idx in past_kv_cache) else None
            x, kv        = layer(x, past_kv)    # x: [batch, seq, hidden_dim]
            kv_cache_dict[idx] = kv             # k/v: [batch, n_kv_heads, total_seq, head_dim]

        x      = self.norm(x)       # [batch, seq, hidden_dim]
        logits = self.lm_head(x)    # [batch, seq, vocab_size]

        return logits, kv_cache_dict


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Running smoke test on device: {device}")

    model = MiniTransformer().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    batch, seq = 2, 16
    ids = torch.randint(0, VOCAB_SIZE, (batch, seq), device=device)

    logits, kv = model(ids)

    # Verify output shapes
    assert logits.shape == (batch, seq, VOCAB_SIZE), f"Unexpected logits shape: {logits.shape}"
    for layer_idx in range(N_LAYERS):
        k, v = kv[layer_idx]
        assert k.shape == (batch, N_KV_HEADS, seq, HEAD_DIM), f"Layer {layer_idx} K shape: {k.shape}"
        assert v.shape == (batch, N_KV_HEADS, seq, HEAD_DIM), f"Layer {layer_idx} V shape: {v.shape}"

    print(f"logits  shape : {tuple(logits.shape)}")
    print(f"KV layer 0  K : {tuple(kv[0][0].shape)}")
    print(f"KV layer 0  V : {tuple(kv[0][1].shape)}")
    print(f"KV layer 1  K : {tuple(kv[1][0].shape)}")
    print(f"KV layer 1  V : {tuple(kv[1][1].shape)}")
    print("Smoke test PASSED.")
