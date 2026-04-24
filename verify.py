"""verify.py — Verify TurboQuant KV cache compression against paper targets.

Run:
    python kv_research/verify.py

What this does:
    1. Build a MiniTransformer and run a forward pass on random token ids
    2. Extract the KV cache from every layer and head
    3. Compress with TurboQuant at TQ3 (3-bit) and TQ4 (4-bit)
    4. Measure:
        - Per-layer MSE between original and reconstructed K and V tensors
        - Attention output difference (cosine similarity and L2 distance)
          by re-running attention with compressed KV
    5. Print a clean results table

Paper targets (on real LLM activations):
    TQ3 MSE ≈ 0.034
    TQ4 MSE ≈ 0.009

Our model uses random weights so activations are not identical to a trained
LLM, but the codebook was built for the correct Beta marginal distribution so
the MSE should be in the same ballpark.
"""

import sys
import os

# Allow running from repo root: python kv_research/verify.py
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from model import (
    MiniTransformer,
    HIDDEN_DIM, N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS, VOCAB_SIZE,
)
from turboquant import TurboQuant

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between two flat tensors."""
    a_flat = a.reshape(-1, a.shape[-1]).float()    # [N, dim]
    b_flat = b.reshape(-1, b.shape[-1]).float()    # [N, dim]
    return F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item()


def l2_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-vector L2 distance."""
    a_flat = a.reshape(-1, a.shape[-1]).float()    # [N, dim]
    b_flat = b.reshape(-1, b.shape[-1]).float()    # [N, dim]
    return (a_flat - b_flat).norm(dim=-1).mean().item()


def attention_output(
    q: torch.Tensor,   # [batch, n_heads, seq, head_dim]
    k: torch.Tensor,   # [batch, n_kv_heads, seq, head_dim]
    v: torch.Tensor,   # [batch, n_kv_heads, seq, head_dim]
    kv_groups: int,
) -> torch.Tensor:
    """Compute scaled dot-product attention output given Q, K, V.

    Returns:
        out : [batch, n_heads, seq, head_dim]
    """
    scale = HEAD_DIM ** -0.5

    if kv_groups > 1:
        k = k.repeat_interleave(kv_groups, dim=1)   # [batch, n_heads, seq, head_dim]
        v = v.repeat_interleave(kv_groups, dim=1)   # [batch, n_heads, seq, head_dim]

    attn_w = torch.matmul(q, k.transpose(-2, -1)) * scale   # [batch, n_heads, seq, seq]
    seq    = q.shape[2]
    mask   = torch.triu(
        torch.full((seq, seq), float("-inf"), device=q.device), diagonal=1
    )                                                         # [seq, seq]
    attn_w = attn_w + mask.unsqueeze(0).unsqueeze(0)
    attn_p = F.softmax(attn_w, dim=-1)                       # [batch, n_heads, seq, seq]
    out    = torch.matmul(attn_p, v)                         # [batch, n_heads, seq, head_dim]
    return out


# ── Main verification ─────────────────────────────────────────────────────────

def run_verification():
    print(f"Device : {device}")
    print(f"Model  : MiniTransformer | layers={N_LAYERS} | heads={N_HEADS} | head_dim={HEAD_DIM}")
    print()

    # ── 1. Build model and run forward pass ───────────────────────────────────
    torch.manual_seed(7)
    model = MiniTransformer().to(device)
    model.eval()

    BATCH, SEQ = 4, 64
    ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ), device=device)  # [batch, seq]

    with torch.no_grad():
        logits, kv_cache = model(ids)
    # kv_cache: {layer_idx: (k, v)}
    # k, v: [batch, n_kv_heads, seq, head_dim]

    print(f"Forward pass complete. KV cache shape: {tuple(kv_cache[0][0].shape)}")
    print()

    # ── 2. Build TurboQuant quantizers ────────────────────────────────────────
    print("Building TurboQuant codebooks (may take a few seconds)...")
    tq3 = TurboQuant(bits=3, dim=HEAD_DIM).to(device)
    tq4 = TurboQuant(bits=4, dim=HEAD_DIM).to(device)
    print()

    # ── 3. Compress each layer's KV cache and measure quality ─────────────────
    # Table header
    col_w = 10
    header = (
        f"{'Layer':<6} {'Tensor':<6} "
        f"{'TQ3 MSE':>{col_w}} {'TQ3 CosSim':>{col_w}} {'TQ3 L2':>{col_w}}   "
        f"{'TQ4 MSE':>{col_w}} {'TQ4 CosSim':>{col_w}} {'TQ4 L2':>{col_w}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    attn_results = {}   # {layer_idx: {bits: cosine_sim of attn output}}

    for layer_idx in range(N_LAYERS):
        k_orig, v_orig = kv_cache[layer_idx]
        # k_orig, v_orig: [batch, n_kv_heads, seq, head_dim]

        for tq, bits in [(tq3, 3), (tq4, 4)]:
            pass   # we'll fill columns together below

        # --- Compress K and V with both bit-widths ---
        results = {}
        for name, tensor in [("K", k_orig), ("V", v_orig)]:
            row = {}
            for tq, bits in [(tq3, 3), (tq4, 4)]:
                with torch.no_grad():
                    indices, scale = tq.compress(tensor)
                    tensor_hat     = tq.decompress(indices, scale)

                mse  = (tensor.float() - tensor_hat.float()).pow(2).mean().item()
                cos  = cosine_sim(tensor, tensor_hat)
                l2   = l2_dist(tensor, tensor_hat)
                row[bits] = (mse, cos, l2, tensor_hat)

            # Print row
            r3 = row[3]
            r4 = row[4]
            print(
                f"{layer_idx:<6} {name:<6} "
                f"{r3[0]:>{col_w}.6f} {r3[1]:>{col_w}.6f} {r3[2]:>{col_w}.4f}   "
                f"{r4[0]:>{col_w}.6f} {r4[1]:>{col_w}.6f} {r4[2]:>{col_w}.4f}"
            )
            results[name] = row

        # -- Attention output comparison (needs Q, which isn't in kv_cache)
        # Re-run just the attention module on this layer to get Q
        with torch.no_grad():
            x    = model.embed(ids)                             # [batch, seq, hidden]
            for i, layer in enumerate(model.layers):
                normed = layer.attn_norm(x)
                q = layer.attn.q_proj(normed)                  # [batch, seq, n_heads*head_dim]
                q = q.view(BATCH, SEQ, N_HEADS, HEAD_DIM).transpose(1, 2)
                # q: [batch, n_heads, seq, head_dim]
                q = layer.attn.rope(q)

                if i == layer_idx:
                    break
                attn_out, _ = layer.attn(normed)
                x = x + attn_out
                x = x + layer.ffn(layer.ffn_norm(x))

        kv_groups = N_HEADS // N_KV_HEADS

        # Original attention output for this layer
        attn_orig = attention_output(q, k_orig, v_orig, kv_groups)
        # [batch, n_heads, seq, head_dim]

        attn_results[layer_idx] = {}
        for bits, tq in [(3, tq3), (4, tq4)]:
            k_hat = results["K"][bits][3]   # compressed K
            v_hat = results["V"][bits][3]   # compressed V
            attn_comp = attention_output(q, k_hat, v_hat, kv_groups)

            cos = cosine_sim(attn_orig, attn_comp)
            l2  = l2_dist(attn_orig, attn_comp)
            attn_results[layer_idx][bits] = (cos, l2)

    print(sep)
    print()

    # ── 4. Attention output quality table ─────────────────────────────────────
    print("Attention output quality (compressed KV vs original KV):")
    print()
    attn_header = (
        f"{'Layer':<6} "
        f"{'TQ3 CosSim':>{col_w}} {'TQ3 L2':>{col_w}}   "
        f"{'TQ4 CosSim':>{col_w}} {'TQ4 L2':>{col_w}}"
    )
    attn_sep = "-" * len(attn_header)
    print(attn_sep)
    print(attn_header)
    print(attn_sep)
    for layer_idx in range(N_LAYERS):
        r3 = attn_results[layer_idx][3]
        r4 = attn_results[layer_idx][4]
        print(
            f"{layer_idx:<6} "
            f"{r3[0]:>{col_w}.6f} {r3[1]:>{col_w}.4f}   "
            f"{r4[0]:>{col_w}.6f} {r4[1]:>{col_w}.4f}"
        )
    print(attn_sep)
    print()

    # ── 5. Summary vs paper targets ───────────────────────────────────────────
    print("Paper targets: TQ3 MSE ≈ 0.034 | TQ4 MSE ≈ 0.009")
    print("(Targets are for trained LLM activations; random-weight model will be close but not exact)")


if __name__ == "__main__":
    run_verification()
