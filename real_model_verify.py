"""real_model_verify.py — TurboQuant verification on Qwen/Qwen3-1.7B real activations.

Model : Qwen/Qwen3-1.7B  (downloads ~3.4 GB on first run via HuggingFace)
Device: MPS (Mac M4)  →  falls back to CPU if MPS is unavailable or OOMs

Architecture details for Qwen3-1.7B
    num_hidden_layers   : 28
    num_attention_heads : 16   (Q heads)
    num_key_value_heads :  8   (GQA — K/V heads != Q heads)
    head_dim            : 128

Hook strategy
    - K: forward hook on  attn.k_norm  output (post-norm, pre-RoPE)
         falls back to    attn.k_proj  + reshape if k_norm absent
    - V: forward hook on  attn.v_proj  + reshape  (no v_norm in Qwen3)
    - Q: forward hook on  attn.q_norm  output (post-norm, pre-RoPE)
         falls back to    attn.q_proj  + reshape if q_norm absent

    K shape captured : [batch=1, num_kv_heads=8, seq_len, head_dim=128]
    V shape captured : [batch=1, num_kv_heads=8, seq_len, head_dim=128]
    Q shape captured : [batch=1, num_q_heads=16, seq_len, head_dim=128]

Run:
    python kv_research/real_model_verify.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from turboquant import TurboQuant

# ── Architecture constants ────────────────────────────────────────────────────

MODEL_ID     = "Qwen/Qwen3-1.7B"
NUM_KV_HEADS = 8
NUM_Q_HEADS  = 16
HEAD_DIM     = 128
NUM_LAYERS   = 28

# ── Test sentences ────────────────────────────────────────────────────────────

SENTENCES = [
    "The transformer architecture revolutionized natural language processing",
    "I walked into a wall and bumped my head hard",
    "KV cache compression reduces memory requirements during inference",
    "The quick brown fox jumps over the lazy dog near the river",
    "Attention mechanisms allow models to focus on relevant context",
]

# ── Quality-gate targets ──────────────────────────────────────────────────────

TQ3_COS_TARGET = 0.97   # TQ3 cosine similarity on real activations
TQ4_COS_TARGET = 0.99   # TQ4 cosine similarity on real activations


# ── Device selection ──────────────────────────────────────────────────────────

def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Device: CPU  (MPS unavailable)")
    return device


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device: torch.device):
    """Load Qwen3-1.7B in fp16.  Downloads ~3.4 GB on first run."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {MODEL_ID} in fp16 ...")
    print("(First run: HuggingFace will download ~3.4 GB — progress shown below)\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    try:
        model = model.to(device)
        print(f"\nModel on {device}")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(
                "\nMPS out of memory — falling back to CPU.\n"
                "Tip: if slow, try: model.to('cpu') explicitly before retrying."
            )
            device = torch.device("cpu")
            model = model.to(device)
        else:
            raise

    model.eval()
    return model, tokenizer, device


# ── Hook registration ─────────────────────────────────────────────────────────

def register_kv_hooks(model, storage: dict) -> list:
    """Attach forward hooks to every attention layer to capture K, V, Q.

    Hooks are registered on:
        k_proj output  [batch, seq_len, num_kv_heads * head_dim] → reshaped
        v_proj output  [batch, seq_len, num_kv_heads * head_dim] → reshaped
        q_proj output  [batch, seq_len, num_q_heads * head_dim]  → reshaped

    Note: k_proj / q_proj are always used (not k_norm / q_norm) because in
    Qwen3 both norms are applied before the heads/seq transpose, so their
    output has seq_len in dim 1 — indistinguishable from num_heads at runtime.
    The proj layers always output a predictable linear shape.

    Args:
        model   : loaded causal LM
        storage : dict[layer_idx -> {"K": Tensor|None, "V": Tensor|None,
                                                        "Q": Tensor|None}]

    Returns:
        hooks : list of torch hook handles — call .remove() to deregister
    """
    hooks = []

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        def make_hooks(idx, attn_mod):

            # -- K hook -------------------------------------------------------
            # Always hook k_proj (not k_norm) — k_norm in Qwen3 is applied
            # before the transpose so its output has seq in dim 1, making it
            # impossible to distinguish seq from num_kv_heads at runtime.
            def k_proj_hook(module, inp, out):
                bsz, seq_len, _ = out.shape            # [batch, seq, kv_h * d]
                k = (
                    out.detach()
                       .float()
                       .view(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM)
                       .transpose(1, 2)
                       .contiguous()
                )                                      # [batch, kv_h, seq, d]
                storage[idx]["K"] = k

            k_handle = attn_mod.k_proj.register_forward_hook(k_proj_hook)

            # -- V hook -------------------------------------------------------
            # No v_norm in Qwen3; hook v_proj and reshape
            def v_hook(module, inp, out):
                bsz, seq_len, _ = out.shape                # [batch, seq, kv_h * d]
                v = (
                    out.detach()
                       .float()
                       .view(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM)
                       .transpose(1, 2)
                       .contiguous()
                )                                          # [batch, kv_h, seq, d]
                storage[idx]["V"] = v

            v_handle = attn_mod.v_proj.register_forward_hook(v_hook)

            # -- Q hook -------------------------------------------------------
            # Always hook q_proj (not q_norm) — q_norm in Qwen3 is applied
            # before the transpose so its output has seq in dim 1, making it
            # impossible to distinguish seq from num_heads at runtime.
            def q_proj_hook(module, inp, out):
                bsz, seq_len, _ = out.shape                # [batch, seq, q_h * d]
                q = (
                    out.detach()
                       .float()
                       .view(bsz, seq_len, NUM_Q_HEADS, HEAD_DIM)
                       .transpose(1, 2)
                       .contiguous()
                )                                          # [batch, q_h, seq, d]
                storage[idx]["Q"] = q

            q_handle = attn_mod.q_proj.register_forward_hook(q_proj_hook)

            return k_handle, v_handle, q_handle

        k_h, v_h, q_h = make_hooks(layer_idx, attn)
        hooks.extend([k_h, v_h, q_h])

    return hooks


# ── Attention output comparison ───────────────────────────────────────────────

def sdpa_cosine_similarity(
    Q:      torch.Tensor,   # [batch, q_heads,  seq, head_dim]
    K_orig: torch.Tensor,   # [batch, kv_heads, seq, head_dim]
    V_orig: torch.Tensor,   # [batch, kv_heads, seq, head_dim]
    K_comp: torch.Tensor,   # same shape as K_orig
    V_comp: torch.Tensor,   # same shape as V_orig
) -> float:
    """Cosine similarity between SDPA outputs with original vs compressed KV.

    GQA expansion: K and V have 8 heads, Q has 16.  Each KV head is broadcast
    to 2 Q heads via repeat_interleave.

    Returns:
        Scalar cosine similarity averaged over all (position × head) pairs.
    """
    # Move everything to CPU for stable SDPA
    K_o = K_orig.cpu()
    V_o = V_orig.cpu()
    K_c = K_comp.cpu()
    V_c = V_comp.cpu()
    Q_c = Q.cpu()

    q_heads  = Q_c.shape[1]   # actual captured Q head count
    kv_heads = K_o.shape[1]   # actual captured KV head count

    # Only expand K/V if Q has more heads than K/V (GQA with q_heads > kv_heads)
    if q_heads != kv_heads and q_heads % kv_heads == 0:
        rep  = q_heads // kv_heads
        K_o  = K_o.repeat_interleave(rep, dim=1)   # [1, q_heads, seq, 128]
        V_o  = V_o.repeat_interleave(rep, dim=1)
        K_c  = K_c.repeat_interleave(rep, dim=1)
        V_c  = V_c.repeat_interleave(rep, dim=1)

    with torch.no_grad():
        out_orig = F.scaled_dot_product_attention(Q_c, K_o, V_o, is_causal=True)
        out_comp = F.scaled_dot_product_attention(Q_c, K_c, V_c, is_causal=True)
        # Both out_*: [1, q_heads, seq, head_dim]

    o = out_orig.reshape(-1, HEAD_DIM)   # [batch * q_heads * seq, head_dim]
    c = out_comp.reshape(-1, HEAD_DIM)

    return F.cosine_similarity(o, c, dim=-1).mean().item()


# ── Outlier diagnosis ─────────────────────────────────────────────────────────

def diagnose_outliers(K: torch.Tensor, V: torch.Tensor) -> None:
    """Print per-dimension stats to understand why quality targets were missed.

    Real LLM activations often have "outlier dimensions" — a small set of
    coordinates with much higher variance than the rest.  These outliers exceed
    the codebook range, inflating quantisation error.  The diagnostics below
    identify how many outlier dimensions exist and which ones.

    Args:
        K : [batch, kv_heads, seq_len, head_dim]  — captured K activations
        V : [batch, kv_heads, seq_len, head_dim]  — captured V activations
    """
    print("\n  Outlier dimension diagnostics:")
    for name, tensor in [("K", K), ("V", V)]:
        flat = tensor.reshape(-1, HEAD_DIM)          # [N, head_dim]  N = b*h*seq
        dim_std = flat.std(dim=0)                    # [head_dim] — per-dim std
        top5_dims = dim_std.topk(5).indices.tolist()
        abs_max = flat.abs().max().item()
        print(
            f"    {name}: max_dim_std={dim_std.max():.4f}  "
            f"mean_dim_std={dim_std.mean():.4f}  "
            f"abs_max={abs_max:.4f}"
        )
        print(f"    {name}: top-5 outlier dimensions → {top5_dims}")

    # Fraction of values outside typical codebook range (±0.5 for dim=128)
    RANGE = 0.5
    for name, tensor in [("K", K), ("V", V)]:
        flat = tensor.reshape(-1, HEAD_DIM)
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = flat / norms                    # unit-norm vectors
        fraction_out = (normalized.abs() > RANGE).float().mean().item()
        print(
            f"    {name}: {fraction_out*100:.2f}% of normalised coordinates "
            f"exceed ±{RANGE} (high = many outliers saturating codebook)"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = select_device()
    model, tokenizer, device = load_model(device)

    # ── Build TurboQuant for head_dim=128 ─────────────────────────────────────
    print("\nBuilding TurboQuant codebooks for dim=128 ...")
    print("(Lloyd-Max requires ~30 s per config on first build)")
    tq3 = TurboQuant.for_dim(3, 128).cpu()   # 8 levels,  3 bits/coord
    tq4 = TurboQuant.for_dim(4, 128).cpu()   # 16 levels, 4 bits/coord
    print()

    # ── Storage for captured KVQ tensors (overwritten each sentence) ──────────
    # Shape per entry:
    #   K, V : [1, num_kv_heads=8, seq_len, head_dim=128]
    #   Q    : [1, num_q_heads=16, seq_len, head_dim=128]
    storage = {i: {"K": None, "V": None, "Q": None} for i in range(NUM_LAYERS)}
    hooks   = register_kv_hooks(model, storage)

    # ── Metric accumulators ────────────────────────────────────────────────────
    # metrics[tag][layer_idx][metric_name] = list of per-sentence values
    METRIC_NAMES = ["k_mse", "k_cos", "v_mse", "v_cos", "attn_cos"]
    metrics = {
        tag: {i: {m: [] for m in METRIC_NAMES} for i in range(NUM_LAYERS)}
        for tag in ["tq3", "tq4"]
    }

    # Retain last captured activations for outlier diagnosis
    last_kv: dict = {}

    # ── Inference loop ────────────────────────────────────────────────────────
    print(f"Running {len(SENTENCES)} sentences through the {NUM_LAYERS}-layer model ...\n")

    with torch.no_grad():
        for s_idx, sentence in enumerate(SENTENCES):
            print(f"  [{s_idx + 1}/{len(SENTENCES)}] {sentence[:72]}")

            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            model(**inputs)

            for layer_idx in range(NUM_LAYERS):
                K = storage[layer_idx]["K"]   # [1, 8, seq_len, 128]
                V = storage[layer_idx]["V"]   # [1, 8, seq_len, 128]
                Q = storage[layer_idx]["Q"]   # [1, 16, seq_len, 128]

                if K is None or V is None or Q is None:
                    print(f"    WARNING: layer {layer_idx} missing tensors — skipping")
                    continue

                # Move to CPU; TurboQuant codebook lives on CPU
                K_cpu = K.cpu()
                V_cpu = V.cpu()
                Q_cpu = Q.cpu()
                last_kv["K"] = K_cpu
                last_kv["V"] = V_cpu

                for tag, tq in [("tq3", tq3), ("tq4", tq4)]:
                    K_rec = tq(K_cpu)   # round-trip: compress → decompress
                    V_rec = tq(V_cpu)   # [1, 8, seq_len, 128]

                    K_flat = K_cpu.reshape(-1, HEAD_DIM)
                    V_flat = V_cpu.reshape(-1, HEAD_DIM)
                    Kr_flat = K_rec.reshape(-1, HEAD_DIM)
                    Vr_flat = V_rec.reshape(-1, HEAD_DIM)

                    k_mse = F.mse_loss(Kr_flat, K_flat).item()
                    k_cos = F.cosine_similarity(Kr_flat, K_flat, dim=-1).mean().item()
                    v_mse = F.mse_loss(Vr_flat, V_flat).item()
                    v_cos = F.cosine_similarity(Vr_flat, V_flat, dim=-1).mean().item()
                    attn_cos = sdpa_cosine_similarity(Q_cpu, K_cpu, V_cpu, K_rec, V_rec)

                    metrics[tag][layer_idx]["k_mse"].append(k_mse)
                    metrics[tag][layer_idx]["k_cos"].append(k_cos)
                    metrics[tag][layer_idx]["v_mse"].append(v_mse)
                    metrics[tag][layer_idx]["v_cos"].append(v_cos)
                    metrics[tag][layer_idx]["attn_cos"].append(attn_cos)

    for h in hooks:
        h.remove()

    # ── Compute per-layer averages over sentences ─────────────────────────────
    def mean(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    avg = {
        tag: {
            layer_idx: {m: mean(metrics[tag][layer_idx][m]) for m in METRIC_NAMES}
            for layer_idx in range(NUM_LAYERS)
        }
        for tag in ["tq3", "tq4"]
    }

    # ── Print results table ───────────────────────────────────────────────────
    W = 108
    print("\n" + "=" * W)
    print(f"{'TurboQuant on Qwen3-1.7B — Real Activation Quality':^{W}}")
    print("=" * W)
    print(
        f"{'Layer':<6}"
        f"{'TQ3 K-MSE':>10}{'TQ3 K-Cos':>10}{'TQ3 V-MSE':>10}{'TQ3 V-Cos':>10}{'TQ3 Attn':>10}"
        f"  |  "
        f"{'TQ4 K-MSE':>10}{'TQ4 K-Cos':>10}{'TQ4 V-MSE':>10}{'TQ4 V-Cos':>10}{'TQ4 Attn':>10}"
    )
    print("-" * W)

    overall_vals = {tag: {m: [] for m in METRIC_NAMES} for tag in ["tq3", "tq4"]}

    for layer_idx in range(NUM_LAYERS):
        # Skip layers with no data (shouldn't happen, but guard anyway)
        if not metrics["tq3"][layer_idx]["k_mse"]:
            continue

        t3 = avg["tq3"][layer_idx]
        t4 = avg["tq4"][layer_idx]

        print(
            f"{layer_idx:<6}"
            f"{t3['k_mse']:10.6f}{t3['k_cos']:10.4f}"
            f"{t3['v_mse']:10.6f}{t3['v_cos']:10.4f}"
            f"{t3['attn_cos']:10.4f}"
            f"  |  "
            f"{t4['k_mse']:10.6f}{t4['k_cos']:10.4f}"
            f"{t4['v_mse']:10.6f}{t4['v_cos']:10.4f}"
            f"{t4['attn_cos']:10.4f}"
        )

        for tag in ["tq3", "tq4"]:
            for m in METRIC_NAMES:
                overall_vals[tag][m].append(avg[tag][layer_idx][m])

    print("-" * W)

    # Overall averages across all layers
    ov3 = {m: mean(overall_vals["tq3"][m]) for m in METRIC_NAMES}
    ov4 = {m: mean(overall_vals["tq4"][m]) for m in METRIC_NAMES}

    print(
        f"{'AVG':<6}"
        f"{ov3['k_mse']:10.6f}{ov3['k_cos']:10.4f}"
        f"{ov3['v_mse']:10.6f}{ov3['v_cos']:10.4f}"
        f"{ov3['attn_cos']:10.4f}"
        f"  |  "
        f"{ov4['k_mse']:10.6f}{ov4['k_cos']:10.4f}"
        f"{ov4['v_mse']:10.6f}{ov4['v_cos']:10.4f}"
        f"{ov4['attn_cos']:10.4f}"
    )
    print("=" * W)

    # ── Quality gates ─────────────────────────────────────────────────────────
    ok = lambda pass_: "✓ PASS" if pass_ else "✗ MISS"

    passes = {
        "tq3_k": ov3["k_cos"] >= TQ3_COS_TARGET,
        "tq3_v": ov3["v_cos"] >= TQ3_COS_TARGET,
        "tq4_k": ov4["k_cos"] >= TQ4_COS_TARGET,
        "tq4_v": ov4["v_cos"] >= TQ4_COS_TARGET,
    }

    print(f"\nQuality Gates  (real activations have outliers — harder than random noise)\n")
    print(f"  TQ3  K cos = {ov3['k_cos']:.4f}   target > {TQ3_COS_TARGET}   {ok(passes['tq3_k'])}")
    print(f"  TQ3  V cos = {ov3['v_cos']:.4f}   target > {TQ3_COS_TARGET}   {ok(passes['tq3_v'])}")
    print(f"  TQ4  K cos = {ov4['k_cos']:.4f}   target > {TQ4_COS_TARGET}   {ok(passes['tq4_k'])}")
    print(f"  TQ4  V cos = {ov4['v_cos']:.4f}   target > {TQ4_COS_TARGET}   {ok(passes['tq4_v'])}")

    if not all(passes.values()) and last_kv:
        diagnose_outliers(last_kv["K"], last_kv["V"])

    print("\nDone.")


if __name__ == "__main__":
    main()
