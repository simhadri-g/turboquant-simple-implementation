"""turboquant.py — TurboQuant KV cache compression.

Algorithm (Zandieh, Daliri, Hadian, Mirrokni — arXiv:2504.19874, 2025):
    1. Normalize  : scale = ||x||_2 per vector;  x_hat = x / scale
    2. Rotate     : z = x_hat @ R.T   (R is a fixed random orthogonal matrix)
    3. Quantize   : scalar Lloyd-Max per coordinate using a pre-built codebook
                    for the Beta((d-1)/2, (d-1)/2) marginal distribution
    4. Dequantize : z_hat = codebook[indices]
    5. Un-rotate  : x_hat_rec = z_hat @ R
    6. Re-scale   : x_rec = x_hat_rec * scale

Key insight: after normalizing x to the unit sphere and applying a random
orthogonal rotation R, each coordinate z_i follows the same marginal
distribution — Beta((d-1)/2, (d-1)/2) scaled to [-1, 1] — regardless of the
original distribution of x. For large d this converges to N(0, 1/d). Because
all coordinates share the same distribution and are approximately independent
in high dimensions, a single scalar codebook can be applied per-coordinate,
reducing optimal vector quantization to scalar quantization.

QJL residual is NOT implemented: community experiments show the 1-bit residual
hurts attention quality in practice (softmax amplifies the quantization noise).

Units / shapes throughout:
    x         : [..., dim]           input KV vectors (any leading batch dims)
    scale     : [..., 1]             per-vector L2 norm
    x_hat     : [..., dim]           unit-norm vectors
    z         : [..., dim]           rotated coordinates
    indices   : [..., dim]           int16 — codebook indices per coordinate
    codebook  : [2**bits]            reconstruction levels (float32)
    R         : [dim, dim]           fixed orthogonal rotation matrix
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


# ── Rotation matrix ───────────────────────────────────────────────────────────

def generate_rotation_matrix(dim: int, seed: int = 42, device: torch.device = None) -> torch.Tensor:
    """Generate a fixed random orthogonal matrix via QR decomposition.

    The matrix is deterministic (fixed seed) so the same rotation is always
    used for both compression and decompression.

    Args:
        dim  : square matrix dimension (e.g., head_dim = 64)
        seed : random seed for reproducibility
        device: target device

    Returns:
        R : [dim, dim]  orthogonal matrix  (R @ R.T == I to float32 precision)
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    g = torch.randn(dim, dim, generator=generator)    # [dim, dim]  Gaussian
    q, _ = torch.linalg.qr(g)                         # [dim, dim]  orthogonal
    if device is not None:
        q = q.to(device)
    return q                                           # [dim, dim]


# ── Lloyd-Max codebook ────────────────────────────────────────────────────────

def build_lloyd_max_codebook(
    bits:      int,
    dim:       int,
    n_samples: int = 200_000,
    n_iter:    int = 200,
    seed:      int = 0,
) -> torch.Tensor:
    """Build an optimal scalar quantizer (Lloyd-Max) for rotated unit-sphere coordinates.

    Distribution: the marginal of each coordinate of a uniformly random unit
    vector in R^d is exactly the scaled Beta((d-1)/2, (d-1)/2) distribution.
    We sample this empirically (draw unit vectors, take one coordinate) and
    run 1-D k-means / Lloyd-Max to convergence.

    For d=64 and bits in {3, 4} this produces codebooks close to:
        TQ3 : 8 levels  covering roughly ±0.27 (≈ ±2σ where σ ≈ 1/√d)
        TQ4 : 16 levels covering roughly ±0.33 (≈ ±2.6σ)

    Args:
        bits      : bit-width (e.g., 3 or 4)
        dim       : vector dimension (e.g., head_dim = 64)
        n_samples : number of empirical samples for distribution estimation
        n_iter    : maximum Lloyd-Max iterations
        seed      : random seed

    Returns:
        codebook : [2**bits]  sorted float32 reconstruction levels
    """
    n_levels = 2 ** bits

    # Sample from the true marginal distribution:
    # draw random unit vectors in R^dim, take the first coordinate (representative
    # of any coordinate by rotational symmetry)
    generator = torch.Generator()
    generator.manual_seed(seed)
    raw    = torch.randn(n_samples, dim, generator=generator)   # [n_samples, dim]
    norms  = raw.norm(dim=-1, keepdim=True)                     # [n_samples, 1]
    unit   = raw / norms                                        # [n_samples, dim]  unit vectors
    samples = unit[:, 0].contiguous()                           # [n_samples]  marginal samples

    # Initialise codebook levels at evenly-spaced quantiles of the empirical CDF
    sorted_samples, _ = samples.sort()                           # [n_samples]
    q_positions = torch.linspace(
        0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels
    )                                                            # [n_levels]
    idx_init   = (q_positions * n_samples).long().clamp(0, n_samples - 1)
    codebook   = sorted_samples[idx_init].clone()               # [n_levels]

    # Lloyd-Max iterations (EM on 1-D data)
    #   E-step : assign each sample to its nearest reconstruction level
    #   M-step : update each level to the mean of its assigned samples
    for _ in range(n_iter):
        # [n_samples, n_levels] → [n_samples]
        dists       = (samples.unsqueeze(1) - codebook.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)                        # [n_samples]

        new_codebook = codebook.clone()
        for k in range(n_levels):
            mask = assignments == k                              # [n_samples] bool
            if mask.any():
                new_codebook[k] = samples[mask].mean()
            # else: leave the level unchanged (empty cell)

        delta = (new_codebook - codebook).abs().max().item()
        codebook = new_codebook
        if delta < 1e-9:
            break

    return codebook.sort().values                                # [n_levels]  ascending


# ── Quantize / Dequantize ─────────────────────────────────────────────────────

def quantize(
    x:        torch.Tensor,    # [..., dim]
    codebook: torch.Tensor,    # [n_levels]
    R:        torch.Tensor,    # [dim, dim]  orthogonal rotation
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress a batch of KV vectors with TurboQuant.

    Steps:
        1. L2-normalise each [dim] vector  →  x_hat  [..., dim]
        2. Rotate: z = x_hat @ R.T         →  z      [..., dim]
        3. Scalar-quantise each coordinate →  indices [..., dim]  (int16)

    Args:
        x        : [..., dim]  raw KV vectors (float32/bfloat16)
        codebook : [n_levels]  reconstruction levels
        R        : [dim, dim]  rotation matrix (same device as x)

    Returns:
        indices : [..., dim]   int16 codebook indices (0 … n_levels-1)
        scale   : [..., 1]     per-vector L2 norm (needed for dequantize)
    """
    orig_dtype = x.dtype
    x = x.float()                                                # [..., dim]

    # Step 1 — per-vector L2 normalisation
    scale  = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)       # [..., 1]
    x_hat  = x / scale                                           # [..., dim]

    # Step 2 — rotation: x_hat @ R.T  (right-multiply by R.T  ==  project onto rows of R)
    z = x_hat @ R.t()                                            # [..., dim]

    # Step 3 — per-coordinate nearest-codebook-entry lookup
    # Expand for broadcasting: z [..., dim, 1]  codebook [n_levels]
    z_exp   = z.unsqueeze(-1)                                    # [..., dim, 1]
    cb_exp  = codebook.to(z.device)                              # [n_levels]
    dists   = (z_exp - cb_exp).abs()                             # [..., dim, n_levels]
    indices = dists.argmin(dim=-1).to(torch.int16)               # [..., dim]

    return indices, scale.to(orig_dtype)


def dequantize(
    indices:  torch.Tensor,    # [..., dim]  int16
    scale:    torch.Tensor,    # [..., 1]    per-vector norm
    codebook: torch.Tensor,    # [n_levels]
    R:        torch.Tensor,    # [dim, dim]  orthogonal rotation
) -> torch.Tensor:
    """Reconstruct KV vectors from TurboQuant-compressed representation.

    Steps:
        1. Lookup reconstruction levels: z_hat = codebook[indices]  [..., dim]
        2. Un-rotate: x_hat_rec = z_hat @ R                         [..., dim]
        3. Re-scale:  x_rec = x_hat_rec * scale                     [..., dim]

    Args:
        indices  : [..., dim]  int16 codebook indices
        scale    : [..., 1]    per-vector L2 norms
        codebook : [n_levels]  reconstruction levels
        R        : [dim, dim]  the same rotation matrix used in quantize()

    Returns:
        x_rec : [..., dim]  float32 reconstructed vectors
    """
    cb = codebook.to(indices.device)                             # [n_levels]
    z_hat = cb[indices.long()]                                   # [..., dim]  float32

    # Un-rotate: z_hat @ R  (inverse of  x_hat @ R.T  is  z_hat @ R)
    x_hat_rec = z_hat @ R.to(z_hat.device)                      # [..., dim]

    # Re-scale by the stored per-vector norm
    x_rec = x_hat_rec * scale.float()                           # [..., dim]

    return x_rec                                                 # [..., dim]


# ── TurboQuant class ──────────────────────────────────────────────────────────

class TurboQuant(nn.Module):
    """TurboQuant: online, data-oblivious KV cache vector quantizer.

    Usage:
        tq = TurboQuant(bits=4, dim=64)
        tq.to(device)

        indices, scale = tq.compress(x)    # x: [..., dim]
        x_hat          = tq.decompress(indices, scale)

    Attributes:
        bits     : int          quantization bit-width
        dim      : int          vector dimension
        codebook : [2**bits]    Lloyd-Max reconstruction levels (registered buffer)
        R        : [dim, dim]   fixed orthogonal rotation matrix (registered buffer)
    """

    def __init__(
        self,
        bits:         int   = 4,
        dim:          int   = 64,
        n_samples:    int   = 200_000,
        n_iter:       int   = 200,
        rotation_seed: int  = 42,
        codebook_seed: int  = 0,
    ):
        super().__init__()
        self.bits = bits
        self.dim  = dim
        n_levels  = 2 ** bits

        # Build codebook (fits the Beta marginal of rotated unit-sphere coordinates)
        codebook = build_lloyd_max_codebook(
            bits=bits, dim=dim,
            n_samples=n_samples, n_iter=n_iter, seed=codebook_seed,
        )                                                         # [n_levels]
        self.register_buffer("codebook", codebook)               # [n_levels]

        # Build fixed rotation matrix
        R = generate_rotation_matrix(dim, seed=rotation_seed)    # [dim, dim]
        self.register_buffer("R", R)                             # [dim, dim]

        print(
            f"TurboQuant(bits={bits}, dim={dim}) | "
            f"{n_levels} levels | "
            f"codebook range [{codebook.min():.4f}, {codebook.max():.4f}]"
        )

    def compress(
        self,
        x: torch.Tensor,    # [..., dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress x to (indices, scale).

        Returns:
            indices : [..., dim]  int16
            scale   : [..., 1]    float32 per-vector norm
        """
        return quantize(x, self.codebook, self.R)

    def decompress(
        self,
        indices: torch.Tensor,   # [..., dim]  int16
        scale:   torch.Tensor,   # [..., 1]    float32
    ) -> torch.Tensor:
        """Reconstruct vectors from (indices, scale).

        Returns:
            x_rec : [..., dim]  float32
        """
        return dequantize(indices, scale, self.codebook, self.R)

    def compression_ratio(self) -> float:
        """Theoretical bits-per-element vs float32 (32 bits)."""
        # Each coordinate stored as `bits` bits; scale stored as 32-bit float
        # but amortised over dim coordinates — negligible for large dim
        return 32.0 / self.bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Round-trip: compress then decompress. Useful for MSE measurement."""
        indices, scale = self.compress(x)
        return self.decompress(indices, scale)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"TurboQuant smoke test on: {device}\n")

    DIM = 64  # head_dim

    for bits in (3, 4):
        tq = TurboQuant(bits=bits, dim=DIM).to(device)

        # Simulate a single KV cache slice: [batch=2, heads=4, seq=16, head_dim=64]
        x = torch.randn(2, 4, 16, DIM, device=device)          # [..., dim]

        indices, scale = tq.compress(x)
        x_hat          = tq.decompress(indices, scale)

        mse = (x - x_hat).pow(2).mean().item()
        cos = torch.nn.functional.cosine_similarity(
            x.reshape(-1, DIM), x_hat.reshape(-1, DIM), dim=-1
        ).mean().item()

        print(
            f"  TQ{bits} | MSE = {mse:.6f} | "
            f"cosine sim = {cos:.6f} | "
            f"compression = {tq.compression_ratio():.1f}x"
        )
        print(f"        indices shape : {tuple(indices.shape)}  dtype={indices.dtype}")
        print(f"        scale   shape : {tuple(scale.shape)}")

    print("\nSmoke test PASSED.")
