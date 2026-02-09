"""
Per-tick update: diffusion then interaction.
Diffusion: golden-decay 3×3 kernel. boundary_consume=True = zero-pad (mass lost at edges, dramatic).
boundary_consume=False = reflective BC (totals conserved).
"""

import numpy as np

from world.grid import Grid

# Fractal interaction: strength; phi_decay for smooth component; edge blend for filaments
DEFAULT_FRACTAL_STRENGTH = 0.3
DEFAULT_PHI_DECAY = 0.618
DEFAULT_FRACTAL_RADIUS = 1  # 3×3 only (fast); kept for UI compatibility, only 1 used
DEFAULT_EDGE_BLEND = 0.7  # 0 = smooth blob, 1 = strong edges (filaments/branching)

# Fixed 3×3 Laplacian: positive where cell > neighbors → edges/boundaries
LAPLACIAN_3X3 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

# Legacy diffusion kernel weights (cardinal 0.2, diagonal 0.05 per neighbor).
_W_CARDINAL, _W_DIAGONAL = 0.2, 0.05


def _convolve3x3_vectorized(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve with 3×3 kernel; same shape; zero-pad."""
    nx, ny = arr.shape
    out = np.zeros_like(arr)
    pad = np.zeros((nx + 2, ny + 2), dtype=arr.dtype)
    pad[1:-1, 1:-1] = arr
    for ki in range(3):
        for kj in range(3):
            out += kernel[ki, kj] * pad[ki : ki + nx, kj : kj + ny]
    return out


def _pad_reflect(arr: np.ndarray) -> np.ndarray:
    """Pad by one cell each side using reflection. Convolution then conserves total."""
    nx, ny = arr.shape
    pad = np.zeros((nx + 2, ny + 2), dtype=arr.dtype)
    pad[1:-1, 1:-1] = arr
    pad[0, 1:-1] = arr[0, :]
    pad[-1, 1:-1] = arr[-1, :]
    pad[1:-1, 0] = arr[:, 0]
    pad[1:-1, -1] = arr[:, -1]
    pad[0, 0], pad[0, -1], pad[-1, 0], pad[-1, -1] = arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]
    return pad


def _convolve3x3_reflect(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve with 3×3 kernel; reflective BC. Conserves total mass."""
    nx, ny = arr.shape
    pad = _pad_reflect(arr)
    out = np.zeros_like(arr)
    for ki in range(3):
        for kj in range(3):
            out += kernel[ki, kj] * pad[ki : ki + nx, kj : kj + ny]
    return out


def _diffuse_legacy(grid: np.ndarray, retain_ratio: float) -> np.ndarray:
    """Legacy diffusion: center retain_ratio, cardinals (1-r)*0.2, diagonals (1-r)*0.05. Zero-pad, vectorized."""
    r = max(0.01, min(0.99, retain_ratio))
    c, d = (1.0 - r) * _W_CARDINAL, (1.0 - r) * _W_DIAGONAL
    kernel = np.array(
        [[d, c, d], [c, r, c], [d, c, d]],
        dtype=np.float64,
    )
    return _convolve3x3_vectorized(grid, kernel)


def _diffuse_channel(grid: np.ndarray, phi_decay: float, boundary_consume: bool) -> np.ndarray:
    """One channel: golden-decay kernel. boundary_consume=True = zero-pad (leaky); False = reflective (conserved)."""
    denom = 1.0 + 8.0 * phi_decay
    kernel = np.array(
        [
            [phi_decay, phi_decay, phi_decay],
            [phi_decay, 1.0, phi_decay],
            [phi_decay, phi_decay, phi_decay],
        ],
        dtype=np.float64,
    ) / denom
    if boundary_consume:
        return _convolve3x3_vectorized(grid, kernel)
    return _convolve3x3_reflect(grid, kernel)


def _seed_modulation(nx: int, ny: int, seed: int) -> np.ndarray:
    """Deterministic per-cell factor from seed to break circular symmetry (0.5–1.5)."""
    I = np.arange(nx, dtype=np.int64).reshape(-1, 1)
    J = np.arange(ny, dtype=np.int64).reshape(1, -1)
    H = np.bitwise_xor(
        np.bitwise_xor(I * 73856093, J * 19349663),
        np.int64(seed) & 0x7FFFFFFF,
    )
    return 0.5 + (np.remainder(np.abs(H), 10000) / 10000.0)  # 0.5 to 1.5


def _interaction_step(
    grid: Grid,
    fractal_strength: float,
    phi_decay: float,
    fractal_radius: int,  # ignored; we use 3×3 only for speed
    edge_blend: float,
    seed: int | None = None,
) -> None:
    """
    Overlap: cancel a fraction of min(pos, neg); redistribute to boundaries of overlap
    (edge + smooth). Optional seed modulates weight per cell to break symmetry.
    Totals conserved.
    """
    pos = grid.positive_energy.copy()
    neg = grid.negative_energy.copy()
    nx, ny = grid.shape
    overlap = np.minimum(pos, neg)
    cancel = fractal_strength * overlap
    pos_new = np.maximum(pos - cancel, 0.0)
    neg_new = np.maximum(neg - cancel, 0.0)
    total_cancel = float(np.sum(cancel))
    if total_cancel < 1e-12:
        grid.positive_energy = pos_new
        grid.negative_energy = neg_new
        return
    # Smooth 3×3 kernel (golden-ratio style: center 1, ring = phi_decay)
    smooth_k = np.array(
        [[phi_decay, phi_decay, phi_decay], [phi_decay, 1.0, phi_decay], [phi_decay, phi_decay, phi_decay]],
        dtype=np.float64,
    )
    smooth_w = _convolve3x3_vectorized(overlap, smooth_k)
    edge_w = _convolve3x3_vectorized(overlap, LAPLACIAN_3X3)
    edge_w = np.maximum(edge_w, 0.0)  # only reinforce boundaries (high Laplacian)
    # Blend: more edge_blend → more filamentary structure
    weight = (1.0 - edge_blend) * smooth_w + edge_blend * (edge_w + 1e-12)
    weight = np.maximum(weight, 0.0)
    # Seed-based modulation: breaks circular symmetry for irregular/fractal-like structure
    if seed is not None:
        weight = weight * _seed_modulation(nx, ny, seed)
    w_sum = np.sum(weight)
    if w_sum > 1e-12:
        weight = weight / w_sum
    else:
        weight = np.full_like(weight, 1.0 / (nx * ny))
    pos_new += weight * total_cancel
    neg_new += weight * total_cancel
    grid.positive_energy = pos_new
    grid.negative_energy = neg_new


def step(
    grid: Grid,
    fractal_strength: float = DEFAULT_FRACTAL_STRENGTH,
    phi_decay: float = DEFAULT_PHI_DECAY,
    fractal_radius: int = DEFAULT_FRACTAL_RADIUS,
    edge_blend: float = DEFAULT_EDGE_BLEND,
    seed: int | None = None,
    boundary_consume: bool = True,
    retain_ratio: float | None = None,
) -> None:
    """One tick: diffuse then interaction. If boundary_consume and retain_ratio given (from old config), use legacy diffusion; else golden-decay."""
    if boundary_consume and retain_ratio is not None:
        grid.positive_energy = _diffuse_legacy(grid.positive_energy, retain_ratio)
        grid.negative_energy = _diffuse_legacy(grid.negative_energy, retain_ratio)
    else:
        grid.positive_energy = _diffuse_channel(grid.positive_energy, phi_decay, boundary_consume)
        grid.negative_energy = _diffuse_channel(grid.negative_energy, phi_decay, boundary_consume)
    _interaction_step(grid, fractal_strength, phi_decay, fractal_radius, edge_blend, seed)
