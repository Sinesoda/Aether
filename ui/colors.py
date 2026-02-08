"""
Display-only normalization: map current grid min–max per channel to 0–1 so small
energies stay visible. Positive = rich red, negative = rich blue (not pure). Where
both energies overlap, use a blackbody-style gradient (absence → intense energy):
black through red, orange, yellow to white, matching how we perceive light energy.
"""

import numpy as np

# Positive energy: pink; negative: cool blue (not pure)
RED = np.array([0.78, 0.38, 0.55], dtype=np.float64)   # deeper pink (positive energy)
BLUE = np.array([0.14, 0.22, 0.78], dtype=np.float64)   # cool blue

# Blackbody (overlap): dark red → red → orange → yellow → white
_ENERGY_STOPS = np.array([
    [0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 0.08, 0.02], [1.0, 0.35, 0.0],
    [1.0, 0.75, 0.2], [1.0, 0.95, 0.7], [1.0, 1.0, 1.0],
], dtype=np.float64)
_ENERGY_T = np.array([0.0, 0.12, 0.28, 0.45, 0.65, 0.85, 1.0], dtype=np.float64)

# Galactic blue (energy 1): pitch black → nebula blue → bright blue → white
_GALACTIC_BLUE_STOPS = np.array([
    [0.0, 0.0, 0.0], [0.06, 0.08, 0.22], [0.12, 0.18, 0.45], [0.25, 0.4, 0.75],
    [0.5, 0.65, 0.95], [0.8, 0.9, 1.0], [1.0, 1.0, 1.0],
], dtype=np.float64)
_GALACTIC_BLUE_T = np.array([0.0, 0.18, 0.38, 0.55, 0.72, 0.88, 1.0], dtype=np.float64)

# Galactic pink (energy 2): pitch black → pink → hot pink → white
_GALACTIC_PINK_STOPS = np.array([
    [0.0, 0.0, 0.0], [0.22, 0.06, 0.2], [0.5, 0.2, 0.45], [0.78, 0.45, 0.6],
    [0.95, 0.75, 0.85], [1.0, 0.92, 0.95], [1.0, 1.0, 1.0],
], dtype=np.float64)
_GALACTIC_PINK_T = np.array([0.0, 0.2, 0.42, 0.6, 0.78, 0.92, 1.0], dtype=np.float64)


def _apply_gradient(t: np.ndarray, stops: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    """Map t in [0,1] to RGB via piecewise-linear stops. t 1D, returns (n, 3)."""
    t = np.clip(np.asarray(t, dtype=np.float64).reshape(-1), 0.0, 1.0)
    out = np.zeros((t.size, 3), dtype=np.float64)
    for i in range(len(t_vals) - 1):
        t0, t1 = t_vals[i], t_vals[i + 1]
        mask = (t >= t0) & (t < t1) if i < len(t_vals) - 2 else (t >= t0)
        s = np.where(mask, (t - t0) / max(1e-9, t1 - t0), 0.0)
        s1 = s[mask].reshape(-1, 1)
        out[mask] = s1 * stops[i + 1] + (1.0 - s1) * stops[i]
    return out


def _energy_gradient(t: np.ndarray) -> np.ndarray:
    """Blackbody-style gradient for overlap (combined energy)."""
    return _apply_gradient(t, _ENERGY_STOPS, _ENERGY_T)


VIEW_MODES = ("default", "unified", "dots", "positive_bw", "negative_bw", "delta_bw")


def energy_to_rgb(
    positive: np.ndarray,
    negative: np.ndarray,
    view_mode: str = "default",
    *,
    unified_show_a: bool = True,
    unified_show_b: bool = True,
    unified_show_ab: bool = True,
) -> np.ndarray:
    """
    Returns (nx, ny, 3) uint8 RGB. view_mode: "default" (red/blue/energy gradient),
    "unified" (overlap + energy1 + energy2 layers, toggled by unified_show_*),
    "positive_bw" / "negative_bw" / "delta_bw" (normalized black–white).
    """
    pos_max = float(np.max(positive))
    neg_max = float(np.max(negative))
    if pos_max > 1e-12:
        p = positive / pos_max
    else:
        p = np.zeros_like(positive)
    if neg_max > 1e-12:
        n = negative / neg_max
    else:
        n = np.zeros_like(negative)

    if view_mode == "positive_bw":
        rgb = np.stack([p, p, p], axis=-1)
        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)
    if view_mode == "negative_bw":
        rgb = np.stack([n, n, n], axis=-1)
        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)
    if view_mode == "delta_bw":
        delta = positive - negative
        d_min, d_max = float(np.min(delta)), float(np.max(delta))
        if d_max - d_min > 1e-12:
            t = (delta - d_min) / (d_max - d_min)
        else:
            t = np.zeros_like(delta)
        rgb = np.stack([t, t, t], axis=-1)
        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)

    # default / unified
    nx, ny = p.shape
    rgb = np.zeros((nx, ny, 3), dtype=np.float64)
    eps = 0.02
    pos_only = (p >= eps) & (n < eps)
    neg_only = (n >= eps) & (p < eps)
    both = (p >= eps) & (n >= eps)
    combined = np.clip((p + n) * 0.5, 0.0, 1.0)
    overlap_weight = np.clip(2.0 * np.minimum(p, n), 0.0, 1.0)

    if view_mode == "unified":
        # p,n already use one global max (above). Cap single-energy by overlap’s typical brightness
        # so they don’t outshine overlap: use 90th percentile of combined in overlap as ref
        rgb = np.zeros((nx, ny, 3), dtype=np.float64)
        if unified_show_ab:
            grad_combined = _energy_gradient(combined.reshape(-1)).reshape(nx, ny, 3)
            rgb += overlap_weight[:, :, np.newaxis] * grad_combined
        if unified_show_a:
            grad_p = _apply_gradient(p.reshape(-1), _GALACTIC_BLUE_STOPS, _GALACTIC_BLUE_T).reshape(nx, ny, 3)
            rgb += (1.0 - overlap_weight)[:, :, np.newaxis] * grad_p
        if unified_show_b:
            grad_n = _apply_gradient(n.reshape(-1), _GALACTIC_PINK_STOPS, _GALACTIC_PINK_T).reshape(nx, ny, 3)
            rgb += (1.0 - overlap_weight)[:, :, np.newaxis] * grad_n
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        # default: red/blue per channel, blackbody in overlap
        rgb[pos_only] = (p[pos_only].reshape(-1, 1) * RED)
        rgb[neg_only] = (n[neg_only].reshape(-1, 1) * BLUE)
        gradient_rgb = _energy_gradient(combined.reshape(-1)).reshape(nx, ny, 3)
        rgb[both] = gradient_rgb[both]

    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)
