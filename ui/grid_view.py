"""Left panel: simulation grid with thin grey border; cell colors from world state."""

import pygame
import numpy as np

from ui.colors import energy_to_rgb

BORDER_COLOR = (80, 80, 80)
BORDER_PX = 1
DOTS_BG = (0, 0, 0)  # pitch black background in dots view
SQUIRCLE_EXPONENT = 2.6  # superellipse exponent > 2 (squircle; larger = closer to square)
SQUIRCLE_POINTS = 48    # polygon points for squircle


def _bilinear_upsample(arr: np.ndarray, scale: int) -> np.ndarray:
    """Upsample 2D array by scale using bilinear interpolation. Returns (nx*scale, ny*scale)."""
    nx, ny = arr.shape
    if scale <= 1:
        return arr
    H, W = nx * scale, ny * scale
    I = np.arange(H, dtype=np.float64)
    J = np.arange(W, dtype=np.float64)
    U = np.clip(I / scale, 0, nx - 1.001)
    V = np.clip(J / scale, 0, ny - 1.001)
    i0 = U.astype(np.int32)
    j0 = V.astype(np.int32)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    su = (U - i0).reshape(-1, 1)
    sv = (V - j0).reshape(1, -1)
    # Broadcast to (H, W): index with (H,1) and (1,W)
    i0_ = i0[:, np.newaxis]
    j0_ = j0[np.newaxis, :]
    i1_ = i1[:, np.newaxis]
    j1_ = j1[np.newaxis, :]
    p00 = arr[i0_, j0_]
    p01 = arr[i0_, j1_]
    p10 = arr[i1_, j0_]
    p11 = arr[i1_, j1_]
    out = (1 - su) * (1 - sv) * p00 + (1 - su) * sv * p01 + su * (1 - sv) * p10 + su * sv * p11
    return out.astype(arr.dtype)


def _squircle_points(cx: float, cy: float, a: float, b: float, n: float, num_points: int) -> list[tuple[float, float]]:
    """Superellipse |x/a|^n + |y/b|^n = 1; n > 2 gives squircle. Returns polygon points."""
    points = []
    for k in range(num_points):
        t = 2 * np.pi * k / num_points
        ct = np.cos(t)
        st = np.sin(t)
        # Parametric: x = a * sign(cos(t)) * |cos(t)|^(2/n), y = b * sign(sin(t)) * |sin(t)|^(2/n)
        x = cx + a * np.sign(ct) * (abs(ct) ** (2.0 / n))
        y = cy + b * np.sign(st) * (abs(st) ** (2.0 / n))
        points.append((x, y))
    return points


def _draw_grid_into_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    rgb: np.ndarray,
    nx: int,
    ny: int,
) -> None:
    """Draw rgb grid (nx, ny, 3) into rect; cell size from rect dimensions."""
    cell_w = max(1, rect.width // ny)
    cell_h = max(1, rect.height // nx)
    for i in range(nx):
        for j in range(ny):
            r, g, b = int(rgb[i, j, 0]), int(rgb[i, j, 1]), int(rgb[i, j, 2])
            x = rect.x + j * cell_w
            y = rect.y + i * cell_h
            pygame.draw.rect(surface, (r, g, b), (x, y, cell_w + 1, cell_h + 1))


def _squircle_g_and_mask(
    cell_h: int, cell_w: int, n: float, antialias: float,
    boundary: np.ndarray,
) -> np.ndarray:
    """Per-cell alpha (nx, cell_h, ny, cell_w). boundary (nx, ny) = size_scale**n; g from unit-radius superellipse."""
    r = max(0.5, min(cell_w, cell_h) * 0.5)
    pi = np.arange(cell_h, dtype=np.float64)[:, np.newaxis]
    pj = np.arange(cell_w, dtype=np.float64)[np.newaxis, :]
    x = (pj + 0.5 - cell_w * 0.5) / r
    y = (pi + 0.5 - cell_h * 0.5) / r
    g = np.abs(x) ** n + np.abs(y) ** n
    # g (ch, cw); boundary (nx, ny) -> (nx, 1, ny, 1); alpha (nx, ch, ny, cw)
    b = boundary[:, np.newaxis, :, np.newaxis]
    g4 = g[np.newaxis, :, np.newaxis, :]
    alpha = np.clip((b + antialias - g4) / (2 * antialias), 0.0, 1.0)
    return alpha


def _draw_grid_dots(
    surface: pygame.Surface,
    rect: pygame.Rect,
    rgb: np.ndarray,
    intensity: np.ndarray,
    nx: int,
    ny: int,
) -> None:
    """Draw grid as squircles per cell; dot size = 0.5 + 0.5*intensity (full at 1, half at 0)."""
    cell_w = max(1, rect.width // ny)
    cell_h = max(1, rect.height // nx)
    size_scale = np.clip(0.5 + 0.5 * intensity, 0.5, 1.0)
    boundary = size_scale ** SQUIRCLE_EXPONENT
    mask4 = _squircle_g_and_mask(cell_h, cell_w, SQUIRCLE_EXPONENT, 0.2, boundary)
    mask5 = mask4[:, :, :, :, np.newaxis]
    rgb5 = rgb[:, np.newaxis, :, np.newaxis, :]
    out = np.clip(rgb5 * mask5, 0, 255).round().astype(np.uint8).reshape(nx * cell_h, ny * cell_w, 3)
    try:
        img = pygame.image.fromstring(out.tobytes(), (ny * cell_w, nx * cell_h), "RGB")
    except TypeError:
        img = pygame.image.frombytes(out.tobytes(), (ny * cell_w, nx * cell_h), "RGB")
    surface.fill(DOTS_BG, rect)
    surface.blit(img, rect.topleft)


def draw_grid(
    surface: pygame.Surface,
    grid_rect: pygame.Rect,
    positive: np.ndarray,
    negative: np.ndarray,
    view_mode: str = "default",
    render_scale: int = 1,
    *,
    unified_show_a: bool = True,
    unified_show_b: bool = True,
    unified_show_ab: bool = True,
) -> None:
    """Draw the grid into grid_rect. render_scale 1 = current (no change). render_scale 2+ = bilinear
    upsample energy data to that resolution, color map, then scale down to grid_rect for sharper visualization of the same tick."""
    nx, ny = positive.shape
    if nx == 0 or ny == 0:
        return
    scale = max(1, min(4, render_scale))
    unified_kw = {"unified_show_a": unified_show_a, "unified_show_b": unified_show_b, "unified_show_ab": unified_show_ab}
    if view_mode == "dots":
        rgb = energy_to_rgb(positive, negative, "unified", **unified_kw)
        # Intensity for dot size: same as Visual (overlap combined); 0 = half size, 1 = full
        pos_max = float(np.max(positive)) if positive.size else 1.0
        neg_max = float(np.max(negative)) if negative.size else 1.0
        p = (positive / pos_max) if pos_max > 1e-12 else np.zeros_like(positive)
        n = (negative / neg_max) if neg_max > 1e-12 else np.zeros_like(negative)
        eps = 0.02
        both = (p >= eps) & (n >= eps)
        intensity = np.where(both, (p + n) * 0.5, 0.0)
        _draw_grid_dots(surface, grid_rect, rgb, intensity, nx, ny)
    elif scale == 1:
        rgb = energy_to_rgb(positive, negative, view_mode, **unified_kw)
        _draw_grid_into_rect(surface, grid_rect, rgb, nx, ny)
    else:
        pos_hr = _bilinear_upsample(positive, scale)
        neg_hr = _bilinear_upsample(negative, scale)
        rgb_hr = energy_to_rgb(pos_hr, neg_hr, view_mode, **unified_kw)
        H, W = rgb_hr.shape[0], rgb_hr.shape[1]
        # pygame: size (width, height); rgb_hr is (H, W, 3) row-major
        try:
            img = pygame.image.fromstring(rgb_hr.tobytes(), (W, H), "RGB")
        except TypeError:
            img = pygame.image.frombytes(rgb_hr.tobytes(), (W, H), "RGB")
        scaled = pygame.transform.smoothscale(img, (grid_rect.width, grid_rect.height))
        surface.blit(scaled, grid_rect.topleft)
    pygame.draw.rect(surface, BORDER_COLOR, grid_rect, BORDER_PX)
