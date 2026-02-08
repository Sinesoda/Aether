"""Tooltip constants, text, and drawing for parameter panel."""

import pygame
from typing import Optional

TOOLTIP_BG = (28, 28, 32)
TOOLTIP_BORDER = (60, 60, 68)
TOOLTIP_TEXT = (240, 240, 235)
TOOLTIP_MINMAX = (150, 150, 148)
TOOLTIP_MAX_WIDTH = 220
TOOLTIP_PADDING = 6
TOOLTIP_OFFSET_Y = 8
TOOLTIP_DESC_MINMAX_GAP = 4

# (description, min_max_meaning). Key = param name matching panel _slider_rects / _tooltip_rects.
PARAM_TOOLTIPS = {
    "fractal_strength": (
        "Fraction of overlap (min of pos and neg at each cell) that is removed from both channels each step, "
        "then redistributed onto the grid using an edge-favoring blend (Laplacian + smooth kernel). "
        "Higher values cancel more overlap and push more energy to boundaries, giving sharper filaments and branches.",
        "Min = almost no interaction; max = strong pull to boundaries, sharp structures.",
    ),
    "phi_decay": (
        "Golden-decay: in diffusion, how much each cell shares with its 8 neighbors (center 1, ring φ); "
        "in interaction, how much the smooth redistribution spreads to neighbors. One parameter for both.",
        "Min = more concentrated (slower spread); max = more spread-out, softer.",
    ),
    "edge_blend": (
        "How much we favor the boundary of the overlap vs a smooth spread when placing the moved energy.",
        "Min = soft, blob-like boundaries; max = sharp edges, filaments and branches.",
    ),
}


def wrap_tooltip_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    words = text.split()
    lines = []
    current: list[str] = []
    for word in words:
        w, _ = font.size(" ".join(current + [word]))
        if current and w > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def draw_tooltip(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    tooltip_raw: Optional[tuple[str, Optional[str]] | str],
    mouse_pos: tuple[int, int],
) -> None:
    if not tooltip_raw:
        return
    desc = tooltip_raw[0] if isinstance(tooltip_raw, tuple) else tooltip_raw
    min_max = tooltip_raw[1] if isinstance(tooltip_raw, tuple) and len(tooltip_raw) > 1 else None
    if not desc:
        return
    mx, my = mouse_pos
    lines_desc = wrap_tooltip_text(desc, font, TOOLTIP_MAX_WIDTH)
    line_h_desc = font.get_height()
    box_w = min(
        TOOLTIP_MAX_WIDTH + 2 * TOOLTIP_PADDING,
        max((font.size(l)[0] for l in lines_desc), default=0) + 2 * TOOLTIP_PADDING,
    )
    box_h = len(lines_desc) * line_h_desc + 2 * TOOLTIP_PADDING
    if min_max:
        lines_mm = wrap_tooltip_text(min_max, small_font, TOOLTIP_MAX_WIDTH)
        line_h_mm = small_font.get_height()
        box_h += TOOLTIP_DESC_MINMAX_GAP + len(lines_mm) * line_h_mm
        box_w = max(
            box_w,
            min(
                TOOLTIP_MAX_WIDTH + 2 * TOOLTIP_PADDING,
                max((small_font.size(l)[0] for l in lines_mm), default=0) + 2 * TOOLTIP_PADDING,
            ),
        )
    tx = mx + 12
    ty = my + TOOLTIP_OFFSET_Y
    sw, sh = surface.get_size()
    if tx + box_w > sw:
        tx = mx - box_w - 12
    if ty + box_h > sh:
        ty = my - box_h - TOOLTIP_OFFSET_Y
    tx = max(0, min(tx, sw - box_w))
    ty = max(0, min(ty, sh - box_h))
    tooltip_rect = pygame.Rect(tx, ty, box_w, box_h)
    pygame.draw.rect(surface, TOOLTIP_BG, tooltip_rect)
    pygame.draw.rect(surface, TOOLTIP_BORDER, tooltip_rect, 1)
    y_off = ty + TOOLTIP_PADDING
    for line in lines_desc:
        surface.blit(font.render(line, True, TOOLTIP_TEXT), (tx + TOOLTIP_PADDING, y_off))
        y_off += line_h_desc
    if min_max:
        y_off += TOOLTIP_DESC_MINMAX_GAP
        for line in lines_mm:
            surface.blit(small_font.render(line, True, TOOLTIP_MINMAX), (tx + TOOLTIP_PADDING, y_off))
            y_off += line_h_mm
