"""Reproducible random initial positions from a seed. Seed -1 = new random each call.
Positions are derived from 0-1 relative coords so the same seed gives the same relative
position for any world size (nx, ny)."""

import random
from typing import Tuple


def pick_initial_positions(
    nx: int, ny: int, seed: int
) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """
    Return (pos_cell, neg_cell, seed_used). If seed == -1, choose a new random seed.
    Positions are computed from normalized 0-1 coords so the same seed gives the same
    relative placement regardless of nx, ny.
    """
    if seed == -1:
        seed_used = random.randint(0, 2**31 - 1)
    else:
        seed_used = seed
    rng = random.Random(seed_used)
    rx1, ry1 = rng.random(), rng.random()
    rx2, ry2 = rng.random(), rng.random()
    while (rx1, ry1) == (rx2, ry2):
        rx2, ry2 = rng.random(), rng.random()
    # Map 0-1 to cell indices; clamp so we stay in [0, nx-1] and [0, ny-1]
    def to_cell(rx: float, ry: float) -> Tuple[int, int]:
        i = min(int(rx * nx), nx - 1) if nx > 0 else 0
        j = min(int(ry * ny), ny - 1) if ny > 0 else 0
        return (i, j)
    pos_cell = to_cell(rx1, ry1)
    neg_cell = to_cell(rx2, ry2)
    return pos_cell, neg_cell, seed_used
