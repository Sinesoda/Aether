"""2D grid of positive and negative energy per cell. Shape (nx, ny) for 3D extension."""

import numpy as np

from world.constants import VACUUM, DEFAULT_NX, DEFAULT_NY


class Grid:
    """Dual energy arrays; totals conserved by diffusion/interaction."""

    __slots__ = ("shape", "positive_energy", "negative_energy")

    def __init__(self, nx: int = DEFAULT_NX, ny: int = DEFAULT_NY) -> None:
        self.shape = (nx, ny)
        self.positive_energy = np.zeros(self.shape, dtype=np.float64)
        self.negative_energy = np.zeros(self.shape, dtype=np.float64)

    def get_cell(self, i: int, j: int) -> tuple[float, float]:
        return float(self.positive_energy[i, j]), float(self.negative_energy[i, j])

    def set_cell(self, i: int, j: int, pos: float, neg: float) -> None:
        self.positive_energy[i, j] = pos
        self.negative_energy[i, j] = neg

    def total_positive(self) -> float:
        return float(np.sum(self.positive_energy))

    def total_negative(self) -> float:
        return float(np.sum(self.negative_energy))

    def set_initial_state(self, pos_cell: tuple[int, int], neg_cell: tuple[int, int]) -> None:
        """One cell +1 positive, one cell -1 negative; rest 0."""
        self.positive_energy.fill(VACUUM)
        self.negative_energy.fill(VACUUM)
        pi, pj = pos_cell
        ni, nj = neg_cell
        nx, ny = self.shape
        if 0 <= pi < nx and 0 <= pj < ny:
            self.positive_energy[pi, pj] = 1.0
        if 0 <= ni < nx and 0 <= nj < ny:
            self.negative_energy[ni, nj] = 1.0
