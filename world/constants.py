"""Simulation constants. Vacuum = 0; energy conserved per type."""

VACUUM = 0.0
# 50% retained in cell, 50% distributed to 8 neighbors; cardinal > diagonal.
RETAIN_RATIO = 0.5
# Neighbor offsets (di, dj): cardinals first, then diagonals. Weights set in diffusion.
CARDINAL_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DEFAULT_NX, DEFAULT_NY = 64, 64
