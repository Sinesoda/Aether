"""World: grid and tick-driven diffusion + interaction."""

from world.grid import Grid
from world.diffusion import step
from world.constants import DEFAULT_NX, DEFAULT_NY, RETAIN_RATIO, VACUUM

__all__ = ["Grid", "step", "DEFAULT_NX", "DEFAULT_NY", "RETAIN_RATIO", "VACUUM"]
