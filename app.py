"""
App shell: display and main loop. Simulation is tick-driven from elapsed time and
tick_rate (independent of frame rate). World, UI, and config are wired here.
"""

import pygame

from world import Grid, step
from world.diffusion import DEFAULT_FRACTAL_STRENGTH, DEFAULT_PHI_DECAY, DEFAULT_EDGE_BLEND
from world.seed_util import pick_initial_positions
from ui.grid_view import draw_grid
from ui.panel import ParamPanel
import config

TITLE = "Aether"
WIDTH, HEIGHT = 960, 640
BACKGROUND = (0, 0, 0)
GRID_PANEL_WIDTH = 640  # left panel for grid; panel is half that (320)


def run() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()
    config.refresh_index()

    cfg = config.load_config()
    nx = cfg["world"].get("nx", 64)
    ny = cfg["world"].get("ny", 64)
    seed = cfg.get("seed", -1)
    if cfg.get("lock_seed") and seed == -1 and "actual_seed_used" in cfg:
        seed_for_init = cfg["actual_seed_used"]
    else:
        seed_for_init = seed
    pos_cell, neg_cell, actual_seed_used = pick_initial_positions(nx, ny, seed_for_init)

    grid = Grid(nx=nx, ny=ny)
    grid.set_initial_state(pos_cell, neg_cell)

    total_ticks = int(cfg.get("tick_count", 0))
    last = config.get_last_config()
    state = config.load_state(last[0], last[1]) if last else None
    if state is not None and state["positive"].shape == (nx, ny):
        grid.positive_energy[:] = state["positive"]
        grid.negative_energy[:] = state["negative"]
        total_ticks = state["tick_count"]
        actual_seed_used = cfg.get("actual_seed_used", actual_seed_used)
    elif last and (state is None or state["positive"].shape != (nx, ny)):
        total_ticks = 0

    grid_rect = pygame.Rect(0, 0, GRID_PANEL_WIDTH, HEIGHT)
    panel_rect = pygame.Rect(GRID_PANEL_WIDTH, 0, WIDTH - GRID_PANEL_WIDTH, HEIGHT)

    def do_restart() -> None:
        nonlocal grid, total_ticks, actual_seed_used
        params = panel.get_params()
        nx, ny = params["nx"], params["ny"]
        s = params.get("seed", -1)
        if s == -1 and params.get("lock_seed"):
            s = actual_seed_used
        grid = Grid(nx=nx, ny=ny)
        pos_cell, neg_cell, actual_seed_used = pick_initial_positions(nx, ny, s)
        grid.set_initial_state(pos_cell, neg_cell)
        total_ticks = 0

    def save_current_config() -> None:
        nonlocal grid
        params = panel.get_params()
        name = (params.get("config_name") or "").strip() or "unnamed"
        actual_seed = actual_seed_used
        cfg = panel._config_dict()
        cfg["actual_seed_used"] = actual_seed
        state = {
            "positive": grid.positive_energy.copy(),
            "negative": grid.negative_energy.copy(),
            "tick_count": total_ticks,
        }
        config.save_config(cfg, actual_seed, name, tick_count=total_ticks, state=state)
        panel.set_selected_config(actual_seed, config._sanitize_name(name) or "unnamed")

    def load_config_callback(seed_id: int, name: str) -> None:
        nonlocal grid, total_ticks, actual_seed_used
        path = config.get_config_path(seed_id, name)
        if not path.exists():
            return
        cfg = config.load_config(path)
        panel.apply_config(cfg)
        panel.params["config_name"] = name
        nx, ny = cfg["world"].get("nx", 64), cfg["world"].get("ny", 64)
        state = config.load_state(seed_id, name)
        if state is not None and state["positive"].shape == (nx, ny):
            if grid.shape != (nx, ny):
                grid = Grid(nx=nx, ny=ny)
            grid.positive_energy[:] = state["positive"]
            grid.negative_energy[:] = state["negative"]
            total_ticks = state["tick_count"]
            actual_seed_used = cfg.get("actual_seed_used", seed_id)
        else:
            do_restart()

    last = config.get_last_config()
    panel = ParamPanel(
        panel_rect,
        {
            "nx": nx,
            "ny": ny,
            "tick_rate": cfg.get("tick_rate", 30),
            "fractal_strength": cfg.get("fractal_strength", DEFAULT_FRACTAL_STRENGTH),
            "phi_decay": cfg.get("phi_decay", DEFAULT_PHI_DECAY),
            "edge_blend": cfg.get("edge_blend", DEFAULT_EDGE_BLEND),
            "seed": cfg.get("actual_seed_used", seed),
            "lock_seed": cfg.get("lock_seed", False),
            "render_scale": cfg.get("render_scale", 1),
            "view_mode": cfg.get("view_mode", "default"),
            "unified_show_a": cfg.get("unified_show_a", True),
            "unified_show_b": cfg.get("unified_show_b", True),
            "unified_show_ab": cfg.get("unified_show_ab", True),
            "config_name": last[1] if last else "",
            "selected_config": last,
            "on_load_config": load_config_callback,
        },
        on_save=save_current_config,
        on_restart=do_restart,
    )

    tick_accum = 0.0
    running = True

    while running:
        dt_ms = clock.tick(60)
        dt_s = dt_ms / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            panel.handle_event(event)

        params = panel.get_params()
        # Live world size: if changed, rebuild grid; same relative seed positions (0-1) so layout preserved
        if (params["nx"], params["ny"]) != grid.shape:
            nx, ny = params["nx"], params["ny"]
            grid = Grid(nx=nx, ny=ny)
            s = params.get("seed", -1)
            if s == -1 and params.get("lock_seed"):
                s = actual_seed_used
            pos_cell, neg_cell, actual_seed_used = pick_initial_positions(nx, ny, s)
            grid.set_initial_state(pos_cell, neg_cell)
            total_ticks = 0

        if not params["paused"]:
            tick_rate = max(1, min(60, params["tick_rate"]))
            tick_accum += dt_s * tick_rate
            # Cap ticks per frame so we never freeze when tick rate exceeds what we can do
            max_ticks_per_frame = max(4, tick_rate // 10)
            num_ticks = min(int(tick_accum), max_ticks_per_frame)
            tick_accum -= num_ticks
            tick_accum = min(tick_accum, max_ticks_per_frame)  # prevent unbounded backlog
            total_ticks += num_ticks
            strength = max(0.0, min(1.0, params["fractal_strength"]))
            phi = max(0.4, min(0.9, params["phi_decay"]))
            edge = max(0.0, min(1.0, params["edge_blend"]))
            for _ in range(num_ticks):
                step(grid, fractal_strength=strength, phi_decay=phi, edge_blend=edge, seed=actual_seed_used)

        screen.fill(BACKGROUND)
        draw_grid(
            screen,
            grid_rect,
            grid.positive_energy,
            grid.negative_energy,
            view_mode=params.get("view_mode", "default"),
            render_scale=params.get("render_scale", 1),
            unified_show_a=params.get("unified_show_a", True),
            unified_show_b=params.get("unified_show_b", True),
            unified_show_ab=params.get("unified_show_ab", True),
        )
        panel.draw(screen, tick_count=total_ticks, actual_used_seed=actual_seed_used)
        panel.draw_tooltip(screen)
        pygame.display.flip()

    pygame.quit()
