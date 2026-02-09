"""Right panel: live-updatable sliders (world X/Y, tick rate), play/pause, save/update settings, config dropdown."""

import pygame
from typing import Callable

import config
from ui import tooltips

FONT_SIZE = 16
TOOLTIP_FONT_SIZE = 19
TOOLTIP_SMALL_FONT_SIZE = 16
LABEL_COLOR = (200, 200, 200)
SLIDER_COLOR = (100, 100, 100)
KNOB_COLOR = (180, 180, 180)
BUTTON_COLOR = (60, 60, 60)
BUTTON_HOVER = (80, 80, 80)

VIEW_MODE_LABELS = {"default": "Default", "unified": "Unified", "dots": "Dots", "positive_bw": "Positive (b/w)", "negative_bw": "Negative (b/w)", "delta_bw": "Delta (b/w)"}
VIEW_MODE_ORDER = ("default", "unified", "dots", "positive_bw", "negative_bw", "delta_bw")


class ParamPanel:
    """State: params dict; draw and handle events. Save and Restart callbacks."""

    def __init__(
        self,
        rect: pygame.Rect,
        initial: dict,
        on_save: Callable[[], None],
        on_restart: Callable[[], None],
    ) -> None:
        self.rect = rect
        self.params = {
            "nx": initial.get("nx", 64),
            "ny": initial.get("ny", 64),
            "tick_rate": initial.get("tick_rate", 30),
            "fractal_strength": initial.get("fractal_strength", 0.3),
            "phi_decay": initial.get("phi_decay", 0.618),
            "edge_blend": initial.get("edge_blend", 0.7),
            "seed": initial.get("seed", -1),
            "view_mode": initial.get("view_mode", "default"),
            "lock_seed": initial.get("lock_seed", False),
            "render_scale": initial.get("render_scale", 1),
            "config_name": initial.get("config_name", ""),
            "unified_show_a": initial.get("unified_show_a", True),
            "unified_show_b": initial.get("unified_show_b", True),
            "unified_show_ab": initial.get("unified_show_ab", True),
            "boundary_consume": initial.get("boundary_consume", True),
            "legacy_diffusion": initial.get("legacy_diffusion", initial.get("retain_ratio") is not None),
            "retain_ratio": initial.get("retain_ratio", 0.5),
            "paused": True,
        }
        self.on_save = on_save
        self.on_restart = on_restart
        self.on_load_config = initial.get("on_load_config")
        self._selected_config: tuple[int, str] | None = initial.get("selected_config")
        self._font = None
        self._slider_rects: dict = {}
        self._button_rects: dict = {}
        self._dragging: str | None = None
        self._seed_focus = False
        self._seed_buffer = ""
        self._config_name_focus = False
        self._config_name_buffer = ""
        self._dropdown_expanded = False
        self._dropdown_option_rects: list[tuple[str, pygame.Rect]] = []
        self._config_dropdown_expanded = False
        self._config_dropdown_option_rects: list[tuple[tuple[int, str], pygame.Rect]] = []
        self._tooltip_rects: dict[str, pygame.Rect] = {}
        self._hover_tooltip_text = None
        self._small_font = None
        self._tooltip_font = None
        self._tooltip_small_font = None

    def _ensure_font(self) -> pygame.font.Font:
        if self._font is None:
            self._font = pygame.font.Font(None, FONT_SIZE)
        return self._font

    def _ensure_small_font(self) -> pygame.font.Font:
        if self._small_font is None:
            self._small_font = pygame.font.Font(None, 14)
        return self._small_font

    def _ensure_tooltip_fonts(self) -> tuple[pygame.font.Font, pygame.font.Font]:
        if self._tooltip_font is None:
            self._tooltip_font = pygame.font.Font(None, TOOLTIP_FONT_SIZE)
            self._tooltip_small_font = pygame.font.Font(None, TOOLTIP_SMALL_FONT_SIZE)
        return self._tooltip_font, self._tooltip_small_font

    def get_params(self) -> dict:
        return self.params.copy()

    def draw(self, surface: pygame.Surface, tick_count: int = 0, actual_used_seed: int | None = None) -> None:
        font = self._ensure_font()
        x, y = self.rect.x + 8, self.rect.y + 6
        line_h = 18
        gap = 4
        self._slider_rects.clear()
        self._button_rects.clear()

        slider_w = self.rect.width - 16 - 44  # leave 44px for value text
        slider_h = 12

        # Tick at top
        label = font.render(f"Tick: {tick_count}", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h + gap

        # Seed (text input): integer or -1 for random each run
        label = font.render("Seed", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h
        seed_rect = pygame.Rect(x, y, 140, 18)
        pygame.draw.rect(surface, SLIDER_COLOR, seed_rect)
        if self._seed_focus:
            display_str = self._seed_buffer if self._seed_buffer else ""
        else:
            if self.params["seed"] == -1 and actual_used_seed is not None:
                display_str = f"-1 ({actual_used_seed})"
            else:
                display_str = str(self.params["seed"])
        t = font.render(display_str[:20], True, LABEL_COLOR)
        surface.blit(t, (seed_rect.x + 4, seed_rect.y + 1))
        self._seed_rect = seed_rect
        y += 18 + gap

        # View mode dropdown
        label = font.render("View", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h
        drop_w, drop_h = 160, 18
        self._dropdown_rect = pygame.Rect(x, y, drop_w, drop_h)
        pygame.draw.rect(surface, SLIDER_COLOR, self._dropdown_rect)
        pygame.draw.polygon(surface, LABEL_COLOR, [(x + drop_w - 12, y + 4), (x + drop_w - 6, y + 4), (x + drop_w - 9, y + 11)])
        current = VIEW_MODE_LABELS.get(self.params["view_mode"], "Default")
        t = font.render(current, True, LABEL_COLOR)
        surface.blit(t, (self._dropdown_rect.x + 4, self._dropdown_rect.y + 2))
        # Unified layer toggles (A, B, A/B) only when view is Unified
        if self.params["view_mode"] == "unified":
            cx = x + drop_w + 10
            box_sz = 14
            for key, label in [("unified_show_a", "A"), ("unified_show_b", "B"), ("unified_show_ab", "A/B")]:
                box = pygame.Rect(cx, y + 2, box_sz, box_sz)
                if self.params[key]:
                    pygame.draw.rect(surface, KNOB_COLOR, box)
                else:
                    pygame.draw.rect(surface, SLIDER_COLOR, box)
                pygame.draw.rect(surface, LABEL_COLOR, box, 1)
                t = font.render(label, True, LABEL_COLOR)
                surface.blit(t, (cx + box_sz + 4, y + 2))
                self._button_rects[key] = box.union(pygame.Rect(cx, y, box_sz + 4 + font.size(label)[0], drop_h))
                cx = self._button_rects[key].right + 6
        y += drop_h + gap
        self._dropdown_option_rects.clear()
        if self._dropdown_expanded:
            for mode in VIEW_MODE_ORDER:
                opt_rect = pygame.Rect(x, y, drop_w, drop_h)
                if opt_rect.collidepoint(pygame.mouse.get_pos()):
                    pygame.draw.rect(surface, BUTTON_HOVER, opt_rect)
                else:
                    pygame.draw.rect(surface, BUTTON_COLOR, opt_rect)
                t = font.render(VIEW_MODE_LABELS[mode], True, LABEL_COLOR)
                surface.blit(t, (opt_rect.x + 4, opt_rect.y + 2))
                self._dropdown_option_rects.append((mode, opt_rect))
                y += drop_h + 1
        y += gap

        # World X
        label = font.render("World X", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h
        sr = _draw_slider(surface, x, y, slider_w, slider_h, self.params["nx"], 8, 128)
        _draw_slider_value(surface, font, x + slider_w + 4, y, str(self.params["nx"]))
        self._slider_rects["nx"] = (sr, 8, 128)
        y += slider_h + gap

        # World Y
        label = font.render("World Y", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h
        sr = _draw_slider(surface, x, y, slider_w, slider_h, self.params["ny"], 8, 128)
        _draw_slider_value(surface, font, x + slider_w + 4, y, str(self.params["ny"]))
        self._slider_rects["ny"] = (sr, 8, 128)
        y += slider_h + gap

        # Render scale
        label = font.render("Render scale", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h
        sr = _draw_slider(surface, x, y, slider_w, slider_h, self.params["render_scale"], 1, 4)
        _draw_slider_value(surface, font, x + slider_w + 4, y, str(self.params["render_scale"]))
        self._slider_rects["render_scale"] = (sr, 1, 4)
        y += slider_h + gap

        # Tick rate 1–60
        label = font.render("Tick rate (1–60)", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h
        sr = _draw_slider(surface, x, y, slider_w, slider_h, self.params["tick_rate"], 1, 60)
        _draw_slider_value(surface, font, x + slider_w + 4, y, str(self.params["tick_rate"]))
        self._slider_rects["tick_rate"] = (sr, 1, 60)
        y += slider_h + gap

        # Fractal strength %
        self._tooltip_rects.clear()
        fs_pct = max(0, min(100, int(self.params["fractal_strength"] * 100)))
        row_y = y
        label = font.render("Fractal strength %", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h
        sr = _draw_slider(surface, x, y, slider_w, slider_h, fs_pct, 0, 100)
        _draw_slider_value(surface, font, x + slider_w + 4, y, f"{fs_pct}%")
        self._slider_rects["fractal_strength"] = (sr, 0, 100)
        self._tooltip_rects["fractal_strength"] = pygame.Rect(x, row_y, self.rect.width - 16, line_h + slider_h + gap)
        y += slider_h + gap

        # Phi decay (Golden decay)
        pd_int = max(40, min(90, int(self.params["phi_decay"] * 100)))
        row_y = y
        label = font.render("Golden decay (×0.01)", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h
        sr = _draw_slider(surface, x, y, slider_w, slider_h, pd_int, 40, 90)
        _draw_slider_value(surface, font, x + slider_w + 4, y, f"0.{pd_int}")
        self._slider_rects["phi_decay"] = (sr, 40, 90)
        self._tooltip_rects["phi_decay"] = pygame.Rect(x, row_y, self.rect.width - 16, line_h + slider_h + gap)
        y += slider_h + gap

        # Edge blend %
        eb_pct = max(0, min(100, int(self.params["edge_blend"] * 100)))
        row_y = y
        label = font.render("Edge blend %", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h
        sr = _draw_slider(surface, x, y, slider_w, slider_h, eb_pct, 0, 100)
        _draw_slider_value(surface, font, x + slider_w + 4, y, f"{eb_pct}%")
        self._slider_rects["edge_blend"] = (sr, 0, 100)
        self._tooltip_rects["edge_blend"] = pygame.Rect(x, row_y, self.rect.width - 16, line_h + slider_h + gap)
        y += slider_h + gap

        # Boundary consume: checked = zero-pad (mass lost at edges, dramatic); unchecked = reflective (conserved)
        row_y = y
        box = pygame.Rect(x, y + 2, 14, 14)
        if self.params["boundary_consume"]:
            pygame.draw.rect(surface, KNOB_COLOR, box)
            pygame.draw.rect(surface, LABEL_COLOR, box, 1)
        else:
            pygame.draw.rect(surface, SLIDER_COLOR, box)
            pygame.draw.rect(surface, LABEL_COLOR, box, 1)
        t = font.render("Boundary consume", True, LABEL_COLOR)
        surface.blit(t, (x + 18, y + 2))
        self._button_rects["boundary_consume"] = box.union(pygame.Rect(x, y, 18 + font.size("Boundary consume")[0], 18))
        y += 18 + gap

        # Retain-ratio diffusion: when enabled, use retain_ratio + cardinal/diagonal weights and show slider
        row_y = y
        box = pygame.Rect(x, y + 2, 14, 14)
        if self.params["legacy_diffusion"]:
            pygame.draw.rect(surface, KNOB_COLOR, box)
            pygame.draw.rect(surface, LABEL_COLOR, box, 1)
        else:
            pygame.draw.rect(surface, SLIDER_COLOR, box)
            pygame.draw.rect(surface, LABEL_COLOR, box, 1)
        t = font.render("Retain-ratio diffusion", True, LABEL_COLOR)
        surface.blit(t, (x + 18, y + 2))
        self._button_rects["legacy_diffusion"] = box.union(pygame.Rect(x, y, 18 + font.size("Retain-ratio diffusion")[0], 18))
        y += 18 + gap
        if self.params["legacy_diffusion"]:
            rr_pct = max(1, min(99, int(self.params["retain_ratio"] * 100)))
            row_y = y
            label = font.render("Retain ratio %", True, LABEL_COLOR)
            surface.blit(label, (x, y))
            y += line_h
            sr = _draw_slider(surface, x, y, slider_w, slider_h, rr_pct, 1, 99)
            _draw_slider_value(surface, font, x + slider_w + 4, y, f"{rr_pct}%")
            self._slider_rects["retain_ratio"] = (sr, 1, 99)
            self._tooltip_rects["retain_ratio"] = pygame.Rect(x, row_y, self.rect.width - 16, line_h + slider_h + gap)
            y += slider_h + gap

        # Start / Pause / Resume and Restart (side by side)
        btn_h = 26
        pause_rect = pygame.Rect(x, y, 100, btn_h)
        color = BUTTON_HOVER if pause_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(surface, color, pause_rect)
        if self.params["paused"]:
            text = "Start" if tick_count == 0 else "Resume"
        else:
            text = "Pause"
        t = font.render(text, True, LABEL_COLOR)
        surface.blit(t, (pause_rect.x + 6, pause_rect.y + 4))
        self._button_rects["pause"] = pause_rect

        restart_rect = pygame.Rect(x + 104, y, 110, btn_h)
        color = BUTTON_HOVER if restart_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(surface, color, restart_rect)
        t = font.render("Restart", True, LABEL_COLOR)
        surface.blit(t, (restart_rect.x + 6, restart_rect.y + 4))
        self._button_rects["restart"] = restart_rect

        # Lock Seed toggle
        lock_x = x + 104 + 110 + 6
        lock_box = pygame.Rect(lock_x, y + 4, 14, 14)
        if self.params["lock_seed"]:
            pygame.draw.rect(surface, KNOB_COLOR, lock_box)
            pygame.draw.rect(surface, LABEL_COLOR, lock_box, 1)
        else:
            pygame.draw.rect(surface, SLIDER_COLOR, lock_box)
            pygame.draw.rect(surface, LABEL_COLOR, lock_box, 1)
        t = font.render("Lock Seed", True, LABEL_COLOR)
        surface.blit(t, (lock_x + 18, y + 4))
        self._button_rects["lock_seed"] = lock_box.union(pygame.Rect(lock_x, y, 88, btn_h))
        y += btn_h + gap

        # Config dropdown (saved configs: "Name (seed)")
        config_list = config.list_configs()
        label = font.render("Config", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h
        drop_w, drop_h = 200, 18
        self._config_dropdown_rect = pygame.Rect(x, y, drop_w, drop_h)
        pygame.draw.rect(surface, SLIDER_COLOR, self._config_dropdown_rect)
        pygame.draw.polygon(surface, LABEL_COLOR, [(x + drop_w - 12, y + 4), (x + drop_w - 6, y + 4), (x + drop_w - 9, y + 11)])
        if self._selected_config is not None:
            sel_name, sel_seed = self._selected_config[1], self._selected_config[0]
            current_config_str = f"{sel_name} ({sel_seed})"
        else:
            current_config_str = "—"
        t = font.render(current_config_str[:28], True, LABEL_COLOR)
        surface.blit(t, (self._config_dropdown_rect.x + 4, self._config_dropdown_rect.y + 2))
        y += drop_h + gap
        self._config_dropdown_option_rects.clear()
        if self._config_dropdown_expanded:
            for seed, name in config_list:
                opt_rect = pygame.Rect(x, y, drop_w, drop_h)
                if opt_rect.collidepoint(pygame.mouse.get_pos()):
                    pygame.draw.rect(surface, BUTTON_HOVER, opt_rect)
                else:
                    pygame.draw.rect(surface, BUTTON_COLOR, opt_rect)
                t = font.render(f"{name} ({seed})"[:28], True, LABEL_COLOR)
                surface.blit(t, (opt_rect.x + 4, opt_rect.y + 2))
                self._config_dropdown_option_rects.append(((seed, name), opt_rect))
                y += drop_h + 1
        y += gap

        # Name field, Save/Update config, and Delete config (when current config exists)
        self._last_actual_seed = actual_used_seed
        label = font.render("Name", True, LABEL_COLOR)
        surface.blit(label, (x, y))
        y += line_h
        name_w = 140
        self._config_name_rect = pygame.Rect(x, y, name_w, 18)
        pygame.draw.rect(surface, SLIDER_COLOR, self._config_name_rect)
        display_name = self._config_name_buffer if self._config_name_focus else (self.params.get("config_name") or "")
        t = font.render(display_name[:24], True, LABEL_COLOR)
        surface.blit(t, (self._config_name_rect.x + 4, self._config_name_rect.y + 1))
        effective_name = (self._config_name_buffer if self._config_name_focus else self.params.get("config_name") or "").strip()
        exists = actual_used_seed is not None and effective_name != "" and config.config_exists(actual_used_seed, effective_name)
        save_btn_w = 120
        btn_rect = pygame.Rect(x + name_w + 6, y, save_btn_w, btn_h)
        btn_text = "Update config" if exists else "Save config"
        color = BUTTON_HOVER if btn_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(surface, color, btn_rect)
        t = font.render(btn_text, True, LABEL_COLOR)
        surface.blit(t, (btn_rect.x + 6, btn_rect.y + 4))
        self._button_rects["save"] = btn_rect
        del_x = x + name_w + 6 + save_btn_w + 6
        if exists:
            del_rect = pygame.Rect(del_x, y, 90, btn_h)
            color = BUTTON_HOVER if del_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
            pygame.draw.rect(surface, color, del_rect)
            t = font.render("Delete config", True, LABEL_COLOR)
            surface.blit(t, (del_rect.x + 6, del_rect.y + 4))
            self._button_rects["delete_config"] = del_rect
        y += max(18, btn_h) + gap

    def update_hover_tooltip(self, pos: tuple[int, int]) -> None:
        self._hover_tooltip_text = None
        for key, r in self._tooltip_rects.items():
            if r.collidepoint(pos):
                self._hover_tooltip_text = tooltips.PARAM_TOOLTIPS.get(key)
                return

    def draw_tooltip(self, surface: pygame.Surface) -> None:
        tf, sf = self._ensure_tooltip_fonts()
        tooltips.draw_tooltip(surface, tf, sf, self._hover_tooltip_text, pygame.mouse.get_pos())

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if event was consumed."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if getattr(self, "_seed_rect", None) and self._seed_rect.collidepoint(event.pos):
                self._seed_focus = True
                self._seed_buffer = str(self.params["seed"])
                if self._config_name_focus:
                    self._apply_config_name_buffer()
                self._config_name_focus = False
                return True
            if getattr(self, "_config_name_rect", None) and self._config_name_rect.collidepoint(event.pos):
                self._config_name_focus = True
                self._config_name_buffer = self.params.get("config_name") or ""
                if self._seed_focus:
                    self._parse_seed_buffer()
                self._seed_focus = False
                return True
            if self._seed_focus:
                self._parse_seed_buffer()
            self._seed_focus = False
            if self._config_name_focus:
                self._apply_config_name_buffer()
            self._config_name_focus = False
            if getattr(self, "_dropdown_rect", None) and self._dropdown_rect.collidepoint(event.pos):
                self._dropdown_expanded = not self._dropdown_expanded
                self._config_dropdown_expanded = False
                return True
            for mode, opt_rect in self._dropdown_option_rects:
                if opt_rect.collidepoint(event.pos):
                    self.params["view_mode"] = mode
                    self._dropdown_expanded = False
                    return True
            self._dropdown_expanded = False
            if getattr(self, "_config_dropdown_rect", None) and self._config_dropdown_rect.collidepoint(event.pos):
                self._config_dropdown_expanded = not self._config_dropdown_expanded
                return True
            for (seed, name), opt_rect in self._config_dropdown_option_rects:
                if opt_rect.collidepoint(event.pos):
                    self._selected_config = (seed, name)
                    self._config_dropdown_expanded = False
                    if self.on_load_config:
                        self.on_load_config(seed, name)
                    return True
            self._config_dropdown_expanded = False
            for key, (slider_rect, lo, hi) in self._slider_rects.items():
                if slider_rect.collidepoint(event.pos):
                    self._dragging = key
                    self._set_slider_value(key, event.pos, slider_rect, lo, hi)
                    return True
            for key, btn_rect in self._button_rects.items():
                if btn_rect.collidepoint(event.pos):
                    if key == "pause":
                        self.params["paused"] = not self.params["paused"]
                    elif key == "restart":
                        self.on_restart()
                    elif key == "lock_seed":
                        self.params["lock_seed"] = not self.params["lock_seed"]
                    elif key == "boundary_consume":
                        self.params["boundary_consume"] = not self.params["boundary_consume"]
                    elif key == "legacy_diffusion":
                        self.params["legacy_diffusion"] = not self.params["legacy_diffusion"]
                    elif key == "save":
                        self.on_save()
                    elif key in ("unified_show_a", "unified_show_b", "unified_show_ab"):
                        self.params[key] = not self.params[key]
                    elif key == "delete_config":
                        if self._config_name_focus:
                            self._apply_config_name_buffer()
                        effective = (self.params.get("config_name") or "").strip() or "unnamed"
                        seed = getattr(self, "_last_actual_seed", None)
                        if seed is not None:
                            key_deleted = (seed, config._sanitize_name(effective) or "unnamed")
                            config.delete_config(seed, effective)
                            if self._selected_config == key_deleted:
                                self._selected_config = None
                    return True
            return False
        if event.type == pygame.KEYDOWN:
            if self._seed_focus:
                if event.key == pygame.K_RETURN:
                    self._seed_focus = False
                    self._parse_seed_buffer()
                    return True
                if event.key == pygame.K_BACKSPACE:
                    self._seed_buffer = self._seed_buffer[:-1]
                    return True
                if event.unicode and (event.unicode.isdigit() or (event.unicode == "-" and not self._seed_buffer)):
                    self._seed_buffer += event.unicode
                    return True
                return True
            if self._config_name_focus:
                if event.key == pygame.K_RETURN:
                    self._config_name_focus = False
                    self._apply_config_name_buffer()
                    return True
                if event.key == pygame.K_BACKSPACE:
                    self._config_name_buffer = self._config_name_buffer[:-1]
                    return True
                if event.unicode and len(self._config_name_buffer) < 48:
                    self._config_name_buffer += event.unicode
                    return True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self._dragging = None
        elif event.type == pygame.MOUSEMOTION:
            self.update_hover_tooltip(event.pos)
            if self._dragging is not None:
                if self._dragging in self._slider_rects:
                    sr, lo, hi = self._slider_rects[self._dragging]
                    self._set_slider_value(self._dragging, event.pos, sr, lo, hi)
                return True
        return False

    def _parse_seed_buffer(self) -> None:
        s = self._seed_buffer.strip()
        if not s:
            return
        try:
            self.params["seed"] = int(s)
        except ValueError:
            pass
        self._seed_buffer = ""

    def _apply_config_name_buffer(self) -> None:
        self.params["config_name"] = self._config_name_buffer.strip()[:64]
        self._config_name_buffer = ""

    def apply_config(self, cfg: dict) -> None:
        """Load a config dict into panel params (e.g. after loading a saved config)."""
        world = cfg.get("world", {})
        self.params["nx"] = world.get("nx", self.params["nx"])
        self.params["ny"] = world.get("ny", self.params["ny"])
        self.params["tick_rate"] = cfg.get("tick_rate", self.params["tick_rate"])
        self.params["fractal_strength"] = cfg.get("fractal_strength", self.params["fractal_strength"])
        self.params["phi_decay"] = cfg.get("phi_decay", self.params["phi_decay"])
        self.params["edge_blend"] = cfg.get("edge_blend", self.params["edge_blend"])
        self.params["boundary_consume"] = cfg.get("boundary_consume", True)
        self.params["legacy_diffusion"] = cfg.get("legacy_diffusion", cfg.get("retain_ratio") is not None)
        self.params["retain_ratio"] = cfg.get("retain_ratio", 0.5)
        self.params["seed"] = cfg.get("seed", self.params["seed"])
        self.params["lock_seed"] = cfg.get("lock_seed", self.params["lock_seed"])
        self.params["render_scale"] = cfg.get("render_scale", self.params["render_scale"])
        vm = cfg.get("view_mode", self.params["view_mode"])
        if vm == "visual":
            vm = "unified"
            self.params["unified_show_a"] = False
            self.params["unified_show_b"] = False
            self.params["unified_show_ab"] = True
        self.params["view_mode"] = vm
        if "actual_seed_used" in cfg:
            self.params["seed"] = cfg["actual_seed_used"]
        if "config_name" in cfg:
            self.params["config_name"] = cfg["config_name"]
        for k in ("unified_show_a", "unified_show_b", "unified_show_ab"):
            if k in cfg:
                self.params[k] = cfg[k]

    def set_selected_config(self, seed: int, name: str) -> None:
        """Called after save so dropdown shows the current config."""
        self._selected_config = (seed, name)

    def _set_slider_value(self, key: str, pos: tuple[int, int], slider_rect: pygame.Rect, lo: int, hi: int) -> None:
        t = (pos[0] - slider_rect.x) / max(1, slider_rect.width - 8)
        t = max(0, min(1, t))
        val = int(lo + t * (hi - lo))
        if key == "fractal_strength":
            self.params[key] = val / 100.0
        elif key == "retain_ratio":
            self.params[key] = val / 100.0
        elif key == "phi_decay":
            self.params[key] = val / 100.0
        elif key == "edge_blend":
            self.params[key] = val / 100.0
        else:
            self.params[key] = val
        if key in ("nx", "ny"):
            self.params[key] = max(4, self.params[key])

    def _config_dict(self) -> dict:
        out = {
            "world": {"nx": self.params["nx"], "ny": self.params["ny"]},
            "tick_rate": self.params["tick_rate"],
            "fractal_strength": self.params["fractal_strength"],
            "phi_decay": self.params["phi_decay"],
            "edge_blend": self.params["edge_blend"],
            "boundary_consume": self.params["boundary_consume"],
            "legacy_diffusion": self.params["legacy_diffusion"],
            "seed": self.params["seed"],
            "lock_seed": self.params["lock_seed"],
            "render_scale": self.params["render_scale"],
            "view_mode": self.params.get("view_mode", "default"),
            "unified_show_a": self.params.get("unified_show_a", True),
            "unified_show_b": self.params.get("unified_show_b", True),
            "unified_show_ab": self.params.get("unified_show_ab", True),
            "initial_pos": [self.params["nx"] // 2, self.params["ny"] // 2],
            "initial_neg": [self.params["nx"] // 2 - 1, self.params["ny"] // 2],
        }
        if self.params.get("legacy_diffusion"):
            out["retain_ratio"] = self.params["retain_ratio"]
        return out


def _draw_slider(
    surface: pygame.Surface, x: int, y: int, w: int, h: int, value: int, vmin: int, vmax: int
) -> pygame.Rect:
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, SLIDER_COLOR, rect)
    t = (value - vmin) / max(1, vmax - vmin)
    knob_x = x + 4 + int(t * (w - 8))
    pygame.draw.rect(surface, KNOB_COLOR, (knob_x, y, 8, h))
    return rect


def _draw_slider_value(
    surface: pygame.Surface, font: pygame.font.Font, x: int, y: int, value_str: str
) -> None:
    text = font.render(value_str, True, LABEL_COLOR)
    surface.blit(text, (x, y))
