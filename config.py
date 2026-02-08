"""Load/save simulation and UI parameters. Configs live in configs/ as {seed}_{name}.json (+ optional .npz state)."""

import json
import re
from pathlib import Path

import numpy as np

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
LAST_FILE = CONFIG_DIR / "last.txt"

# In-memory index of (seed, name) so we avoid disk access for exists/dropdown.
_CONFIG_INDEX: set[tuple[int, str]] = set()


def refresh_index() -> None:
    """Rebuild _CONFIG_INDEX from disk. Call at startup and after external changes."""
    global _CONFIG_INDEX
    _CONFIG_INDEX = set()
    if not CONFIG_DIR.exists():
        return
    for f in CONFIG_DIR.glob("*.json"):
        if f.name.endswith(".json"):
            stem = f.stem
            if "_" not in stem:
                continue
            first, rest = stem.split("_", 1)
            try:
                _CONFIG_INDEX.add((int(first), rest))
            except ValueError:
                continue


def _sanitize_name(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s-]+", "_", s).strip("_")
    return s[:64] or "unnamed"


def config_id(seed: int, name: str) -> str:
    return f"{seed}_{_sanitize_name(name)}"


def get_config_path(seed: int, name: str) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR / f"{config_id(seed, name)}.json"


def get_state_path(seed: int, name: str) -> Path:
    return CONFIG_DIR / f"{config_id(seed, name)}.npz"


def list_configs() -> list[tuple[int, str]]:
    """Return (seed, name) for each saved config, from in-memory index."""
    return sorted(_CONFIG_INDEX, key=lambda x: (x[1].lower(), x[0]))


def get_last_config() -> tuple[int, str] | None:
    if not LAST_FILE.exists():
        return None
    try:
        raw = LAST_FILE.read_text().strip()
        if "_" not in raw:
            return None
        first, rest = raw.split("_", 1)
        return (int(first), rest)
    except (ValueError, OSError):
        return None


def set_last_config(seed: int, name: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LAST_FILE.write_text(config_id(seed, name))


def load_config(path: Path | str | None = None) -> dict:
    if path is not None:
        p = Path(path)
        if not p.exists():
            return _default_config()
        with open(p, "r") as f:
            return _merge_defaults(json.load(f))
    last = get_last_config()
    if last is None:
        return _default_config()
    p = get_config_path(last[0], last[1])
    if not p.exists():
        return _default_config()
    with open(p, "r") as f:
        return _merge_defaults(json.load(f))


def save_config(
    params: dict,
    actual_seed: int,
    name: str,
    tick_count: int = 0,
    state: dict | None = None,
) -> None:
    """Save config and optional state. actual_seed used for filename; name from UI."""
    cid = _sanitize_name(name) or "unnamed"
    path = get_config_path(actual_seed, name)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = {**params, "actual_seed_used": actual_seed, "tick_count": tick_count}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    if state is not None:
        np.savez_compressed(
            get_state_path(actual_seed, name),
            positive=state["positive"],
            negative=state["negative"],
            tick_count=np.int64(state["tick_count"]),
        )
    set_last_config(actual_seed, name)
    _CONFIG_INDEX.add((actual_seed, _sanitize_name(name) or "unnamed"))


def load_state(seed: int, name: str) -> dict | None:
    """Return {'positive': ndarray, 'negative': ndarray, 'tick_count': int} or None."""
    p = get_state_path(seed, name)
    if not p.exists():
        return None
    try:
        data = np.load(p, allow_pickle=False)
        return {
            "positive": data["positive"].copy(),
            "negative": data["negative"].copy(),
            "tick_count": int(data["tick_count"]),
        }
    except (KeyError, OSError, ValueError):
        return None


def _default_config() -> dict:
    return {
        "world": {"nx": 64, "ny": 64},
        "tick_rate": 30,
        "retain_ratio": 0.5,
        "fractal_strength": 0.3,
        "phi_decay": 0.618,
        "edge_blend": 0.7,
        "seed": -1,
        "lock_seed": False,
        "render_scale": 1,
        "view_mode": "default",
        "unified_show_a": True,
        "unified_show_b": True,
        "unified_show_ab": True,
        "initial_pos": [32, 32],
        "initial_neg": [31, 32],
    }


def _merge_defaults(data: dict) -> dict:
    d = _default_config()
    if "world" in data:
        d["world"] = {**d["world"], **data["world"]}
    for k in (
        "tick_rate", "retain_ratio", "fractal_strength", "phi_decay", "edge_blend",
        "seed", "lock_seed", "actual_seed_used", "render_scale", "view_mode",
        "unified_show_a", "unified_show_b", "unified_show_ab",
        "initial_pos", "initial_neg", "tick_count",
    ):
        if k in data:
            d[k] = data[k]
    return d


def config_exists(actual_seed: int, name: str) -> bool:
    """Use in-memory index; no disk access."""
    key = (actual_seed, _sanitize_name(name) or "unnamed")
    return key in _CONFIG_INDEX


def delete_config(seed: int, name: str) -> None:
    """Remove config and state from disk and index. Clear last if this was last."""
    key = (seed, _sanitize_name(name) or "unnamed")
    _CONFIG_INDEX.discard(key)
    p = get_config_path(seed, name)
    if p.exists():
        p.unlink(missing_ok=True)
    sp = get_state_path(seed, name)
    if sp.exists():
        sp.unlink(missing_ok=True)
    last = get_last_config()
    if last and last == key:
        if LAST_FILE.exists():
            LAST_FILE.unlink(missing_ok=True)
