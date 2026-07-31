"""Named parameter presets: save/load/list/delete/compare full Settings
snapshots, so you can design a system by trial (tweak, save as "v2", tweak
again, compare "v1" vs "v2" vs "aggressive" side by side) rather than
hand-editing YAML each time."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from settings import Settings, DEFAULT_SETTINGS_PATH

PRESETS_DIR = Path(__file__).parent / "presets_store"
BUILTIN_PRESETS_DIR = Path(__file__).parent / "config"


def _preset_path(name: str) -> Path:
    safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_", " ")).strip().replace(" ", "_")
    return PRESETS_DIR / f"{safe_name}.yaml"


def list_presets() -> list:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def save_preset(name: str, settings: Settings) -> Path:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    path = _preset_path(name)
    settings.save(path)
    return path


def load_preset(name: str) -> Settings:
    return Settings.load(_preset_path(name))


def delete_preset(name: str) -> None:
    path = _preset_path(name)
    if path.exists():
        path.unlink()


def load_default() -> Settings:
    return Settings.load(DEFAULT_SETTINGS_PATH)


def list_builtin_presets() -> list:
    """Built-in presets ship in config/ (unlike user presets in the
    gitignored presets_store/), so they're always present after a fresh
    clone/deploy. Named preset_<name>.yaml; returns just <name>."""
    return sorted(
        p.stem.replace("preset_", "", 1) for p in BUILTIN_PRESETS_DIR.glob("preset_*.yaml")
    )


def load_builtin_preset(name: str) -> Settings:
    return Settings.load(BUILTIN_PRESETS_DIR / f"preset_{name}.yaml")


def _resolve_preset(name: str) -> Settings:
    """Accepts either a plain user-preset name or a built-in's display
    name (the "⭐ " prefix app.py uses in its selectbox)."""
    if name.startswith("⭐ "):
        return load_builtin_preset(name[2:])
    return load_preset(name)


def compare_presets(names: list) -> pd.DataFrame:
    """Flatten each named preset's settings into one row for a side-by-side
    comparison table (used by the Designer/Settings tabs). `names` may mix
    user-preset names and "⭐ "-prefixed built-in names."""
    rows = []
    for name in names:
        settings = _resolve_preset(name)
        flat = {"preset": name}
        for section, section_obj in settings.to_dict().items():
            for field_name, value in section_obj.items():
                flat[f"{section}.{field_name}"] = value
        rows.append(flat)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("preset").T
