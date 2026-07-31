"""Named parameter presets: save/load/list/delete/compare full Settings
snapshots, so you can design a system by trial (tweak, save as "v2", tweak
again, compare "v1" vs "v2" vs "aggressive" side by side) rather than
hand-editing YAML each time."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from settings import Settings, DEFAULT_SETTINGS_PATH

PRESETS_DIR = Path(__file__).parent / "presets_store"


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


def compare_presets(names: list) -> pd.DataFrame:
    """Flatten each named preset's settings into one row for a side-by-side
    comparison table (used by the Designer/Settings tabs)."""
    rows = []
    for name in names:
        settings = load_preset(name)
        flat = {"preset": name}
        for section, section_obj in settings.to_dict().items():
            for field_name, value in section_obj.items():
                flat[f"{section}.{field_name}"] = value
        rows.append(flat)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("preset").T
