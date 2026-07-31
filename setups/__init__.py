from setups import (
    breakout,
    episodic_pivot,
    parabolic_short,
    cup_with_handle,
    double_bottom,
    flat_base,
    ascending_base,
    high_tight_flag,
)
from setups.explanations import PATTERN_EXPLANATIONS

# Registry used by scanner.py, backtest/*, and app.py so adding a new setup
# later only means adding one entry here.
SETUP_REGISTRY = {
    "breakout": {
        "label": "Breakout",
        "module": breakout,
        "settings_attr": "breakout",
        "side": "long",
    },
    "episodic_pivot": {
        "label": "Episodic Pivot",
        "module": episodic_pivot,
        "settings_attr": "episodic_pivot",
        "side": "long",
    },
    "parabolic_short": {
        "label": "Parabolic Short",
        "module": parabolic_short,
        "settings_attr": "parabolic_short",
        "side": "short",
    },
    "cup_with_handle": {
        "label": "Cup with Handle (O'Neil)",
        "module": cup_with_handle,
        "settings_attr": "cup_with_handle",
        "side": "long",
    },
    "double_bottom": {
        "label": "Double Bottom (O'Neil)",
        "module": double_bottom,
        "settings_attr": "double_bottom",
        "side": "long",
    },
    "flat_base": {
        "label": "Flat Base (O'Neil)",
        "module": flat_base,
        "settings_attr": "flat_base",
        "side": "long",
    },
    "ascending_base": {
        "label": "Ascending Base (O'Neil)",
        "module": ascending_base,
        "settings_attr": "ascending_base",
        "side": "long",
    },
    "high_tight_flag": {
        "label": "High Tight Flag (O'Neil)",
        "module": high_tight_flag,
        "settings_attr": "high_tight_flag",
        "side": "long",
    },
}
