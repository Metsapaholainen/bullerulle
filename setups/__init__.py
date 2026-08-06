from setups import (
    squeeze,
    pullback,
    breakout,
    episodic_pivot,
    parabolic_short,
    cup_with_handle,
    double_bottom,
    flat_base,
    ascending_base,
    high_tight_flag,
    momentum_burst,
    vcp,
    spring,
)
from setups.explanations import PATTERN_EXPLANATIONS

# Registry used by scanner.py, backtest/*, and app.py so adding a new setup
# later only means adding one entry here. Squeeze/Pullback are listed FIRST
# so every list(SETUP_REGISTRY.keys())-driven selectbox in app.py defaults
# to Squeeze with no extra index= plumbing needed. "tier" distinguishes the
# two primary, currently-recommended systems from the 11 older setups kept
# for reference (measured negative, too small a sample, or -- Breakout --
# real edge but chases an already-confirmed move); see app.py for how tier
# drives the Scanner/Pattern-library UI.
SETUP_REGISTRY = {
    "squeeze": {
        "label": "Squeeze",
        "module": squeeze,
        "settings_attr": "squeeze",
        "side": "long",
        "tier": "primary",
    },
    "pullback": {
        "label": "Pullback",
        "module": pullback,
        "settings_attr": "pullback",
        "side": "long",
        "tier": "primary",
    },
    "breakout": {
        "label": "Breakout",
        "module": breakout,
        "settings_attr": "breakout",
        "side": "long",
        "tier": "secondary",
    },
    "episodic_pivot": {
        "label": "Episodic Pivot",
        "module": episodic_pivot,
        "settings_attr": "episodic_pivot",
        "side": "long",
        "tier": "secondary",
    },
    "parabolic_short": {
        "label": "Parabolic Short",
        "module": parabolic_short,
        "settings_attr": "parabolic_short",
        "side": "short",
        "tier": "secondary",
    },
    "cup_with_handle": {
        "label": "Cup with Handle (O'Neil)",
        "module": cup_with_handle,
        "settings_attr": "cup_with_handle",
        "side": "long",
        "tier": "secondary",
    },
    "double_bottom": {
        "label": "Double Bottom (O'Neil)",
        "module": double_bottom,
        "settings_attr": "double_bottom",
        "side": "long",
        "tier": "secondary",
    },
    "flat_base": {
        "label": "Flat Base (O'Neil)",
        "module": flat_base,
        "settings_attr": "flat_base",
        "side": "long",
        "tier": "secondary",
    },
    "ascending_base": {
        "label": "Ascending Base (O'Neil)",
        "module": ascending_base,
        "settings_attr": "ascending_base",
        "side": "long",
        "tier": "secondary",
    },
    "high_tight_flag": {
        "label": "High Tight Flag (O'Neil)",
        "module": high_tight_flag,
        "settings_attr": "high_tight_flag",
        "side": "long",
        "tier": "secondary",
    },
    "momentum_burst": {
        "label": "Momentum Burst (Stockbee)",
        "module": momentum_burst,
        "settings_attr": "momentum_burst",
        "side": "long",
        "tier": "secondary",
    },
    "vcp": {
        "label": "VCP (Minervini)",
        "module": vcp,
        "settings_attr": "vcp",
        "side": "long",
        "tier": "secondary",
    },
    "spring": {
        "label": "Spring (Wyckoff)",
        "module": spring,
        "settings_attr": "spring",
        "side": "long",
        "tier": "secondary",
    },
}
