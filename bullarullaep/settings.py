"""Settings for BullaRullaEP -- forked from the parent project's
`EpisodicPivotSettings` (settings.py:119-152) and extended with the
catalyst-classifier and social-sentiment fields this project adds.

Reuses `UniverseSettings`/`BacktestSettings` from the parent `settings.py`
rather than redefining them -- universe filtering and backtest/exit
mechanics don't need reinventing for an EP-only project.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from settings import BacktestSettings, UniverseSettings

DEFAULT_EP_SETTINGS_PATH = Path(__file__).parent / "config" / "default_ep_settings.yaml"


@dataclass
class EPSettings:
    enabled: bool = True
    min_gap_pct: float = 10.0
    # Raised from the parent project's EpisodicPivotSettings default of 2.0:
    # research (Qullamaggie/Stockbee sources) says ideal EPs trade multiples
    # of 10x average, and an EOD full-day ratio is a much weaker proxy for
    # that than a real premarket read -- a full day still doing 3x average
    # is already a meaningfully rarer, stronger day. Pass 2's premarket
    # confirmation (scan_confirm.py) is the sharper, more honest test of
    # "huge volume fast"; this hard gate is just the first, coarser cut.
    min_volume_ratio: float = 3.0
    volume_avg_period: int = 50
    lookback_base_days: int = 20
    max_prior_range_pct: float = 20.0
    min_price: float = 1.0
    # "Best EPs are on stocks that have gone sideways for 3-6 months or
    # more" -- the neglect-period check, a longer/coarser window than
    # lookback_base_days' tight-basing check above.
    quiet_base_lookback_days: int = 100
    max_quiet_base_run_pct: float = 40.0
    # "Avoid stocks that have already made a big move from a previous EP" --
    # soft penalty (halves score), not a hard exclude.
    prior_ep_lookback_days: int = 252
    prior_ep_min_gap_pct: float = 8.0
    max_prior_ep_count: int = 0
    # Growth-quality scoring/gating, from point-in-time fundamentals.
    min_growth_pct_floor: float = 25.0
    ideal_growth_pct: float = 100.0
    require_growth_floor: bool = True
    min_eps_beat_pct_for_bonus: float = 20.0
    require_eps_beat: bool = False
    # Backward-looking catalyst confirmation: does this gap coincide with a
    # just-reported earnings catalyst? 1 = today or yesterday (covers both
    # a before-market-open report today and an after-market-close report
    # yesterday). See data/fundamentals_cache.py::get_earnings_dates_window.
    catalyst_earnings_window_days: int = 1
    # LLM catalyst classifier (catalyst_classifier.py) -- a scoring bonus,
    # never a hard gate (a misclassification should never silently kill a
    # real trade -- same philosophy the parent project already uses for
    # growth/EPS-beat).
    enable_catalyst_classifier: bool = True
    min_catalyst_confidence: float = 0.5
    catalyst_bonus_weight: float = 2.0
    # Exploratory social-sentiment chatter-spike signal. OFF by default:
    # FMP's /historical/social-sentiment endpoint returned 403 Forbidden on
    # this project's current plan tier when checked live during build --
    # confirmed unavailable, not just unverified. Leave off until/unless
    # the plan is upgraded; the plumbing (data/fmp_client.py::social_sentiment)
    # is otherwise already there.
    enable_social_sentiment: bool = False
    social_sentiment_spike_threshold: float = 2.0  # multiple of the symbol's own trailing baseline


def _ep_universe_defaults() -> UniverseSettings:
    """EP-specific universe defaults, looser than the parent project's
    swing-trading defaults (settings.py:19-35) in the direction "neglect"
    actually points: institutionally-ignored stocks skew smaller and
    thinner than the parent's $300M/$20M-dollar-volume floors assume. Still
    keeps a real floor so a match is actually tradeable/exitable."""
    return UniverseSettings(
        min_price=2.0,
        min_market_cap=50_000_000,
        max_market_cap=0,  # no upper bound -- a big-cap surprise beat is still a real EP
        min_avg_dollar_volume=3_000_000,
        max_symbols=6000,
    )


@dataclass
class EPConfig:
    universe: UniverseSettings = field(default_factory=_ep_universe_defaults)
    ep: EPSettings = field(default_factory=EPSettings)
    backtest: BacktestSettings = field(default_factory=lambda: BacktestSettings(
        # The user explicitly wants "stop buy WITH LIMIT" -- the parent
        # project's own default (max_gap_fill_pct=0.0) disables the limit
        # cap entirely, so this needs a real nonzero starting value here.
        # Treat as a tunable starting point, to be calibrated once real
        # backtest results exist, not a final answer.
        max_gap_fill_pct=2.5,
    ))

    @classmethod
    def from_dict(cls, d: dict) -> "EPConfig":
        return cls(
            universe=UniverseSettings(**d.get("universe", {})),
            ep=EPSettings(**d.get("ep", {})),
            backtest=BacktestSettings(**d.get("backtest", {})),
        )

    def to_dict(self) -> dict:
        return {"universe": asdict(self.universe), "ep": asdict(self.ep), "backtest": asdict(self.backtest)}

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EPConfig":
        path = Path(path) if path else DEFAULT_EP_SETTINGS_PATH
        if not path.exists():
            return cls()
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def save(self, path: Optional[Path] = None) -> None:
        path = Path(path) if path else DEFAULT_EP_SETTINGS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
