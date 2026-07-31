"""Typed settings loaded from config/default_settings.yaml (or a saved preset).

Every threshold used anywhere in the scanner/backtester flows through this
module so the Streamlit app, the CLI, and the reverse parameter finder all
share one schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import copy

import yaml

DEFAULT_SETTINGS_PATH = Path(__file__).parent / "config" / "default_settings.yaml"


@dataclass
class UniverseSettings:
    min_price: float = 1.0
    # Default bounds scope the universe to small/mid-caps -- large/mega-caps
    # rarely produce the big % moves these setups are looking for. Set
    # max_market_cap to 0 to remove the upper bound entirely.
    min_market_cap: float = 300_000_000
    max_market_cap: float = 10_000_000_000
    min_avg_dollar_volume: float = 20_000_000  # "strongest liquidity" floor
    exchanges: list = field(default_factory=lambda: ["NASDAQ", "NYSE", "AMEX"])
    exclude_etf: bool = True
    exclude_funds: bool = True
    max_symbols: int = 1500
    # Multi-timeframe momentum pre-filter: before pattern-scanning the whole
    # market, narrow down to the top `momentum_top_pct`% movers (by trailing
    # return) in any of these lookback windows -- ~1/3/6/12/18 months.
    momentum_timeframes_days: list = field(default_factory=lambda: [21, 63, 126, 252, 378])
    momentum_top_pct: float = 2.0


@dataclass
class RSRatingSettings:
    lookback_periods: list = field(default_factory=lambda: [63, 126, 189, 252])
    period_weights: list = field(default_factory=lambda: [0.4, 0.2, 0.2, 0.2])


@dataclass
class BreakoutSettings:
    enabled: bool = True
    min_adr_pct: float = 4.0
    adr_lookback_days: int = 20
    prior_move_lookback_days: int = 63
    prior_move_min_pct: float = 30.0
    prior_move_max_pct: float = 100.0
    min_consolidation_days: int = 10
    max_consolidation_days: int = 40
    max_consolidation_range_pct: float = 25.0
    require_above_ema10: bool = True
    require_above_ema20: bool = True
    ema_fast_period: int = 10
    ema_slow_period: int = 20
    breakout_volume_ratio_min: float = 1.5
    volume_avg_period: int = 50
    min_rs_rating: float = 90.0


@dataclass
class EpisodicPivotSettings:
    enabled: bool = True
    min_gap_pct: float = 10.0
    min_volume_ratio: float = 2.0
    volume_avg_period: int = 50
    lookback_base_days: int = 20
    max_prior_range_pct: float = 20.0
    min_price: float = 1.0


@dataclass
class ParabolicShortSettings:
    enabled: bool = True
    ema_period: int = 20
    min_extension_adr_multiple: float = 3.0
    min_run_up_pct: float = 100.0
    run_up_lookback_days: int = 20
    consecutive_up_days_min: int = 3


@dataclass
class CupWithHandleSettings:
    enabled: bool = True
    cup_lookback_days: int = 130          # ~26 weeks, the window searched for the cup
    left_peak_days: int = 40              # portion of that window treated as "the left side high"
    handle_lookback_days: int = 15        # ~3 weeks, the most recent sub-window treated as the handle
    min_cup_depth_pct: float = 12.0
    max_cup_depth_pct: float = 35.0
    max_handle_depth_pct: float = 15.0
    min_recovery_pct: float = 85.0        # handle's peak must recover to at least this % of the left high
    handle_upper_half_only: bool = True   # handle's low must stay above the cup's midpoint
    prior_uptrend_lookback_days: int = 60
    min_prior_uptrend_pct: float = 30.0
    breakout_volume_ratio_min: float = 1.4
    volume_avg_period: int = 50


@dataclass
class DoubleBottomSettings:
    enabled: bool = True
    pattern_lookback_days: int = 50       # ~10 weeks, split into three segments: low1 / peak / low2
    min_depth_pct: float = 10.0
    max_depth_pct: float = 40.0
    max_low_difference_pct: float = 8.0   # how far the second low may differ from the first
    breakout_volume_ratio_min: float = 1.4
    volume_avg_period: int = 50


@dataclass
class FlatBaseSettings:
    enabled: bool = True
    base_days: int = 25                   # ~5 weeks, O'Neil's minimum flat-base length
    max_range_pct: float = 15.0
    prior_move_lookback_days: int = 40
    min_prior_move_pct: float = 20.0
    breakout_volume_ratio_min: float = 1.4
    volume_avg_period: int = 50


@dataclass
class AscendingBaseSettings:
    enabled: bool = True
    pattern_lookback_days: int = 70       # ~14 weeks across 3 segments (a staircase of pullbacks)
    min_segment_depth_pct: float = 5.0
    max_segment_depth_pct: float = 25.0
    breakout_volume_ratio_min: float = 1.4
    volume_avg_period: int = 50


@dataclass
class HighTightFlagSettings:
    enabled: bool = True
    run_up_lookback_days: int = 30        # ~4-8 weeks for the prior explosive move
    min_run_up_pct: float = 100.0
    flag_lookback_days: int = 20          # ~3-5 weeks for the tight flag/consolidation
    max_flag_depth_pct: float = 25.0
    breakout_volume_ratio_min: float = 1.4
    volume_avg_period: int = 50


@dataclass
class BacktestSettings:
    initial_capital: float = 100_000
    risk_pct_per_trade: float = 1.0
    commission_per_share: float = 0.0
    slippage_pct: float = 0.1
    # Initial stop placement. "adr_multiple": entry_price -/+ ADR% * stop_adr_multiple
    # (the original heuristic). "low_of_signal_day": the signal day's actual low
    # (high for shorts), capped so it's never more than stop_adr_multiple ADRs away --
    # this is how Qullamaggie himself describes setting stops ("set stop at low of
    # day... stop should be no more than the ATR").
    stop_mode: str = "low_of_signal_day"
    stop_adr_multiple: float = 1.0
    partial_profit_r_multiple: float = 1.0
    partial_profit_fraction: float = 0.5
    # Fallback partial-profit trigger by time elapsed, independent of whether
    # the R-multiple target has been hit -- "sell 1/3 to 1/2 of the position
    # after 3-5 days, then move the stop to break even." Whichever of the R
    # target or this day count is hit first takes the partial. 0 disables.
    partial_profit_max_days: int = 5
    # After the partial is taken, move the stop on the remainder to breakeven
    # (entry price) -- "take 10-30% profit after 1-3 days, then move stop-loss
    # to breakeven."
    move_stop_to_breakeven_after_partial: bool = True
    trail_ema_period: int = 10  # Qullamaggie: "use 10-day SMA as trailing stop"
    trail_ma_type: str = "sma"  # "sma" | "ema"
    max_positions: int = 10
    # Never risk more than this % of equity on one position ("don't put more
    # than 20% of your account into any one share"), regardless of what the
    # risk-based share count would otherwise imply.
    max_position_pct_of_equity: float = 20.0
    # Skip a signal if the signal day's own range already exceeds this many
    # ADRs -- an anti-chase filter ("if price change on day is more than the
    # ATR, skip it"). Set to 0 to disable.
    avoid_chase_adr_multiple: float = 1.5
    entry_delay_days: int = 1


@dataclass
class Settings:
    universe: UniverseSettings = field(default_factory=UniverseSettings)
    rs_rating: RSRatingSettings = field(default_factory=RSRatingSettings)
    breakout: BreakoutSettings = field(default_factory=BreakoutSettings)
    episodic_pivot: EpisodicPivotSettings = field(default_factory=EpisodicPivotSettings)
    parabolic_short: ParabolicShortSettings = field(default_factory=ParabolicShortSettings)
    cup_with_handle: CupWithHandleSettings = field(default_factory=CupWithHandleSettings)
    double_bottom: DoubleBottomSettings = field(default_factory=DoubleBottomSettings)
    flat_base: FlatBaseSettings = field(default_factory=FlatBaseSettings)
    ascending_base: AscendingBaseSettings = field(default_factory=AscendingBaseSettings)
    high_tight_flag: HighTightFlagSettings = field(default_factory=HighTightFlagSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)

    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> "Settings":
        return copy.deepcopy(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        return cls(
            universe=UniverseSettings(**d.get("universe", {})),
            rs_rating=RSRatingSettings(**d.get("rs_rating", {})),
            breakout=BreakoutSettings(**d.get("breakout", {})),
            episodic_pivot=EpisodicPivotSettings(**d.get("episodic_pivot", {})),
            parabolic_short=ParabolicShortSettings(**d.get("parabolic_short", {})),
            cup_with_handle=CupWithHandleSettings(**d.get("cup_with_handle", {})),
            double_bottom=DoubleBottomSettings(**d.get("double_bottom", {})),
            flat_base=FlatBaseSettings(**d.get("flat_base", {})),
            ascending_base=AscendingBaseSettings(**d.get("ascending_base", {})),
            high_tight_flag=HighTightFlagSettings(**d.get("high_tight_flag", {})),
            backtest=BacktestSettings(**d.get("backtest", {})),
        )

    @classmethod
    def load(cls, path: Path | str = DEFAULT_SETTINGS_PATH) -> "Settings":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    def save(self, path: Path | str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def load_default_settings() -> Settings:
    return Settings.load(DEFAULT_SETTINGS_PATH)
