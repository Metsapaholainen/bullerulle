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
    # NOTE on the loosened defaults below (min_adr_pct, prior_move_min_pct,
    # max_consolidation_range_pct, min_rs_rating): these were each tightened
    # to Qullamaggie's own stated numbers, which is correct in isolation but
    # wrong as an AND-chain -- stacked, they cut real scans to ~5 matches per
    # 400 symbols. Each of those criteria is now *scored* in setup_quality.py
    # (0-6.5 -> 1-6 stars) instead of being a cliff, so a stock with RS 84
    # still surfaces and simply ranks below one with RS 98. Filter wide, rank
    # hard: use the Scanner's "minimum stars" control to tighten, not these.
    enabled: bool = True
    min_adr_pct: float = 3.5
    adr_lookback_days: int = 20
    prior_move_lookback_days: int = 63
    prior_move_min_pct: float = 20.0
    prior_move_max_pct: float = 100.0
    min_consolidation_days: int = 10
    max_consolidation_days: int = 40
    max_consolidation_range_pct: float = 32.0
    require_above_ema10: bool = True
    require_above_ema20: bool = True
    ema_fast_period: int = 10
    ema_slow_period: int = 20
    breakout_volume_ratio_min: float = 1.5
    volume_avg_period: int = 50
    min_rs_rating: float = 80.0
    # "Surf the rising 10- and 20-day moving averages" -- being *above* the
    # EMA (require_above_ema10/20 above) isn't the same as the EMA itself
    # actually sloping upward. These check the EMA's own trend. Opt-in
    # (default off): combined with the existing RS>=90/ADR/tight-range
    # filters, turning all of these on by default cut real-world match
    # rates to roughly a third of before -- available to try, not forced.
    require_rising_ema10: bool = False
    require_rising_ema20: bool = False
    ema_slope_lookback_days: int = 5
    min_ema_slope_pct: float = 0.0
    # "An orderly pullback and consolidation with higher lows" -- distinct
    # from max_consolidation_range_pct's tightness check, which says nothing
    # about whether the base's lows are actually ascending vs. choppy.
    # Opt-in (default off) -- see note above.
    require_higher_lows: bool = False
    higher_lows_tolerance_pct: float = 0.0
    # The prior move must be a genuine net advance ("a big move HIGHER"),
    # not just a wide high-low range (prior_move_min_pct/max_pct above),
    # which a choppy or even net-down stock could otherwise satisfy.
    # Opt-in (default off) -- see note above.
    require_net_prior_advance: bool = False
    min_prior_net_advance_pct: float = 0.0


@dataclass
class EpisodicPivotSettings:
    enabled: bool = True
    min_gap_pct: float = 10.0
    min_volume_ratio: float = 2.0
    volume_avg_period: int = 50
    lookback_base_days: int = 20
    max_prior_range_pct: float = 20.0
    min_price: float = 1.0
    # "Best EPs are on stocks that have gone sideways for 3-6 months or
    # more" -- a separate, longer window from lookback_base_days' tightness
    # check above, which only looks at the tail end right before the gap.
    quiet_base_lookback_days: int = 100
    max_quiet_base_run_pct: float = 40.0
    # "Avoid stocks that have already made a big move from a previous EP" --
    # a soft penalty (halves score), not a hard exclude, per the article.
    prior_ep_lookback_days: int = 252
    prior_ep_min_gap_pct: float = 8.0
    max_prior_ep_count: int = 0
    # Growth-quality scoring, opt-in via the `fundamentals` kwarg to scan().
    # "Ideal EPS/revenue growth is triple digit YoY, but mid/high double
    # digits works really well too."
    min_growth_pct_floor: float = 25.0
    ideal_growth_pct: float = 100.0
    # "There HAS to be big growth numbers" -- the article treats growth as a
    # requirement, not just a nice-to-have, so this now defaults on. Still
    # safe with no fundamentals fetched: the filter only activates when
    # growth_pct is not None (see setups/episodic_pivot.py).
    require_growth_floor: bool = True
    # "A significant beat to analyst expectations" -- scored as a bonus on
    # top of growth quality; optionally a hard filter too (off by default,
    # since earnings-beat data coverage is spotty for smaller/newer names).
    min_eps_beat_pct_for_bonus: float = 20.0
    require_eps_beat: bool = False


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
class MomentumBurstSettings:
    """Stockbee (Pradeep Bonde) "Momentum Burst" -- a 3-5 day swing off a
    range-expansion day out of a quiet base.

    Bonde's published scan has been unchanged for 14+ years and is only three
    conditions: `c/c1 >= 1.04 and v > v1 and v > 100000`. Everything else
    below is his separately-published 8-point quality checklist made
    quantitative. Included here because it is the closest published system to
    "hold 3-5 days, sell half, trail the rest" with an EOD scan and a
    stop-buy entry -- and because Qullamaggie credits Bonde as a source.

    His words on the hold: "3 to 5 days moves of 8% to 40%." On entry timing:
    take day 1 of the burst, not day 2 or 3 (that's max_consecutive_up_days)."""
    enabled: bool = True
    # --- Bonde's core scan, verbatim ---
    min_gain_pct: float = 4.0             # c/c1 >= 1.04
    require_volume_above_prior: bool = True   # v > v1
    min_volume: float = 100_000           # v > 100000
    # --- liquidity, ours: the Qullamaggie community's usual $3M/day floor ---
    min_dollar_volume: float = 3_000_000
    # --- the 8-point quality checklist ---
    # IMPORTANT: Bonde's published SCAN is only the three conditions above.
    # The checklist below is how he then picks among the scan's results by
    # eye -- it was never meant as an AND-chain. Encoded as hard gates at his
    # descriptive values it cut a real 272-symbol universe from 44 candidates
    # with a valid trigger day to zero. So these thresholds are set to
    # exclude clear violations only, and the ranking work is done by
    # setup_quality.py's star score plus this setup's own `score` column.
    # 1. "stock should have range expansion on breakout day"
    min_range_expansion_ratio: float = 1.3
    range_expansion_lookback: int = 5
    # 3. "day before breakout should be narrow range day or negative day"
    require_quiet_prior_day: bool = True
    # 4. "pre breakout there should not be many big range moves or breakdowns"
    max_base_daily_move_pct: float = 12.0
    # 5. "stock should have linearity in prior uptrend before the consolidation"
    min_efficiency_ratio: float = 0.15
    prior_uptrend_lookback_days: int = 40
    # 6. "correction or consolidation should be orderly during the entire move"
    consolidation_days: int = 10          # "5 to 10 days of orderly consolidation"
    max_consolidation_range_pct: float = 25.0
    # 7. "volume during consolidation should be preferably orderly and lower"
    require_volume_dryup: bool = True
    volume_dryup_ratio_max: float = 1.15
    # 8. "stock should close near high on breakout day"
    min_close_position_pct: float = 60.0  # close in the top 40% of the day's range
    # "avoid day 2/3 entries" -- the burst is already underway by then and the
    # stop is too far from the entry to be worth taking.
    max_consecutive_up_days: int = 3
    # --- shared plumbing ---
    volume_avg_period: int = 50
    min_adr_pct: float = 3.0
    min_rs_rating: float = 60.0
    adr_lookback_days: int = 20


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
    # What to do when the signal day's low sits FURTHER than stop_adr_multiple
    # ADRs from where we actually got filled.
    #   "cap":  silently tighten the stop to the ADR distance. Original
    #           behaviour -- but the stop then no longer sits under the
    #           signal-day low, so an ordinary pullback into the base takes
    #           you out of a trade that was still working.
    #   "skip": don't take the trade. This is the actual rule ("the stop
    #           should be no more than the ATR") -- if it would be wider,
    #           the setup isn't takeable at acceptable risk.
    # Matters much more under entry_mode="stop_buy", where the higher fill
    # makes the cap bind on nearly every trade, silently degenerating
    # stop_mode="low_of_signal_day" into "adr_multiple".
    stop_exceeds_adr_action: str = "cap"
    # How the entry actually gets filled.
    #   "next_open": fill unconditionally at the next bar's open. What every
    #                result recorded before this flag existed was measured
    #                with -- keep it to reproduce those numbers.
    #   "stop_buy":  a resting premarket buy-stop entry_buffer_pct above the
    #                signal day's high (mirror: a sell-stop below the low for
    #                shorts). If price never trades through the trigger there
    #                is no trade -- which is the whole point. "next_open"
    #                fills every signal including the ones that opened and
    #                immediately rolled over, and those are exactly the
    #                failed breakouts a real stop order never buys.
    # Default is "stop_buy" because it is how this app's user actually
    # trades; it also lowers trade counts substantially, which is expected.
    entry_mode: str = "stop_buy"
    # How far beyond the signal-day extreme the resting order sits. A few
    # basis points, enough that a single tick touching the exact high doesn't
    # fill you on a level that was never really taken out.
    entry_buffer_pct: float = 0.05
    # Stand aside when the bar opens more than this % past the trigger. A gap
    # that far through the order fills you at a price the setup never
    # justified, and leaves the stop an unacceptable distance away. This is
    # the closest daily-bar substitute for the opening-range-high filter
    # (skipping a stock that gaps up then fails) -- it is not equivalent.
    # 0 disables: you take the gapped fill.
    max_gap_fill_pct: float = 0.0
    # Check whether the ENTRY bar itself traded back through the stop.
    # Off by default, and worth understanding before turning on: the exits
    # loop runs before the entries loop, so a position opened on bar D isn't
    # exit-checked until D+1. Under stop_buy you enter near the TOP of the
    # entry bar's range, so a bar that pokes the trigger and then closes at
    # its low blows through the stop with zero modelling. Leaving this off
    # inflates the win rate; turning it on is more honest but strictly
    # pessimistic (it assumes the low came after the fill, which isn't
    # knowable from a daily bar).
    check_same_bar_stop: bool = False
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
    # "Buy no more than 1% of the average volume" (Qullamaggie's Laws of
    # Swing) -- caps share count independent of the risk-based sizing
    # above. 0 disables.
    max_position_pct_of_avg_volume: float = 1.0
    # Per the source article, EP's stop rule is worded identically to
    # Breakout's: "the stop is at the lows of the day" -- no documented
    # carve-out for the gap or the opening candle specifically. The default
    # (None, i.e. `stop_mode` as-is -- the signal day's whole-day low,
    # ADR-capped) already matches this. This override exists only as an
    # optional alternative to experiment with in backtests, not because the
    # default is wrong; set to e.g. "adr_multiple" to try it.
    ep_stop_mode_override: Optional[str] = None


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
    momentum_burst: MomentumBurstSettings = field(default_factory=MomentumBurstSettings)
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
            momentum_burst=MomentumBurstSettings(**d.get("momentum_burst", {})),
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
