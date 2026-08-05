# Qullamaggie-Style Momentum Scanner & Backtester

A local scanner + backtester for Kristjan Kullamägi ("Qullamaggie")-style momentum
swing setups: **Breakout**, **Episodic Pivot**, and **Parabolic Short**, plus five
O'Neil chart patterns and Stockbee's **Momentum Burst**. All thresholds are
adjustable, presets are saveable/comparable, and there's a reverse parameter
finder that reverse-engineers what settings would have captured a specific
stock's past move.

Data comes from the [Financial Modeling Prep](https://financialmodelingprep.com/) API.

## Two things worth knowing up front

**Filter wide, rank hard — but rank, don't filter.** Every one of Qullamaggie's
criteria is real, but chained together with AND they encode "perfect or nothing"
— at his own stated numbers the Breakout scan returned about 5 matches per 400
symbols. So the hard gates are deliberately loose and `setup_quality.py` scores
each match 0–6.5 across seven of those criteria (prior move, above a rising
50-day, RS, base tightness, volume dry-up then expansion, MA alignment,
linearity), bucketed into 1–6 stars.

⚠️ **The stars are a reading order, not a quality gate.** Backtested across star
thresholds, Breakout's expectancy goes 0.473R (all matches) → 0.185R (3★+) →
0.269R (4★+) while its win rate rises the whole way (54.8% → 60.0%). The score
rewards tight, orderly setups that win more often but travel less far, and
Breakout's edge is the handful of trades that run +3.6R — so filtering on it
trades away the tail that pays for everything. Cup with Handle is the exception
(0.39 → 0.54 at 3★+); on Double Bottom it's noise. Both star sliders default to
0. Use them to decide what to look at first.

*The star scale is this project's encoding of his criteria.* He uses "first-tier"
/ "second-tier" language and never publishes a numeric grade; the popular
"5-star setup" scales come from third-party TradingView indicators.

**Backtests model a stop-buy, not a next-open fill.** `entry_mode` defaults to
`"stop_buy"`: a resting order just above the signal day's high that produces no
trade at all if price never trades through it. Those non-fills are exactly the
failed breakouts, so this cuts trade counts substantially versus the old
unconditional next-open fill — that drop is the filter working. Set
`entry_mode: next_open` to reproduce pre-2026 numbers.

One caveat the app states in the UI too: a real stop-buy gives up the
opening-range-high filter that is part of Qullamaggie's actual edge (he skips a
stock that gaps up and then fails to clear its early high). `max_gap_fill_pct`
is a blunt substitute, not an equivalent.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env and set FMP_API_KEY=<your key>
```

## Usage

### Streamlit app (recommended)

```bash
streamlit run app.py
```

Tabs:
- **Scanner** — today's matches against the current settings, ranked by a 1–6
  star quality score, with charts. The star filter and top-N cap are the
  intended way to tighten: the scan itself runs wide on purpose (see
  "Filter wide, rank hard" below).
- **📋 Tomorrow's Orders** — the scan turned into stop-buy tickets you can place
  before the open: trigger, stop-limit cap, initial stop, share count for your
  account and risk %, and the sell-half target. Built for trading this without
  watching the market intraday. Includes a written comparison of which setup
  actually suits that workflow.
- **Designer** — tweak any parameter and instantly see updated scan matches and
  backtest stats (data is cached locally so this doesn't re-hit the API).
- **Backtest & Optimizer** — run a full historical backtest for a setup, or
  sweep a grid of parameter combinations and rank the results.
- **Parameter Finder** — give it a symbol and the approximate date of a known
  good move; it reports the indicator values that stock actually had at that
  moment, then searches for threshold sets that would flag it, validated
  against a broad-universe backtest so you can tell a real edge from an
  overfit to one trade.
- **Settings** — edit/save/load named presets.

### CLI (headless)

```bash
python cli.py scan                       # ranked CSV of today's matches
python cli.py scan --preset aggressive   # use a saved preset
python cli.py backtest --setup breakout --start 2018-01-01 --end 2024-12-31
```

## Project layout

See the plan / architecture notes for the full file breakdown:
- `data/` — FMP client, local OHLCV cache, universe builder
- `indicators.py` — ADR%, EMAs, prior-move %, consolidation tightness, RS rating,
  gap %, volume ratio, volume dry-up, SMA slope, Kaufman efficiency ratio
- `setups/` — one detector per pattern (breakout, episodic_pivot,
  parabolic_short, momentum_burst, and five O'Neil patterns), all registered in
  `setups/__init__.py`
- `setup_quality.py` — the 0–6.5 quality score and its 1–6 star bucketing
- `scanner.py` — orchestrates a full scan, plus "approaching pivot"
- `backtest/` — engine, stats, grid optimizer, reverse parameter finder.
  `engine.resolve_entry_and_stop()` is the single source of truth for whether a
  signal is tradeable, at what fill, and with what stop — the engine, the Orders
  tab, `stop_calculator.py` and the chart overlay all go through it
- `presets.py` — named settings presets
- `app.py` / `cli.py` — interfaces
