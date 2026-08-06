# Qullamaggie-Style Momentum Scanner & Backtester

A local scanner + backtester for momentum swing trading. Two primary systems are
scanned by default: **Squeeze** (fires on the quiet/coiled day, before a breakout
prints, so you catch the move live instead of chasing an already-confirmed high)
and **Pullback** (buy the first shallow pullback to a rising EMA in an
already-confirmed leader, never chasing a new high). Eleven older
Kristjan Kullamägi ("Qullamaggie")-style and O'Neil-style setups -- **Breakout**,
**Episodic Pivot**, **Parabolic Short**, five O'Neil chart patterns, Stockbee's
**Momentum Burst**, Minervini's **VCP**, and a Wyckoff-style **Spring** -- are
kept, off by default, for reference (see "Which system should I actually trade?"
in the app for the measured status of each). All thresholds are adjustable,
presets are saveable/comparable, and there's a reverse parameter finder that
reverse-engineers what settings would have captured a specific stock's past move.

⚠️ **Honest headline, measured not guessed: on this data, none of these systems**
**beat simply buying and holding the index.** SPY returned +25.0% CAGR and QQQ
+30.2% CAGR over the same 2022-2026 measured window; Breakout's own full-universe
CAGR was -9.4%, Squeeze's was -30.7%. A positive per-trade expectancy (R-multiple)
is not the same as the portfolio actually compounding faster than a passive
index -- position sizing and correlation across many simultaneously-open
small-cap positions matter enormously. Squeeze in particular fires far more often
than Breakout (a "quiet, coiled" state is structurally more common than a
"confirmed breakout"), which keeps more correlated positions open at once than
the shared default position sizing (10 concurrent slots, 20% of equity each) was
built for -- its entry timing is a real fix for "the breakout already happened,"
its position sizing is not yet. See `setups/explanations.py` and the app's
"Which system should I actually trade?" panel for the full numbers, including
where each system fell short.

**Anti-crowd features on the older setups (opt-in, off by default):** VCP and
Spring detect structurally different, less-common setups (a genuine multi-leg
contraction, and a shakeout-before-breakout that fires before price even breaks
its base). A recent-fakeout filter and a close-confirmation requirement on
Breakout/VCP target the fact that a bare resistance level is an easy, gameable
trigger. A regime/crowding filter can downsize or gate new entries during a
market drawdown or high-volatility stretch, per the momentum-crash literature
(Daniel & Moskowitz, NBER w20439/JFE 2016). Treat all of this as exploratory, not
proven.

Data comes from the [Financial Modeling Prep](https://financialmodelingprep.com/) API.

## Two things worth knowing up front

**Filter wide, rank hard — but rank, don't filter.** Every one of Qullamaggie's
criteria is real, but chained together with AND they encode "perfect or nothing"
— at his own stated numbers the Breakout scan returned about 5 matches per 400
symbols. So the hard gates are deliberately loose and `setup_quality.py` scores
each match 0–6.5 across seven of those criteria (prior move, above a rising
50-day, RS, base tightness, volume dry-up then expansion, MA alignment,
linearity), bucketed into 1–6 stars.

⚠️ **The stars are a reading order, not a quality gate.** Correlated all seven
components against the eventual trade R-multiple across 135 real Breakout
trades (and again against just the 26 trail-exit trades, where the edge lives)
— every component came back statistically indistinguishable from zero on trade
*magnitude*. One component, prior-move size, showed a real, monotonic
relationship to a *different* thing: the probability of becoming a trail-exit
winner at all (trail-exit rate climbed 5.9% → 20.6% → 21.2% → 29.4% across
quartiles). That's now reflected in the scoring (see below), and a real
measurement bug was fixed along the way — tightness was silently 0.0 for every
single trade because it measured the full base instead of the short recent
window Breakout's own detector actually checks. Neither fix repaired filtering:
raising the bar to 3★ still roughly halves expectancy, and filtering on
prior-move alone (no other component involved) never beat taking every match
either. An earlier claim that Cup with Handle was an exception to this did not
survive the retiering (a shared-component change moved it from "helps" to
"hurts"), which itself suggests that was small-sample noise, not a real effect.
Both star sliders default to 0. Use them to decide what to look at first, for
any setup.

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
