# Qullamaggie-Style Momentum Scanner & Backtester

A local scanner + backtester for Kristjan Kullamägi ("Qullamaggie")-style momentum
swing setups: **Breakout**, **Episodic Pivot**, and **Parabolic Short**. All
thresholds are adjustable, presets are saveable/comparable, and there's a
reverse parameter finder that reverse-engineers what settings would have
captured a specific stock's past move.

Data comes from the [Financial Modeling Prep](https://financialmodelingprep.com/) API.

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
- **Scanner** — today's matches against the current settings, ranked, with charts.
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
- `indicators.py` — ADR%, EMAs, prior-move %, consolidation tightness, RS rating, gap %, volume ratio
- `setups/` — breakout / episodic_pivot / parabolic_short detectors
- `scanner.py` — orchestrates a full scan
- `backtest/` — engine, stats, grid optimizer, reverse parameter finder
- `presets.py` — named settings presets
- `app.py` / `cli.py` — interfaces
