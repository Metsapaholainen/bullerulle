"""Idealized, schematic diagrams of each pattern -- NOT real stock data.

Used in the Pattern Library so "what does a Cup with Handle actually look
like" has a picture next to the paragraph, not just a description. Curves
are hand-built smooth shapes with a bit of texture (a sine ripple) so bases
don't read as a perfectly flat line, but they're illustrative geometry, not
a real ticker's price history.

Each diagram also marks the ideal *entry zone* -- where you'd actually pull
the trigger -- as a shaded vertical band (green for a long breakout entry,
orange for the Parabolic Short's reversal entry), plus a dotted line at the
resistance/pivot level being broken, where one applies.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

N = 200  # points per diagram

# Reusable entry-zone styling: (fill_color, border_color, text_color)
_BUY_ENTRY_STYLE = ("rgba(50,205,50,0.28)", "rgba(50,205,50,0.9)", "#39ff14")
_SHORT_ENTRY_STYLE = ("rgba(255,140,0,0.28)", "rgba(255,140,0,0.9)", "#ffa500")


def _curve(control_points: list, ripple_regions: list = None) -> np.ndarray:
    """control_points: [(x_frac 0-1, y), ...] smoothly interpolated over N
    points. ripple_regions: [(x0_frac, x1_frac, amplitude), ...] adds a small
    sine wiggle over that span so a "tight base" doesn't look like a ruler-flat line."""
    positions = np.array([p for p, _ in control_points]) * (N - 1)
    values = np.array([v for _, v in control_points])
    x = np.arange(N)
    y = np.interp(x, positions, values)
    if ripple_regions:
        for x0f, x1f, amp in ripple_regions:
            x0, x1 = int(x0f * (N - 1)), int(x1f * (N - 1))
            span = max(x1 - x0, 1)
            t = np.linspace(0, 3.6 * np.pi, span)
            y[x0:x1] += amp * np.sin(t) * np.linspace(1, 0.3, span)
    return y


def _figure(
    y: np.ndarray,
    regions: list,
    marker_x_frac: float = 1.0,
    marker_label: str = "Breakout",
    marker_color: str = "lime",
    pivot: tuple = None,
    entry_zone: tuple = None,
) -> go.Figure:
    """regions: [(x0f, x1f, label, fill_color), ...] -- the base/cup/flag shading.
    pivot: (price_level, x0f, x1f) -- dotted resistance line to show what's being broken.
    entry_zone: (x0f, x1f, label, style) where style is one of the _*_ENTRY_STYLE tuples above."""
    x = np.arange(len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=2.5, color="#4da6ff"), showlegend=False))
    for x0f, x1f, label, color in regions:
        fig.add_vrect(
            x0=x0f * (N - 1), x1=x1f * (N - 1), fillcolor=color, line_width=0,
            annotation_text=label, annotation_position="top left", annotation_font_size=11,
        )
    if pivot:
        pivot_value, px0f, px1f = pivot
        fig.add_shape(
            type="line", x0=px0f * (N - 1), x1=px1f * (N - 1), y0=pivot_value, y1=pivot_value,
            line=dict(color="rgba(255,255,255,0.6)", width=1.5, dash="dot"),
        )
    if entry_zone:
        ex0f, ex1f, elabel, (efill, eborder, etext) = entry_zone
        fig.add_vrect(
            x0=ex0f * (N - 1), x1=ex1f * (N - 1), fillcolor=efill, line_width=1.5, line_color=eborder,
            annotation_text=elabel, annotation_position="bottom left",
            annotation_font_size=10, annotation_font_color=etext,
        )
    marker_idx = int(round(marker_x_frac * (len(y) - 1)))
    fig.add_trace(
        go.Scatter(
            x=[marker_idx], y=[y[marker_idx]], mode="markers+text",
            marker=dict(symbol="star", size=14, color=marker_color, line=dict(width=1, color="black")),
            text=[marker_label], textposition="top center", textfont=dict(color=marker_color, size=11),
            showlegend=False,
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _breakout_diagram():
    y = _curve(
        [(0, 1.0), (0.35, 1.55), (0.55, 1.42), (0.8, 1.52), (0.87, 1.56), (1.0, 1.8)],
        ripple_regions=[(0.35, 0.8, 0.03)],
    )
    return _figure(
        y,
        [(0.0, 0.35, "Prior move", "rgba(100,149,237,0.10)"), (0.35, 0.8, "Base", "rgba(100,149,237,0.18)")],
        marker_x_frac=0.87, marker_label="Breakout entry",
        pivot=(1.56, 0.35, 0.87),
        entry_zone=(0.83, 0.92, "Entry zone", _BUY_ENTRY_STYLE),
    )


def _episodic_pivot_diagram():
    y = _curve([(0, 1.0), (0.55, 1.02), (0.6, 1.45), (1.0, 1.62)], ripple_regions=[(0.0, 0.55, 0.03)])
    return _figure(
        y,
        [(0.0, 0.55, "Quiet base", "rgba(100,149,237,0.15)")],
        marker_x_frac=0.6, marker_label="Gap up on volume",
        pivot=(1.05, 0.0, 0.6),
        entry_zone=(0.57, 0.66, "Entry (gap day)", _BUY_ENTRY_STYLE),
    )


def _parabolic_short_diagram():
    y = _curve([(0, 1.0), (0.45, 1.25), (0.7, 1.95), (0.85, 2.5), (1.0, 2.05)])
    return _figure(
        y, [(0.45, 0.85, "Extension", "rgba(255,0,0,0.10)")],
        marker_x_frac=0.85, marker_label="Overextended -- short entry", marker_color="orange",
        entry_zone=(0.81, 0.89, "Short entry zone", _SHORT_ENTRY_STYLE),
    )


def _cup_with_handle_diagram():
    y = _curve(
        [
            (0, 1.0), (0.2, 1.6), (0.35, 1.28), (0.475, 1.15), (0.6, 1.28), (0.75, 1.58),
            (0.82, 1.48), (0.9, 1.53), (0.97, 1.62), (1.0, 1.72),
        ],
        ripple_regions=[(0.35, 0.6, 0.02), (0.75, 0.9, 0.02)],
    )
    return _figure(
        y,
        [(0.2, 0.75, "Cup", "rgba(100,149,237,0.12)"), (0.75, 0.9, "Handle", "rgba(255,165,0,0.30)")],
        marker_x_frac=0.97, marker_label="Breakout entry",
        pivot=(1.6, 0.2, 0.97),
        entry_zone=(0.93, 1.0, "Entry zone", _BUY_ENTRY_STYLE),
    )


def _double_bottom_diagram():
    y = _curve(
        [(0, 1.5), (0.15, 1.0), (0.4, 1.32), (0.65, 0.95), (0.85, 1.25), (0.9, 1.33), (1.0, 1.55)],
        ripple_regions=[(0.05, 0.2, 0.02), (0.55, 0.7, 0.02)],
    )
    return _figure(
        y,
        [(0.0, 0.9, "W pattern", "rgba(100,149,237,0.15)")],
        marker_x_frac=0.9, marker_label="Breakout entry",
        pivot=(1.32, 0.4, 0.9),
        entry_zone=(0.86, 0.94, "Entry zone", _BUY_ENTRY_STYLE),
    )


def _flat_base_diagram():
    y = _curve(
        [(0, 1.0), (0.3, 1.55), (0.55, 1.58), (0.85, 1.56), (0.92, 1.6), (1.0, 1.85)],
        ripple_regions=[(0.3, 0.85, 0.025)],
    )
    return _figure(
        y,
        [(0.0, 0.3, "Prior move", "rgba(100,149,237,0.10)"), (0.3, 0.85, "Flat base", "rgba(100,149,237,0.18)")],
        marker_x_frac=0.92, marker_label="Breakout entry",
        pivot=(1.6, 0.3, 0.92),
        entry_zone=(0.88, 0.96, "Entry zone", _BUY_ENTRY_STYLE),
    )


def _ascending_base_diagram():
    y = _curve(
        [(0, 1.0), (0.18, 1.3), (0.32, 1.16), (0.5, 1.48), (0.65, 1.36), (0.83, 1.62), (0.9, 1.55), (0.96, 1.65), (1.0, 1.78)],
        ripple_regions=[(0.0, 0.83, 0.015)],
    )
    return _figure(
        y,
        [(0.0, 0.9, "Ascending base (higher lows/highs)", "rgba(100,149,237,0.15)")],
        marker_x_frac=0.96, marker_label="Breakout entry",
        pivot=(1.62, 0.83, 0.96),
        entry_zone=(0.92, 0.99, "Entry zone", _BUY_ENTRY_STYLE),
    )


def _high_tight_flag_diagram():
    y = _curve(
        [(0, 1.0), (0.55, 2.6), (0.68, 2.44), (0.8, 2.5), (0.87, 2.58), (0.93, 2.7), (1.0, 2.95)],
        ripple_regions=[(0.55, 0.87, 0.03)],
    )
    return _figure(
        y,
        [(0.0, 0.55, "Explosive run-up", "rgba(255,0,0,0.08)"), (0.55, 0.87, "Tight flag", "rgba(255,165,0,0.30)")],
        marker_x_frac=0.93, marker_label="Breakout entry",
        pivot=(2.58, 0.55, 0.93),
        entry_zone=(0.89, 0.97, "Entry zone", _BUY_ENTRY_STYLE),
    )


def _momentum_burst_diagram():
    # Deliberately drawn on a shorter horizon than the other diagrams: the
    # whole point of this setup is that the base is ~10 days and the move
    # you're capturing is 3-5 days, not 3-5 weeks.
    y = _curve(
        [(0, 1.0), (0.35, 1.42), (0.5, 1.38), (0.62, 1.41), (0.72, 1.39), (0.78, 1.40), (0.82, 1.52), (1.0, 1.72)],
        ripple_regions=[(0.5, 0.78, 0.008)],
    )
    return _figure(
        y,
        [
            (0.0, 0.35, "Linear prior advance", "rgba(100,149,237,0.10)"),
            (0.5, 0.78, "Quiet base, volume dries up", "rgba(100,149,237,0.20)"),
        ],
        marker_x_frac=0.82, marker_label="Range-expansion day",
        pivot=(1.41, 0.5, 0.82),
        entry_zone=(0.80, 0.88, "Stop-buy above the high", _BUY_ENTRY_STYLE),
    )


def _vcp_diagram():
    # Three successive pullback legs, each SHALLOWER than the last (a
    # narrowing amplitude staircase around a common resistance), ending in a
    # tight low-volume pivot and breakout -- the visual signature that
    # distinguishes a genuine multi-leg VCP from a single tight-range snapshot.
    y = _curve(
        [
            (0, 1.0), (0.15, 1.5),
            (0.25, 1.2), (0.35, 1.48),      # leg 1: deep pullback
            (0.45, 1.30), (0.55, 1.47),     # leg 2: shallower pullback
            (0.63, 1.38), (0.70, 1.46),     # leg 3: shallowest pullback
            (0.78, 1.44), (0.85, 1.50), (1.0, 1.75),
        ],
        ripple_regions=[(0.25, 0.35, 0.02), (0.45, 0.55, 0.012), (0.63, 0.70, 0.006)],
    )
    return _figure(
        y,
        [
            (0.0, 0.15, "Prior move", "rgba(100,149,237,0.10)"),
            (0.15, 0.78, "Contracting legs (each shallower, quieter)", "rgba(100,149,237,0.15)"),
            (0.78, 0.85, "Pivot", "rgba(255,165,0,0.30)"),
        ],
        marker_x_frac=0.85, marker_label="Breakout entry",
        pivot=(1.50, 0.15, 0.85),
        entry_zone=(0.82, 0.90, "Entry zone", _BUY_ENTRY_STYLE),
    )


def _spring_diagram():
    # A quiet range with volume drying up, a brief false breakdown BELOW
    # range support, then a reclaim back into the range -- match fires on the
    # reclaim day, deliberately before price has broken out of anything.
    y = _curve(
        [
            (0, 1.3), (0.3, 1.32), (0.55, 1.28), (0.7, 1.31),
            (0.78, 1.05), (0.83, 1.02),     # the undercut -- briefly below range support
            (0.88, 1.30), (0.95, 1.34), (1.0, 1.36),
        ],
        ripple_regions=[(0.0, 0.7, 0.012)],
    )
    return _figure(
        y,
        [(0.0, 0.78, "Quiet range, volume drying up", "rgba(100,149,237,0.15)")],
        marker_x_frac=0.88, marker_label="Reclaim entry", marker_color="lime",
        pivot=(1.28, 0.0, 0.88),
        entry_zone=(0.85, 0.91, "Entry zone (reclaim day)", _BUY_ENTRY_STYLE),
    )


def _squeeze_diagram():
    # A prior advance into a tight, quiet coil -- the marker sits ON the
    # coiled day itself, BEFORE the pivot line is crossed, with the stop-buy
    # entry zone resting just above it. This is the visual contrast with
    # Breakout's diagram: the star/marker there sits ABOVE the pivot line
    # (already broken out); here it sits AT/BELOW it (still coiled).
    y = _curve(
        [(0, 1.0), (0.4, 1.5), (0.55, 1.48), (0.7, 1.50), (0.82, 1.49), (0.9, 1.505), (1.0, 1.51)],
        ripple_regions=[(0.55, 0.9, 0.006)],
    )
    return _figure(
        y,
        [
            (0.0, 0.4, "Prior move", "rgba(100,149,237,0.10)"),
            (0.55, 0.9, "Coiled -- range tight vs. own ADR", "rgba(100,149,237,0.20)"),
        ],
        marker_x_frac=0.9, marker_label="Squeeze: stop-buy rests above here", marker_color="lime",
        pivot=(1.52, 0.55, 1.0),
        entry_zone=(0.87, 0.94, "Resting stop-buy zone", _BUY_ENTRY_STYLE),
    )


def _pullback_diagram():
    # An extended move away from a rising EMA, then a shallow pullback BACK
    # to the EMA on light volume -- entry is AT the pocket, not at a new
    # high, which is what makes this style structurally unable to feel
    # "late" the way a breakout entry can.
    y = _curve(
        [(0, 1.0), (0.3, 1.05), (0.55, 1.45), (0.68, 1.42), (0.8, 1.18), (0.88, 1.16), (1.0, 1.30)],
        ripple_regions=[(0.0, 0.3, 0.01)],
    )
    ema = _curve([(0, 1.0), (0.3, 1.03), (0.55, 1.10), (0.8, 1.16), (1.0, 1.20)])
    fig = _figure(
        y,
        [(0.3, 0.68, "Extension away from the EMA", "rgba(100,149,237,0.10)")],
        marker_x_frac=0.88, marker_label="Pullback entry (at the EMA)", marker_color="lime",
        entry_zone=(0.85, 0.92, "Entry zone", _BUY_ENTRY_STYLE),
    )
    fig.add_trace(go.Scatter(x=list(range(N)), y=ema, mode="lines", line=dict(width=1.5, color="orange", dash="dot"), showlegend=False))
    return fig


PATTERN_DIAGRAMS = {
    "squeeze": _squeeze_diagram,
    "pullback": _pullback_diagram,
    "breakout": _breakout_diagram,
    "episodic_pivot": _episodic_pivot_diagram,
    "parabolic_short": _parabolic_short_diagram,
    "cup_with_handle": _cup_with_handle_diagram,
    "double_bottom": _double_bottom_diagram,
    "flat_base": _flat_base_diagram,
    "ascending_base": _ascending_base_diagram,
    "high_tight_flag": _high_tight_flag_diagram,
    "momentum_burst": _momentum_burst_diagram,
    "vcp": _vcp_diagram,
    "spring": _spring_diagram,
}


def get_pattern_diagram(setup_key: str):
    """Returns a plotly Figure schematically illustrating the pattern, or
    None if there's no diagram registered for that setup key."""
    builder = PATTERN_DIAGRAMS.get(setup_key)
    return builder() if builder else None
