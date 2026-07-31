"""Idealized, schematic diagrams of each pattern -- NOT real stock data.

Used in the Pattern Library so "what does a Cup with Handle actually look
like" has a picture next to the paragraph, not just a description. Curves
are hand-built smooth shapes with a bit of texture (a sine ripple) so bases
don't read as a perfectly flat line, but they're illustrative geometry, not
a real ticker's price history.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

N = 200  # points per diagram


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


def _figure(y: np.ndarray, regions: list, marker_label: str = "Breakout", marker_color: str = "lime") -> go.Figure:
    x = np.arange(len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=2.5, color="#4da6ff"), showlegend=False))
    for x0f, x1f, label, color in regions:
        fig.add_vrect(
            x0=x0f * (N - 1), x1=x1f * (N - 1), fillcolor=color, line_width=0,
            annotation_text=label, annotation_position="top left", annotation_font_size=11,
        )
    fig.add_trace(
        go.Scatter(
            x=[len(y) - 1], y=[y[-1]], mode="markers+text",
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
        [(0, 1.0), (0.35, 1.55), (0.55, 1.42), (0.85, 1.52), (1.0, 1.85)],
        ripple_regions=[(0.35, 0.85, 0.05)],
    )
    return _figure(y, [(0.0, 0.35, "Prior move", "rgba(100,149,237,0.10)"), (0.35, 0.85, "Base", "rgba(100,149,237,0.18)")])


def _episodic_pivot_diagram():
    y = _curve([(0, 1.0), (0.6, 1.02), (0.63, 1.42), (1.0, 1.5)], ripple_regions=[(0.0, 0.6, 0.03)])
    return _figure(y, [(0.0, 0.6, "Quiet base", "rgba(100,149,237,0.15)")], marker_label="Gap up on volume")


def _parabolic_short_diagram():
    y = _curve([(0, 1.0), (0.45, 1.25), (0.75, 2.15), (1.0, 3.0)])
    return _figure(
        y, [(0.45, 1.0, "Extension", "rgba(255,0,0,0.10)")],
        marker_label="Overextended -- short candidate", marker_color="orange",
    )


def _cup_with_handle_diagram():
    y = _curve(
        [(0, 1.0), (0.25, 1.55), (0.45, 1.08), (0.65, 1.52), (0.8, 1.38), (0.95, 1.53), (1.0, 1.7)],
        ripple_regions=[(0.8, 0.95, 0.03)],
    )
    return _figure(
        y,
        [(0.25, 0.8, "Cup", "rgba(100,149,237,0.12)"), (0.8, 0.95, "Handle", "rgba(255,165,0,0.30)")],
    )


def _double_bottom_diagram():
    y = _curve([(0, 1.35), (0.2, 1.0), (0.45, 1.28), (0.7, 1.03), (0.85, 1.18), (1.0, 1.45)])
    return _figure(y, [(0.0, 0.85, "W pattern", "rgba(100,149,237,0.15)")])


def _flat_base_diagram():
    y = _curve(
        [(0, 1.0), (0.3, 1.55), (0.85, 1.58), (1.0, 1.85)],
        ripple_regions=[(0.3, 0.85, 0.045)],
    )
    return _figure(y, [(0.0, 0.3, "Prior move", "rgba(100,149,237,0.10)"), (0.3, 0.85, "Flat base", "rgba(100,149,237,0.18)")])


def _ascending_base_diagram():
    y = _curve(
        [(0, 1.0), (0.18, 1.3), (0.32, 1.16), (0.5, 1.48), (0.65, 1.36), (0.83, 1.62), (0.95, 1.55), (1.0, 1.85)],
        ripple_regions=[(0.0, 0.95, 0.02)],
    )
    return _figure(y, [(0.0, 0.95, "Ascending base (higher lows/highs)", "rgba(100,149,237,0.15)")])


def _high_tight_flag_diagram():
    y = _curve(
        [(0, 1.0), (0.55, 2.6), (0.75, 2.42), (0.9, 2.56), (1.0, 2.95)],
        ripple_regions=[(0.55, 0.9, 0.04)],
    )
    return _figure(y, [(0.0, 0.55, "Explosive run-up", "rgba(255,0,0,0.08)"), (0.55, 0.9, "Tight flag", "rgba(255,165,0,0.30)")])


PATTERN_DIAGRAMS = {
    "breakout": _breakout_diagram,
    "episodic_pivot": _episodic_pivot_diagram,
    "parabolic_short": _parabolic_short_diagram,
    "cup_with_handle": _cup_with_handle_diagram,
    "double_bottom": _double_bottom_diagram,
    "flat_base": _flat_base_diagram,
    "ascending_base": _ascending_base_diagram,
    "high_tight_flag": _high_tight_flag_diagram,
}


def get_pattern_diagram(setup_key: str):
    """Returns a plotly Figure schematically illustrating the pattern, or
    None if there's no diagram registered for that setup key."""
    builder = PATTERN_DIAGRAMS.get(setup_key)
    return builder() if builder else None
