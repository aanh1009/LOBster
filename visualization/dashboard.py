"""
Plotly Dash Dashboard
======================

Renders a multi-panel interactive dashboard from a completed SimulationResult.
Launch via:

    python -m visualization.dashboard          (uses cached result)
    from visualization.dashboard import launch_dashboard; launch_dashboard(result)

Panels
------
1. Order Book Depth  — live bid/ask depth at simulation end
2. Mid-Price & Spread — time series from LOB snapshots
3. Hawkes Intensity  — fitted λ*(t) overlaid on event arrivals
4. Kyle's Lambda     — rolling 5-min estimate of price impact
5. VPIN Rolling      — volume-bucket VPIN over time
6. P&L               — strategy mark-to-market (if backtest was run)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from feed.simulator import SimulationResult
    from backtest.engine import BacktestResult


# ---------------------------------------------------------------------------
# Lazy import of Dash to avoid import error when dash is not installed
# ---------------------------------------------------------------------------

def _require_dash():
    try:
        import dash
        from dash import dcc, html
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return dash, dcc, html, go, make_subplots
    except ImportError:
        raise ImportError(
            "Dash is required for the dashboard. Install with:\n"
            "  pip install dash plotly"
        )


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_depth_chart(result: "SimulationResult") -> "go.Figure":
    """Bid/ask volume by price level — ladder view."""
    _, _, _, go, make_subplots = _require_dash()

    bid_depth = result.book.bid_depth(20)
    ask_depth = result.book.ask_depth(20)

    bid_prices = [p for p, _ in bid_depth]
    bid_vols   = [v for _, v in bid_depth]
    ask_prices = [p for p, _ in ask_depth]
    ask_vols   = [v for _, v in ask_depth]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bid_prices, y=bid_vols,
        name="Bid", marker_color="rgba(0,200,100,0.7)",
        orientation="v",
    ))
    fig.add_trace(go.Bar(
        x=ask_prices, y=ask_vols,
        name="Ask", marker_color="rgba(220,50,50,0.7)",
        orientation="v",
    ))
    fig.update_layout(
        title="Order Book Depth (Top 20 Levels)",
        xaxis_title="Price",
        yaxis_title="Volume (shares)",
        barmode="overlay",
        template="plotly_dark",
        height=350,
    )
    return fig


def build_midprice_spread_chart(result: "SimulationResult") -> "go.Figure":
    """Mid-price and spread time series from LOB snapshots."""
    _, _, _, go, make_subplots = _require_dash()

    snaps = result.book.snapshots
    if not snaps:
        return go.Figure()

    ts      = [s.timestamp / 1e9 for s in snaps]
    ts_rel  = [t - ts[0] for t in ts]
    mids    = [s.mid_price   for s in snaps]
    spreads = [s.spread      for s in snaps]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Mid-Price", "Bid-Ask Spread (bps)"),
                        vertical_spacing=0.08)

    fig.add_trace(go.Scatter(x=ts_rel, y=mids, mode="lines",
                             line=dict(color="cyan", width=1),
                             name="Mid-Price"), row=1, col=1)

    # Convert spread to bps
    spread_bps = [
        (sp / mp * 10000) if (sp is not None and mp is not None and mp > 0) else None
        for sp, mp in zip(spreads, mids)
    ]
    fig.add_trace(go.Scatter(x=ts_rel, y=spread_bps, mode="lines",
                             line=dict(color="orange", width=1),
                             name="Spread (bps)"), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=450,
        title="Mid-Price & Spread Over Simulation Time",
        xaxis2_title="Simulation time (s)",
    )
    return fig


def build_hawkes_chart(
    bid_times: np.ndarray,
    ask_times: np.ndarray,
    bid_params: dict,
    ask_params: dict,
    duration_s: float,
    n_points: int = 500,
) -> "go.Figure":
    """Hawkes intensity λ*(t) over time with event rug plots."""
    from hawkes.process import intensity as hawkes_intensity
    _, _, _, go, make_subplots = _require_dash()

    t_grid = np.linspace(0, min(duration_s, 600), n_points)  # first 10 min

    lam_bid = hawkes_intensity(t_grid, bid_times, **bid_params)
    lam_ask = hawkes_intensity(t_grid, ask_times, **ask_params)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Bid-Side Intensity λ*(t)",
                                        "Ask-Side Intensity λ*(t)"),
                        vertical_spacing=0.10)

    # Intensity curves
    fig.add_trace(go.Scatter(x=t_grid, y=lam_bid, mode="lines",
                             line=dict(color="limegreen", width=1.5),
                             name="λ_bid(t)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_grid, y=lam_ask, mode="lines",
                             line=dict(color="tomato", width=1.5),
                             name="λ_ask(t)"), row=2, col=1)

    # Event rug plots (first 600s only)
    bt = bid_times[bid_times <= 600]
    at = ask_times[ask_times <= 600]

    fig.add_trace(go.Scatter(
        x=bt, y=np.zeros(len(bt)),
        mode="markers", marker=dict(symbol="line-ns-open", size=6, color="limegreen"),
        name="Bid events", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=at, y=np.zeros(len(at)),
        mode="markers", marker=dict(symbol="line-ns-open", size=6, color="tomato"),
        name="Ask events", showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=450,
        title="Hawkes Process: Conditional Intensity (first 10 min)",
        xaxis2_title="Simulation time (s)",
    )
    return fig


def build_kyle_vpin_chart(
    kyle_result: dict,
    vpin_result: dict,
    mid_prices: np.ndarray,
    signed_volumes: np.ndarray,
) -> "go.Figure":
    """Kyle's lambda regression scatter + VPIN rolling series."""
    _, _, _, go, make_subplots = _require_dash()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(
                            f"Kyle's λ Regression  (λ={kyle_result['lambda']:.6f}, "
                            f"R²={kyle_result['r_squared']:.3f})",
                            f"Rolling VPIN  (mean={vpin_result['vpin']:.3f})"
                        ))

    # --- Kyle scatter ---
    delta_p = np.diff(mid_prices)
    sv      = signed_volumes[: len(delta_p)]
    lam     = kyle_result["lambda"]
    intercept = kyle_result["intercept"]
    x_line  = np.array([sv.min(), sv.max()])
    y_line  = intercept + lam * x_line

    fig.add_trace(go.Scatter(
        x=sv.tolist(), y=delta_p.tolist(),
        mode="markers", marker=dict(size=3, color="cyan", opacity=0.4),
        name="(OFI, ΔP)", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_line.tolist(), y=y_line.tolist(),
        mode="lines", line=dict(color="yellow", width=2),
        name=f"OLS fit (λ={lam:.5f})", showlegend=False,
    ), row=1, col=1)

    # --- VPIN rolling ---
    vpin_vals = vpin_result["vpin_rolling"]
    fig.add_trace(go.Scatter(
        x=list(range(len(vpin_vals))), y=vpin_vals,
        mode="lines", line=dict(color="orange", width=1),
        name="VPIN", showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=vpin_result["vpin"], line_dash="dot",
                  line_color="white", annotation_text="mean", row=1, col=2)

    fig.update_layout(template="plotly_dark", height=350,
                      title="Market Microstructure Metrics")
    fig.update_xaxes(title_text="Signed Volume (shares)", row=1, col=1)
    fig.update_yaxes(title_text="ΔMid-Price ($)",         row=1, col=1)
    fig.update_xaxes(title_text="Volume Bucket #",        row=1, col=2)
    fig.update_yaxes(title_text="VPIN",                   row=1, col=2)
    return fig


def build_pnl_chart(backtest_results: List["BacktestResult"]) -> "go.Figure":
    """Overlay P&L curves for all backtested strategies."""
    _, _, _, go, _ = _require_dash()

    COLORS = ["#00e5ff", "#ff6d00", "#76ff03", "#d500f9"]
    fig = go.Figure()
    for i, br in enumerate(backtest_results):
        if not br.pnl_series:
            continue
        times = [t for t, _ in br.pnl_series]
        pnls  = [p for _, p in br.pnl_series]
        fig.add_trace(go.Scatter(
            x=times, y=pnls,
            mode="lines",
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            name=f"{br.strategy_name}  (Sharpe {br.sharpe_ratio:.2f})",
        ))

    fig.update_layout(
        template="plotly_dark", height=350,
        title="Strategy P&L (Mark-to-Market)",
        xaxis_title="Simulation time (s)",
        yaxis_title="P&L ($)",
    )
    return fig


# ---------------------------------------------------------------------------
# Dashboard launcher
# ---------------------------------------------------------------------------

def launch_dashboard(
    result: "SimulationResult",
    bid_times: Optional[np.ndarray] = None,
    ask_times: Optional[np.ndarray] = None,
    bid_params: Optional[dict] = None,
    ask_params: Optional[dict] = None,
    kyle_result: Optional[dict] = None,
    vpin_result: Optional[dict] = None,
    mid_prices: Optional[np.ndarray] = None,
    signed_volumes: Optional[np.ndarray] = None,
    backtest_results: Optional[List["BacktestResult"]] = None,
    port: int = 8050,
    debug: bool = False,
) -> None:
    """
    Build and serve the Dash dashboard.
    Open http://localhost:8050 in your browser.
    """
    dash, dcc, html, go, _ = _require_dash()

    app = dash.Dash(__name__, title="LOB Engine Dashboard")

    children = [
        html.H1("Limit Order Book Engine — Microstructure Dashboard",
                style={"textAlign": "center", "color": "#00e5ff",
                       "fontFamily": "monospace", "marginBottom": "20px"}),

        # Row 1: depth + mid-price
        html.Div([
            html.Div([dcc.Graph(figure=build_depth_chart(result))],
                     style={"width": "40%", "display": "inline-block"}),
            html.Div([dcc.Graph(figure=build_midprice_spread_chart(result))],
                     style={"width": "60%", "display": "inline-block"}),
        ]),
    ]

    # Row 2: Hawkes (only if params available)
    if all(x is not None for x in [bid_times, ask_times, bid_params, ask_params]):
        children.append(dcc.Graph(figure=build_hawkes_chart(
            bid_times, ask_times, bid_params, ask_params, result.sim_duration_s
        )))

    # Row 3: Kyle + VPIN
    if all(x is not None for x in [kyle_result, vpin_result, mid_prices, signed_volumes]):
        children.append(dcc.Graph(figure=build_kyle_vpin_chart(
            kyle_result, vpin_result, mid_prices, signed_volumes
        )))

    # Row 4: P&L
    if backtest_results:
        children.append(dcc.Graph(figure=build_pnl_chart(backtest_results)))

    app.layout = html.Div(
        children,
        style={"backgroundColor": "#111", "padding": "20px"},
    )

    print(f"\n  Dashboard running at  http://localhost:{port}")
    print("  Press Ctrl+C to stop.\n")
    app.run(debug=debug, port=port)
