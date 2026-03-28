from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
import numpy as np

if TYPE_CHECKING:
    from feed.simulator import SimulationResult
    from backtest.engine import BacktestResult

def _require_dash():
    try:
        import dash
        from dash import dcc, html
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return dash, dcc, html, go, make_subplots
    except ImportError:
        raise ImportError("Install with: pip install dash plotly")

def build_depth_chart(result):
    _, _, _, go, _ = _require_dash()
    bid_depth = result.book.bid_depth(20)
    ask_depth = result.book.ask_depth(20)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[p for p,_ in bid_depth], y=[v for _,v in bid_depth],
                         name="Bid", marker_color="rgba(0,200,100,0.7)"))
    fig.add_trace(go.Bar(x=[p for p,_ in ask_depth], y=[v for _,v in ask_depth],
                         name="Ask", marker_color="rgba(220,50,50,0.7)"))
    fig.update_layout(title="Order Book Depth (Top 20 Levels)",
                      xaxis_title="Price", yaxis_title="Volume",
                      barmode="overlay", template="plotly_dark", height=350)
    return fig

def build_midprice_spread_chart(result):
    _, _, _, go, make_subplots = _require_dash()
    snaps = result.book.snapshots
    if not snaps:
        return go.Figure()
    ts = [s.timestamp / 1e9 for s in snaps]
    ts_rel = [t - ts[0] for t in ts]
    mids = [s.mid_price for s in snaps]
    spreads = [s.spread for s in snaps]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Mid-Price", "Spread (bps)"))
    fig.add_trace(go.Scatter(x=ts_rel, y=mids, mode="lines",
                             line=dict(color="cyan", width=1)), row=1, col=1)
    spread_bps = [(sp/mp*10000) if sp and mp else None for sp, mp in zip(spreads, mids)]
    fig.add_trace(go.Scatter(x=ts_rel, y=spread_bps, mode="lines",
                             line=dict(color="orange", width=1)), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=450,
                      title="Mid-Price and Spread")
    return fig

def launch_dashboard(result, port=8050, **kwargs):
    dash, dcc, html, go, _ = _require_dash()
    app = dash.Dash(__name__, title="LOB Dashboard")
    app.layout = html.Div([
        html.H1("Limit Order Book Dashboard",
                style={"textAlign":"center","color":"#00e5ff","fontFamily":"monospace"}),
        html.Div([
            html.Div([dcc.Graph(figure=build_depth_chart(result))],
                     style={"width":"40%","display":"inline-block"}),
            html.Div([dcc.Graph(figure=build_midprice_spread_chart(result))],
                     style={"width":"60%","display":"inline-block"}),
        ]),
    ], style={"backgroundColor":"#111","padding":"20px"})
    print(f"\n  Dashboard running at http://localhost:{port}\n")
    app.run(debug=False, port=port)
