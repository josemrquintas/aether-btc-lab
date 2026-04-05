#!/usr/bin/env python3
"""Live BTC Dashboard — multi-timeframe candlestick chart with indicators.

Reads from DB (populated by scripts/refresh_btc.py).
Auto-refreshes every 60 seconds.

Usage:
    python dashboards/btc_live.py
    python dashboards/btc_live.py --port 8051
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
load_dotenv(_project_root / ".env")

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from dash import Dash, Input, Output, callback, dcc, html  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from sqlalchemy import text  # noqa: E402

from aether_btc.data.database import (  # noqa: E402
    TIMEFRAMES,
    _indicators_table_name,
    _live_state_table_name,
    _candle_table_name,
    get_engine,
    init_db,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INCREASING_COLOR = "#26a69a"
DECREASING_COLOR = "#ef5350"
VOLUME_INCREASING = "rgba(38, 166, 154, 0.5)"
VOLUME_DECREASING = "rgba(239, 83, 80, 0.5)"
BUY_COLOR = "#00e676"
SHORT_COLOR = "#ff1744"

INDICATOR_COLORS = {
    "sma_20": "#ffeb3b", "sma_50": "#ff9800", "sma_200": "#e91e63",
    "bb_upper": "rgba(33, 150, 243, 0.4)", "bb_middle": "rgba(33, 150, 243, 0.8)",
    "bb_lower": "rgba(33, 150, 243, 0.4)",
    "supertrend": "#00e676",
    "rsi": "#ffc107",
    "macd": "#2196f3", "macd_signal": "#ff5722",
    "macd_histogram_pos": "rgba(38, 166, 154, 0.6)",
    "macd_histogram_neg": "rgba(239, 83, 80, 0.6)",
    "adx": "#26c6da", "adx_pos": "#66bb6a", "adx_neg": "#ef5350",
    "stoch_k": "#29b6f6", "stoch_d": "#ef5350",
}

INDICATOR_CATALOG = {
    "SMA 20": {"cols": ["sma_20"], "type": "overlay"},
    "SMA 50": {"cols": ["sma_50"], "type": "overlay"},
    "SMA 200": {"cols": ["sma_200"], "type": "overlay"},
    "Bollinger Bands": {"cols": ["bb_upper", "bb_middle", "bb_lower"], "type": "overlay"},
    "Supertrend": {"cols": ["supertrend", "supertrend_direction"], "type": "overlay"},
    "RSI": {"cols": ["rsi"], "type": "sub"},
    "MACD": {"cols": ["macd", "macd_signal", "macd_histogram"], "type": "sub"},
    "ADX": {"cols": ["adx", "adx_pos", "adx_neg"], "type": "sub"},
    "Stochastic": {"cols": ["stoch_k", "stoch_d"], "type": "sub"},
}

DEFAULT_INDICATORS = ["SMA 20", "SMA 50", "Bollinger Bands", "RSI", "MACD"]

SIGNAL_SOURCES = {
    "GA Model": {"type": "ga"},
    "Momentum": {"col": "sig_momentum", "conf_col": "sig_momentum_conf"},
    "Mean Reversion": {"col": "sig_mean_reversion", "conf_col": "sig_mean_reversion_conf"},
    "Trend Following": {"col": "sig_trend_following", "conf_col": "sig_trend_following_conf"},
    "Volatility Breakout": {"col": "sig_volatility_breakout", "conf_col": "sig_volatility_breakout_conf"},
    "Funding/Volume": {"col": "sig_funding_volume", "conf_col": "sig_funding_volume_conf"},
    "Consensus (2+)": {"type": "consensus"},
    "None": {"type": "none"},
}

# Timeframe display labels
TIMEFRAME_OPTIONS = [
    {"label": "15 min", "value": "15m"},
    {"label": "1 hour", "value": "1h"},
    {"label": "4 hours", "value": "4h"},
    {"label": "1 day", "value": "1d"},
]

# Default date range per timeframe (in days)
DEFAULT_DATE_RANGE = {"15m": 7, "1h": 14, "4h": 30, "1d": 90}

# Date range options per timeframe
DATE_RANGE_OPTIONS = {
    "15m": [
        {"label": "24h", "value": 1},
        {"label": "3d", "value": 3},
        {"label": "7d", "value": 7},
        {"label": "14d", "value": 14},
        {"label": "30d", "value": 30},
    ],
    "1h": [
        {"label": "3d", "value": 3},
        {"label": "7d", "value": 7},
        {"label": "14d", "value": 14},
        {"label": "30d", "value": 30},
        {"label": "90d", "value": 90},
    ],
    "4h": [
        {"label": "7d", "value": 7},
        {"label": "14d", "value": 14},
        {"label": "30d", "value": 30},
        {"label": "90d", "value": 90},
        {"label": "180d", "value": 180},
    ],
    "1d": [
        {"label": "30d", "value": 30},
        {"label": "90d", "value": 90},
        {"label": "180d", "value": 180},
        {"label": "1y", "value": 365},
        {"label": "2y", "value": 730},
    ],
}

DARK_BG = "#0d1117"
CARD_BG = "#161b22"
BORDER = "#30363d"
TEXT_PRIMARY = "#c9d1d9"
TEXT_SECONDARY = "#8b949e"
ACCENT = "#58a6ff"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRIMARY, family="monospace"),
    margin=dict(l=40, r=10, t=10, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
    xaxis_rangeslider_visible=False,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Aether BTC — Live Dashboard"

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            body { margin: 0; overflow-x: hidden; }
            .Select-control { background-color: #161b22 !important; border-color: #30363d !important; }
            .Select-value-label, .Select-placeholder { color: #c9d1d9 !important; }
            .Select-menu-outer { background-color: #161b22 !important; border-color: #30363d !important; }
            .VirtualizedSelectOption { color: #c9d1d9 !important; }
            .VirtualizedSelectFocusedOption { background-color: #21262d !important; }
            @media (max-width: 768px) {
                .js-plotly-plot .plotly .modebar { display: none !important; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={"fontFamily": "monospace", "padding": "12px", "backgroundColor": DARK_BG,
           "color": TEXT_PRIMARY, "minHeight": "100vh",
           "maxWidth": "100vw", "overflowX": "hidden"},
    children=[
        dcc.Interval(id="refresh", interval=60_000, n_intervals=0),

        # Signal banner
        html.Div(
            id="signal-banner",
            style={
                "display": "flex", "flexWrap": "wrap",
                "justifyContent": "space-between", "alignItems": "center",
                "gap": "12px",
                "padding": "12px 16px", "marginBottom": "12px",
                "backgroundColor": CARD_BG, "borderRadius": "8px", "border": f"1px solid {BORDER}",
            },
            children=[
                html.Div([
                    html.H2("BTC/USDT Live", style={"margin": "0", "color": ACCENT, "fontSize": "clamp(16px, 4vw, 24px)"}),
                    html.Span(id="last-update", style={"color": TEXT_SECONDARY, "fontSize": "11px"}),
                ]),
                html.Div(id="signal-stats", style={
                    "display": "flex", "flexWrap": "wrap", "gap": "16px",
                }),
            ],
        ),

        # Controls row
        html.Div(
            style={
                "display": "flex", "flexWrap": "wrap", "gap": "12px", "marginBottom": "12px",
                "padding": "10px 16px", "alignItems": "flex-end",
                "backgroundColor": CARD_BG, "borderRadius": "8px", "border": f"1px solid {BORDER}",
            },
            children=[
                html.Div([
                    html.Label("Timeframe", style={"fontSize": "11px", "color": TEXT_SECONDARY}),
                    dcc.Dropdown(
                        id="timeframe-selector",
                        options=TIMEFRAME_OPTIONS,
                        value="15m",
                        clearable=False,
                        style={"minWidth": "100px", "color": "#000"},
                    ),
                ], style={"flex": "0 0 auto"}),
                html.Div([
                    html.Label("Date Range", style={"fontSize": "11px", "color": TEXT_SECONDARY}),
                    dcc.Dropdown(
                        id="date-range",
                        options=DATE_RANGE_OPTIONS["15m"],
                        value=7,
                        clearable=False,
                        style={"minWidth": "90px", "color": "#000"},
                    ),
                ], style={"flex": "0 0 auto"}),
                html.Div([
                    html.Label("Signal Source", style={"fontSize": "11px", "color": TEXT_SECONDARY}),
                    dcc.Dropdown(
                        id="signal-source",
                        options=[{"label": k, "value": k} for k in SIGNAL_SOURCES],
                        value="GA Model",
                        clearable=False,
                        style={"minWidth": "140px", "color": "#000"},
                    ),
                ], style={"flex": "0 0 auto"}),
                html.Div([
                    html.Label("Indicators", style={"fontSize": "11px", "color": TEXT_SECONDARY}),
                    dcc.Dropdown(
                        id="indicator-selector",
                        options=[{"label": k, "value": k} for k in INDICATOR_CATALOG],
                        value=DEFAULT_INDICATORS,
                        multi=True,
                        style={"minWidth": "200px", "color": "#000"},
                    ),
                ], style={"flex": "1 1 200px"}),
            ],
        ),

        # Main chart
        html.Div(
            style={"backgroundColor": CARD_BG, "borderRadius": "8px",
                    "border": f"1px solid {BORDER}", "padding": "8px",
                    "overflowX": "auto"},
            children=[
                dcc.Graph(
                    id="main-chart",
                    config={"displayModeBar": "hover", "scrollZoom": True,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                    style={"minHeight": "400px"},
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def _get_live_state(engine, interval: str = "15m") -> dict | None:
    table = _live_state_table_name(interval)
    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT * FROM {table} WHERE pair = 'BTCUSDT' LIMIT 1")
        ).fetchone()

    if not row:
        return None
    return dict(row._mapping)


def _get_ohlc_with_indicators(engine, pair: str, days: int, interval: str = "15m") -> pd.DataFrame:
    table = _indicators_table_name(interval)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT timestamp, open, high, low, close, volume,
                       indicators, signal_type, signal_score, signal_confidence
                FROM {table}
                WHERE pair = :pair AND timestamp >= :cutoff
                ORDER BY timestamp
            """),
            {"pair": pair, "cutoff": cutoff},
        ).fetchall()

    if not rows:
        return _get_raw_candles(engine, pair, days, interval)

    records = []
    for r in rows:
        row_dict = dict(r._mapping)
        ind = row_dict.pop("indicators", None)
        if ind:
            if isinstance(ind, str):
                ind = json.loads(ind)
            row_dict.update(ind)
        records.append(row_dict)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def _get_raw_candles(engine, pair: str, days: int, interval: str = "15m") -> pd.DataFrame:
    """Fallback: load raw candles if indicators table is empty."""
    table = _candle_table_name(interval)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT timestamp, open, high, low, close, volume
                FROM {table}
                WHERE pair = :pair AND timestamp >= :cutoff
                ORDER BY timestamp
            """),
            {"pair": pair, "cutoff": cutoff},
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r._mapping) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Chart building
# ---------------------------------------------------------------------------
def _add_candlestick(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing_line_color=INCREASING_COLOR,
            decreasing_line_color=DECREASING_COLOR,
            name="OHLC", showlegend=False,
        ),
        row=row, col=1,
    )


def _add_volume(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    colors = [
        VOLUME_INCREASING if c >= o else VOLUME_DECREASING
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], marker_color=colors,
               name="Volume", showlegend=False),
        row=row, col=1,
    )


def _add_overlay(fig: go.Figure, df: pd.DataFrame, name: str, row: int) -> None:
    entry = INDICATOR_CATALOG.get(name)
    if not entry:
        return

    if name == "Bollinger Bands":
        for col, dash, show in [("bb_upper", None, False), ("bb_lower", None, True), ("bb_middle", "dot", False)]:
            if col not in df.columns:
                continue
            kwargs = dict(
                x=df.index, y=df[col], mode="lines",
                line=dict(width=1, color=INDICATOR_COLORS.get(col, "#fff"), dash=dash),
                name=name if show else f"{name} {col.split('_')[1].title()}",
                showlegend=show,
            )
            if col == "bb_lower":
                kwargs["fill"] = "tonexty"
                kwargs["fillcolor"] = "rgba(33, 150, 243, 0.08)"
            fig.add_trace(go.Scatter(**kwargs), row=row, col=1)
        return

    if name == "Supertrend":
        if "supertrend" in df.columns and "supertrend_direction" in df.columns:
            st = df[["supertrend", "supertrend_direction"]].dropna()
            if not st.empty:
                bull = st[st["supertrend_direction"] == 1]
                bear = st[st["supertrend_direction"] == -1]
                if not bull.empty:
                    fig.add_trace(go.Scatter(
                        x=bull.index, y=bull["supertrend"], mode="markers+lines",
                        line=dict(width=1.5, color="#00e676"), marker=dict(size=0),
                        name="ST Bull", showlegend=False,
                    ), row=row, col=1)
                if not bear.empty:
                    fig.add_trace(go.Scatter(
                        x=bear.index, y=bear["supertrend"], mode="markers+lines",
                        line=dict(width=1.5, color="#ff1744"), marker=dict(size=0),
                        name="ST Bear", showlegend=False,
                    ), row=row, col=1)
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines",
                    line=dict(width=1.5, color="#00e676"), name="Supertrend",
                ), row=row, col=1)
        return

    for col in entry["cols"]:
        if col in df.columns:
            color = INDICATOR_COLORS.get(col, "#ffffff")
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines",
                line=dict(width=1, color=color), name=name,
            ), row=row, col=1)


def _add_signal_markers(fig: go.Figure, df: pd.DataFrame, row: int, signal_source: str = "GA Model") -> None:
    source = SIGNAL_SOURCES.get(signal_source, {})
    source_type = source.get("type", "strategy")

    if source_type == "none":
        return

    if source_type == "ga":
        if "signal_type" not in df.columns:
            return
        buy_mask = df["signal_type"] == "BUY"
        short_mask = df["signal_type"] == "SHORT"
        conf_col = "signal_confidence"

    elif source_type == "consensus":
        buy_cols = [c for c in df.columns if c.startswith("sig_") and not c.endswith("_conf")
                    and c != "signal_type"]
        if not buy_cols:
            return
        sig_sum = df[buy_cols].sum(axis=1)
        buy_mask = sig_sum >= 2
        short_mask = sig_sum <= -2
        conf_cols = [c for c in df.columns if c.startswith("sig_") and c.endswith("_conf")]
        if conf_cols:
            df = df.copy()
            df["_consensus_conf"] = df[conf_cols].max(axis=1)
            conf_col = "_consensus_conf"
        else:
            conf_col = None

    else:
        sig_col = source.get("col", "")
        conf_col_name = source.get("conf_col", "")
        if sig_col not in df.columns:
            return
        buy_mask = df[sig_col] == 1
        short_mask = df[sig_col] == -1
        conf_col = conf_col_name if conf_col_name in df.columns else None

    buys = df[buy_mask]
    if not buys.empty:
        hover_parts = [f"<b>BUY</b> ({signal_source})"]
        customdata = None
        if conf_col and conf_col in buys.columns:
            hover_parts.append("Conf: %{customdata[0]:.3f}")
            customdata = buys[[conf_col]].fillna(0).values
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["low"] * 0.998,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10, color=BUY_COLOR),
            name="BUY",
            customdata=customdata,
            hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
        ), row=row, col=1)

    shorts = df[short_mask]
    if not shorts.empty:
        hover_parts = [f"<b>SHORT</b> ({signal_source})"]
        customdata = None
        if conf_col and conf_col in shorts.columns:
            hover_parts.append("Conf: %{customdata[0]:.3f}")
            customdata = shorts[[conf_col]].fillna(0).values
        fig.add_trace(go.Scatter(
            x=shorts.index, y=shorts["high"] * 1.002,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10, color=SHORT_COLOR),
            name="SHORT",
            customdata=customdata,
            hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
        ), row=row, col=1)


def _add_rsi(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    if "rsi" not in df.columns:
        return
    fig.add_trace(go.Scatter(
        x=df.index, y=df["rsi"], mode="lines",
        line=dict(width=1.5, color=INDICATOR_COLORS["rsi"]),
        name="RSI", showlegend=False,
    ), row=row, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=row, col=1)
    fig.update_yaxes(range=[0, 100], row=row, col=1)


def _add_macd(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    if "macd" not in df.columns:
        return
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd"], mode="lines",
        line=dict(width=1.5, color=INDICATOR_COLORS["macd"]),
        name="MACD", showlegend=False,
    ), row=row, col=1)
    if "macd_signal" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd_signal"], mode="lines",
            line=dict(width=1.5, color=INDICATOR_COLORS["macd_signal"]),
            name="Signal", showlegend=False,
        ), row=row, col=1)
    if "macd_histogram" in df.columns:
        colors = [
            INDICATOR_COLORS["macd_histogram_pos"] if v >= 0
            else INDICATOR_COLORS["macd_histogram_neg"]
            for v in df["macd_histogram"].fillna(0)
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["macd_histogram"],
            marker_color=colors, name="Histogram", showlegend=False,
        ), row=row, col=1)


def _add_adx(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    if "adx" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["adx"], mode="lines",
            line=dict(width=1.5, color=INDICATOR_COLORS["adx"]),
            name="ADX", showlegend=False,
        ), row=row, col=1)
    if "adx_pos" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["adx_pos"], mode="lines",
            line=dict(width=1, color=INDICATOR_COLORS["adx_pos"], dash="dash"),
            name="+DI", showlegend=False,
        ), row=row, col=1)
    if "adx_neg" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["adx_neg"], mode="lines",
            line=dict(width=1, color=INDICATOR_COLORS["adx_neg"], dash="dash"),
            name="-DI", showlegend=False,
        ), row=row, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=row, col=1)


def _add_stochastic(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    if "stoch_k" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["stoch_k"], mode="lines",
            line=dict(width=1.5, color=INDICATOR_COLORS["stoch_k"]),
            name="%K", showlegend=False,
        ), row=row, col=1)
    if "stoch_d" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["stoch_d"], mode="lines",
            line=dict(width=1.5, color=INDICATOR_COLORS["stoch_d"]),
            name="%D", showlegend=False,
        ), row=row, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=row, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=row, col=1)
    fig.update_yaxes(range=[0, 100], row=row, col=1)


SUB_PANEL_RENDERERS = {
    "RSI": _add_rsi,
    "MACD": _add_macd,
    "ADX": _add_adx,
    "Stochastic": _add_stochastic,
}


def build_chart(df: pd.DataFrame, selected_indicators: list[str], signal_source: str = "GA Model") -> go.Figure:
    """Build the complete financial chart with dynamic subplots."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            annotations=[dict(text="No data available", showarrow=False,
                              xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=20))],
        )
        return fig

    sub_panels = []
    for name in selected_indicators:
        entry = INDICATOR_CATALOG.get(name)
        if entry and entry["type"] == "sub":
            sub_panels.append(name)

    n_rows = 2 + len(sub_panels)
    row_heights = [0.55, 0.15] + [0.15] * len(sub_panels)
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    subplot_titles = ["", "Volume"] + sub_panels

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    _add_candlestick(fig, df, row=1)

    for name in selected_indicators:
        entry = INDICATOR_CATALOG.get(name)
        if entry and entry["type"] == "overlay":
            _add_overlay(fig, df, name, row=1)

    _add_signal_markers(fig, df, row=1, signal_source=signal_source)
    _add_volume(fig, df, row=2)

    for i, name in enumerate(sub_panels):
        renderer = SUB_PANEL_RENDERERS.get(name)
        if renderer:
            renderer(fig, df, row=3 + i)

    fig.update_layout(
        **PLOT_LAYOUT,
        template="plotly_dark",
        height=600 + len(sub_panels) * 120,
        uirevision="chart",
    )

    for annotation in fig.layout.annotations:
        annotation.font = dict(size=11, color=TEXT_SECONDARY)

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stat_box(label: str, value: str, color: str = TEXT_PRIMARY) -> html.Div:
    return html.Div([
        html.Div(label, className="stat-label",
                 style={"fontSize": "10px", "color": TEXT_SECONDARY, "textTransform": "uppercase"}),
        html.Div(value, className="stat-value",
                 style={"fontSize": "clamp(14px, 3vw, 20px)", "fontWeight": "bold",
                         "color": color, "whiteSpace": "nowrap"}),
    ], className="stat-box", style={"minWidth": "70px"})


def _signal_color(signal_type: str) -> str:
    if signal_type == "BUY":
        return BUY_COLOR
    elif signal_type == "SHORT":
        return SHORT_COLOR
    return TEXT_SECONDARY


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# Main chart update
@callback(
    Output("last-update", "children"),
    Output("signal-stats", "children"),
    Output("main-chart", "figure"),
    Output("date-range", "options"),
    Output("date-range", "value"),
    Input("refresh", "n_intervals"),
    Input("date-range", "value"),
    Input("signal-source", "value"),
    Input("indicator-selector", "value"),
    Input("timeframe-selector", "value"),
)
def update(n_intervals, days, signal_source, selected_indicators, timeframe):
    engine = get_engine()
    selected_indicators = selected_indicators or DEFAULT_INDICATORS
    signal_source = signal_source or "GA Model"
    timeframe = timeframe or "15m"

    # Live state
    state = _get_live_state(engine, timeframe)

    if state:
        signal_type = state.get("signal_type", "HOLD")
        price = state.get("price", 0)
        score = state.get("signal_score", 0.5)
        confidence = state.get("signal_confidence", 0)
        model_name = state.get("model_name", "-")
        updated = state.get("updated_at", "")

        strategy_signals = state.get("strategy_signals")
        if isinstance(strategy_signals, str):
            strategy_signals = json.loads(strategy_signals)

        source_cfg = SIGNAL_SOURCES.get(signal_source, {})
        if source_cfg.get("type") not in ("ga", "none", "consensus", None):
            col = source_cfg.get("col", "")
            strat_name = col.replace("sig_", "") if col else ""
            if strategy_signals and strat_name in strategy_signals:
                s = strategy_signals[strat_name]
                sig_val = s.get("signal", 0)
                sig_conf = s.get("confidence", 0)
                if sig_val == 1:
                    signal_type = "BUY"
                elif sig_val == -1:
                    signal_type = "SHORT"
                else:
                    signal_type = "HOLD"
                confidence = sig_conf

        tf_label = timeframe.upper()
        stats = [
            _stat_box("Timeframe", tf_label, ACCENT),
            _stat_box("Signal", signal_type, _signal_color(signal_type)),
            _stat_box("Price", f"${price:,.2f}"),
            _stat_box("Score", f"{score:.4f}"),
            _stat_box("Confidence", f"{confidence:.1%}"),
            _stat_box("Source", signal_source, ACCENT),
        ]
        update_text = f"Last update: {updated}"
    else:
        tf_label = timeframe.upper()
        stats = [
            _stat_box("Timeframe", tf_label, ACCENT),
            _stat_box("Status", "No data", TEXT_SECONDARY),
        ]
        update_text = f"Waiting for first {timeframe} refresh..."

    # Date range options for the selected timeframe
    tf_options = DATE_RANGE_OPTIONS.get(timeframe, DATE_RANGE_OPTIONS["15m"])
    tf_default = DEFAULT_DATE_RANGE.get(timeframe, 7)
    # Keep user's selection if it's valid for this timeframe, otherwise use default
    valid_values = [o["value"] for o in tf_options]
    if days not in valid_values:
        days = tf_default

    # Chart data
    df = _get_ohlc_with_indicators(engine, "BTCUSDT", days, interval=timeframe)
    fig = build_chart(df, selected_indicators, signal_source=signal_source)

    return update_text, stats, fig, tf_options, days


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC Live Dashboard")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_db()
    print(f"BTC Live Dashboard running at http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
