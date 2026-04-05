#!/usr/bin/env python3
"""Live Dash dashboard for walk-forward GA training.

Polls the database every 2 seconds to show:
  1. Header: run status, elapsed time, current window
  2. GA fitness chart for the current window (best + avg per generation)
  3. Per-window results table (grows as windows complete)
  4. OOS return bar chart (green=positive, red=negative)

Usage:
    python dashboards/walk_forward_monitor.py
    python dashboards/walk_forward_monitor.py --port 8051
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Add project root so imports work when running from any directory
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
load_dotenv(_project_root / ".env")

import plotly.graph_objects as go  # noqa: E402
from dash import Dash, Input, Output, callback, dash_table, dcc, html  # noqa: E402
from sqlalchemy import text  # noqa: E402

from aether_btc.data.database import get_session, init_db  # noqa: E402

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Aether BTC — Walk-Forward Monitor"

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={"fontFamily": "monospace", "padding": "20px", "backgroundColor": "#0d1117", "color": "#c9d1d9", "minHeight": "100vh"},
    children=[
        dcc.Interval(id="refresh", interval=2000, n_intervals=0),

        # Header
        html.Div(
            id="header",
            style={
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "padding": "16px 24px", "marginBottom": "20px",
                "backgroundColor": "#161b22", "borderRadius": "8px", "border": "1px solid #30363d",
            },
            children=[
                html.Div([
                    html.H2("Walk-Forward GA Training", style={"margin": "0", "color": "#58a6ff"}),
                    html.Span(id="run-id", style={"color": "#8b949e", "fontSize": "14px"}),
                ]),
                html.Div(id="header-stats", style={"display": "flex", "gap": "32px"}),
            ],
        ),

        # Progress bar
        html.Div(id="progress-bar-container", style={"marginBottom": "20px"}),

        # GA Fitness Chart
        html.Div(
            style={"marginBottom": "20px", "backgroundColor": "#161b22", "borderRadius": "8px", "border": "1px solid #30363d", "padding": "16px"},
            children=[
                html.H3("GA Fitness — Current Window", style={"margin": "0 0 8px 0", "color": "#58a6ff"}),
                dcc.Graph(id="fitness-chart", config={"displayModeBar": False}, style={"height": "320px"}),
            ],
        ),

        # Per-Window Results Table
        html.Div(
            style={"marginBottom": "20px", "backgroundColor": "#161b22", "borderRadius": "8px", "border": "1px solid #30363d", "padding": "16px"},
            children=[
                html.H3("Per-Window Results", style={"margin": "0 0 8px 0", "color": "#58a6ff"}),
                html.Div(id="results-table"),
            ],
        ),

        # OOS Bar Chart
        html.Div(
            style={"backgroundColor": "#161b22", "borderRadius": "8px", "border": "1px solid #30363d", "padding": "16px"},
            children=[
                html.H3("OOS Return by Window", style={"margin": "0 0 8px 0", "color": "#58a6ff"}),
                dcc.Graph(id="oos-bar-chart", config={"displayModeBar": False}, style={"height": "280px"}),
            ],
        ),
    ],
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", family="monospace"),
    margin=dict(l=50, r=20, t=10, b=40),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def _stat_box(label: str, value: str, color: str = "#c9d1d9") -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": "#8b949e", "textTransform": "uppercase"}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": "bold", "color": color}),
    ])


def _get_latest_run_id(session) -> str | None:
    """Find the most recent walk-forward run_id."""
    row = session.execute(
        text("SELECT run_id FROM walk_forward_progress ORDER BY created_at DESC LIMIT 1")
    ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Callback — polls every 2s
# ---------------------------------------------------------------------------
@callback(
    Output("run-id", "children"),
    Output("header-stats", "children"),
    Output("progress-bar-container", "children"),
    Output("fitness-chart", "figure"),
    Output("results-table", "children"),
    Output("oos-bar-chart", "figure"),
    Input("refresh", "n_intervals"),
)
def update(n_intervals):
    session = get_session()
    try:
        return _update_inner(session)
    finally:
        session.close()


def _update_inner(session):
    run_id = _get_latest_run_id(session)

    empty_fig = go.Figure(layout=PLOT_LAYOUT)

    empty_progress = html.Div()

    if not run_id:
        return (
            "Waiting for training run...",
            [_stat_box("Status", "IDLE", "#8b949e")],
            empty_progress,
            empty_fig,
            html.Div("No data yet.", style={"color": "#8b949e", "padding": "20px"}),
            empty_fig,
        )

    # ---- Window-level data ----
    windows = session.execute(
        text("""
            SELECT current_window, total_windows, status, train_start, train_end,
                   test_start, test_end, train_fitness, oos_sharpe, oos_return,
                   oos_max_dd, oos_trades, oos_win_rate, oos_calmar, generations_run,
                   window_elapsed_seconds, created_at
            FROM walk_forward_progress
            WHERE run_id = :rid
            ORDER BY current_window
        """),
        {"rid": run_id},
    ).fetchall()

    if not windows:
        return (
            f"Run: {run_id}",
            [_stat_box("Status", "STARTING", "#e3b341")],
            empty_progress,
            empty_fig,
            html.Div("Waiting for first window...", style={"color": "#8b949e", "padding": "20px"}),
            empty_fig,
        )

    total_windows = windows[0][1]
    completed = [w for w in windows if w[2] == "complete"]
    active = [w for w in windows if w[2] != "complete"]
    current_win = active[0] if active else completed[-1] if completed else windows[-1]
    current_idx = current_win[0]

    # Elapsed from first window created_at
    first_created = windows[0][16]
    if isinstance(first_created, str):
        first_created = datetime.fromisoformat(first_created)
    elapsed = datetime.now(timezone.utc) - first_created.replace(tzinfo=timezone.utc) if first_created.tzinfo is None else datetime.now(timezone.utc) - first_created
    elapsed_str = f"{int(elapsed.total_seconds()) // 60}m {int(elapsed.total_seconds()) % 60}s"

    is_all_done = len(completed) == total_windows
    status_text = "COMPLETE" if is_all_done else "TRAINING"
    status_color = "#3fb950" if is_all_done else "#e3b341"

    # ---- ETA calculation ----
    eta_str = "-"
    if completed and not is_all_done:
        avg_window_secs = sum(w[15] for w in completed if w[15]) / len(completed)
        remaining_windows = total_windows - len(completed)
        # Account for partial progress on current window via GA generations
        current_gen_count = 0
        current_total_gens = 0
        if active:
            window_run_id_eta = f"wf_{run_id}_w{current_idx}"
            ga_count = session.execute(
                text("SELECT COUNT(*), MAX(generation) FROM ga_training_progress WHERE run_id = :rid"),
                {"rid": window_run_id_eta},
            ).fetchone()
            if ga_count and ga_count[0] > 0:
                current_gen_count = ga_count[1] + 1  # 0-indexed
                # Estimate total gens from completed windows' average
                avg_gens = sum(w[14] for w in completed if w[14]) / len(completed) if completed else 50
                current_total_gens = max(avg_gens, current_gen_count)
                partial_done = current_gen_count / current_total_gens
                remaining_windows = (total_windows - len(completed) - partial_done)

        eta_secs = remaining_windows * avg_window_secs
        if eta_secs >= 3600:
            eta_str = f"{eta_secs / 3600:.1f}h"
        else:
            eta_str = f"{eta_secs / 60:.0f}m"
    elif is_all_done:
        eta_str = "Done"

    # ---- Header stats ----
    header_stats = [
        _stat_box("Status", status_text, status_color),
        _stat_box("Window", f"{current_idx + 1} / {total_windows}"),
        _stat_box("Elapsed", elapsed_str),
        _stat_box("Completed", str(len(completed))),
        _stat_box("ETA", eta_str, "#58a6ff" if not is_all_done else "#3fb950"),
    ]

    # ---- GA Fitness chart for current window ----
    window_run_id = f"wf_{run_id}_w{current_idx}"
    ga_rows = session.execute(
        text("""
            SELECT generation, best_fitness, avg_fitness
            FROM ga_training_progress
            WHERE run_id = :rid
            ORDER BY generation
        """),
        {"rid": window_run_id},
    ).fetchall()

    if ga_rows:
        gens = [r[0] for r in ga_rows]
        best = [r[1] for r in ga_rows]
        avg = [r[2] for r in ga_rows]
        fitness_fig = go.Figure(layout=PLOT_LAYOUT)
        fitness_fig.add_trace(go.Scatter(x=gens, y=best, mode="lines", name="Best", line=dict(color="#3fb950", width=2)))
        fitness_fig.add_trace(go.Scatter(x=gens, y=avg, mode="lines", name="Avg", line=dict(color="#58a6ff", width=1, dash="dot")))
        fitness_fig.update_layout(xaxis_title="Generation", yaxis_title="Fitness")
    else:
        fitness_fig = empty_fig

    # ---- Per-window results table ----
    if completed:
        table_data = []
        for w in completed:
            table_data.append({
                "Window": w[0] + 1,
                "Train Period": f"{w[3]} → {w[4]}",
                "Test Period": f"{w[5]} → {w[6]}",
                "Train Fit": f"{w[7]:.3f}" if w[7] is not None else "-",
                "OOS Sharpe": f"{w[8]:.2f}" if w[8] is not None else "-",
                "OOS Return": f"{w[9]:.1%}" if w[9] is not None else "-",
                "OOS MaxDD": f"{w[10]:.1%}" if w[10] is not None else "-",
                "Trades": w[11] or 0,
                "Win Rate": f"{w[12]:.1%}" if w[12] is not None else "-",
                "Time": f"{w[15]:.0f}s" if w[15] is not None else "-",
            })

        results_table = dash_table.DataTable(
            data=table_data,
            columns=[{"name": k, "id": k} for k in table_data[0]],
            style_header={
                "backgroundColor": "#21262d", "color": "#58a6ff",
                "fontWeight": "bold", "border": "1px solid #30363d",
                "fontFamily": "monospace", "fontSize": "13px",
            },
            style_cell={
                "backgroundColor": "#0d1117", "color": "#c9d1d9",
                "border": "1px solid #21262d", "textAlign": "center",
                "fontFamily": "monospace", "fontSize": "13px",
                "padding": "8px",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#161b22"},
            ],
        )
    else:
        results_table = html.Div("No windows completed yet.", style={"color": "#8b949e", "padding": "20px"})

    # ---- OOS return bar chart ----
    if completed:
        win_labels = [f"W{w[0] + 1}" for w in completed]
        oos_returns = [w[9] if w[9] is not None else 0 for w in completed]
        bar_colors = ["#3fb950" if r >= 0 else "#f85149" for r in oos_returns]

        oos_fig = go.Figure(layout=PLOT_LAYOUT)
        oos_fig.add_trace(go.Bar(
            x=win_labels, y=[r * 100 for r in oos_returns],
            marker_color=bar_colors, text=[f"{r:.1%}" for r in oos_returns],
            textposition="outside", textfont=dict(color="#c9d1d9", size=11),
        ))
        oos_fig.update_layout(xaxis_title="Window", yaxis_title="OOS Return (%)")
    else:
        oos_fig = empty_fig

    # ---- Progress bar ----
    pct = len(completed) / total_windows * 100 if total_windows else 0
    progress_bar = html.Div(
        style={
            "backgroundColor": "#161b22", "borderRadius": "8px",
            "border": "1px solid #30363d", "padding": "12px 24px",
        },
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"},
                children=[
                    html.Span(f"Overall Progress", style={"color": "#8b949e", "fontSize": "12px"}),
                    html.Span(
                        f"{len(completed)}/{total_windows} windows — {pct:.0f}%",
                        style={"color": "#c9d1d9", "fontSize": "12px"},
                    ),
                ],
            ),
            html.Div(
                style={
                    "width": "100%", "height": "8px", "backgroundColor": "#21262d",
                    "borderRadius": "4px", "overflow": "hidden",
                },
                children=[
                    html.Div(style={
                        "width": f"{pct}%", "height": "100%",
                        "backgroundColor": "#3fb950" if is_all_done else "#58a6ff",
                        "borderRadius": "4px", "transition": "width 0.5s ease",
                    }),
                ],
            ),
        ],
    )

    return (
        f"Run: {run_id}",
        header_stats,
        progress_bar,
        fitness_fig,
        results_table,
        oos_fig,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward training dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_db()
    print(f"Dashboard running at http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
