# -*- coding: utf-8 -*-
"""
econ_plots.py
=============
Interactive Plotly-based economics visualisations for the HyOps Streamlit app.

Chart types
-----------
  - Waterfall: revenue -> staff cost -> queue cost -> net result
  - Stacked bar: revenue vs cost breakdown per schedule
  - Net result spread: box plot across seeds per schedule
  - Sensitivity: net result vs arrival rate, per schedule
  - Sensitivity by pattern: isolate timing effect
  - Volume x timing grid: grouped bars
  - Arrival pattern preview: intensity curve over 24h

All functions return a plotly.graph_objects.Figure.
Use st.plotly_chart(fig, use_container_width=True) in Streamlit.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Colour palette ──────────────────────────────────────────
_PALETTE = [
    "#1D9E75", "#533483", "#4682B4", "#EF9F27",
    "#D85A30", "#E24B4A", "#639922", "#8B4513",
    "#2196F3", "#FF6B6B", "#4ECDC4", "#45B7D1",
]
_REVENUE = "#1D9E75"
_STAFF   = "#4682B4"
_QUEUE   = "#E24B4A"
_NET     = "#2C2C2A"

_LAYOUT = dict(
    plot_bgcolor="#F8F7F4",
    paper_bgcolor="white",
    font=dict(size=12),
    hovermode="x unified",
    margin=dict(l=60, r=20, t=50, b=60),
)

_MARKERS = ["circle", "square", "diamond", "cross", "triangle-up",
            "triangle-down", "star", "hexagon"]


def _fmt_kr(v):
    if abs(v) >= 1e6:
        return f"{v/1e6:,.1f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:,.0f}k"
    return f"{v:,.0f}"


# ──────────────────────────────────────────────────────────────
# 0. Arrival pattern preview
# ──────────────────────────────────────────────────────────────

def plot_arrival_pattern(arrival_pattern, avg_arrivals_per_day=None, title=None):
    hours = np.linspace(0, 24, 24 * 12, endpoint=False)
    intensity = np.array([arrival_pattern.rate_at_hour(h) for h in hours])

    if avg_arrivals_per_day is not None:
        y = intensity * avg_arrivals_per_day / 24.0
        y_label = "Expected arrivals / hour"
    else:
        y = intensity
        y_label = "Relative intensity"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=y, mode="lines", fill="tozeroy",
        line=dict(color=_STAFF, width=2),
        fillcolor="rgba(70, 130, 180, 0.15)",
        hovertemplate="Hour: %{x:.1f}<br>" + y_label + ": %{y:.2f}<extra></extra>",
    ))

    if arrival_pattern.pattern_type in ("single_peak", "double_peak"):
        fig.add_vline(x=arrival_pattern.peak_hour, line_dash="dash",
                      line_color=_QUEUE,
                      annotation_text=f"Peak 1: {arrival_pattern.peak_hour:.1f}h")
    if arrival_pattern.pattern_type == "double_peak":
        fig.add_vline(x=arrival_pattern.peak_hour_2, line_dash="dash",
                      line_color="#EF9F27",
                      annotation_text=f"Peak 2: {arrival_pattern.peak_hour_2:.1f}h")

    fig.update_layout(
        **_LAYOUT, height=350, showlegend=False,
        xaxis=dict(title="Hour of day", tickmode="linear", dtick=2, range=[0, 24]),
        yaxis=dict(title=y_label),
        title=title or f"Arrival timing — {arrival_pattern.pattern_type}",
    )
    return fig


# ──────────────────────────────────────────────────────────────
# 1. Box plot — spread of net result across seeds
# ──────────────────────────────────────────────────────────────

def plot_net_result_spread(headline_df, schedule_order, arrival_rate, n_runs,
                           value_col="net_kr_annual"):
    fig = go.Figure()

    for i, lbl in enumerate(schedule_order):
        vals = headline_df[headline_df["schedule_label"] == lbl][value_col].values
        fig.add_trace(go.Box(
            y=vals, name=lbl,
            marker_color=_PALETTE[i % len(_PALETTE)],
            boxmean=True,
            hovertemplate="%{y:,.0f} kr/yr<extra>" + lbl + "</extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(
        **_LAYOUT, height=450, showlegend=False,
        title=f"Net Result Spread  ({arrival_rate} arrivals/day, n={n_runs} seeds)",
        yaxis=dict(title="Net result (kr/year)", tickformat=",.0f"),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# 2. Stacked bar — revenue vs cost per schedule
# ──────────────────────────────────────────────────────────────

def plot_stacked_bar(summary_df, title_suffix=""):
    labels = summary_df["schedule_label"].astype(str).tolist()
    revenue = summary_df["revenue_kr_annual"].values
    staff   = summary_df["staff_cost_kr_annual"].values
    queue   = summary_df["queue_cost_kr_annual"].values
    net     = summary_df["net_kr_annual"].values

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels, y=revenue, name="Revenue",
        marker_color=_REVENUE, opacity=0.9,
        hovertemplate="%{x}<br>Revenue: %{y:,.0f} kr/yr<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=-staff, name="Staff cost",
        marker_color=_STAFF, opacity=0.9,
        customdata=staff,
        hovertemplate="%{x}<br>Staff cost: %{customdata:,.0f} kr/yr<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=-queue, name="Queue cost",
        marker_color=_QUEUE, opacity=0.9,
        customdata=queue,
        hovertemplate="%{x}<br>Queue cost: %{customdata:,.0f} kr/yr<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=net, name="Net result",
        mode="markers+text",
        marker=dict(color=_NET, size=12, symbol="diamond"),
        text=[_fmt_kr(v) for v in net],
        textposition="top center",
        textfont=dict(size=11, color=_NET),
        hovertemplate="%{x}<br>Net: %{y:,.0f} kr/yr<extra></extra>",
    ))

    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        **_LAYOUT, height=480, barmode="relative",
        title=f"Annualised Revenue vs. Cost by Staff Schedule{title_suffix}",
        yaxis=dict(title="kr / year", tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# 3. Waterfall — single schedule build-up
# ──────────────────────────────────────────────────────────────

def plot_waterfall(row, title=None):
    rev   = row["revenue_kr_annual"]
    staff = row["staff_cost_kr_annual"]
    queue = row["queue_cost_kr_annual"]

    fig = go.Figure(go.Waterfall(
        x=["Revenue", "Staff cost", "Queue cost", "Net result"],
        measure=["absolute", "relative", "relative", "total"],
        y=[rev, -staff, -queue, 0],
        text=[_fmt_kr(rev), _fmt_kr(-staff), _fmt_kr(-queue),
              _fmt_kr(rev - staff - queue)],
        textposition="outside",
        textfont=dict(size=12, color=_NET),
        connector=dict(line=dict(color="#888", width=1, dash="dot")),
        increasing=dict(marker_color=_REVENUE),
        decreasing=dict(marker_color=_QUEUE),
        totals=dict(marker_color=_NET),
        hovertemplate="%{x}: %{y:,.0f} kr/yr<extra></extra>",
    ))

    label = row.get("schedule_label", "") if hasattr(row, "get") else row["schedule_label"]
    fig.update_layout(
        **_LAYOUT, height=450, showlegend=False,
        title=title or f"Net Result Build-up — {label}",
        yaxis=dict(title="kr / year", tickformat=",.0f"),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# 4. Waterfall grid — all schedules side by side
# ──────────────────────────────────────────────────────────────

def plot_waterfall_grid(summary_df, title_suffix=""):
    n = len(summary_df)
    labels = summary_df["schedule_label"].astype(str).tolist()

    fig = make_subplots(rows=1, cols=n, subplot_titles=labels, shared_yaxes=True)

    for i, (_, row) in enumerate(summary_df.iterrows()):
        rev   = row["revenue_kr_annual"]
        staff = row["staff_cost_kr_annual"]
        queue = row["queue_cost_kr_annual"]

        fig.add_trace(go.Waterfall(
            x=["Rev", "Staff", "Queue", "Net"],
            measure=["absolute", "relative", "relative", "total"],
            y=[rev, -staff, -queue, 0],
            text=[_fmt_kr(rev), _fmt_kr(-staff), _fmt_kr(-queue),
                  _fmt_kr(rev - staff - queue)],
            textposition="outside", textfont=dict(size=9),
            connector=dict(line=dict(color="#888", width=0.8, dash="dot")),
            increasing=dict(marker_color=_REVENUE),
            decreasing=dict(marker_color=_QUEUE),
            totals=dict(marker_color=_NET),
            showlegend=False,
        ), row=1, col=i + 1)

    fig.update_layout(
        **_LAYOUT, height=450,
        title_text=f"Net Result Build-up — All Schedules{title_suffix}",
    )
    fig.update_yaxes(tickformat=",.0f", row=1, col=1)
    return fig


# ──────────────────────────────────────────────────────────────
# 5. Sensitivity — net result vs arrival rate
# ──────────────────────────────────────────────────────────────

def plot_sensitivity(econ_df, schedule_order, value_col="net_kr_annual",
                     value_label="Net result (kr/year)",
                     title="Net Annual Result vs. Arrival Rate",
                     filters=None):
    df = econ_df.copy()
    if filters:
        for col, val in filters.items():
            if col in df.columns and val is not None:
                df = df[df[col] == val]

    sens = (
        df.groupby(["arrival_rate", "schedule_label"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    sens["se95"] = 1.96 * sens["std"] / np.sqrt(sens["count"].clip(lower=1))

    fig = go.Figure()
    for i, label in enumerate(schedule_order):
        sub = sens[sens["schedule_label"] == label].sort_values("arrival_rate")
        color = _PALETTE[i % len(_PALETTE)]

        if len(sub) > 1 and sub["std"].notna().any():
            fig.add_trace(go.Scatter(
                x=list(sub["arrival_rate"]) + list(sub["arrival_rate"][::-1]),
                y=list(sub["mean"] + sub["se95"]) + list((sub["mean"] - sub["se95"])[::-1]),
                fill="toself", fillcolor=color, opacity=0.1,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=sub["arrival_rate"], y=sub["mean"],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=8, symbol=_MARKERS[i % len(_MARKERS)]),
            hovertemplate=(f"{label}<br>Arrivals/day: %{{x}}<br>"
                           f"{value_label}: %{{y:,.0f}}<extra></extra>"),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(
        **_LAYOUT, height=480,
        title=title,
        xaxis=dict(title="Average arrivals per day"),
        yaxis=dict(title=value_label, tickformat=",.0f"),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# 5b. Sensitivity by arrival pattern
# ──────────────────────────────────────────────────────────────

def plot_sensitivity_by_pattern(econ_df, pattern_order, value_col="net_kr_annual",
                                value_label="Net result (kr/year)",
                                schedule_label=None, title=None, filters=None):
    df = econ_df.copy()
    if schedule_label is not None:
        df = df[df["schedule_label"] == schedule_label]
    if filters:
        for col, val in filters.items():
            if col in df.columns and val is not None:
                df = df[df[col] == val]

    sens = (
        df.groupby(["arrival_rate", "arrival_pattern_label"])[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    sens["se95"] = 1.96 * sens["std"] / np.sqrt(sens["count"].clip(lower=1))

    fig = go.Figure()
    for i, label in enumerate(pattern_order):
        sub = sens[sens["arrival_pattern_label"] == label].sort_values("arrival_rate")
        color = _PALETTE[i % len(_PALETTE)]

        if len(sub) > 1 and sub["std"].notna().any():
            fig.add_trace(go.Scatter(
                x=list(sub["arrival_rate"]) + list(sub["arrival_rate"][::-1]),
                y=list(sub["mean"] + sub["se95"]) + list((sub["mean"] - sub["se95"])[::-1]),
                fill="toself", fillcolor=color, opacity=0.1,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=sub["arrival_rate"], y=sub["mean"],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2.5),
            marker=dict(color=color, size=8, symbol=_MARKERS[i % len(_MARKERS)]),
            hovertemplate=(f"{label}<br>Arrivals/day: %{{x}}<br>"
                           f"{value_label}: %{{y:,.0f}}<extra></extra>"),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    default_title = f"{value_label.split(' (')[0]} vs. Arrival Rate, by Timing Pattern"
    if schedule_label:
        default_title += f"  (schedule = {schedule_label})"
    fig.update_layout(
        **_LAYOUT, height=480,
        title=title or default_title,
        xaxis=dict(title="Average arrivals per day"),
        yaxis=dict(title=value_label, tickformat=",.0f"),
        legend=dict(title="Arrival pattern"),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# 5c. Volume x timing grid
# ──────────────────────────────────────────────────────────────

def plot_volume_timing_grid(econ_df, arrival_rates, pattern_order,
                            value_col="net_kr_annual",
                            value_label="Net result (kr/year)",
                            schedule_label=None, title=None):
    df = econ_df
    if schedule_label is not None:
        df = df[df["schedule_label"] == schedule_label]

    grp = (
        df.groupby(["arrival_rate", "arrival_pattern_label"])[value_col]
        .mean()
        .reset_index()
    )

    fig = go.Figure()
    for i, pattern in enumerate(pattern_order):
        vals = []
        for rate in arrival_rates:
            match = grp[(grp["arrival_rate"] == rate)
                        & (grp["arrival_pattern_label"] == pattern)]
            vals.append(match[value_col].iloc[0] if not match.empty else np.nan)
        fig.add_trace(go.Bar(
            x=[str(r) for r in arrival_rates], y=vals, name=pattern,
            marker_color=_PALETTE[i % len(_PALETTE)],
            hovertemplate=(f"{pattern}<br>Rate: %{{x}}/day<br>"
                           f"{value_label}: %{{y:,.0f}}<extra></extra>"),
        ))

    fig.add_hline(y=0, line_color="black", line_width=1)
    default_title = "Volume vs. Timing"
    if schedule_label:
        default_title += f"  (schedule = {schedule_label})"
    fig.update_layout(
        **_LAYOUT, height=450, barmode="group",
        title=title or default_title,
        xaxis=dict(title="Average arrivals per day"),
        yaxis=dict(title=value_label, tickformat=",.0f"),
        legend=dict(title="Arrival pattern"),
    )
    return fig
