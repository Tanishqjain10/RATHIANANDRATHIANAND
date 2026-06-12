"""
app.py – Anand Rathi Equity MF Model Portfolio Dashboard
=========================================================
Run: streamlit run app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import logging
from datetime import datetime
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from schemes import SCHEMES, CATEGORY_COLORS
from ingestion import fetch_scheme_data, get_holdings, fetch_benchmark_data, fallback_holdings
from cleaning import (
    build_summary_df, to_display_df,
    compute_overlap_matrix, fmt_pct, fmt_cr
)

logging.basicConfig(level=logging.WARNING)

# ─── Try to import auto-refresh (with fallback) ───────────────────────────────
AUTO_REFRESH_AVAILABLE = False
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anand Rathi | MF Model Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
/* Dark header */
.main-header {
    background: linear-gradient(135deg, #0a2342 0%, #1a3a6b 60%, #204e8a 100%);
    padding: 1.5rem 2rem; border-radius: 12px;
    color: white; margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.main-header h1 { margin:0; font-size: 1.7rem; font-weight: 700; }
.main-header p { margin:0; font-size: 0.85rem; opacity: 0.8; margin-top:4px; }
/* KPI cards */
.kpi-card {
    background: white; border-radius: 10px; padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 4px solid #1a3a6b;
    height: 100%;
}
.kpi-label { font-size:0.72rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; }
.kpi-value { font-size:1.55rem; font-weight:700; color:#0a2342; line-height:1.2; margin-top:2px; }
.kpi-sub { font-size:0.72rem; color:#94a3b8; margin-top:3px; }
/* Section headers */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #0a2342;
    border-bottom: 2px solid #1a3a6b; padding-bottom: 6px;
    margin: 1.5rem 0 0.8rem 0;
}
/* Positive / Negative */
.pos { color: #16a34a; font-weight: 600; }
.neg { color: #dc2626; font-weight: 600; }
/* Sidebar */
[data-testid="stSidebar"] { background: #0a2342; }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; font-size:0.8rem; }
/* Table */
.stDataFrame { border-radius: 8px; overflow: hidden; }
/* Tabs */
.stTabs [data-baseweb="tab"] { font-size: 0.82rem; font-weight:600; }
.stTabs [aria-selected="true"] { color: #1a3a6b !important; border-bottom-color: #1a3a6b !important; }
/* Live badge */
.live-badge {
    display: inline-flex; align-items:center; gap:6px;
    background: #dcfce7; color:#15803d; border-radius:20px;
    padding:3px 10px; font-size:0.72rem; font-weight:600;
}
.live-dot { width:8px; height:8px; border-radius:50%; background:#16a34a;
    animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
/* Source note */
.source-note { font-size:0.7rem; color:#94a3b8; font-style:italic; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
REFRESH_INTERVAL = 300
REFRESH_INTERVAL_MS = REFRESH_INTERVAL * 1000
REFRESH_LABEL = "5 min"
CACHE_VERSION = "holdings-fallback-v2"

# ─── Data loading (cached) ───────────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner=False)
def load_all_data(cache_version=CACHE_VERSION):
    all_data, all_holdings = [], {}
    progress = st.progress(0, text="Fetching live data from all configured URLs...")
    for i, scheme in enumerate(SCHEMES):
        progress.progress((i + 1) / len(SCHEMES),
                          text=f"Loading {scheme['short_name']}…")
        data = fetch_scheme_data(scheme)
        all_data.append(data)
        all_holdings[scheme["mc_id"]] = get_holdings(scheme)
        time.sleep(0.15)
    progress.empty()
    return all_data, all_holdings


@st.cache_data(ttl=REFRESH_INTERVAL)
def get_summary_df(all_data):
    return build_summary_df(all_data)


@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner=False)
def load_benchmark_data():
    return fetch_benchmark_data()


# ─── Plotly theme helper ──────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font_family="Inter",
    paper_bgcolor="white",
    plot_bgcolor="#f8fafc",
    title_font_size=14,
    title_font_color="#0a2342",
    margin=dict(l=10, r=10, t=40, b=10),
)


def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#e2e8f0", linecolor="#cbd5e1")
    fig.update_yaxes(gridcolor="#e2e8f0", linecolor="#cbd5e1")
    return fig


def num_col(frame, column):
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_mean(frame, column):
    values = num_col(frame, column).dropna()
    return None if values.empty else values.mean()


def safe_sum(frame, column):
    values = num_col(frame, column).dropna()
    return None if values.empty else values.sum()


def weighted_mean(frame, column, weight_column="weight"):
    values = num_col(frame, column)
    weights = num_col(frame, weight_column)
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return safe_mean(frame, column)
    return np.average(values[valid], weights=weights[valid])


def fmt_signed_pct(val, decimals=2):
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:+.{decimals}f}%"


def render_metric_grid(items, columns=3):
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, item in zip(cols, items[start:start + columns]):
            label, value, sub = item
            col.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    auto_refresh = st.toggle("Auto Refresh (5 min)", value=True)
    if st.button("🔄 Force Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Filters")
    cat_filter = st.multiselect(
        "Category",
        options=sorted({s["category"] for s in SCHEMES}),
        default=[],
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; opacity:0.7; line-height:1.6;">
    <b>Data Sources</b><br>
    • Live NAV/returns: <b>mfapi.in</b><br>
    • Scheme pages: Value Research (best effort)<br>
    • Benchmark: Not configured
    </div>
    """, unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📊 Anand Rathi | Equity MF Model Portfolio Dashboard</h1>
  <p>Oct'25 – Dec'26 &nbsp;|&nbsp; 14 Curated Schemes &nbsp;|&nbsp; Live Data Feed</p>
</div>
""", unsafe_allow_html=True)

# ─── Auto Refresh Logic (with fallback) ──────────────────────────────────────
if AUTO_REFRESH_AVAILABLE and auto_refresh:
    st_autorefresh(interval=REFRESH_INTERVAL_MS, limit=100, key="autorefresh")
else:
    if auto_refresh:
        st.markdown(f"""
        <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {REFRESH_INTERVAL_MS});
        </script>
        """, unsafe_allow_html=True)

# ─── Load data ───────────────────────────────────────────────────────────────
with st.spinner("Loading live fund data from Value Research…"):
    all_data, all_holdings = load_all_data()
benchmark_data = load_benchmark_data()

for scheme in SCHEMES:
    sid = scheme.get("mc_id")
    if sid and not all_holdings.get(sid, {}).get("top_holdings"):
        all_holdings[sid] = fallback_holdings(sid)

df = get_summary_df(all_data)

if cat_filter:
    df_filtered = df[df["category"].isin(cat_filter)]
    filtered_data = [d for d in all_data if d.get("category") in cat_filter]
else:
    df_filtered = df
    filtered_data = all_data

latest_fetch = max(
    (d for d in all_data if d.get("fetched_at")),
    key=lambda d: d["fetched_at"],
    default={},
)
last_updated = latest_fetch.get("last_updated", "Not fetched yet")
total_provided_urls = int(df.get("provided_url_count", pd.Series(dtype=int)).sum())
total_fetched_urls = int(df.get("fetched_url_count", pd.Series(dtype=int)).sum())
total_nav_feeds = int(df.get("mfapi_fetched", pd.Series(dtype=bool)).fillna(False).sum())

# ─── Status bar ──────────────────────────────────────────────────────────────
c1, c2 = st.columns([8, 2])
with c1:
    st.markdown(f"""
    <span class="live-badge">
        <span class="live-dot"></span> LIVE &nbsp;|&nbsp; Last Updated: {last_updated}
    </span>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"<p class='source-note' style='text-align:right'>Auto-refresh: {'ON' if auto_refresh else 'OFF'} ({REFRESH_LABEL})</p>",
                unsafe_allow_html=True)

# ─── Section A: KPI Summary ───────────────────────────────────────────────────
with st.expander("Live API health", expanded=True):
    api_health = df[[
        "short_name",
        "mfapi_id",
        "mfapi_fetched",
        "nav",
        "nav_date",
        "ret_1y",
        "cagr_3y",
        "last_updated",
    ]].copy()
    api_health["Status"] = np.where(api_health["mfapi_fetched"], "OK", "Check")
    api_health = api_health.rename(columns={
        "short_name": "Fund",
        "mfapi_id": "API ID",
        "nav": "NAV",
        "nav_date": "NAV Date",
        "ret_1y": "1Y Return",
        "cagr_3y": "3Y CAGR",
        "last_updated": "Last Updated",
    })
    api_health = api_health[["Fund", "API ID", "Status", "NAV", "NAV Date", "1Y Return", "3Y CAGR", "Last Updated"]]
    st.dataframe(api_health, use_container_width=True, hide_index=True)
    st.download_button(
        "Download live API health CSV",
        data=api_health.to_csv(index=False).encode("utf-8"),
        file_name="live_api_health.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("Priority live feed: mfapi.in. If every row shows OK and NAV/NAV Date are filled, the live API is working.")

st.markdown("<div class='section-header'>A · Portfolio Summary</div>", unsafe_allow_html=True)

valid_cagr5 = df["cagr_5y"].dropna()
valid_std = df["std_dev"].dropna()
valid_exp = df["expense_ratio"].dropna()
valid_cagr3 = df["cagr_3y"].dropna()

kpi_data = [
    ("Total Schemes", str(len(SCHEMES)), "in model portfolio"),
    ("Avg 5Y CAGR", fmt_pct(valid_cagr5.mean(), 1) if not valid_cagr5.empty else "N/A", "across all schemes"),
    ("Avg Expense", fmt_pct(valid_exp.mean(), 2) if not valid_exp.empty else "N/A", "average TER"),
    ("Avg 3Y CAGR", fmt_pct(valid_cagr3.mean(), 1) if not valid_cagr3.empty else "N/A", "3-year CAGR"),
    ("Avg Std Dev", fmt_pct(valid_std.mean(), 1) if not valid_std.empty else "N/A", "annualised volatility"),
    ("Live APIs", f"{total_nav_feeds}/{len(df)}", "NAV/returns via mfapi.in"),
]

kpi_cols = st.columns(len(kpi_data))
for col, (label, value, sub) in zip(kpi_cols, kpi_data):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Comparison Table",
    "🏗️ Portfolio Analysis",
    "💸 Fund Flows",
    "🔄 Stock Movements",
    "🔁 Overlap Matrix",
    "📈 Benchmark",
    "⚠️ Risk Analysis",
    "📊 Charts",
])

# ─── Tab 1: Comparison Table ──────────────────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-header'>B · Scheme Comparison Table</div>", unsafe_allow_html=True)

    disp = to_display_df(df_filtered).reset_index(drop=True)

    # Portfolio Average Row
    avg_row = {
        "Fund": "PORTFOLIO AVERAGE",
        "Category": "",
        "Wt %": "",
        "NAV (₹)": round(df["nav"].mean(), 2) if not df["nav"].empty else None,
        "AUM (Cr)": round(df["aum_cr"].mean(), 0) if not df["aum_cr"].empty else None,
        "Exp Ratio": round(df["expense_ratio"].mean(), 2) if not df["expense_ratio"].empty else None,
        "1M Ret": round(df["ret_1m"].mean(), 2) if not df["ret_1m"].empty else None,
        "3M Ret": round(df["ret_3m"].mean(), 2) if not df["ret_3m"].empty else None,
        "6M Ret": round(df["ret_6m"].mean(), 2) if not df["ret_6m"].empty else None,
        "1Y Ret": round(df["ret_1y"].mean(), 2) if not df["ret_1y"].empty else None,
        "3Y CAGR": round(df["cagr_3y"].mean(), 2) if not df["cagr_3y"].empty else None,
        "5Y CAGR": round(df["cagr_5y"].mean(), 2) if not df["cagr_5y"].empty else None,
        "Std Dev": round(df["std_dev"].mean(), 2) if not df["std_dev"].empty else None,
    }

    # Portfolio Total Row
    total_row = {
        "Fund": "PORTFOLIO TOTAL",
        "Category": "",
        "Wt %": "100%",
        "AUM (Cr)": round(df["aum_cr"].sum(), 0) if not df["aum_cr"].empty else None,
    }

    disp = pd.concat([disp, pd.DataFrame([avg_row, total_row])], ignore_index=True)

    st.dataframe(
        disp,
        use_container_width=True,
        height=620,
        column_config={
            "Fund": st.column_config.TextColumn("Fund", width="medium"),
            "NAV (₹)": st.column_config.NumberColumn("NAV (₹)", format="₹%.2f"),
            "Wt %": st.column_config.NumberColumn("Wt %", format="%d%%"),
        },
    )
    st.download_button(
        "Download comparison CSV",
        data=disp.to_csv(index=False).encode("utf-8"),
        file_name="scheme_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("""
    <p class="source-note">
    ℹ️ NAV, return, CAGR, and volatility values are sourced from the live <b>mfapi.in</b> feed.<br>
    Last two rows = <b>Portfolio Average</b> and <b>Portfolio Total</b> (calculated dynamically).
    </p>
    """, unsafe_allow_html=True)

# ─── Tab 2: Portfolio Analysis ────────────────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-header'>C · Aggregated Portfolio View</div>", unsafe_allow_html=True)

    avg_aum = safe_mean(df_filtered, "aum_cr")
    avg_expense = safe_mean(df_filtered, "expense_ratio")
    avg_std = safe_mean(df_filtered, "std_dev")
    metric_items = [
        ("Avg AUM", fmt_cr(avg_aum), "available source-page data"),
        ("Avg Exp Ratio", fmt_pct(avg_expense, 2), "available source-page data"),
        ("Avg 3M Return", fmt_signed_pct(weighted_mean(df_filtered, "ret_3m")), "weighted by model allocation"),
        ("Avg 1Y Return", fmt_signed_pct(weighted_mean(df_filtered, "ret_1y")), "weighted by model allocation"),
        ("Avg 3Y CAGR", fmt_signed_pct(weighted_mean(df_filtered, "cagr_3y")), "weighted by model allocation"),
        ("Avg 5Y CAGR", fmt_signed_pct(weighted_mean(df_filtered, "cagr_5y")), "weighted by model allocation"),
        ("Avg Std Dev", fmt_pct(avg_std, 2), "annualised volatility"),
        ("Live APIs", f"{total_nav_feeds}/{len(df)}", "mfapi NAV/returns feed"),
        ("Last Updated", last_updated, "auto-refresh every 5 minutes"),
    ]
    render_metric_grid(metric_items, columns=3)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("<div class='section-header'>Category Allocation</div>", unsafe_allow_html=True)
        category_df = (
            df_filtered.groupby("category", as_index=False)["weight"]
            .sum()
            .sort_values("weight", ascending=False)
        )
        if category_df.empty:
            st.warning("No category allocation available for the selected filters.")
        else:
            fig = px.pie(
                category_df,
                names="category",
                values="weight",
                hole=0.45,
                color="category",
                color_discrete_map=CATEGORY_COLORS,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(apply_theme(fig), use_container_width=True)

    with right:
        st.markdown("<div class='section-header'>Return Profile</div>", unsafe_allow_html=True)
        return_cols = ["ret_1m", "ret_3m", "ret_6m", "ret_1y", "cagr_3y", "cagr_5y"]
        return_rows = []
        for col in return_cols:
            val = weighted_mean(df_filtered, col)
            if val is not None and not pd.isna(val):
                return_rows.append({"Metric": col.replace("ret_", "").replace("cagr_", "").upper(), "Return": val})
        if return_rows:
            ret_fig = px.bar(pd.DataFrame(return_rows), x="Metric", y="Return", text="Return")
            ret_fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(apply_theme(ret_fig), use_container_width=True)
        else:
            st.warning("Return profile is not available for the selected filters.")

    st.markdown("<div class='section-header'>Risk vs Return</div>", unsafe_allow_html=True)
    risk_df = df_filtered.copy()
    risk_df["cagr_3y_num"] = num_col(risk_df, "cagr_3y")
    risk_df["std_dev_num"] = num_col(risk_df, "std_dev")
    risk_df = risk_df.dropna(subset=["cagr_3y_num", "std_dev_num"])
    if risk_df.empty:
        st.warning("Risk-return chart needs CAGR and volatility data.")
    else:
        risk_fig = px.scatter(
            risk_df,
            x="std_dev_num",
            y="cagr_3y_num",
            size="weight",
            color="category",
            hover_name="short_name",
            color_discrete_map=CATEGORY_COLORS,
            labels={"std_dev_num": "Std Dev (%)", "cagr_3y_num": "3Y CAGR (%)"},
        )
        st.plotly_chart(apply_theme(risk_fig), use_container_width=True)

    insight_left, insight_right = st.columns([1, 1])
    with insight_left:
        st.markdown("<div class='section-header'>Top / Bottom 1Y Performers</div>", unsafe_allow_html=True)
        perf_cols = ["short_name", "category", "weight", "ret_1y", "cagr_3y", "std_dev"]
        perf_df = df_filtered[[c for c in perf_cols if c in df_filtered.columns]].copy()
        perf_df["ret_1y_num"] = num_col(perf_df, "ret_1y")
        perf_df = perf_df.dropna(subset=["ret_1y_num"]).sort_values("ret_1y_num", ascending=False)
        if perf_df.empty:
            st.warning("1Y performance data is not available yet.")
        else:
            top_bottom = pd.concat([perf_df.head(3), perf_df.tail(3)]).drop_duplicates()
            top_bottom = top_bottom.rename(columns={
                "short_name": "Fund",
                "category": "Category",
                "weight": "Wt %",
                "ret_1y": "1Y Return",
                "cagr_3y": "3Y CAGR",
                "std_dev": "Std Dev",
            })
            st.dataframe(top_bottom[["Fund", "Category", "Wt %", "1Y Return", "3Y CAGR", "Std Dev"]], use_container_width=True, hide_index=True)

    with insight_right:
        st.markdown("<div class='section-header'>Weighted Return Contribution</div>", unsafe_allow_html=True)
        contrib_df = df_filtered[["short_name", "category", "weight", "ret_1y"]].copy()
        contrib_df["weight_num"] = num_col(contrib_df, "weight")
        contrib_df["ret_1y_num"] = num_col(contrib_df, "ret_1y")
        contrib_df = contrib_df.dropna(subset=["weight_num", "ret_1y_num"])
        if contrib_df.empty:
            st.warning("Contribution data needs both weights and 1Y returns.")
        else:
            contrib_df["Contribution"] = contrib_df["weight_num"] * contrib_df["ret_1y_num"] / 100
            contrib_df = contrib_df.sort_values("Contribution", ascending=False)
            contrib_fig = px.bar(
                contrib_df,
                x="Contribution",
                y="short_name",
                color="category",
                orientation="h",
                labels={"short_name": "Fund", "Contribution": "Contribution to 1Y Return (%)"},
                color_discrete_map=CATEGORY_COLORS,
            )
            st.plotly_chart(apply_theme(contrib_fig), use_container_width=True)

# ─── Tab 3 to 8 ───────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-header'>Fund Flows</div>", unsafe_allow_html=True)
    flow_df = df_filtered[["short_name", "category", "weight", "ret_1m", "ret_3m", "ret_6m", "ret_1y"]].copy()
    for col_name in ["weight", "ret_1m", "ret_3m", "ret_6m", "ret_1y"]:
        flow_df[f"{col_name}_num"] = num_col(flow_df, col_name)
    flow_df["1M Flow Score"] = flow_df["weight_num"] * flow_df["ret_1m_num"] / 100
    flow_df["3M Flow Score"] = flow_df["weight_num"] * flow_df["ret_3m_num"] / 100
    flow_df["6M Flow Score"] = flow_df["weight_num"] * flow_df["ret_6m_num"] / 100
    flow_df["1Y Flow Score"] = flow_df["weight_num"] * flow_df["ret_1y_num"] / 100
    score_cols = ["1M Flow Score", "3M Flow Score", "6M Flow Score", "1Y Flow Score"]
    flow_summary = [
        ("1M Net Flow Proxy", fmt_signed_pct(flow_df["1M Flow Score"].sum()), "weight x live NAV return"),
        ("3M Net Flow Proxy", fmt_signed_pct(flow_df["3M Flow Score"].sum()), "weight x live NAV return"),
        ("6M Net Flow Proxy", fmt_signed_pct(flow_df["6M Flow Score"].sum()), "weight x live NAV return"),
        ("1Y Net Flow Proxy", fmt_signed_pct(flow_df["1Y Flow Score"].sum()), "weight x live NAV return"),
    ]
    render_metric_grid(flow_summary, columns=4)
    flow_view = flow_df.rename(columns={
        "short_name": "Fund",
        "category": "Category",
        "weight": "Wt %",
        "ret_1m": "1M Ret",
        "ret_3m": "3M Ret",
        "ret_6m": "6M Ret",
        "ret_1y": "1Y Ret",
    })[["Fund", "Category", "Wt %", "1M Ret", "3M Ret", "6M Ret", "1Y Ret", *score_cols]]
    st.dataframe(flow_view, use_container_width=True, hide_index=True)
    flow_chart = flow_df.dropna(subset=["3M Flow Score"]).sort_values("3M Flow Score", ascending=False)
    if not flow_chart.empty:
        fig = px.bar(
            flow_chart,
            x="3M Flow Score",
            y="short_name",
            color="category",
            orientation="h",
            labels={"short_name": "Fund", "3M Flow Score": "3M Flow Proxy (%)"},
            color_discrete_map=CATEGORY_COLORS,
        )
        st.plotly_chart(apply_theme(fig), use_container_width=True)
    st.caption("This is a live model-flow proxy from NAV returns and portfolio weights, not AMC subscription/redemption flow data.")

with tabs[3]:
    st.markdown("<div class='section-header'>Stock Movements</div>", unsafe_allow_html=True)
    holdings_rows = []
    for scheme in filtered_data:
        sid = scheme.get("mc_id")
        holding_info = all_holdings.get(sid, {})
        for stock, pct in holding_info.get("top_holdings", []):
            holdings_rows.append({
                "Fund": scheme.get("short_name"),
                "Category": scheme.get("category"),
                "Stock": stock,
                "Holding %": pct,
                "Weighted Exposure": pct * scheme.get("weight", 0) / 100,
                "Source": holding_info.get("holdings_source", "Value Research"),
            })
    holdings_df = pd.DataFrame(holdings_rows)
    if holdings_df.empty:
        st.warning("Holdings could not be fetched from the source pages in this run.")
    else:
        exposure = (
            holdings_df.groupby("Stock", as_index=False)
            .agg({"Weighted Exposure": "sum", "Fund": "nunique"})
            .rename(columns={"Fund": "Funds Holding"})
            .sort_values("Weighted Exposure", ascending=False)
        )
        stock_cols = st.columns([1, 1])
        with stock_cols[0]:
            st.markdown("<div class='section-header'>Top Portfolio Stock Exposures</div>", unsafe_allow_html=True)
            st.dataframe(exposure.head(20), use_container_width=True, hide_index=True)
        with stock_cols[1]:
            fig = px.bar(
                exposure.head(15),
                x="Weighted Exposure",
                y="Stock",
                orientation="h",
                color="Funds Holding",
                labels={"Weighted Exposure": "Weighted Exposure (%)"},
            )
            st.plotly_chart(apply_theme(fig), use_container_width=True)
        st.markdown("<div class='section-header'>Latest Top Holdings By Fund</div>", unsafe_allow_html=True)
        st.dataframe(holdings_df.sort_values(["Fund", "Holding %"], ascending=[True, False]), use_container_width=True, hide_index=True)
        snapshot_count = sum(1 for h in all_holdings.values() if h.get("holdings_snapshot"))
        st.caption(f"Holdings source: Value Research pages with snapshot fallback. Snapshot fallback used for {snapshot_count}/{len(all_holdings)} funds in this run.")

with tabs[4]:
    st.markdown("<div class='section-header'>Overlap Matrix</div>", unsafe_allow_html=True)
    scheme_ids = [d.get("mc_id") for d in filtered_data if d.get("mc_id")]
    has_holdings = any(all_holdings.get(sid, {}).get("top_holdings") for sid in scheme_ids)
    if has_holdings:
        overlap = compute_overlap_matrix(all_holdings, scheme_ids)
        name_map = {d.get("mc_id"): d.get("short_name") for d in filtered_data}
        overlap = overlap.rename(index=name_map, columns=name_map)
        fig = px.imshow(
            overlap.astype(float),
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(color="Overlap %"),
        )
        st.plotly_chart(apply_theme(fig), use_container_width=True)
        st.dataframe(overlap, use_container_width=True)
    else:
        st.info("Holdings feed is not configured yet, so stock overlap is not calculated. Category allocation and risk-return views are still live.")

with tabs[5]:
    st.markdown("<div class='section-header'>Benchmark</div>", unsafe_allow_html=True)
    if not benchmark_data.get("fetched"):
        st.warning("Benchmark feed could not be fetched in this run.")
    else:
        bench_items = [
            ("Benchmark", benchmark_data["label"], benchmark_data.get("latest_date", "")),
            ("Latest Level", f"{benchmark_data.get('latest'):,.2f}", benchmark_data.get("symbol", "")),
            ("1Y Return", fmt_signed_pct(benchmark_data.get("ret_1y")), "live benchmark"),
            ("3Y CAGR", fmt_signed_pct(benchmark_data.get("cagr_3y")), "live benchmark"),
            ("5Y CAGR", fmt_signed_pct(benchmark_data.get("cagr_5y")), "live benchmark"),
            ("Std Dev", fmt_pct(benchmark_data.get("std_dev"), 2), "annualised"),
        ]
        render_metric_grid(bench_items, columns=3)
        comp_rows = []
        for label, fund_col, bench_key in [
            ("1Y Return", "ret_1y", "ret_1y"),
            ("3Y CAGR", "cagr_3y", "cagr_3y"),
            ("5Y CAGR", "cagr_5y", "cagr_5y"),
        ]:
            fund_val = weighted_mean(df_filtered, fund_col)
            bench_val = benchmark_data.get(bench_key)
            comp_rows.append({
                "Metric": label,
                "Portfolio": fund_val,
                "Benchmark": bench_val,
                "Alpha": None if fund_val is None or bench_val is None else fund_val - bench_val,
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        series = benchmark_data.get("series", pd.Series(dtype=float))
        if not series.empty:
            bench_plot = series.tail(252).reset_index()
            bench_plot.columns = ["Date", "Level"]
            fig = px.line(bench_plot, x="Date", y="Level", title="Nifty 50 - Last 1 Year")
            st.plotly_chart(apply_theme(fig), use_container_width=True)

with tabs[6]:
    st.markdown("<div class='section-header'>Risk Analysis</div>", unsafe_allow_html=True)
    risk_table = df_filtered[["short_name", "category", "weight", "cagr_3y", "cagr_5y", "std_dev"]].copy()
    risk_table = risk_table.rename(columns={
        "short_name": "Fund",
        "category": "Category",
        "weight": "Wt %",
        "cagr_3y": "3Y CAGR",
        "cagr_5y": "5Y CAGR",
        "std_dev": "Std Dev",
    })
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

with tabs[7]:
    st.markdown("<div class='section-header'>Charts</div>", unsafe_allow_html=True)
    chart_df = df_filtered.copy()
    chart_df["ret_1y_num"] = num_col(chart_df, "ret_1y")
    chart_df["cagr_3y_num"] = num_col(chart_df, "cagr_3y")
    chart_df["cagr_5y_num"] = num_col(chart_df, "cagr_5y")

    top_chart, category_chart = st.columns([1, 1])
    with top_chart:
        perf_df = chart_df.dropna(subset=["cagr_3y_num"]).sort_values("cagr_3y_num", ascending=False)
        if perf_df.empty:
            st.warning("Performance chart needs 3Y CAGR data.")
        else:
            fig = px.bar(
                perf_df,
                x="cagr_3y_num",
                y="short_name",
                color="category",
                orientation="h",
                labels={"cagr_3y_num": "3Y CAGR (%)", "short_name": "Fund"},
                color_discrete_map=CATEGORY_COLORS,
            )
            st.plotly_chart(apply_theme(fig), use_container_width=True)

    with category_chart:
        category_df = (
            chart_df.groupby("category", as_index=False)["weight"]
            .sum()
            .sort_values("weight", ascending=False)
        )
        if category_df.empty:
            st.warning("Category chart needs allocation data.")
        else:
            fig = px.treemap(category_df, path=["category"], values="weight", color="category", color_discrete_map=CATEGORY_COLORS)
            st.plotly_chart(apply_theme(fig), use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<p class="source-note" style="text-align:center; margin-top:20px;">
Auto-refresh every 5 minutes • Live API source: mfapi.in
</p>
""", unsafe_allow_html=True)
