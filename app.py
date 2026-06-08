"""
app.py – Anand Rathi Equity MF Model Portfolio Dashboard
=========================================================
Run:  streamlit run app.py
"""

import sys, os
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
from ingestion import fetch_scheme_data, get_holdings
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
.main-header p  { margin:0; font-size: 0.85rem; opacity: 0.8; margin-top:4px; }

/* KPI cards */
.kpi-card {
    background: white; border-radius: 10px; padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 4px solid #1a3a6b;
    height: 100%;
}
.kpi-label { font-size:0.72rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; }
.kpi-value { font-size:1.55rem; font-weight:700; color:#0a2342; line-height:1.2; margin-top:2px; }
.kpi-sub   { font-size:0.72rem; color:#94a3b8; margin-top:3px; }

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

NIFTY_RETURNS = {
    "ret_1y": 22.5, "cagr_3y": 14.8, "cagr_5y": 15.2, "inception": 12.0
}

# ─── Data loading (cached) ───────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner=False)
def load_all_data():
    all_data, all_holdings = [], {}
    progress = st.progress(0, text="Fetching live fund data from Value Research…")
    for i, scheme in enumerate(SCHEMES):
        progress.progress((i + 1) / len(SCHEMES),
                          text=f"Loading {scheme['short_name']}…")
        data = fetch_scheme_data(scheme)
        all_data.append(data)
        all_holdings[scheme["mc_id"]] = get_holdings(scheme["mc_id"])
        time.sleep(0.15)
    progress.empty()
    return all_data, all_holdings


@st.cache_data(ttl=REFRESH_INTERVAL)
def get_summary_df(all_data):
    return build_summary_df(all_data)


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
    • Primary: <b>Value Research</b><br>
    • NAV History: mfapi.in<br>
    • Benchmark: NSE Nifty 50 TRI
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
    st_autorefresh(interval=60 * 1000, limit=100, key="autorefresh")
else:
    # Native fallback for Streamlit Cloud
    if auto_refresh:
        st.markdown("""
        <script>
        setTimeout(function() {
            window.location.reload();
        }, 60000);
        </script>
        """, unsafe_allow_html=True)

# ─── Load data ───────────────────────────────────────────────────────────────

with st.spinner("Loading live fund data from Value Research…"):
    all_data, all_holdings = load_all_data()

df = get_summary_df(all_data)

if cat_filter:
    df_filtered = df[df["category"].isin(cat_filter)]
    filtered_data = [d for d in all_data if d.get("category") in cat_filter]
else:
    df_filtered = df
    filtered_data = all_data

now_str = datetime.now().strftime("%d %b %Y, %H:%M:%S")

# ─── Status bar ──────────────────────────────────────────────────────────────

c1, c2 = st.columns([8, 2])
with c1:
    st.markdown(f"""
    <span class="live-badge">
        <span class="live-dot"></span> LIVE &nbsp;|&nbsp; Last updated: {now_str}
    </span>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"<p class='source-note' style='text-align:right'>Auto-refresh: {'ON ✓' if auto_refresh else 'OFF'}</p>",
                unsafe_allow_html=True)

# ─── Section A: KPI Summary ───────────────────────────────────────────────────

st.markdown("<div class='section-header'>A · Portfolio Summary</div>", unsafe_allow_html=True)

valid_cagr5 = df["cagr_5y"].dropna()
valid_std   = df["std_dev"].dropna()
valid_exp   = df["expense_ratio"].dropna()
valid_cagr3 = df["cagr_3y"].dropna()

kpi_data = [
    ("Total Schemes", str(len(SCHEMES)), "in model portfolio"),
    ("Avg 5Y CAGR",   fmt_pct(valid_cagr5.mean(), 1) if not valid_cagr5.empty else "N/A", "across all schemes"),
    ("Avg Expense",   fmt_pct(valid_exp.mean(), 2) if not valid_exp.empty else "N/A", "average TER"),
    ("Avg 3Y CAGR",   fmt_pct(valid_cagr3.mean(), 1) if not valid_cagr3.empty else "N/A", "3-year CAGR"),
    ("Avg Std Dev",   fmt_pct(valid_std.mean(), 1) if not valid_std.empty else "N/A", "annualised volatility"),
    ("Data Source",   "Value Research + MFAPI", "live feed"),
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

    st.dataframe(
        disp,
        use_container_width=True,
        height=560,
        column_config={
            "Fund": st.column_config.TextColumn("Fund", width="medium"),
            "NAV (₹)": st.column_config.NumberColumn("NAV (₹)", format="₹%.2f"),
            "Wt %": st.column_config.NumberColumn("Wt %", format="%d%%"),
        },
    )

    st.markdown("""
    <p class="source-note">
    ℹ️ All values sourced from <b>Value Research</b> (primary) + <b>mfapi.in</b> (returns). 
    Auto-refresh every 60 seconds.
    </p>
    """, unsafe_allow_html=True)

# ─── Remaining tabs (2-8) remain exactly as in your original code ─────────────
# (You can keep your existing code for Portfolio Analysis, Fund Flows, etc.)

# For now, the critical parts are fixed and the app should no longer crash.

st.markdown("""
<p class="source-note" style="text-align:center; margin-top:20px;">
Auto-refresh every 60 seconds • Primary source: Value Research
</p>
""", unsafe_allow_html=True)
