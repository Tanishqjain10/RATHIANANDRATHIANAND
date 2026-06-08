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

# ─── Data loading (cached) ───────────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner=False)
def load_all_data():
    all_data, all_holdings = [], {}
    progress = st.progress(0, text="Fetching live data from all configured URLs...")
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
with st.expander("Live data source audit", expanded=False):
    audit_cols = [
        "short_name",
        "provided_url_count",
        "fetched_url_count",
        "failed_url_count",
        "mfapi_fetched",
        "last_updated",
    ]
    audit_df = df[[c for c in audit_cols if c in df.columns]].rename(columns={
        "short_name": "Fund",
        "provided_url_count": "URLs Provided",
        "fetched_url_count": "URLs Fetched",
        "failed_url_count": "URLs Pending",
        "mfapi_fetched": "NAV Feed",
        "last_updated": "Last Updated",
    })
    st.dataframe(audit_df, use_container_width=True, hide_index=True)
    st.caption("The dashboard refreshes cached live data every 5 minutes and uses the URLs configured in schemes.py.")

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
    ("Live Sources", f"{total_fetched_urls}/{total_provided_urls}", f"NAV feeds {total_nav_feeds}/{len(df)}"),
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

    st.markdown("""
    <p class="source-note">
    ℹ️ All values sourced from <b>Value Research</b> (primary) + <b>mfapi.in</b> (returns).<br>
    Last two rows = <b>Portfolio Average</b> and <b>Portfolio Total</b> (calculated dynamically).
    </p>
    """, unsafe_allow_html=True)

# ─── Tab 2: Portfolio Analysis ────────────────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-header'>C · Portfolio Analysis</div>", unsafe_allow_html=True)
    st.info("Portfolio Analysis tab content goes here (your original code can be placed in this block).")

# ─── Tab 3 to 8 ───────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-header'>Fund Flows</div>", unsafe_allow_html=True)
    st.info("Fund Flows content goes here.")

with tabs[3]:
    st.markdown("<div class='section-header'>Stock Movements</div>", unsafe_allow_html=True)
    st.info("Stock Movements content goes here.")

with tabs[4]:
    st.markdown("<div class='section-header'>Overlap Matrix</div>", unsafe_allow_html=True)
    st.info("Overlap Matrix content goes here.")

with tabs[5]:
    st.markdown("<div class='section-header'>Benchmark</div>", unsafe_allow_html=True)
    st.info("Benchmark content goes here.")

with tabs[6]:
    st.markdown("<div class='section-header'>Risk Analysis</div>", unsafe_allow_html=True)
    st.info("Risk Analysis content goes here.")

with tabs[7]:
    st.markdown("<div class='section-header'>Charts</div>", unsafe_allow_html=True)
    st.info("Charts content goes here.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<p class="source-note" style="text-align:center; margin-top:20px;">
Auto-refresh every 5 minutes • Primary source: Value Research + mfapi.in
</p>
""", unsafe_allow_html=True)
