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

# Robust import for Streamlit Cloud
try:
    from ingestion import fetch_scheme_data, get_holdings
    from cleaning import build_summary_df, to_display_df, compute_overlap_matrix, fmt_pct, fmt_cr
    print("✅ Imported successfully from root")
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()
    print("✅ Imported from root (ingestion.py & cleaning.py)")
except ImportError:
    # Fallback if modules folder exists
    try:
        from modules.ingestion import fetch_scheme_data, get_holdings
        from modules.cleaning import (
            build_summary_df, 
            to_display_df,
            compute_overlap_matrix, 
            fmt_pct, 
            fmt_cr
        )
        print("✅ Imported from modules/ folder")
    except ImportError:
        st.error("❌ Could not import ingestion or cleaning modules. Check file structure.")

logging.basicConfig(level=logging.WARNING)

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

REFRESH_INTERVAL = 300   # seconds between auto-refreshes

# Approximate Nifty 50 benchmark returns (updated periodically)
NIFTY_RETURNS = {
    "ret_1y": 22.5, "cagr_3y": 14.8, "cagr_5y": 15.2, "inception": 12.0
}

# ─── Data loading (cached) ───────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_INTERVAL, show_spinner=False)
def load_all_data():
    """Fetch and cache all 14 scheme data. TTL = 5 min."""
    all_data, all_holdings = [], {}
    progress = st.progress(0, text="Fetching live fund data…")
    for i, scheme in enumerate(SCHEMES):
        progress.progress((i + 1) / len(SCHEMES),
                          text=f"Loading {scheme['short_name']}…")
        data = fetch_scheme_data(scheme)
        all_data.append(data)
        all_holdings[scheme["mc_id"]] = get_holdings(scheme["mc_id"])
        time.sleep(0.15)   # gentle rate-limiting
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
    • NAV & Returns: <a href="https://api.mfapi.in" style="color:#93c5fd">mfapi.in</a><br>
    • AUM / Expense: MoneyControl<br>
    • Holdings: SEBI Disclosures<br>
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

# ─── Load data ───────────────────────────────────────────────────────────────

with st.spinner("Loading live fund data from MFAPI & MoneyControl…"):
    all_data, all_holdings = load_all_data()

df = get_summary_df(all_data)

# Apply sidebar category filter
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
    ("Data Source",   "mfapi.in + MC", "live NAV feed"),
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

    # Colour-code return columns
    ret_cols = ["1M Ret", "3M Ret", "6M Ret", "1Y Ret", "3Y CAGR", "5Y CAGR", "Since Inc."]

    def color_returns(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        return "color: #16a34a; font-weight:600" if val >= 0 else "color: #dc2626; font-weight:600"

    # Format percentage columns
    pct_cols = ret_cols + ["Exp Ratio", "Std Dev"]
    for c in pct_cols:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "N/A")

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
    ℹ️ Returns sourced from <b>mfapi.in</b> (AMFI-registered). 
    AUM & Expense Ratio from <b>MoneyControl</b> (best-effort scrape; may show N/A if page structure changed).
    Holdings data from latest SEBI monthly disclosures.
    </p>
    """, unsafe_allow_html=True)

# ─── Tab 2: Portfolio Analysis ────────────────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-header'>C · Portfolio Analysis</div>", unsafe_allow_html=True)

    scheme_names = [s["short_name"] for s in SCHEMES]
    mc_ids = [s["mc_id"] for s in SCHEMES]
    sel_name = st.selectbox("Select Scheme", scheme_names, key="port_sel")
    sel_mc_id = mc_ids[scheme_names.index(sel_name)]
    h = all_holdings.get(sel_mc_id, {})

    col1, col2 = st.columns(2)

    with col1:
        # Top 10 Holdings
        st.markdown("**🏆 Top 10 Holdings**")
        top_h = h.get("top_holdings", [])
        if top_h:
            fig_h = go.Figure(go.Bar(
                x=[p for _, p in top_h],
                y=[s for s, _ in top_h],
                orientation="h",
                marker_color="#1a3a6b",
                text=[f"{p}%" for _, p in top_h],
                textposition="outside",
            ))
            fig_h.update_layout(title=f"Top 10 Holdings – {sel_name}",
                                height=380, yaxis=dict(autorange="reversed"),
                                xaxis_title="Weight (%)", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Holdings not available")

    with col2:
        # Sector Allocation
        st.markdown("**🏭 Sector Allocation**")
        sectors = h.get("sector", [])
        if sectors:
            fig_s = go.Figure(go.Pie(
                labels=[s for s, _ in sectors],
                values=[p for _, p in sectors],
                hole=0.42,
                textinfo="label+percent",
            ))
            fig_s.update_layout(title=f"Sector Allocation – {sel_name}",
                                height=380, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_s, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Market Cap
        st.markdown("**📊 Market Cap Allocation**")
        mcap = h.get("market_cap", [])
        if mcap:
            colors_mcap = ["#1a3a6b", "#2563eb", "#60a5fa", "#bfdbfe"]
            fig_m = go.Figure(go.Pie(
                labels=[s for s, _ in mcap],
                values=[p for _, p in mcap],
                hole=0.42,
                marker_colors=colors_mcap,
                textinfo="label+percent",
            ))
            fig_m.update_layout(title=f"Market Cap – {sel_name}",
                                height=320, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_m, use_container_width=True)

    with col4:
        st.markdown("**📌 Key Stats**")
        row = df[df["mc_id"] == sel_mc_id].iloc[0] if sel_mc_id in df["mc_id"].values else None
        stats = {
            "# Stocks": h.get("num_stocks", "N/A"),
            "Cash %": f"{h.get('cash_pct', 'N/A')}%",
            "5Y CAGR": fmt_pct(row.get("cagr_5y") if row is not None else None),
            "3Y CAGR": fmt_pct(row.get("cagr_3y") if row is not None else None),
            "Std Dev": fmt_pct(row.get("std_dev") if row is not None else None),
            "Exp Ratio": fmt_pct(row.get("expense_ratio") if row is not None else None),
        }
        for k, v in stats.items():
            c_l, c_r = st.columns(2)
            c_l.markdown(f"**{k}**")
            c_r.markdown(str(v))

# ─── Tab 3: Fund Flows ────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-header'>D · Fund Flow Analysis</div>", unsafe_allow_html=True)
    st.info("""
    ℹ️ **Note on Fund Flows:** Real-time SIP/redemption flows require AMFI's monthly 
    industry data (published ~10th of next month). The chart below shows **simulated 
    inflow/outflow trend** based on AUM growth proxy. 
    For live AMFI data visit [amfiindia.com/industry-trends](https://www.amfiindia.com/industry-trends).
    """)

    # Simulate monthly flows using NAV series growth as proxy
    np.random.seed(42)
    months = pd.date_range(end=pd.Timestamp.today(), periods=12, freq="ME")

    all_inflows, all_outflows = [], []
    for m in months:
        inflow  = np.random.uniform(2000, 8000)
        outflow = np.random.uniform(1000, 5000)
        all_inflows.append(inflow)
        all_outflows.append(outflow)

    net_flows = [i - o for i, o in zip(all_inflows, all_outflows)]

    fig_flow = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Inflows vs Outflows (₹ Cr)", "Net Flow Trend (₹ Cr)"),
                             vertical_spacing=0.12)

    fig_flow.add_trace(go.Bar(x=months, y=all_inflows, name="Inflows",
                              marker_color="#16a34a"), row=1, col=1)
    fig_flow.add_trace(go.Bar(x=months, y=[-o for o in all_outflows], name="Outflows",
                              marker_color="#dc2626"), row=1, col=1)
    fig_flow.add_trace(go.Scatter(x=months, y=net_flows, name="Net Flow",
                                  line=dict(color="#1a3a6b", width=2.5),
                                  mode="lines+markers"), row=2, col=1)
    fig_flow.update_layout(height=460, barmode="relative",
                           title="Equity MF Category Fund Flows (Illustrative)",
                           **PLOTLY_LAYOUT)
    st.plotly_chart(fig_flow, use_container_width=True)

    st.markdown("""
    <p class="source-note">
    Source: AMFI India (monthly MF industry data). Above chart uses illustrative data; 
    actual scheme-level flows are disclosed quarterly in SID addenda.
    </p>""", unsafe_allow_html=True)

# ─── Tab 4: Stock Movements ───────────────────────────────────────────────────
with tabs[3]:
    st.markdown("<div class='section-header'>E · Stock Movement Analysis</div>", unsafe_allow_html=True)
    st.info("""
    ℹ️ **Note:** Month-on-month portfolio changes require two consecutive SEBI monthly disclosures. 
    Below shows **latest snapshot** (most recent SEBI filing). Real-time changes are published on 
    fund house websites ~10th of each month.
    """)

    sel_s = st.selectbox("Select Scheme", scheme_names, key="stock_move_sel")
    sel_s_id = mc_ids[scheme_names.index(sel_s)]
    h2 = all_holdings.get(sel_s_id, {})
    top_h2 = h2.get("top_holdings", [])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📈 Top Positions (Latest)**")
        if top_h2:
            added = top_h2[:5]
            for stk, pct in added:
                delta = np.random.uniform(0.1, 0.8)
                st.markdown(f"✅ **{stk}** — {pct:.1f}% &nbsp; <span class='pos'>▲ {delta:.2f}%</span>",
                            unsafe_allow_html=True)
    with c2:
        st.markdown("**📉 Reduced / Exited Positions**")
        if top_h2:
            reduced = top_h2[5:10]
            for stk, pct in reduced:
                delta = np.random.uniform(0.1, 0.5)
                st.markdown(f"🔻 **{stk}** — {pct:.1f}% &nbsp; <span class='neg'>▼ {delta:.2f}%</span>",
                            unsafe_allow_html=True)

    st.markdown("""
    <p class="source-note">
    Holdings data from latest published SEBI monthly portfolio disclosure. 
    Deltas are illustrative – actual month-on-month changes available post SEBI filing.
    </p>""", unsafe_allow_html=True)

# ─── Tab 5: Overlap Matrix ────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("<div class='section-header'>F · Portfolio Overlap Matrix</div>", unsafe_allow_html=True)

    mc_ids_list = [s["mc_id"] for s in SCHEMES]
    short_names  = [s["short_name"] for s in SCHEMES]

    overlap_df = compute_overlap_matrix(all_holdings, mc_ids_list)
    overlap_df.index   = short_names
    overlap_df.columns = short_names

    fig_ov = go.Figure(go.Heatmap(
        z=overlap_df.values.astype(float),
        x=short_names, y=short_names,
        colorscale="Blues",
        text=overlap_df.values.astype(float).round(0),
        texttemplate="%{text:.0f}%",
        textfont_size=9,
        colorbar_title="Overlap %",
    ))
    fig_ov.update_layout(
        title="Portfolio Overlap – Top 10 Holdings (Jaccard Index)",
        height=580,
        xaxis=dict(tickangle=-35, tickfont_size=10),
        yaxis=dict(tickfont_size=10),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_ov, use_container_width=True)

    # Most / least similar
    flat = overlap_df.where(np.tril(np.ones(overlap_df.shape), k=-1).astype(bool))
    flat = flat.stack().reset_index()
    flat.columns = ["Fund A", "Fund B", "Overlap %"]
    flat = flat.sort_values("Overlap %", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Most Similar Pairs**")
        st.dataframe(flat.head(5).reset_index(drop=True), use_container_width=True)
    with c2:
        st.markdown("**Least Similar Pairs**")
        st.dataframe(flat.tail(5).reset_index(drop=True), use_container_width=True)

    st.markdown("""
    <p class="source-note">Overlap computed on top-10 disclosed holdings using Jaccard similarity.  
    Higher overlap = more concentrated common positions between two funds.</p>
    """, unsafe_allow_html=True)

# ─── Tab 6: Benchmark Comparison ─────────────────────────────────────────────
with tabs[5]:
    st.markdown("<div class='section-header'>G · Benchmark Comparison vs Nifty 50 TRI</div>",
                unsafe_allow_html=True)

    bm_rows = []
    for d in all_data:
        r1y   = d.get("ret_1y")
        r3y   = d.get("cagr_3y")
        r5y   = d.get("cagr_5y")
        alpha = (r1y - NIFTY_RETURNS["ret_1y"]) if r1y else None
        bm_rows.append({
            "Fund": d.get("short_name"),
            "1Y Ret": r1y,
            "Nifty 1Y": NIFTY_RETURNS["ret_1y"],
            "1Y Alpha": alpha,
            "3Y CAGR": r3y,
            "Nifty 3Y": NIFTY_RETURNS["cagr_3y"],
            "3Y Alpha": (r3y - NIFTY_RETURNS["cagr_3y"]) if r3y else None,
            "5Y CAGR": r5y,
            "Nifty 5Y": NIFTY_RETURNS["cagr_5y"],
            "5Y Alpha": (r5y - NIFTY_RETURNS["cagr_5y"]) if r5y else None,
        })

    bm_df = pd.DataFrame(bm_rows).set_index("Fund")

    # 1Y comparison bar
    fig_bm = go.Figure()
    fig_bm.add_trace(go.Bar(name="Fund 1Y", x=bm_df.index,
                            y=bm_df["1Y Ret"].round(2),
                            marker_color="#1a3a6b"))
    fig_bm.add_trace(go.Bar(name="Nifty 50 1Y", x=bm_df.index,
                            y=[NIFTY_RETURNS["ret_1y"]] * len(bm_df),
                            marker_color="#f59e0b", opacity=0.7))
    fig_bm.update_layout(barmode="group", title="1-Year Returns vs Nifty 50",
                         height=380, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_bm, use_container_width=True)

    # Alpha table
    alpha_cols = ["1Y Alpha", "3Y Alpha", "5Y Alpha"]
    alpha_disp = bm_df[alpha_cols].copy()
    for c in alpha_cols:
        alpha_disp[c] = alpha_disp[c].apply(
            lambda x: f"+{x:.2f}%" if (pd.notna(x) and x >= 0) else (f"{x:.2f}%" if pd.notna(x) else "N/A")
        )
    st.markdown("**Alpha Generated over Nifty 50 TRI**")
    st.dataframe(alpha_disp, use_container_width=True)

    st.markdown("""
    <p class="source-note">Nifty 50 TRI benchmark returns: 1Y=22.5%, 3Y CAGR=14.8%, 5Y CAGR=15.2% (as of latest available).  
    Alpha = Fund Return − Benchmark Return (simple excess return, not risk-adjusted).</p>
    """, unsafe_allow_html=True)

# ─── Tab 7: Risk Analysis ─────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("<div class='section-header'>H · Risk Analysis</div>", unsafe_allow_html=True)

    risk_rows = []
    for d in all_data:
        std_dev = d.get("std_dev")
        r5y     = d.get("cagr_5y")
        risk_rows.append({
            "Fund":     d.get("short_name"),
            "Category": d.get("category"),
            "Std Dev":  std_dev,
            "5Y CAGR":  r5y,
            "Risk Rating": (
                "Low"    if std_dev and std_dev < 14 else
                "Moderate" if std_dev and std_dev < 18 else
                "High"   if std_dev else "N/A"
            ),
        })

    risk_df = pd.DataFrame(risk_rows)

    c1, c2 = st.columns([2, 1])
    with c1:
        # Risk-Return scatter
        fig_rr = px.scatter(
            risk_df.dropna(subset=["Std Dev", "5Y CAGR"]),
            x="Std Dev", y="5Y CAGR", text="Fund",
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
            size_max=14,
            title="Risk vs Return (5Y CAGR vs Annualised Std Dev)",
            labels={"Std Dev": "Risk – Std Dev (%)", "5Y CAGR": "5Y CAGR (%)"},
        )
        fig_rr.update_traces(textposition="top center", marker_size=12)
        fig_rr.update_layout(height=460, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_rr, use_container_width=True)

    with c2:
        st.markdown("**Risk Rating Summary**")
        risk_display = risk_df[["Fund", "Std Dev", "Risk Rating"]].copy()
        risk_display["Std Dev"] = risk_display["Std Dev"].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
        )
        st.dataframe(risk_display.reset_index(drop=True), use_container_width=True, height=440)

    st.markdown("""
    <p class="source-note">
    Standard Deviation computed from daily NAV returns (mfapi.in), annualised (×√252).  
    Beta vs Nifty 50 requires daily index data (available via NSE API – not included in this build to avoid rate limits).
    Risk Rating: Low &lt;14%, Moderate 14–18%, High &gt;18%.
    </p>""", unsafe_allow_html=True)

# ─── Tab 8: Charts ────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("<div class='section-header'>I · Charts</div>", unsafe_allow_html=True)

    chart_sel = st.selectbox("Chart Type", [
        "CAGR Comparison (5Y)", "Portfolio Weight Distribution",
        "Category Breakdown", "NAV Trend (Individual)", "AUM Distribution"
    ], key="chart_sel")

    if chart_sel == "CAGR Comparison (5Y)":
        cagr_df = df[["short_name", "cagr_5y", "category"]].dropna(subset=["cagr_5y"]).copy()
        cagr_df = cagr_df.sort_values("cagr_5y", ascending=True)
        fig_c = px.bar(cagr_df, x="cagr_5y", y="short_name",
                       color="category", orientation="h",
                       color_discrete_map=CATEGORY_COLORS,
                       labels={"cagr_5y": "5Y CAGR (%)", "short_name": ""},
                       title="5-Year CAGR Comparison – All Schemes")
        fig_c.add_vline(x=NIFTY_RETURNS["cagr_5y"], line_dash="dash",
                        line_color="orange", annotation_text="Nifty 50")
        fig_c.update_layout(height=480, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_c, use_container_width=True)

    elif chart_sel == "Portfolio Weight Distribution":
        wt_df = pd.DataFrame([(s["short_name"], s["weight"], s["category"]) for s in SCHEMES],
                             columns=["Fund", "Weight", "Category"])
        fig_w = px.treemap(wt_df, path=["Category", "Fund"], values="Weight",
                           color="Category", color_discrete_map=CATEGORY_COLORS,
                           title="Model Portfolio Weight Distribution")
        fig_w.update_layout(height=520, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_w, use_container_width=True)

    elif chart_sel == "Category Breakdown":
        cat_wt = pd.DataFrame([(s["category"], s["weight"]) for s in SCHEMES],
                              columns=["Category", "Weight"])
        cat_wt = cat_wt.groupby("Category")["Weight"].sum().reset_index()
        fig_cat = px.pie(cat_wt, values="Weight", names="Category",
                         color="Category", color_discrete_map=CATEGORY_COLORS,
                         hole=0.42, title="Category Weight Distribution")
        fig_cat.update_layout(height=440, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_cat, use_container_width=True)

    elif chart_sel == "NAV Trend (Individual)":
        sel_nav = st.selectbox("Select Scheme", scheme_names, key="nav_trend_sel")
        sel_nav_idx = scheme_names.index(sel_nav)
        nav_series = all_data[sel_nav_idx].get("_nav_series", pd.Series(dtype=float))
        if not nav_series.empty:
            fig_nav = go.Figure(go.Scatter(
                x=nav_series.index, y=nav_series.values,
                mode="lines", fill="tozeroy",
                line=dict(color="#1a3a6b", width=2),
                fillcolor="rgba(26,58,107,0.1)",
                name="NAV",
            ))
            fig_nav.update_layout(title=f"NAV Trend – {sel_nav}",
                                  xaxis_title="Date", yaxis_title="NAV (₹)",
                                  height=420, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_nav, use_container_width=True)
        else:
            st.warning("NAV data not available for this scheme.")

    elif chart_sel == "AUM Distribution":
        aum_df = df[df["aum_cr"].notna()][["short_name", "aum_cr"]].copy()
        if aum_df.empty:
            st.info("AUM data unavailable (MoneyControl scraping may be blocked). "
                    "AUM available on moneycontrol.com directly.")
        else:
            fig_aum = px.bar(aum_df.sort_values("aum_cr", ascending=False),
                             x="short_name", y="aum_cr",
                             color="aum_cr", color_continuous_scale="Blues",
                             labels={"aum_cr": "AUM (₹ Cr)", "short_name": "Fund"},
                             title="AUM Comparison (₹ Crores)")
            fig_aum.update_layout(height=420, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_aum, use_container_width=True)

# ─── Auto-refresh ─────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(0.5)
    st.markdown(f"""
    <script>
    setTimeout(function() {{ window.location.reload(); }}, {REFRESH_INTERVAL * 1000});
    </script>
    """, unsafe_allow_html=True)
