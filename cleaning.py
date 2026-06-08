"""
modules/cleaning.py
-------------------
Cleans and normalises raw ingested scheme data into a tidy DataFrame.
"""

import re
import pandas as pd
import numpy as np
from typing import List


# ── AUM parsing ──────────────────────────────────────────────────────────────

def parse_aum(raw: str) -> float | None:
    """Convert strings like '₹12,345 Cr' or '1234.56' to float (crores)."""
    if not raw:
        return None
    raw = str(raw).replace(",", "").replace("₹", "").strip()
    m = re.search(r"([\d.]+)\s*(Cr|Lakh|K|M|B)?", raw, re.I)
    if not m:
        return None
    val = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    if suffix == "lakh":
        val /= 100          # convert lakh → crore
    elif suffix in ("k",):
        val /= 1000         # thousands → assume crores already in Cr context
    elif suffix == "m":
        val = val / 10      # million ≈ 10 lakh = 0.1 crore (USD context – ignore)
    return round(val, 2)


# ── Formatter helpers ─────────────────────────────────────────────────────────

def fmt_pct(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}%"


def fmt_cr(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if val >= 1_00_000:
        return f"₹{val/1_00_000:.1f}L Cr"
    if val >= 1_000:
        return f"₹{val/1_000:.1f}K Cr"
    return f"₹{val:.0f} Cr"


# ── DataFrame builder ─────────────────────────────────────────────────────────

DISPLAY_COLS = [
    "sr", "short_name", "category", "weight",
    "nav", "nav_date",
    "aum_cr",
    "expense_ratio",
    "inception_cagr", "ret_1m", "ret_3m", "ret_6m", "ret_1y",
    "cagr_3y", "cagr_5y",
    "std_dev",
    "cat_rank",
    "num_stocks",
    "fund_manager",
    "launch_date",
    "last_updated",
    "fetched_url_count",
    "provided_url_count",
    "mfapi_fetched",
]

RENAME = {
    "short_name":     "Fund",
    "category":       "Category",
    "weight":         "Wt %",
    "nav":            "NAV (₹)",
    "nav_date":       "NAV Date",
    "aum_cr":         "AUM (Cr)",
    "expense_ratio":  "Exp Ratio",
    "inception_cagr": "Since Inc.",
    "ret_1m":         "1M Ret",
    "ret_3m":         "3M Ret",
    "ret_6m":         "6M Ret",
    "ret_1y":         "1Y Ret",
    "cagr_3y":        "3Y CAGR",
    "cagr_5y":        "5Y CAGR",
    "std_dev":        "Std Dev",
    "cat_rank":       "Cat Rank",
    "num_stocks":     "# Stocks",
    "fund_manager":   "Fund Manager",
    "launch_date":    "Launch",
    "last_updated":   "Last Updated",
    "fetched_url_count": "URLs Fetched",
    "provided_url_count": "URLs Provided",
    "mfapi_fetched":  "NAV Feed",
}


def build_summary_df(all_data: list) -> pd.DataFrame:
    """Convert list of enriched scheme dicts to a clean summary DataFrame."""
    rows = []
    for d in all_data:
        aum_cr = None
        aum_raw = d.get("aum_raw", "")
        if aum_raw:
            aum_cr = parse_aum(aum_raw)

        row = {
            "sr":            d.get("sr"),
            "short_name":    d.get("short_name", d.get("name", "")),
            "name":          d.get("name", ""),
            "category":      d.get("category", ""),
            "weight":        d.get("weight"),
            "mc_id":         d.get("mc_id"),
            "nav":           d.get("nav"),
            "nav_date":      d.get("nav_date"),
            "aum_cr":        aum_cr,
            "expense_ratio": d.get("expense_ratio"),
            "inception_cagr":d.get("inception_cagr"),
            "ret_1m":        d.get("ret_1m"),
            "ret_3m":        d.get("ret_3m"),
            "ret_6m":        d.get("ret_6m"),
            "ret_1y":        d.get("ret_1y"),
            "cagr_3y":       d.get("cagr_3y"),
            "cagr_5y":       d.get("cagr_5y"),
            "std_dev":       d.get("std_dev"),
            "cat_rank":      d.get("cat_rank"),
            "num_stocks":    d.get("num_stocks"),
            "fund_manager":  d.get("fund_manager", ""),
            "launch_date":   d.get("launch_date", ""),
            "last_updated":   d.get("last_updated", ""),
            "provided_urls":  d.get("provided_urls", []),
            "fetched_urls":   d.get("fetched_urls", []),
            "failed_urls":    d.get("failed_urls", []),
            "provided_url_count": d.get("provided_url_count", 0),
            "fetched_url_count":  d.get("fetched_url_count", 0),
            "failed_url_count":   d.get("failed_url_count", 0),
            "mfapi_id":      d.get("mfapi_id"),
            "mfapi_fetched": d.get("mfapi_fetched", False),
            "fetched_at":     d.get("fetched_at"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Subset & rename for display in Streamlit tables."""
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    out = df[cols].copy()
    out = out.rename(columns=RENAME)
    return out


# ── Overlap Matrix ────────────────────────────────────────────────────────────

def compute_overlap_matrix(holdings_map: dict, scheme_ids: list) -> pd.DataFrame:
    """
    Compute pairwise Jaccard overlap of top-10 holdings per scheme.
    holdings_map: {mc_id: [(stock_name, pct), ...]}
    Returns DataFrame with scheme short names as index/columns.
    """
    sets = {}
    for sid in scheme_ids:
        h = holdings_map.get(sid, {}).get("top_holdings", [])
        sets[sid] = set(s for s, _ in h)

    matrix = pd.DataFrame(index=scheme_ids, columns=scheme_ids, dtype=float)
    for a in scheme_ids:
        for b in scheme_ids:
            if a == b:
                matrix.loc[a, b] = 100.0
            else:
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b])
                matrix.loc[a, b] = round(inter / union * 100, 1) if union else 0.0
    return matrix
