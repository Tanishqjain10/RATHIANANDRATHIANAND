"""
cleaning.py
-----------
Cleans and normalises raw ingested scheme data into a tidy DataFrame.
"""

import re
import pandas as pd
import numpy as np


def parse_aum(raw: str) -> float | None:
    """Convert strings like 'Rs.12,345 Cr' or '1234.56' to float crores."""
    if not raw:
        return None
    cleaned = (
        str(raw)
        .replace(",", "")
        .replace("Rs.", "")
        .replace("INR", "")
        .replace("₹", "")
        .replace("â‚¹", "")
        .strip()
    )
    match = re.search(r"([\d.]+)\s*(Cr|Crore|Lakh|K|M|B)?", cleaned, re.I)
    if not match:
        return None
    val = float(match.group(1))
    suffix = (match.group(2) or "Cr").lower()
    if suffix == "lakh":
        val /= 100
    elif suffix == "k":
        val /= 1000
    elif suffix == "m":
        val /= 10
    elif suffix == "b":
        val *= 100
    return round(val, 2)


def fmt_pct(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}%"


def fmt_cr(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if val >= 100_000:
        return f"Rs.{val / 100_000:.1f}L Cr"
    if val >= 1_000:
        return f"Rs.{val / 1_000:.1f}K Cr"
    return f"Rs.{val:.0f} Cr"


DISPLAY_COLS = [
    "sr",
    "short_name",
    "category",
    "weight",
    "nav",
    "nav_date",
    "aum_cr",
    "expense_ratio",
    "inception_cagr",
    "ret_1m",
    "ret_3m",
    "ret_6m",
    "ret_1y",
    "cagr_3y",
    "cagr_5y",
    "std_dev",
    "cat_rank",
    "num_stocks",
    "fund_manager",
    "launch_date",
    "last_updated",
    "fetched_url_count",
    "provided_url_count",
    "metadata_source",
    "mfapi_fetched",
    "mfapi_scheme_name",
]

RENAME = {
    "short_name": "Fund",
    "category": "Category",
    "weight": "Wt %",
    "nav": "NAV (Rs.)",
    "nav_date": "NAV Date",
    "aum_cr": "AUM (Cr)",
    "expense_ratio": "Exp Ratio",
    "inception_cagr": "Since Inc.",
    "ret_1m": "1M Ret",
    "ret_3m": "3M Ret",
    "ret_6m": "6M Ret",
    "ret_1y": "1Y Ret",
    "cagr_3y": "3Y CAGR",
    "cagr_5y": "5Y CAGR",
    "std_dev": "Std Dev",
    "cat_rank": "Cat Rank",
    "num_stocks": "# Stocks",
    "fund_manager": "Fund Manager",
    "launch_date": "Launch",
    "last_updated": "Last Updated",
    "fetched_url_count": "URLs Fetched",
    "provided_url_count": "URLs Provided",
    "metadata_source": "AUM/TER Source",
    "mfapi_fetched": "NAV Feed",
    "mfapi_scheme_name": "mfapi Scheme",
}


def build_summary_df(all_data: list) -> pd.DataFrame:
    """Convert enriched scheme dicts to a clean summary DataFrame."""
    rows = []
    for d in all_data:
        aum_cr = d.get("aum_cr")
        if aum_cr is None and d.get("aum_raw"):
            aum_cr = parse_aum(d.get("aum_raw"))

        row = {
            "sr": d.get("sr"),
            "short_name": d.get("short_name", d.get("name", "")),
            "name": d.get("name", ""),
            "category": d.get("category", ""),
            "weight": d.get("weight"),
            "mc_id": d.get("mc_id"),
            "nav": d.get("nav"),
            "nav_date": d.get("nav_date"),
            "aum_cr": aum_cr,
            "expense_ratio": d.get("expense_ratio"),
            "inception_cagr": d.get("inception_cagr"),
            "ret_1m": d.get("ret_1m"),
            "ret_3m": d.get("ret_3m"),
            "ret_6m": d.get("ret_6m"),
            "ret_1y": d.get("ret_1y"),
            "cagr_3y": d.get("cagr_3y"),
            "cagr_5y": d.get("cagr_5y"),
            "std_dev": d.get("std_dev"),
            "cat_rank": d.get("cat_rank"),
            "num_stocks": d.get("num_stocks"),
            "fund_manager": d.get("fund_manager", ""),
            "launch_date": d.get("launch_date", ""),
            "last_updated": d.get("last_updated", ""),
            "provided_urls": d.get("provided_urls", []),
            "fetched_urls": d.get("fetched_urls", []),
            "failed_urls": d.get("failed_urls", []),
            "provided_url_count": d.get("provided_url_count", 0),
            "fetched_url_count": d.get("fetched_url_count", 0),
            "failed_url_count": d.get("failed_url_count", 0),
            "metadata_source": d.get(
                "metadata_source",
                "Value Research" if d.get("aum_raw") or d.get("expense_ratio") else "",
            ),
            "metadata_fetched": d.get("metadata_fetched", False),
            "metadata_snapshot": d.get("metadata_snapshot", False),
            "mfapi_id": d.get("mfapi_id"),
            "mfapi_scheme_name": d.get("mfapi_scheme_name", ""),
            "mfapi_fetched": d.get("mfapi_fetched", False),
            "fetched_at": d.get("fetched_at"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Subset and rename columns for Streamlit tables."""
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    return df[cols].copy().rename(columns=RENAME)


def compute_overlap_matrix(holdings_map: dict, scheme_ids: list) -> pd.DataFrame:
    """
    Compute pairwise Jaccard overlap of top holdings per scheme.
    holdings_map: {mc_id: {"top_holdings": [(stock_name, pct), ...]}}
    """
    sets = {}
    for sid in scheme_ids:
        holdings = holdings_map.get(sid, {}).get("top_holdings", [])
        sets[sid] = {stock for stock, _ in holdings}

    matrix = pd.DataFrame(index=scheme_ids, columns=scheme_ids, dtype=float)
    for left in scheme_ids:
        for right in scheme_ids:
            if left == right:
                matrix.loc[left, right] = 100.0
                continue
            intersection = len(sets[left] & sets[right])
            union = len(sets[left] | sets[right])
            matrix.loc[left, right] = round(intersection / union * 100, 1) if union else 0.0
    return matrix
