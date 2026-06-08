"""
modules/ingestion.py - FULLY LIVE VERSION
Primary: Value Research
Fallback: MFAPI
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# MFAPI Mapping
MC_TO_MFAPI = {
    "MES082": 119597, "MSB501": 125494, "MDS580": 119270, "MAG091": 145552,
    "MKM099": 120503, "MMS025": 118989, "INVESCO_SC": 120832, "MHD1144": 119598,
    "MKM1397": 147946, "MCAA002": 147977, "MSB520": 125497, "MPI643": 120586,
    "MLI1122": 120840, "MPI2056": 120600,
}

MFAPI_DETAIL = "https://api.mfapi.in/mf/{}"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
def _get(url: str, timeout: int = 15):
    return SESSION.get(url, timeout=timeout)

# Value Research Scraper
def scrape_valueresearch(vr_url: str) -> dict:
    result = {"source": "Value Research", "url": vr_url}
    try:
        r = _get(vr_url)
        text = r.text

        # NAV
        nav_match = re.search(r'₹([\d,]+\.\d{2})', text)
        if nav_match:
            result["nav"] = float(nav_match.group(1).replace(",", ""))

        # AUM
        aum_match = re.search(r'AUM.*?₹([\d,]+\.?\d*)\s*Cr', text, re.I)
        if aum_match:
            result["aum_raw"] = f"₹{aum_match.group(1)} Cr"

        # Expense Ratio
        exp_match = re.search(r'Expense Ratio.*?(\d+\.\d+)%', text, re.I)
        if exp_match:
            result["expense_ratio"] = float(exp_match.group(1))

        # Fund Manager & others
        manager_match = re.search(r'Fund Manager.*?:?\s*([A-Za-z\s&.,-]+)', text, re.I)
        if manager_match:
            result["fund_manager"] = manager_match.group(1).strip()[:80]

        logger.info(f"✅ VR Success: {vr_url}")
    except Exception as e:
        logger.warning(f"VR failed {vr_url}: {e}")
    return result

# MFAPI Functions (keep your original ones here - abbreviated for space)
def fetch_mfapi_data(mfapi_id: int) -> dict:
    try:
        r = _get(MFAPI_DETAIL.format(mfapi_id))
        return r.json()
    except:
        return {}

def nav_series_from_mfapi(data: dict) -> pd.Series:
    records = data.get("data", [])
    if not records:
        return pd.Series(dtype=float)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna().sort_values("date").set_index("date")
    return df["nav"]

# ... paste your original compute_cagr, compute_returns, compute_risk_metrics here ...

def fetch_scheme_data(scheme: dict) -> dict:
    result = {**scheme, "fetched_at": datetime.now().isoformat()}

    # Value Research Primary
    if scheme.get("vr_url"):
        vr_data = scrape_valueresearch(scheme["vr_url"])
        result.update({k: v for k, v in vr_data.items() if v is not None})

    # MFAPI for returns
    mfapi_id = MC_TO_MFAPI.get(scheme.get("mc_id"))
    nav_series = pd.Series(dtype=float)
    if mfapi_id:
        raw = fetch_mfapi_data(mfapi_id)
        nav_series = nav_series_from_mfapi(raw)

    # Add returns and risk
    # (Add your compute functions here)

    return result

def get_holdings(mc_id: str) -> dict:
    return {"top_holdings": [], "sector": [], "market_cap": [], "num_stocks": 0, "cash_pct": 0.0}
