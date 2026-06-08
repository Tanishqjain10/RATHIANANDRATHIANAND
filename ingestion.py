"""
modules/ingestion.py - IMPROVED LIVE SCRAPER
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

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

MC_TO_MFAPI = { ... }  # your existing mapping

MFAPI_DETAIL = "https://api.mfapi.in/mf/{}"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
def _get(url: str, timeout: int = 20):
    return SESSION.get(url, timeout=timeout)

def scrape_valueresearch(vr_url: str) -> dict:
    result = {"source": "Value Research", "url": vr_url}
    try:
        r = _get(vr_url)
        soup = BeautifulSoup(r.text, "lxml")

        # Better extraction using labels
        text = r.text.lower()

        # NAV
        nav_match = re.search(r'current nav.*?₹?([\d,]+\.\d{2})', r.text, re.I)
        if nav_match:
            result["nav"] = float(nav_match.group(1).replace(",", ""))

        # AUM
        aum_match = re.search(r'aum.*?₹?([\d,]+\.?\d*)\s*cr', r.text, re.I)
        if aum_match:
            result["aum_raw"] = f"₹{aum_match.group(1)} Cr"

        # Expense Ratio
        exp_match = re.search(r'expense ratio.*?(\d+\.\d+)%', r.text, re.I)
        if exp_match:
            result["expense_ratio"] = float(exp_match.group(1))

        # Fund Manager
        manager = soup.find(string=re.compile("Fund Manager", re.I))
        if manager:
            parent = manager.find_parent()
            if parent:
                result["fund_manager"] = parent.get_text(strip=True)[:80]

        # Launch Date
        launch = soup.find(string=re.compile("Launch Date|Inception", re.I))
        if launch:
            parent = launch.find_parent()
            if parent:
                result["launch_date"] = parent.get_text(strip=True)[:30]

        logger.info(f"✅ Improved VR scrape: {vr_url}")
    except Exception as e:
        logger.warning(f"VR scrape failed: {e}")

    return result

# Keep your original MFAPI functions (fetch_mfapi_data, nav_series_from_mfapi, compute_returns, etc.)

def fetch_scheme_data(scheme: dict) -> dict:
    result = {**scheme, "fetched_at": datetime.now().isoformat()}

    if scheme.get("vr_url"):
        vr_data = scrape_valueresearch(scheme["vr_url"])
        result.update({k: v for k, v in vr_data.items() if v is not None})

    # MFAPI for returns
    mfapi_id = MC_TO_MFAPI.get(scheme.get("mc_id"))
    nav_series = pd.Series(dtype=float)
    if mfapi_id:
        raw = fetch_mfapi_data(mfapi_id)
        nav_series = nav_series_from_mfapi(raw)

    result.update(compute_returns(nav_series))
    result.update(compute_risk_metrics(nav_series))

    if not nav_series.empty:
        cutoff = nav_series.index[-1] - timedelta(days=3 * 365)
        result["_nav_series"] = nav_series[nav_series.index >= cutoff]

    return result

def get_holdings(mc_id: str) -> dict:
    return {"top_holdings": [], "sector": [], "market_cap": [], "num_stocks": 0, "cash_pct": 0.0}
