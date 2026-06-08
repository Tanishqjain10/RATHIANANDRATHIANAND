"""
modules/ingestion.py - Fully Live with Value Research
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

MFAPI_DETAIL = "https://api.mfapi.in/mf/{}"

MC_TO_MFAPI = { ... }  # your existing mapping

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
def _get(url: str, timeout: int = 15) -> requests.Response:
    return SESSION.get(url, timeout=timeout)

def scrape_valueresearch(vr_url: str) -> dict:
    result = {"source": "Value Research", "url": vr_url}
    try:
        r = _get(vr_url)
        text = r.text
        soup = BeautifulSoup(r.text, "lxml")

        # NAV, AUM, Expense, Manager, Launch, Rank
        nav_match = re.search(r'₹([\d,]+\.\d{2})', text)
        if nav_match: result["nav"] = float(nav_match.group(1).replace(",", ""))

        aum_match = re.search(r'AUM.*?₹([\d,]+\.?\d*)\s*Cr', text, re.I)
        if aum_match: result["aum_raw"] = f"₹{aum_match.group(1)} Cr"

        exp_match = re.search(r'Expense Ratio.*?(\d+\.\d+)%', text, re.I)
        if exp_match: result["expense_ratio"] = float(exp_match.group(1))

        # Holdings from portfolio section
        top_holdings = []
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                name = cells[0].get_text(strip=True)
                pct_text = cells[-1].get_text(strip=True)
                pct_match = re.search(r'[\d.]+', pct_text)
                if name and pct_match and len(top_holdings) < 10:
                    try:
                        top_holdings.append((name, round(float(pct_match.group()), 2)))
                    except:
                        pass
        if top_holdings:
            result["top_holdings"] = top_holdings

        logger.info(f"VR live fetch successful: {vr_url}")
    except Exception as e:
        logger.warning(f"VR failed {vr_url}: {e}")

    return result

# ... (keep your existing MFAPI functions: fetch_mfapi_data, nav_series_from_mfapi, compute_returns, etc.)

def fetch_scheme_data(scheme: dict) -> dict:
    result = {**scheme, "fetched_at": datetime.now().isoformat()}

    if scheme.get("vr_url"):
        vr_data = scrape_valueresearch(scheme["vr_url"])
        result.update(vr_data)

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
    """Holdings now fetched live from VR in fetch_scheme_data"""
    return {"top_holdings": [], "sector": [], "market_cap": [], "num_stocks": 0, "cash_pct": 0.0}
