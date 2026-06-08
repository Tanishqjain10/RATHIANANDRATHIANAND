"""
ingestion.py - FINAL PRODUCTION VERSION
Value Research via Playwright (JS-rendered pages)
"""

import re
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict

import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)

MFAPI_DETAIL = "https://api.mfapi.in/mf/{}"

MC_TO_MFAPI = {
    "MES082": 119597, "MSB501": 125494, "MDS580": 119270, "MAG091": 145552,
    "MKM099": 120503, "MMS025": 118989, "INVESCO_SC": 120832, "MHD1144": 119598,
    "MKM1397": 147946, "MCAA002": 147977, "MSB520": 125497, "MPI643": 120586,
    "MLI1122": 120840, "MPI2056": 120600,
}

# ====================== PLAYWRIGHT SCRAPER ======================

async def scrape_valueresearch_playwright(vr_url: str) -> Dict:
    result = {"source": "Value Research (Playwright)", "url": vr_url}
    if not PLAYWRIGHT_AVAILABLE:
        result["error"] = "Playwright not available"
        return result

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(vr_url, timeout=45000)
            await page.wait_for_load_state("networkidle", timeout=30000)

            content = await page.content()

            # NAV
            nav_match = re.search(r'The latest declared NAV of .*? is ₹?([\d,]+\.\d{2})', content, re.I)
            if nav_match:
                nav = float(nav_match.group(1).replace(",", ""))
                if nav > 0:
                    result["nav"] = nav

            # AUM
            aum_match = re.search(r'The fund has an overall AUM \(Assets Under Management\) of ₹?([\d,]+\.?\d*)\s*Cr', content, re.I)
            if aum_match:
                result["aum_raw"] = f"₹{aum_match.group(1)} Cr"

            # Expense Ratio
            exp_match = re.search(r'The fund has an expense ratio of (\d+\.\d+)%', content, re.I)
            if exp_match:
                exp = float(exp_match.group(1))
                if 0 < exp <= 5:
                    result["expense_ratio"] = exp

            # Fund Manager
            manager_match = re.search(r'it is currently managed by ([A-Za-z\s&.,-]+)', content, re.I)
            if manager_match:
                result["fund_manager"] = manager_match.group(1).strip()[:120]

            # Launch Date
            launch_match = re.search(r'Launched on ([A-Za-z]+\s+\d{1,2},?\s+\d{4})', content, re.I)
            if launch_match:
                result["launch_date"] = launch_match.group(1)

            await browser.close()
            logger.info(f"✅ Playwright Success: {vr_url}")

    except Exception as e:
        logger.error(f"Playwright failed for {vr_url}: {e}")
        result["error"] = str(e)

    return result

def scrape_valueresearch(vr_url: str) -> Dict:
    """Sync wrapper for Playwright"""
    try:
        return asyncio.run(scrape_valueresearch_playwright(vr_url))
    except Exception as e:
        logger.warning(f"Playwright wrapper failed: {e}")
        return {"source": "Value Research", "error": str(e)}

# ====================== MFAPI + CALCULATIONS ======================

def fetch_mfapi_data(mfapi_id: int) -> dict:
    import requests
    try:
        r = requests.get(MFAPI_DETAIL.format(mfapi_id), timeout=15)
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

def compute_cagr(series: pd.Series, years: float) -> Optional[float]:
    if series.empty or len(series) < 2:
        return None
    end_date = series.index[-1]
    start_date = end_date - timedelta(days=int(years * 365.25))
    sub = series[series.index >= start_date]
    if len(sub) < 2:
        return None
    start_nav = sub.iloc[0]
    end_nav = sub.iloc[-1]
    actual_years = (sub.index[-1] - sub.index[0]).days / 365.25
    if actual_years <= 0 or start_nav <= 0:
        return None
    return ((end_nav / start_nav) ** (1 / actual_years) - 1) * 100

def compute_returns(series: pd.Series) -> dict:
    if series.empty:
        return {}
    latest_nav = series.iloc[-1]
    latest_date = series.index[-1]

    def pct(days):
        target = latest_date - timedelta(days=days)
        sub = series[series.index <= target]
        if sub.empty:
            return None
        old_nav = sub.iloc[-1]
        return ((latest_nav / old_nav) - 1) * 100 if old_nav > 0 else None

    return {
        "nav": round(latest_nav, 4),
        "nav_date": latest_date.strftime("%d %b %Y"),
        "ret_1m": pct(30),
        "ret_3m": pct(91),
        "ret_6m": pct(183),
        "ret_1y": pct(365),
        "cagr_3y": compute_cagr(series, 3),
        "cagr_5y": compute_cagr(series, 5),
        "inception_cagr": compute_cagr(series, (series.index[-1] - series.index[0]).days / 365.25),
    }

def compute_risk_metrics(series: pd.Series) -> dict:
    if len(series) < 30:
        return {}
    daily_ret = series.pct_change().dropna()
    std_dev = round(daily_ret.std() * (252 ** 0.5) * 100, 2)
    return {"std_dev": std_dev}

# ====================== MAIN FETCH ======================

def fetch_scheme_data(scheme: dict) -> dict:
    result = {**scheme, "fetched_at": datetime.now().isoformat()}

    if scheme.get("vr_url"):
        vr_data = scrape_valueresearch(scheme["vr_url"])
        result.update({k: v for k, v in vr_data.items() if v is not None})

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
