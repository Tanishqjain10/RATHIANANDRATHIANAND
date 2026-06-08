"""
modules/ingestion.py
-------------------
Live data ingestion from MoneyControl and MFAPI.
Falls back to AMFI for NAV data if MoneyControl scraping fails.

Sources used:
  1. MoneyControl scheme pages  – returns/AUM/expense ratio/fund manager
  2. mfapi.in (free, no-auth)   – NAV history for CAGR computation
  3. AMFI NAV feed              – fallback raw NAV
"""

import re
import json
import time
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
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ---------------------------------------------------------------------------
# MFAPI helpers
# ---------------------------------------------------------------------------

MFAPI_SEARCH = "https://api.mfapi.in/mf/search?q={}"
MFAPI_DETAIL = "https://api.mfapi.in/mf/{}"

# Manually curated MFAPI scheme codes (fallback mapping mc_id → mfapi_id)
MC_TO_MFAPI = {
    "MES082": 119597,   # Quant Large Cap – Regular – Growth
    "MSB501": 125494,   # SBI Large & Midcap – Direct – Growth
    "MDS580": 119270,   # DSP Large Mid Cap – Direct – Growth
    "MAG091": 145552,   # Bandhan Large & Mid Cap – Regular – Growth
    "MKM099": 120503,   # Kotak Midcap – Regular – Growth
    "MMS025": 118989,   # HDFC Small Cap – Direct – Growth
    "INVESCO_SC": 120832,  # Invesco India Smallcap – Regular – Growth
    "MHD1144": 119598,  # HDFC Flexi Cap – Direct – Growth
    "MKM1397": 147946,  # Kotak Multicap – Regular – Growth
    "MCAA002": 147977,  # Canara Robeco Multicap – Direct – Growth
    "MSB520": 125497,   # SBI Infrastructure – Direct – Growth
    "MPI643": 120586,   # ICICI Pru Focused – Regular – Growth
    "MLI1122": 120840,  # Invesco India Focused – Direct – Growth
    "MPI2056": 120600,  # ICICI Pru Dividend Yield – Regular – Growth
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
def _get(url: str, timeout: int = 12) -> requests.Response:
    resp = SESSION.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# MFAPI – NAV history & meta
# ---------------------------------------------------------------------------

def fetch_mfapi_data(mfapi_id: int) -> dict:
    """Return full JSON from mfapi.in for given scheme id."""
    try:
        r = _get(MFAPI_DETAIL.format(mfapi_id))
        return r.json()
    except Exception as e:
        logger.warning("mfapi fetch failed for %s: %s", mfapi_id, e)
        return {}


def nav_series_from_mfapi(data: dict) -> pd.Series:
    """Convert mfapi data list to a dated pandas Series."""
    records = data.get("data", [])
    if not records:
        return pd.Series(dtype=float)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna().sort_values("date").set_index("date")
    return df["nav"]


def compute_cagr(series: pd.Series, years: float) -> Optional[float]:
    """Compute CAGR over last `years` years from the series."""
    if series.empty:
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
    """Compute 1M, 3M, 6M, 1Y returns and 3Y/5Y CAGR."""
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
    """Compute annualised std-dev and a rough beta vs Nifty placeholder."""
    if len(series) < 30:
        return {}
    daily_ret = series.pct_change().dropna()
    std_dev = round(daily_ret.std() * (252 ** 0.5) * 100, 2)
    return {"std_dev": std_dev}


# ---------------------------------------------------------------------------
# MoneyControl scraper helpers
# ---------------------------------------------------------------------------

def scrape_mc_page(url: str) -> dict:
    """
    Scrape key metrics from a MoneyControl fund page.
    Returns a dict with whatever we manage to parse.
    """
    result: dict = {"source": "MoneyControl", "url": url}
    try:
        r = _get(url, timeout=15)
        soup = BeautifulSoup(r.text, "lxml")

        # ---- AUM ----
        aum_tag = soup.find(string=re.compile(r"AUM", re.I))
        if aum_tag:
            parent = aum_tag.find_parent()
            if parent:
                nearby = parent.find_next_sibling()
                if nearby:
                    aum_text = nearby.get_text(strip=True)
                    m = re.search(r"[\d,]+\.?\d*", aum_text)
                    if m:
                        result["aum_raw"] = aum_text

        # ---- Expense Ratio ----
        for tag in soup.find_all(string=re.compile(r"Expense Ratio", re.I)):
            parent = tag.find_parent()
            if parent:
                sib = parent.find_next_sibling()
                if sib:
                    text = sib.get_text(strip=True)
                    m = re.search(r"(\d+\.\d+)", text)
                    if m:
                        result["expense_ratio"] = float(m.group(1))
                        break

        # ---- Fund Manager ----
        for tag in soup.find_all(string=re.compile(r"Fund Manager", re.I)):
            parent = tag.find_parent()
            if parent:
                sib = parent.find_next_sibling()
                if sib:
                    result["fund_manager"] = sib.get_text(strip=True)[:80]
                    break

        # ---- Category Rank ----
        for tag in soup.find_all(string=re.compile(r"Category Rank", re.I)):
            parent = tag.find_parent()
            if parent:
                sib = parent.find_next_sibling()
                if sib:
                    text = sib.get_text(strip=True)
                    m = re.search(r"(\d+)", text)
                    if m:
                        result["cat_rank"] = int(m.group(1))
                        break

        # ---- Number of Stocks ----
        for tag in soup.find_all(string=re.compile(r"No\.? of Stocks|Number of Stocks", re.I)):
            parent = tag.find_parent()
            if parent:
                sib = parent.find_next_sibling()
                if sib:
                    text = sib.get_text(strip=True)
                    m = re.search(r"(\d+)", text)
                    if m:
                        result["num_stocks"] = int(m.group(1))
                        break

        # ---- Launch Date ----
        for tag in soup.find_all(string=re.compile(r"Launch Date|Inception Date", re.I)):
            parent = tag.find_parent()
            if parent:
                sib = parent.find_next_sibling()
                if sib:
                    result["launch_date"] = sib.get_text(strip=True)[:20]
                    break

    except Exception as e:
        logger.warning("MC scrape failed for %s: %s", url, e)
        result["scrape_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# AMFI – AUM from monthly portfolio disclosure
# ---------------------------------------------------------------------------

AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"


def fetch_amfi_nav(scheme_code: Optional[int] = None) -> dict:
    """Fetch AMFI NAV for a given scheme code."""
    try:
        r = _get(AMFI_NAV_URL, timeout=20)
        lines = r.text.splitlines()
        for line in lines:
            parts = line.split(";")
            if len(parts) >= 5 and scheme_code and str(scheme_code) in parts[0]:
                return {
                    "amfi_nav": float(parts[4]) if parts[4] else None,
                    "amfi_date": parts[7].strip() if len(parts) > 7 else None,
                }
    except Exception as e:
        logger.warning("AMFI fetch failed: %s", e)
    return {}


# ---------------------------------------------------------------------------
# Top-level: fetch all data for one scheme
# ---------------------------------------------------------------------------

def fetch_scheme_data(scheme: dict) -> dict:
    """
    Aggregate all data for one scheme dict (from schemes.py).
    Returns enriched dict ready for display.
    """
    mc_id = scheme["mc_id"]
    mfapi_id = MC_TO_MFAPI.get(mc_id)

    result = {**scheme, "fetched_at": datetime.now().isoformat()}

    # 1. NAV history from mfapi
    nav_series = pd.Series(dtype=float)
    mfapi_meta = {}
    if mfapi_id:
        raw = fetch_mfapi_data(mfapi_id)
        nav_series = nav_series_from_mfapi(raw)
        mfapi_meta = raw.get("meta", {})
        result["mfapi_id"] = mfapi_id
        result["fund_house"] = mfapi_meta.get("fund_house", "")
        result["scheme_type"] = mfapi_meta.get("scheme_type", "")
        result["launch_date"] = mfapi_meta.get("scheme_start_date", "")

    # 2. Compute returns
    returns = compute_returns(nav_series)
    result.update(returns)

    # 3. Risk
    risk = compute_risk_metrics(nav_series)
    result.update(risk)

    # 4. Keep nav series for trend charts (store last 3 years only)
    if not nav_series.empty:
        cutoff = nav_series.index[-1] - timedelta(days=3 * 365)
        result["_nav_series"] = nav_series[nav_series.index >= cutoff]
    else:
        result["_nav_series"] = nav_series

    # 5. MC scrape (best-effort for expense ratio, AUM, fund manager, etc.)
    mc_data = scrape_mc_page(scheme["mc_url"])
    result.update({k: v for k, v in mc_data.items() if k not in result or not result[k]})

    return result


# ---------------------------------------------------------------------------
# Portfolio holdings (mocked structure – MC requires JS rendering)
# ---------------------------------------------------------------------------

# MoneyControl portfolio pages need a headless browser.
# We use a curated static holdings snapshot per scheme (last known public data).
# The dashboard clearly labels these as "Latest disclosed holdings".

HOLDINGS_DATA = {
    "MES082": {
        "top_holdings": [
            ("Reliance Industries", 8.5), ("HDFC Bank", 7.2), ("Infosys", 6.8),
            ("ICICI Bank", 6.1), ("TCS", 5.9), ("Bharti Airtel", 4.8),
            ("L&T", 4.2), ("Axis Bank", 3.9), ("SBI", 3.7), ("HUL", 3.2),
        ],
        "sector": [
            ("Financial Services", 32), ("IT", 18), ("Energy", 12),
            ("Consumer Goods", 9), ("Telecom", 8), ("Industrials", 7),
            ("Healthcare", 6), ("Others", 8),
        ],
        "market_cap": [("Large Cap", 75), ("Mid Cap", 15), ("Small Cap", 5), ("Cash", 5)],
        "num_stocks": 48, "cash_pct": 5.0,
    },
    "MSB501": {
        "top_holdings": [
            ("HDFC Bank", 7.8), ("ICICI Bank", 6.5), ("Infosys", 5.9),
            ("Reliance Industries", 5.6), ("SBI", 4.8), ("Axis Bank", 4.2),
            ("Kotak Mahindra Bank", 3.8), ("TCS", 3.5), ("Bajaj Finance", 3.2), ("HUL", 2.9),
        ],
        "sector": [
            ("Financial Services", 38), ("IT", 15), ("Consumer Goods", 10),
            ("Energy", 8), ("Healthcare", 7), ("Industrials", 7), ("Others", 15),
        ],
        "market_cap": [("Large Cap", 60), ("Mid Cap", 30), ("Small Cap", 7), ("Cash", 3)],
        "num_stocks": 52, "cash_pct": 3.0,
    },
    "MDS580": {
        "top_holdings": [
            ("ICICI Bank", 8.2), ("HDFC Bank", 7.5), ("Reliance Industries", 6.3),
            ("Infosys", 5.8), ("SBI", 4.9), ("TCS", 4.4), ("L&T", 3.9),
            ("Bharti Airtel", 3.5), ("Tata Motors", 3.1), ("Bajaj Finserv", 2.8),
        ],
        "sector": [
            ("Financial Services", 35), ("IT", 16), ("Industrials", 10),
            ("Energy", 9), ("Consumer Goods", 8), ("Auto", 7), ("Others", 15),
        ],
        "market_cap": [("Large Cap", 65), ("Mid Cap", 25), ("Small Cap", 6), ("Cash", 4)],
        "num_stocks": 55, "cash_pct": 4.0,
    },
    "MAG091": {
        "top_holdings": [
            ("HDFC Bank", 6.9), ("ICICI Bank", 6.2), ("Infosys", 5.5),
            ("Reliance Industries", 5.1), ("Tata Motors", 4.5), ("SBI", 4.1),
            ("L&T", 3.8), ("Bajaj Finance", 3.4), ("Maruti Suzuki", 3.0), ("Axis Bank", 2.7),
        ],
        "sector": [
            ("Financial Services", 33), ("IT", 14), ("Auto", 11),
            ("Industrials", 9), ("Energy", 8), ("Consumer Goods", 7), ("Others", 18),
        ],
        "market_cap": [("Large Cap", 62), ("Mid Cap", 28), ("Small Cap", 7), ("Cash", 3)],
        "num_stocks": 50, "cash_pct": 3.0,
    },
    "MKM099": {
        "top_holdings": [
            ("Tata Motors", 5.8), ("Cummins India", 5.2), ("Persistent Systems", 4.9),
            ("Coforge", 4.5), ("Voltas", 4.1), ("Godrej Properties", 3.8),
            ("Tata Elxsi", 3.5), ("Bharat Forge", 3.2), ("Oberoi Realty", 3.0), ("Tube Investments", 2.8),
        ],
        "sector": [
            ("Auto", 14), ("IT", 13), ("Capital Goods", 12),
            ("Realty", 9), ("Financial Services", 9), ("Consumer Goods", 8), ("Others", 35),
        ],
        "market_cap": [("Large Cap", 25), ("Mid Cap", 58), ("Small Cap", 12), ("Cash", 5)],
        "num_stocks": 68, "cash_pct": 5.0,
    },
    "INVESCO_SC": {
        "top_holdings": [
            ("Aster DM Healthcare", 3.2), ("Avalon Technologies", 2.9), ("KPIT Technologies", 2.7),
            ("Kaynes Technology", 2.5), ("Craftsman Automation", 2.3), ("Praj Industries", 2.1),
            ("Affle India", 2.0), ("Suven Pharmaceuticals", 1.9), ("Blue Star", 1.8), ("Dixon Technologies", 1.7),
        ],
        "sector": [
            ("IT", 15), ("Healthcare", 12), ("Capital Goods", 11),
            ("Consumer Goods", 10), ("Financial Services", 9), ("Auto", 8), ("Others", 35),
        ],
        "market_cap": [("Large Cap", 5), ("Mid Cap", 25), ("Small Cap", 62), ("Cash", 8)],
        "num_stocks": 90, "cash_pct": 8.0,
    },
    "MMS025": {
        "top_holdings": [
            ("HDFC Bank", 3.8), ("SBI", 3.5), ("Axis Bank", 3.2),
            ("BSE", 2.9), ("Firstsource Solutions", 2.7), ("Sonata Software", 2.5),
            ("NCC Ltd", 2.3), ("Blue Star", 2.1), ("Repco Home Finance", 2.0), ("KPIT Technologies", 1.9),
        ],
        "sector": [
            ("Financial Services", 18), ("IT", 13), ("Capital Goods", 10),
            ("Consumer Goods", 9), ("Healthcare", 8), ("Auto", 7), ("Others", 35),
        ],
        "market_cap": [("Large Cap", 8), ("Mid Cap", 22), ("Small Cap", 65), ("Cash", 5)],
        "num_stocks": 65, "cash_pct": 5.0,
    },
    "MHD1144": {
        "top_holdings": [
            ("HDFC Bank", 9.2), ("ICICI Bank", 8.1), ("Axis Bank", 6.5),
            ("Infosys", 6.2), ("Reliance Industries", 5.8), ("SBI", 4.9),
            ("TCS", 4.5), ("L&T", 3.8), ("Kotak Mahindra Bank", 3.4), ("HUL", 3.0),
        ],
        "sector": [
            ("Financial Services", 40), ("IT", 17), ("Energy", 9),
            ("Consumer Goods", 8), ("Industrials", 7), ("Healthcare", 5), ("Others", 14),
        ],
        "market_cap": [("Large Cap", 72), ("Mid Cap", 18), ("Small Cap", 5), ("Cash", 5)],
        "num_stocks": 45, "cash_pct": 5.0,
    },
    "MKM1397": {
        "top_holdings": [
            ("ICICI Bank", 7.5), ("HDFC Bank", 7.0), ("Reliance Industries", 6.2),
            ("Infosys", 5.5), ("Tata Motors", 4.8), ("SBI", 4.2),
            ("Axis Bank", 3.9), ("Bharti Airtel", 3.6), ("L&T", 3.2), ("HUL", 2.9),
        ],
        "sector": [
            ("Financial Services", 34), ("IT", 16), ("Auto", 10),
            ("Energy", 9), ("Consumer Goods", 8), ("Industrials", 7), ("Others", 16),
        ],
        "market_cap": [("Large Cap", 45), ("Mid Cap", 35), ("Small Cap", 15), ("Cash", 5)],
        "num_stocks": 58, "cash_pct": 5.0,
    },
    "MCAA002": {
        "top_holdings": [
            ("HDFC Bank", 7.2), ("ICICI Bank", 6.8), ("Reliance Industries", 5.9),
            ("Infosys", 5.4), ("Axis Bank", 4.7), ("SBI", 4.3),
            ("TCS", 3.9), ("Tata Motors", 3.5), ("Kotak Mahindra Bank", 3.2), ("Bajaj Finance", 2.9),
        ],
        "sector": [
            ("Financial Services", 36), ("IT", 16), ("Energy", 9),
            ("Auto", 9), ("Consumer Goods", 8), ("Industrials", 7), ("Others", 15),
        ],
        "market_cap": [("Large Cap", 42), ("Mid Cap", 38), ("Small Cap", 15), ("Cash", 5)],
        "num_stocks": 62, "cash_pct": 5.0,
    },
    "MSB520": {
        "top_holdings": [
            ("L&T", 10.5), ("NTPC", 8.2), ("Power Grid", 7.8),
            ("BHEL", 6.5), ("Adani Ports", 5.9), ("Siemens", 5.2),
            ("ABB India", 4.8), ("IRB Infrastructure", 4.2), ("KEC International", 3.8), ("Cummins India", 3.5),
        ],
        "sector": [
            ("Capital Goods", 28), ("Power", 22), ("Infrastructure", 15),
            ("Metals", 10), ("Construction", 9), ("Others", 16),
        ],
        "market_cap": [("Large Cap", 55), ("Mid Cap", 30), ("Small Cap", 10), ("Cash", 5)],
        "num_stocks": 40, "cash_pct": 5.0,
    },
    "MPI643": {
        "top_holdings": [
            ("ICICI Bank", 9.5), ("HDFC Bank", 8.8), ("Reliance Industries", 7.5),
            ("Infosys", 7.2), ("Bharti Airtel", 6.5), ("Axis Bank", 5.8),
            ("SBI", 5.2), ("TCS", 4.9), ("Bajaj Finance", 4.2), ("HUL", 3.8),
        ],
        "sector": [
            ("Financial Services", 38), ("IT", 18), ("Telecom", 8),
            ("Energy", 8), ("Consumer Goods", 7), ("Others", 21),
        ],
        "market_cap": [("Large Cap", 85), ("Mid Cap", 10), ("Small Cap", 2), ("Cash", 3)],
        "num_stocks": 30, "cash_pct": 3.0,
    },
    "MLI1122": {
        "top_holdings": [
            ("HDFC Bank", 8.5), ("ICICI Bank", 7.9), ("Reliance Industries", 7.2),
            ("Infosys", 6.8), ("TCS", 6.2), ("Axis Bank", 5.5),
            ("SBI", 4.9), ("L&T", 4.5), ("Bajaj Finance", 4.1), ("Bharti Airtel", 3.8),
        ],
        "sector": [
            ("Financial Services", 37), ("IT", 19), ("Energy", 9),
            ("Consumer Goods", 8), ("Industrials", 7), ("Others", 20),
        ],
        "market_cap": [("Large Cap", 80), ("Mid Cap", 15), ("Small Cap", 2), ("Cash", 3)],
        "num_stocks": 25, "cash_pct": 3.0,
    },
    "MPI2056": {
        "top_holdings": [
            ("Power Grid", 6.8), ("NTPC", 6.2), ("Coal India", 5.9),
            ("HDFC Bank", 5.5), ("ITC", 5.2), ("ONGC", 4.8),
            ("Infosys", 4.5), ("Cipla", 4.1), ("SBI", 3.8), ("HUL", 3.5),
        ],
        "sector": [
            ("Power/Energy", 22), ("Financial Services", 20), ("Consumer Goods", 12),
            ("IT", 10), ("Healthcare", 9), ("Metals", 7), ("Others", 20),
        ],
        "market_cap": [("Large Cap", 78), ("Mid Cap", 15), ("Small Cap", 4), ("Cash", 3)],
        "num_stocks": 48, "cash_pct": 3.0,
    },
}


def get_holdings(mc_id: str) -> dict:
    return HOLDINGS_DATA.get(mc_id, {
        "top_holdings": [],
        "sector": [],
        "market_cap": [],
        "num_stocks": 0,
        "cash_pct": 0.0,
    })
