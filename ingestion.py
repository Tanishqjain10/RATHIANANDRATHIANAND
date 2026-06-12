"""
ingestion.py - PRODUCTION VERSION
Value Research Primary (requests + BeautifulSoup)
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List

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

MFAPI_DETAIL = "https://api.mfapi.in/mf/{}"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{}"

SCHEME_URL_FIELDS = (
    "vr_url",
    "vr_performance",
    "vr_risk",
    "vr_portfolio",
    "vr_other",
)

MC_TO_MFAPI = {
    "MES082": 119597, "MSB501": 125494, "MDS580": 119270, "MAG091": 145552,
    "MKM099": 120503, "MMS025": 118989, "INVESCO_SC": 120832, "MHD1144": 119598,
    "MKM1397": 147946, "MCAA002": 147977, "MSB520": 125497, "MPI643": 120586,
    "MLI1122": 120840, "MPI2056": 120600,
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _get(url: str, timeout: int = 20):
    return SESSION.get(url, timeout=timeout)

def scrape_valueresearch(vr_url: str) -> Dict:
    result = {"source": "Value Research", "url": vr_url}
    try:
        r = _get(vr_url)
        result["status_code"] = r.status_code
        result["fetch_ok"] = 200 <= r.status_code < 400
        text = r.text

        # NAV
        nav_match = re.search(r'The latest declared NAV of .*? is ₹?([\d,]+\.\d{2})', text, re.I)
        if nav_match:
            nav = float(nav_match.group(1).replace(",", ""))
            if nav > 0:
                result["nav"] = nav

        # AUM
        aum_match = re.search(r'The fund has an overall AUM \(Assets Under Management\) of ₹?([\d,]+\.?\d*)\s*Cr', text, re.I)
        if aum_match:
            result["aum_raw"] = f"₹{aum_match.group(1)} Cr"

        # Expense Ratio
        exp_match = re.search(r'The fund has an expense ratio of (\d+\.\d+)%', text, re.I)
        if exp_match:
            exp = float(exp_match.group(1))
            if 0 < exp <= 5:
                result["expense_ratio"] = exp

        # Fund Manager
        manager_match = re.search(r'it is currently managed by ([A-Za-z\s&.,-]+)', text, re.I)
        if manager_match:
            result["fund_manager"] = manager_match.group(1).strip()[:120]

        # Launch Date
        launch_match = re.search(r'Launched on ([A-Za-z]+\s+\d{1,2},?\s+\d{4})', text, re.I)
        if launch_match:
            result["launch_date"] = launch_match.group(1)

        logger.info(f"VR fetch success: {vr_url}")
    except Exception as e:
        result["fetch_ok"] = False
        result["error"] = str(e)
        logger.warning(f"VR failed {vr_url}: {e}")

    return result

def provided_urls_for_scheme(scheme: dict) -> List[str]:
    urls = []
    for field in SCHEME_URL_FIELDS:
        url = scheme.get(field)
        if url and url not in urls:
            urls.append(url)
    return urls

def scrape_all_valueresearch_urls(scheme: dict) -> Dict:
    urls = provided_urls_for_scheme(scheme)
    merged = {
        "source": "Value Research",
        "provided_urls": urls,
        "fetched_urls": [],
        "failed_urls": [],
    }

    for url in urls:
        data = scrape_valueresearch(url)
        if data.get("fetch_ok"):
            merged["fetched_urls"].append(url)
        else:
            merged["failed_urls"].append(url)

        has_live_fields = any(
            key in data for key in ("nav", "aum_raw", "expense_ratio", "fund_manager", "launch_date")
        )
        if has_live_fields:
            for key, value in data.items():
                if value is not None and key not in ("source", "url", "fetch_ok", "status_code", "error"):
                    merged[key] = value

    merged["provided_url_count"] = len(urls)
    merged["fetched_url_count"] = len(merged["fetched_urls"])
    merged["failed_url_count"] = len(merged["failed_urls"])
    return merged

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
    daily_ret = series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    if daily_ret.empty:
        return {}
    std_dev = round(daily_ret.std() * (252 ** 0.5) * 100, 2)
    return {"std_dev": std_dev}

def parse_top_holdings_from_valueresearch(vr_portfolio_url: str) -> dict:
    result = {
        "top_holdings": [],
        "num_stocks": 0,
        "holdings_source": "Value Research",
        "holdings_fetched": False,
    }
    if not vr_portfolio_url:
        return result

    try:
        r = _get(vr_portfolio_url)
        soup = BeautifulSoup(r.text, "html.parser")
        heading = soup.find(string=re.compile(r"What are the top holdings", re.I))
        table = heading.find_parent().find_next("table") if heading else None
        if not table:
            return result

        holdings = []
        for tr in table.select("tbody tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            name = re.sub(r"\s+", " ", tds[0].get_text(" ", strip=True))
            pct_text = tds[1].get_text(" ", strip=True)
            pct_match = re.search(r"[-+]?\d+\.?\d*", pct_text)
            if not name or not pct_match:
                continue
            holdings.append((name, float(pct_match.group(0))))

        result["top_holdings"] = holdings
        result["num_stocks"] = len(holdings)
        result["holdings_fetched"] = bool(holdings)
    except Exception as e:
        result["holdings_error"] = str(e)
        logger.warning(f"Holdings failed {vr_portfolio_url}: {e}")
    return result

def yahoo_series(symbol: str, years: int = 5) -> pd.Series:
    try:
        url = YAHOO_CHART.format(symbol)
        r = _get(f"{url}?range={years}y&interval=1d")
        payload = r.json()
        result = payload.get("chart", {}).get("result", [{}])[0]
        timestamps = result.get("timestamp", [])
        closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        if not timestamps or not closes:
            return pd.Series(dtype=float)
        idx = pd.to_datetime(timestamps, unit="s").tz_localize("UTC").tz_convert(None).normalize()
        series = pd.Series(closes, index=idx, dtype=float).dropna().sort_index()
        return series[~series.index.duplicated(keep="last")]
    except Exception as e:
        logger.warning(f"Benchmark failed {symbol}: {e}")
        return pd.Series(dtype=float)

def fetch_benchmark_data(symbol: str = "%5ENSEI", label: str = "Nifty 50") -> dict:
    series = yahoo_series(symbol)
    latest = series.iloc[-1] if not series.empty else None
    latest_date = series.index[-1].strftime("%d %b %Y") if not series.empty else None
    data = {
        "label": label,
        "symbol": symbol,
        "latest": round(latest, 2) if latest is not None else None,
        "latest_date": latest_date,
        "fetched": not series.empty,
        "series": series,
    }
    data.update(compute_returns(series))
    data.update(compute_risk_metrics(series))
    return data

def fetch_scheme_data(scheme: dict) -> dict:
    fetched_at = datetime.now()
    result = {
        **scheme,
        "fetched_at": fetched_at.isoformat(),
        "last_updated": fetched_at.strftime("%d %b %Y, %H:%M:%S"),
    }

    if provided_urls_for_scheme(scheme):
        vr_data = scrape_all_valueresearch_urls(scheme)
        result.update({k: v for k, v in vr_data.items() if v is not None})

    mfapi_id = MC_TO_MFAPI.get(scheme.get("mc_id"))
    nav_series = pd.Series(dtype=float)
    if mfapi_id:
        raw = fetch_mfapi_data(mfapi_id)
        nav_series = nav_series_from_mfapi(raw)
        result["mfapi_id"] = mfapi_id
        result["mfapi_fetched"] = not nav_series.empty

    result.update(compute_returns(nav_series))
    result.update(compute_risk_metrics(nav_series))

    if not nav_series.empty:
        cutoff = nav_series.index[-1] - timedelta(days=3 * 365)
        result["_nav_series"] = nav_series[nav_series.index >= cutoff]

    return result

def get_holdings(scheme_or_id) -> dict:
    if isinstance(scheme_or_id, dict):
        return parse_top_holdings_from_valueresearch(scheme_or_id.get("vr_portfolio", ""))
    return {"top_holdings": [], "sector": [], "market_cap": [], "num_stocks": 0, "cash_pct": 0.0}
