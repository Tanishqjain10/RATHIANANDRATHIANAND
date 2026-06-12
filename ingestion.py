"""
ingestion.py - PRODUCTION VERSION
Value Research Primary (requests + BeautifulSoup)
"""

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from holdings_snapshot import HOLDINGS_SNAPSHOT

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

MFAPI_DETAIL = "https://api.mfapi.in/mf/{}"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{}"
GROWW_FUND_PAGE = "https://groww.in/mutual-funds/{}"

SCHEME_URL_FIELDS = (
    "vr_url",
    "vr_performance",
    "vr_risk",
    "vr_portfolio",
    "vr_other",
)

MC_TO_MFAPI = {
    "MES082": 150440,       # quant Large Cap Fund - Growth Option - Direct Plan
    "MSB501": 119721,       # SBI Large & Midcap Fund - Direct Plan - Growth
    "MDS580": 119218,       # DSP Large & Mid Cap Fund - Direct Plan - Growth
    "MAG091": 118419,       # Bandhan Large & Mid Cap Fund - Direct Plan - Growth
    "MKM099": 119775,       # Kotak Midcap Fund - Direct Plan - Growth
    "INVESCO_SC": 145137,   # Invesco India Smallcap Fund - Direct Plan - Growth
    "MMS025": 130503,       # HDFC Small Cap Fund - Growth Option - Direct Plan
    "MHD1144": 118955,      # HDFC Flexi Cap Fund - Growth Option - Direct Plan
    "MKM1397": 149185,      # Kotak Multicap Fund - Direct Plan - Growth
    "MCAA002": 151824,      # Canara Robeco Multi Cap Fund - Direct Plan - Growth
    "MSB520": 119700,       # SBI Infrastructure Fund - Direct Plan - Growth
    "MPI643": 120722,       # ICICI Prudential Focused Equity Fund - Direct Plan - Growth
    "MLI1122": 148481,      # Invesco India Focused Fund - Direct Plan - Growth
    "MPI2056": 129312,      # ICICI Prudential Dividend Yield Equity Fund Direct Plan Growth
}

GROWW_SLUGS = {
    "MES082": "quant-large-cap-fund-direct-growth",
    "MDS580": "dsp-large-mid-cap-fund-direct-plan-growth",
    "MAG091": "bandhan-large-mid-cap-fund-direct-growth",
    "MKM099": "kotak-midcap-fund-direct-growth",
    "INVESCO_SC": "invesco-india-smallcap-fund-direct-growth",
    "MMS025": "hdfc-small-cap-fund-direct-growth",
    "MHD1144": "hdfc-equity-fund-direct-growth",
    "MKM1397": "kotak-multicap-fund-direct-growth",
    "MCAA002": "canara-robeco-multi-cap-fund-direct-growth",
    "MSB520": "sbi-infrastructure-fund-direct-growth",
    "MPI643": "icici-prudential-focused-bluechip-equity-fund-direct-growth",
    "MLI1122": "invesco-india-focused-fund-direct-growth",
    "MPI2056": "icici-prudential-dividend-yield-equity-fund-direct-growth",
}

METADATA_SNAPSHOT = {
    "MES082": {"aum_cr": 2320.0, "expense_ratio": 0.42},
    "MSB501": {"aum_cr": 32850.0, "expense_ratio": 0.64},
    "MDS580": {"aum_cr": 29850.0, "expense_ratio": 0.53},
    "MAG091": {"aum_cr": 3750.0, "expense_ratio": 0.44},
    "MKM099": {"aum_cr": 66000.0, "expense_ratio": 0.32},
    "INVESCO_SC": {"aum_cr": 6400.0, "expense_ratio": 0.39},
    "MMS025": {"aum_cr": 36000.0, "expense_ratio": 0.62},
    "MHD1144": {"aum_cr": 80000.0, "expense_ratio": 0.58},
    "MKM1397": {"aum_cr": 18500.0, "expense_ratio": 0.40},
    "MCAA002": {"aum_cr": 4300.0, "expense_ratio": 0.53},
    "MSB520": {"aum_cr": 5600.0, "expense_ratio": 0.87},
    "MPI643": {"aum_cr": 9300.0, "expense_ratio": 0.52},
    "MLI1122": {"aum_cr": 1150.0, "expense_ratio": 0.50},
    "MPI2056": {"aum_cr": 5300.0, "expense_ratio": 0.52},
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
        if "Just a moment" in text and "challenges.cloudflare.com" in text:
            result["fetch_ok"] = False
            result["error"] = "Value Research Cloudflare challenge"
            return result

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

def _walk_json(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_json(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_json(value)

def _number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).replace(",", "").strip()
    match = re.search(r"[-+]?\d+\.?\d*", text)
    return float(match.group(0)) if match else None

def fetch_groww_metadata(slug: str) -> dict:
    result = {"metadata_source": "Groww", "metadata_fetched": False}
    if not slug:
        return result
    try:
        r = _get(GROWW_FUND_PAGE.format(slug))
        result["groww_url"] = GROWW_FUND_PAGE.format(slug)
        if r.status_code != 200 or "404 - Page Not Found" in r.text[:40000]:
            result["metadata_error"] = f"Groww returned {r.status_code}"
            return result
        match = re.search(
            r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            r.text,
            re.S,
        )
        if not match:
            result["metadata_error"] = "Groww page JSON not found"
            return result

        payload = json.loads(match.group(1))
        found_aum = None
        found_expense = None
        for node in _walk_json(payload):
            lowered = {str(k).lower(): v for k, v in node.items()}
            if found_aum is None:
                for key in ("aum", "scheme_aum", "fund_aum"):
                    if key in lowered:
                        found_aum = _number(lowered[key])
                        break
            if found_expense is None:
                for key in ("expense_ratio", "expenseRatio".lower(), "ter"):
                    if key in lowered:
                        found_expense = _number(lowered[key])
                        break
            if found_aum is not None and found_expense is not None:
                break

        if found_aum is not None and found_aum > 0:
            result["aum_cr"] = found_aum
            result["aum_raw"] = f"{found_aum:.2f} Cr"
        if found_expense is not None and 0 < found_expense <= 5:
            result["groww_expense_ratio"] = found_expense
        result["metadata_fetched"] = "aum_cr" in result
    except Exception as e:
        result["metadata_error"] = str(e)
        logger.warning(f"Groww metadata failed {slug}: {e}")
    return result

def metadata_snapshot(mc_id: str) -> dict:
    snapshot = METADATA_SNAPSHOT.get(mc_id, {})
    if not snapshot:
        return {"metadata_source": "No metadata fallback", "metadata_fetched": False}
    return {
        "aum_cr": snapshot.get("aum_cr"),
        "aum_raw": f"{snapshot.get('aum_cr'):.2f} Cr" if snapshot.get("aum_cr") else None,
        "expense_ratio": snapshot.get("expense_ratio"),
        "metadata_source": "Verified metadata snapshot",
        "metadata_fetched": True,
        "metadata_snapshot": True,
    }

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

    if not result.get("aum_raw") or result.get("expense_ratio") is None:
        had_value_research_expense = result.get("expense_ratio") is not None
        groww_data = fetch_groww_metadata(GROWW_SLUGS.get(scheme.get("mc_id"), ""))
        groww_used_for_aum = False
        for key, value in groww_data.items():
            if value is None:
                continue
            if key in ("aum_cr", "aum_raw", "expense_ratio"):
                if not result.get("aum_raw") and key in ("aum_cr", "aum_raw"):
                    result[key] = value
                    groww_used_for_aum = True
                elif result.get("expense_ratio") is None and key == "expense_ratio":
                    result[key] = value
            elif key not in result:
                result[key] = value
        if groww_used_for_aum and had_value_research_expense:
            result["metadata_source"] = "Groww AUM + Value Research TER"

    if not result.get("aum_raw") or result.get("expense_ratio") is None:
        fallback = metadata_snapshot(scheme.get("mc_id", ""))
        fallback_used = False
        existing_metadata_source = result.get("metadata_source")
        for key, value in fallback.items():
            if value is None:
                continue
            if key in ("aum_cr", "aum_raw", "expense_ratio"):
                if not result.get("aum_raw") and key in ("aum_cr", "aum_raw"):
                    result[key] = value
                    fallback_used = True
                elif result.get("expense_ratio") is None and key == "expense_ratio":
                    result[key] = value
                    fallback_used = True
            elif key not in result or (fallback_used and key.startswith("metadata_")):
                result[key] = value
        if fallback_used:
            if existing_metadata_source == "Groww" and result.get("aum_raw"):
                result["metadata_source"] = "Groww AUM + verified TER snapshot"
            else:
                result["metadata_source"] = fallback.get("metadata_source", "Verified metadata snapshot")
            result["metadata_fetched"] = fallback.get("metadata_fetched", True)
            result["metadata_snapshot"] = True

    mfapi_id = MC_TO_MFAPI.get(scheme.get("mc_id"))
    nav_series = pd.Series(dtype=float)
    if mfapi_id:
        raw = fetch_mfapi_data(mfapi_id)
        result["mfapi_scheme_name"] = raw.get("meta", {}).get("scheme_name")
        nav_series = nav_series_from_mfapi(raw)
        result["mfapi_id"] = mfapi_id
        result["mfapi_fetched"] = not nav_series.empty

    result.update(compute_returns(nav_series))
    result.update(compute_risk_metrics(nav_series))

    if not nav_series.empty:
        cutoff = nav_series.index[-1] - timedelta(days=3 * 365)
        result["_nav_series"] = nav_series[nav_series.index >= cutoff]

    return result

def fallback_holdings(mc_id: str) -> dict:
    snapshot = HOLDINGS_SNAPSHOT.get(mc_id, {})
    holdings = snapshot.get("top_holdings", [])
    return {
        "top_holdings": holdings,
        "sector": [],
        "market_cap": [],
        "num_stocks": len(holdings),
        "cash_pct": 0.0,
        "holdings_source": "Value Research snapshot",
        "holdings_source_url": snapshot.get("source_url", ""),
        "holdings_fetched": bool(holdings),
        "holdings_snapshot": True,
    }

def get_holdings(scheme_or_id) -> dict:
    if isinstance(scheme_or_id, dict):
        live = parse_top_holdings_from_valueresearch(scheme_or_id.get("vr_portfolio", ""))
        if live.get("top_holdings"):
            live["holdings_source_url"] = scheme_or_id.get("vr_portfolio", "")
            live["holdings_snapshot"] = False
            return live
        return fallback_holdings(scheme_or_id.get("mc_id", ""))
    return fallback_holdings(str(scheme_or_id))
