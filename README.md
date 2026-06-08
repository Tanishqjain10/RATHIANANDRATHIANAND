# Anand Rathi Equity MF Model Portfolio Dashboard

A production-ready, **live-data** Streamlit dashboard for the 14-scheme equity mutual fund model portfolio (Oct'25–Dec'26).

---

## 📁 Folder Structure

```
mf_dashboard/
├── app.py                  # Main Streamlit dashboard
├── schemes.py              # Master scheme registry (14 funds + weights)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── modules/
    ├── __init__.py
    ├── ingestion.py        # Live data fetching (mfapi.in + MoneyControl)
    └── cleaning.py         # Data normalisation + overlap matrix
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10 or 3.11
- pip

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run locally
```bash
cd mf_dashboard
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this folder to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, main file `app.py`.
4. Click **Deploy** — live in ~2 minutes.

---

## 🐳 Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
```

```bash
docker build -t mf-dashboard .
docker run -p 8501:8501 mf-dashboard
```

---

## 📡 Data Sources

| Data | Source | Refresh |
|------|--------|---------|
| NAV History & Returns | [mfapi.in](https://api.mfapi.in) | Real-time (Streamlit cache 5 min) |
| AUM, Expense Ratio, Fund Manager | MoneyControl (scrape) | Best-effort |
| Holdings & Sector Allocation | SEBI Monthly Disclosures | Monthly snapshot (hardcoded latest) |
| Fund Flows | AMFI India (illustrative) | Monthly |
| Benchmark | Nifty 50 TRI (hardcoded latest) | Quarterly update |

---

## ⚠️ Known Limitations & Workarounds

| Issue | Reason | Workaround |
|-------|--------|------------|
| AUM shows N/A | MoneyControl uses dynamic JS rendering | Install Playwright + use `playwright fetch` branch |
| Beta not shown | Requires NSE daily index data | Subscribe NSE Data APIs or use nsepy library |
| Fund flows are illustrative | AMFI publishes only category-level monthly data | Use AMFI Excel download + parse |
| Holdings are static | MC portfolio pages need headless browser | Schedule monthly Selenium scraper |
| Invesco Smallcap URL (Google redirect) | Google share link doesn't resolve to MC directly | Manually replaced with MC direct URL in code |

---

## 🔧 Configuration

Edit `app.py` top section:
```python
REFRESH_INTERVAL = 300   # seconds (5 minutes)
NIFTY_RETURNS = { ... }  # Update benchmark returns quarterly
```

Edit `schemes.py` to add/remove schemes or update weights.

---

## 📊 Dashboard Sections

| Tab | Content |
|-----|---------|
| Summary KPIs | Total schemes, Avg 5Y CAGR, Avg Expense, Std Dev |
| Comparison Table | All 14 schemes with full metrics, sortable |
| Portfolio Analysis | Holdings, Sector, Market Cap per scheme |
| Fund Flows | Inflow/Outflow trend (AMFI data) |
| Stock Movements | Latest additions/exits per scheme |
| Overlap Matrix | Jaccard overlap heatmap for all 14×14 pairs |
| Benchmark | Alpha vs Nifty 50 TRI |
| Risk Analysis | Risk-Return scatter, Std Dev ratings |
| Charts | CAGR bar, Treemap, NAV trend, AUM |
