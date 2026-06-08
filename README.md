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
| Scheme facts, AUM, TER, manager, launch details | All Value Research URLs configured in `schemes.py` | Automatic, cached 5 min |
| NAV history, NAV date, returns, CAGR, volatility | [mfapi.in](https://api.mfapi.in) | Automatic, cached 5 min |
| Last updated status and URL fetch audit | Runtime ingestion metadata | Shown on every dashboard refresh |

---

## ⚠️ Known Limitations & Workarounds

| Issue | Reason | Workaround |
|-------|--------|------------|
| AUM shows N/A | Source page markup may change or block scraping | Check the relevant Value Research URL in `schemes.py` |
| Beta not shown | Requires NSE daily index data | Subscribe NSE Data APIs or use nsepy library |
| Fund flows are illustrative | AMFI publishes only category-level monthly data | Use AMFI Excel download + parse |
| Holdings are empty | No holdings endpoint is configured yet | Add a live holdings disclosure URL per scheme |

---

## 🔧 Configuration

Edit `app.py` top section:
```python
REFRESH_INTERVAL = 300   # seconds (5 minutes)
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
