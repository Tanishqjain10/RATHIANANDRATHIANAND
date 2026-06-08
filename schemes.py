"""
schemes.py – Master registry with Value Research URLs
"""

SCHEMES = [
    {"sr": 1, "name": "Quant Large Cap Fund-Reg(G)", "short_name": "Quant Large Cap", "category": "Large Cap", "weight": 10, "mc_id": "MES082", "vr_url": "https://www.valueresearchonline.com/funds/42367/quant-large-cap-fund-direct-plan/"},
    {"sr": 2, "name": "SBI Large & Midcap Fund-Reg(G)", "short_name": "SBI Large & Midcap", "category": "Large & Mid Cap", "weight": 6, "mc_id": "MSB501", "vr_url": "https://www.valueresearchonline.com/funds/16240/sbi-large-midcap-fund-direct-plan/"},
    {"sr": 3, "name": "DSP Large & Mid Cap Fund-Reg(G)", "short_name": "DSP Large & Mid Cap", "category": "Large & Mid Cap", "weight": 9, "mc_id": "MDS580", "vr_url": "https://www.valueresearchonline.com/funds/16425/dsp-large-mid-cap-fund-direct-plan/"},
    {"sr": 4, "name": "Bandhan Large & Mid Cap Fund-Reg(G)", "short_name": "Bandhan L&M Cap", "category": "Large & Mid Cap", "weight": 7, "mc_id": "MAG091", "vr_url": "https://www.valueresearchonline.com/funds/16474/bandhan-large-mid-cap-fund-direct-plan/"},
    {"sr": 5, "name": "Kotak Midcap Fund-Reg(G)", "short_name": "Kotak Midcap", "category": "Mid Cap", "weight": 6, "mc_id": "MKM099", "vr_url": "https://www.valueresearchonline.com/funds/17134/kotak-midcap-fund-direct-plan/"},
    {"sr": 6, "name": "Invesco India Smallcap Fund-Reg(G)", "short_name": "Invesco Smallcap", "category": "Small Cap", "weight": 7, "mc_id": "INVESCO_SC", "vr_url": "https://www.valueresearchonline.com/funds/37843/invesco-india-smallcap-fund-direct-plan/"},
    {"sr": 7, "name": "HDFC Small Cap Fund-Reg(G)", "short_name": "HDFC Small Cap", "category": "Small Cap", "weight": 5, "mc_id": "MMS025", "vr_url": "https://www.valueresearchonline.com/funds/16617/hdfc-small-cap-fund-direct-plan/"},
    {"sr": 8, "name": "HDFC Flexi Cap Fund(G)", "short_name": "HDFC Flexi Cap", "category": "Flexi Cap", "weight": 9, "mc_id": "MHD1144", "vr_url": "https://www.valueresearchonline.com/funds/16026/hdfc-flexi-cap-fund-direct-plan/"},
    {"sr": 9, "name": "Kotak Multicap Fund-Reg(G)", "short_name": "Kotak Multicap", "category": "Multi Cap", "weight": 7, "mc_id": "MKM1397", "vr_url": "https://www.valueresearchonline.com/funds/41707/kotak-multicap-fund-direct-plan/"},
    {"sr": 10, "name": "Canara Rob Multi Cap Fund-Reg(G)", "short_name": "Canara Rob Multicap", "category": "Multi Cap", "weight": 7, "mc_id": "MCAA002", "vr_url": "https://www.valueresearchonline.com/funds/43569/canara-robeco-multi-cap-fund-regular-plan/"},
    {"sr": 11, "name": "SBI Infrastructure Fund-Reg(G)", "short_name": "SBI Infrastructure", "category": "Multi Cap*", "weight": 6, "mc_id": "MSB520", "vr_url": "https://www.valueresearchonline.com/funds/17161/sbi-infrastructure-fund-direct-plan/"},
    {"sr": 12, "name": "ICICI Pru Focused Equity Fund(G)", "short_name": "ICICI Pru Focused", "category": "Focused", "weight": 7, "mc_id": "MPI643", "vr_url": "https://www.valueresearchonline.com/funds/17412/icici-prudential-focused-equity-fund-direct-plan/"},
    {"sr": 13, "name": "Invesco India Focused Fund-Reg(G)", "short_name": "Invesco Focused", "category": "Focused", "weight": 7, "mc_id": "MLI1122", "vr_url": "https://www.valueresearchonline.com/funds/41096/invesco-india-focused-fund-direct-plan/"},
    {"sr": 14, "name": "ICICI Pru Dividend Yield Equity Fund(G)", "short_name": "ICICI Pru Div Yield", "category": "Dividend Yield", "weight": 7, "mc_id": "MPI2056", "vr_url": "https://www.valueresearchonline.com/funds/26271/icici-prudential-dividend-yield-equity-fund-direct-plan/"},
]

BENCHMARK = {"name": "Nifty 50 TRI", "symbol": "^NSEI"}

CATEGORY_COLORS = {
    "Large Cap": "#1f77b4", "Large & Mid Cap": "#ff7f0e", "Mid Cap": "#2ca02c",
    "Small Cap": "#d62728", "Flexi Cap": "#9467bd", "Multi Cap": "#8c564b",
    "Multi Cap*": "#e377c2", "Focused": "#7f7f7f", "Dividend Yield": "#bcbd22",
}
