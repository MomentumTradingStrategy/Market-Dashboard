import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Overview Dashboard", layout="wide")

# =========================
# Styling
# =========================
st.markdown("""
<style>
.block-container {max-width:1700px; padding-top:1rem;}
.section-title {font-weight:800; font-size:1.1rem; margin:0.4rem 0;}
.hr {border-top:1px solid rgba(255,255,255,0.12); margin:10px 0;}
.card {border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:10px; margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

BENCHMARK = "SPY"

def asof():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# =========================
# Tickers (exact list)
# =========================
TICKERS = [
"SPY","QQQ","DIA","IWM","RSP","QQQE","EDOW","MDY","IWN","IWO",
"XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU",
"SOXX","SMH","XSD","IGV","XSW","IGM","VGT","XT","CIBR","BOTZ","AIQ",
"XTL","VOX","FCOM","FDN","SOCL","XRT","IBUY","CARZ","IDRV","ITB","XHB","PEJ",
"VDC","FSTA","KXI","PBJ","VPU","FUTY","IDU","IYE","VDE","XOP","IEO","OIH","IXC",
"IBB","XBI","PBE","IDNA","IHI","XHE","XHS","XPH","FHLC","PINK",
"KBE","KRE","IAT","KIE","IAI","KCE","IYG","VFH",
"ITA","PPA","XAR","IYT","XTN","VIS","FIDU",
"XME","GDX","SIL","SLX","PICK","VAW",
"VNQ","IYR","REET","SRVR","HOMZ","SCHH","NETL",
"GLD","SLV","UNG","USO","DBA","CORN","DBB","PALL","URA","UGA","CPER","CATL","HOGS",
"SOYB","WEAT","DBC",
"IEMG","EUE","C6E","FEZ","E40","DAX","ISF",
"FXI","EEM","EWJ","EWU","EWZ","EWG","EWT","EWH","EWI","EWW","PIN","IDX",
"EWY","EWA","EWM","EWS","EWC","EWP","EZA","EWL",
"UUP","FXE","FXY","FXB","FXA","FXF","FXC",
"IBIT","ETHA",
"TLT","BND","SHY","IEF","SGOV","IEI","TLH","AGG","MUB","GOVT","IGSB","USHY","IGIB"
]

# =========================
# Grouping
# =========================
SUB_LEFT = {
"Semiconductors":["SOXX","SMH","XSD"],
"Software / Cloud / Broad Tech":["IGV","XSW","IGM","VGT","XT"],
"Cyber Security":["CIBR"],
"AI / Robotics / Automation":["BOTZ","AIQ"],
"Telecom & Communication":["XTL","VOX","FCOM"],
"Internet / Media / Social":["FDN","SOCL"],
"Retail":["XRT","IBUY"],
"Autos / EV":["CARZ","IDRV"],
"Homebuilders / Construction":["ITB","XHB"],
"Leisure & Entertainment":["PEJ"],
"Consumer Staples":["VDC","FSTA","KXI","PBJ"],
"Utilities":["VPU","FUTY","IDU"],
"Energy":["IYE","VDE"],
"E&P":["XOP","IEO"],
"Oil Services":["OIH"],
"Global Energy":["IXC"]
}

SUB_RIGHT = {
"Biotech / Genomics":["IBB","XBI","PBE","IDNA"],
"Medical Equipment":["IHI","XHE"],
"Health Care Services":["XHS"],
"Pharmaceuticals":["XPH"],
"Health Care Broad":["FHLC","PINK"],
"Banks":["KBE","KRE","IAT"],
"Insurance":["KIE"],
"Capital Markets":["IAI","KCE"],
"Diversified Financials":["IYG"],
"Broad Financials":["VFH"],
"Aerospace & Defense":["ITA","PPA","XAR"],
"Transportation":["IYT","XTN"],
"Industrials":["VIS","FIDU"],
"Materials":["XME","GDX","SIL","SLX","PICK","VAW"],
"Real Estate":["VNQ","IYR","REET"],
"Specialty REITs":["SRVR","HOMZ","SCHH","NETL"]
}

# =========================
# Data
# =========================
@st.cache_data(ttl=3600)
def prices(tickers, period):
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].ffill()
    return pd.DataFrame({tickers[0]: df["Close"]})

def build(df, tickers):
    b = df[BENCHMARK]
    rows = []
    for t in tickers:
        if t not in df: continue
        c = df[t]
        r = {
            "Ticker":t,
            "Price":c.iloc[-1],
            "%1D":c.pct_change(1).iloc[-1],
            "%1W":c.pct_change(5).iloc[-1],
            "%1M":c.pct_change(21).iloc[-1],
            "%3M":c.pct_change(63).iloc[-1],
            "%6M":c.pct_change(126).iloc[-1],
            "%1Y":c.pct_change(252).iloc[-1],
            "RS 1W":((c/c.shift(5))/(b/b.shift(5))).iloc[-1],
            "RS 1M":((c/c.shift(21))/(b/b.shift(21))).iloc[-1],
            "RS 3M":((c/c.shift(63))/(b/b.shift(63))).iloc[-1],
            "RS 6M":((c/c.shift(126))/(b/b.shift(126))).iloc[-1],
            "RS 1Y":((c/c.shift(252))/(b/b.shift(252))).iloc[-1],
        }
        rows.append(r)
    d = pd.DataFrame(rows)
    for col in ["RS 1W","RS 1M","RS 3M","RS 6M","RS 1Y"]:
        d[col]=(d[col].rank(pct=True)*99).round().clip(1,99)
    return d

# =========================
# UI
# =========================
st.title("Market Overview Dashboard")
st.caption(f"As of {asof()} • Auto data: Yahoo Finance • RS Benchmark: SPY")

with st.sidebar:
    period = st.selectbox("Price history (used only for calc)",["1y","2y","5y"],1)
    if st.button("Refresh Data"):
        st.cache_data.clear()

dfp = prices(list(set(TICKERS+[BENCHMARK])), period)

major = build(dfp, TICKERS[:10])
sectors = build(dfp, TICKERS[10:21])

def grouped(groups):
    rows=[]
    for g,ts in groups.items():
        rows.append({"Ticker":g})
        rows += build(dfp, ts).to_dict("records")
    return pd.DataFrame(rows)

left,right = st.columns([3.6,1.4])

with left:
    st.markdown("### Major U.S. Indexes")
    st.dataframe(major, use_container_width=True, height=300)
    st.markdown("---")
    st.markdown("### U.S. Sectors")
    st.dataframe(sectors, use_container_width=True, height=340)
    st.markdown("---")
    st.markdown("### U.S. Sub-Sectors / Industry Groups")
    c1,c2 = st.columns(2)
    with c1: st.dataframe(grouped(SUB_LEFT), use_container_width=True, height=900)
    with c2: st.dataframe(grouped(SUB_RIGHT), use_container_width=True, height=900)

with right:
    st.markdown("### Manual Inputs")
    st.info("Right-panel logic unchanged")

