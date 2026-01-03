import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Overview Dashboard", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1700px;}
.small-muted {opacity: 0.75; font-size: 0.9rem;}
.section-title {font-weight: 800; font-size: 1.05rem; margin: 0.35rem 0 0.4rem 0;}
.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 10px;
}
.card h4 {margin: 0 0 8px 0; font-size: 0.95rem;}
.hr {border-top: 1px solid rgba(255,255,255,0.10); margin: 10px 0;}
[data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

BENCHMARK = "SPY"

def _asof_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ============================================================
# YOUR TICKER LIST (1 per line)
# ============================================================
TICKERS_RAW = r"""
SPY
QQQ
DIA
IWM
RSP
QQQE
EDOW
MDY
IWN
IWO
XLC
XLY
XLP
XLE
XLF
XLV
XLI
XLB
XLRE
XLK
XLU
SOXX
SMH
XSD
IGV
XSW
IGM
VGT
XT
CIBR
BOTZ
AIQ
XTL
VOX
FCOM
FDN
SOCL
XRT
IBUY
CARZ
IDRV
ITB
XHB
PEJ
VDC
FSTA
KXI
PBJ
VPU
FUTY
IDU
IYE
VDE
XOP
IEO
OIH
IXC
IBB
XBI
PBE
IDNA
IHI
XHE
XHS
XPH
FHLC
PINK
KBE
KRE
IAT
KIE
IAI
KCE
IYG
VFH
ITA
PPA
XAR
IYT
XTN
VIS
FIDU
XME
GDX
SIL
SLX
PICK
VAW
VNQ
IYR
REET
SRVR
HOMZ
SCHH
NETL
GLD
SLV
UNG
USO
DBA
CORN
DBB
PALL
URA
UGA
CPER
CATL
HOGS
SLX
SOYB
WEAT
DBC
IEMG
EUE
C6E
FEZ
E40
DAX
ISF
FXI
EEM
EWJ
EWU
EWZ
EWG
EWT
EWH
EWI
EWW
PIN
IDX
EWY
EWA
EWM
EWS
EWC
EWP
EZA
EWL
UUP
FXE
FXY
FXB
FXA
FXF
FXC
IBIT
ETHA
TLT
BND
SHY
IEF
SGOV
IEI
TLH
AGG
MUB
GOVT
IGSB
USHY
IGIB
""".strip()

def parse_ticker_list(raw: str) -> list[str]:
    out = []
    for ln in raw.splitlines():
        t = ln.strip().upper()
        if t:
            out.append(t)
    # keep order, remove duplicates
    seen = set()
    uniq = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq

# ===========================
# SUB-SECTOR / INDUSTRY GROUP MAP (EXCEL-LIKE)
# ===========================
SUBSECTOR_LEFT = {
    "Semiconductors": ["SOXX","SMH","XSD"],
    "Software / Cloud / Broad Tech": ["IGV","XSW","IGM","VGT","XT"],
    "Cyber Security": ["CIBR"],
    "AI / Robotics / Automation": ["BOTZ","AIQ"],
    "Telecom & Communication": ["XTL","VOX","FCOM"],
    "Internet / Media / Social": ["FDN","SOCL"],
    "Retail": ["XRT","IBUY"],
    "Autos / EV": ["IDRV"],  # CARZ not in your current list
    "Homebuilders / Construction": ["ITB","XHB"],
    "Leisure & Entertainment": ["PEJ"],
    "XLP Consumer Staples": ["VDC","FSTA","KXI","PBJ"],
    "XLU Utilities": ["VPU","FUTY","IDU"],
    "XLE Energy": ["IYE","VDE"],
    "Exploration & Production": ["XOP","IEO"],
    "Oil Services": ["OIH"],
    "Global Energy": ["IXC"],
}

SUBSECTOR_RIGHT = {
    "Biotechnology / Genomics": ["IBB","XBI","PBE","IDNA"],
    "Medical Equipment": ["IHI","XHE"],
    "Health Care Providers / Services": ["XHS"],
    "Pharmaceuticals": ["XPH"],
    "Broad / Alternative Health": ["FHLC","PINK"],
    "Banks": ["KBE","KRE","IAT"],
    "Insurance": ["KIE"],
    "Capital Markets / Brokerage": ["IAI","KCE"],
    "Diversified Financial Services": ["IYG"],
    "Broad Financials": ["VFH"],
    "Aerospace & Defense": ["ITA","PPA","XAR"],
    "Transportation": ["IYT","XTN"],
    "Broad Industrials": ["VIS","FIDU"],
    "XLB Materials": ["XME","GDX","SIL","SLX","PICK","VAW"],
    "XLRE Real Estate": ["VNQ","IYR","REET"],
    "Specialty REITs": ["SRVR","HOMZ","SCHH","NETL"],
}

# Anything not used above but still in tickers can show in a final “Other Assets” section if you want.

# -----------------------------
# Data pulls
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_prices(tickers, period="2y"):
    df = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("No data returned from price source.")

    if isinstance(df.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                closes[t] = df[(t, "Close")]
        close_df = pd.DataFrame(closes)
    else:
        close_df = pd.DataFrame({tickers[0]: df["Close"]})

    return close_df.dropna(how="all").ffill()

@st.cache_data(show_spinner=False, ttl=24*60*60)
def fetch_names(tickers: list[str]) -> dict[str, str]:
    names = {t: t for t in tickers}
    for t in tickers:
        try:
            inf = yf.Ticker(t).info
            n = inf.get("shortName") or inf.get("longName")
            if n:
                names[t] = str(n)
        except Exception:
            pass
    # small cleanups for your top rows
    names.setdefault("SPY","S&P 500")
    names.setdefault("QQQ","Nasdaq-100")
    names.setdefault("DIA","Dow")
    names.setdefault("IWM","Russell 2000")
    names.setdefault("RSP","S&P 500 EW")
    return names

# -----------------------------
# Metrics
# -----------------------------
def _ret(close: pd.Series, periods: int):
    return close.pct_change(periods=periods)

def _ratio_rs(close_t: pd.Series, close_b: pd.Series, periods: int):
    t = close_t / close_t.shift(periods)
    b = close_b / close_b.shift(periods)
    return (t / b) - 1

SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline_from_series(s: pd.Series, n=21) -> str:
    s = s.dropna().tail(n)
    if s.empty or s.nunique() == 1:
        return ""
    lo, hi = float(s.min()), float(s.max())
    if hi - lo <= 1e-12:
        return ""
    scaled = (s - lo) / (hi - lo)
    idx = (scaled * (len(SPARK_CHARS)-1)).round().astype(int).clip(0, len(SPARK_CHARS)-1)
    return "".join(SPARK_CHARS[i] for i in idx)

def build_table(p: pd.DataFrame, tickers: list[str], name_map: dict[str, str]) -> pd.DataFrame:
    horizons = {"% 1D": 1, "% 1W": 5, "% 1M": 21, "% 3M": 63, "% 6M": 126, "% 1Y": 252}
    rs_horizons = {"RS 3M": 63, "RS 6M": 126, "RS 1Y": 252}

    b = p[BENCHMARK]
    rows = []
    rs1m_raw_vals = []

    # first pass: compute raw rs1m for rank
    for t in tickers:
        if t not in p.columns:
            rs1m_raw_vals.append((t, np.nan))
            continue
        rr = _ratio_rs(p[t], b, 21)
        val = float(rr.dropna().iloc[-1]) if rr.dropna().shape[0] else np.nan
        rs1m_raw_vals.append((t, val))

    rs1m_series = pd.Series({t:v for t,v in rs1m_raw_vals}, dtype="float64")
    rs1m_pct = rs1m_series.rank(pct=True) * 100.0

    # build rows
    for t in tickers:
        if t not in p.columns:
            continue

        close = p[t]
        price = float(close.dropna().iloc[-1]) if close.dropna().shape[0] else np.nan

        # RS 1M raw series for sparkline (ratio over last 21d window)
        rs_ratio_series = (close / close.shift(21)) / (b / b.shift(21))
        spark = sparkline_from_series(rs_ratio_series, n=21)

        rec = {
            "Ticker": t,
            "Name": name_map.get(t, t),
            "Price": price,
            "Relative Strength 1mo": spark,
            "RS STS %": float(rs1m_pct.get(t, np.nan)),
        }

        for col, n in rs_horizons.items():
            rr = _ratio_rs(close, b, n)
            rec[col] = float(rr.dropna().iloc[-1]) if rr.dropna().shape[0] else np.nan

        for col, n in horizons.items():
            r = _ret(close, n)
            rec[col] = float(r.dropna().iloc[-1]) if r.dropna().shape[0] else np.nan

        rows.append(rec)

    df = pd.DataFrame(rows)

    # RS 3M/6M/1Y => convert to 1–99 rank like your sheet
    for col in ["RS 3M", "RS 6M", "RS 1Y"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            pct = s.rank(pct=True)
            df[col] = (pct * 99).round(0).clip(1, 99)

    df["RS STS %"] = pd.to_numeric(df["RS STS %"], errors="coerce").round(0)

    return df

# -----------------------------
# Styling
# -----------------------------
def _rs_color(v):
    try:
        v = float(v)
    except:
        return ""
    x = (v - 1) / 98.0
    r = int(255 * (1 - x) if x < 0.5 else 255 * (1 - (x - 0.5) * 2))
    g = int(255 * x if x < 0.5 else 255)
    b = 60
    return f"background-color: rgb({r},{g},{b}); color: #0B0B0B; font-weight: 800;"

def _pct_color(v):
    try:
        v = float(v)
    except:
        return ""
    if np.isnan(v):
        return ""
    if v > 0:
        return "color: #7CFC9A; font-weight: 700;"
    if v < 0:
        return "color: #FF6B6B; font-weight: 700;"
    return ""

def _sts_color(v):
    try:
        v = float(v)
    except:
        return ""
    # same heat style as RS ranks
    x = v / 100.0
    r = int(255 * (1 - x) if x < 0.5 else 255 * (1 - (x - 0.5) * 2))
    g = int(255 * x if x < 0.5 else 255)
    b = 60
    return f"background-color: rgb({r},{g},{b}); color: #0B0B0B; font-weight: 800;"

def style_df(df: pd.DataFrame):
    fmt = {
        "Price": "${:,.2f}",
        "RS STS %": "{:.0f}%",
        "% 1D": "{:.2%}", "% 1W": "{:.2%}", "% 1M": "{:.2%}",
        "% 3M": "{:.2%}", "% 6M": "{:.2%}", "% 1Y": "{:.2%}",
        "RS 3M": "{:.0f}", "RS 6M": "{:.0f}", "RS 1Y": "{:.0f}",
    }
    rs_cols = ["RS 3M", "RS 6M", "RS 1Y"]
    pct_cols = ["% 1D", "% 1W", "% 1M", "% 3M", "% 6M", "% 1Y"]

    sty = df.style.format(fmt)

    if "RS STS %" in df.columns:
        sty = sty.applymap(_sts_color, subset=["RS STS %"])
    for c in rs_cols:
        if c in df.columns:
            sty = sty.applymap(_rs_color, subset=[c])
    for c in pct_cols:
        if c in df.columns:
            sty = sty.applymap(_pct_color, subset=[c])

    return sty

# -----------------------------
# Right panel manual inputs
# -----------------------------
DEFAULT_RIGHT = {
    "Market Exposure": {"IBD Exposure": "40-60%", "Selected": "X"},
    "Market Type": {"Type": "Bull Quiet"},
    "Trend Condition (QQQ)": {"Above 5DMA": "Yes", "Above 10DMA": "Yes", "Above 20DMA": "Yes", "Above 50DMA": "Yes", "Above 200DMA": "No"},
    "52-Week High/Low": {"Daily": 231, "Weekly": 811, "Monthly": -828},
    "Market Indicators": {"VIX": 16.34, "PCC": 0.67, "Up/Down Vol Ratio": 2.36, "A/D Ratio": 2.20},
    "Macro": {"Fed Funds": 4.09, "M2 Money": 22.2, "10yr": 4.02},
    "Breadth & Participation": {"% Price Above 10DMA": 56, "% Price Above 20DMA": 49, "% Price Above 50DMA": 58, "% Price Above 200DMA": 68},
    "Composite Model": {"Monetary Policy": "Neutral", "Liquidity Flow": "Good", "Rates & Credit": "Good", "Tape Strength": "Good", "Sentiment": "Neutral", "Total Score": 8.5},
    "Hot Sectors / Industry Groups": {"Notes": "Type here..."},
    "Market Correlations": {"Correlated": "Dow, Nasdaq", "Uncorrelated": "Dollar, Bonds"},
}

def init_right_state():
    if "right_panel" not in st.session_state:
        st.session_state.right_panel = DEFAULT_RIGHT

def right_panel_ui():
    init_right_state()
    rp = st.session_state.right_panel

    st.markdown(
        '<div class="card"><h4>Top-Right Manual Inputs</h4>'
        '<div class="small-muted">You update only these. Everything else pulls & calculates automatically.</div></div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "Download settings.json",
        data=json.dumps(rp, indent=2),
        file_name="dashboard_settings.json",
        mime="application/json",
        use_container_width=True,
    )

    up = st.file_uploader("Import settings JSON (optional)", type=["json"])
    if up is not None:
        try:
            st.session_state.right_panel = json.loads(up.read().decode("utf-8"))
            rp = st.session_state.right_panel
            st.success("Imported settings.")
        except Exception as e:
            st.error(f"Import failed: {e}")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    for block in rp.keys():
        st.markdown(f'<div class="card"><h4>{block}</h4>', unsafe_allow_html=True)
        data = rp.get(block, {})
        kv = pd.DataFrame({"Metric": list(data.keys()), "Value": list(data.values())})
        edited = st.data_editor(kv, hide_index=True, use_container_width=True, num_rows="dynamic", key=f"ed_{block}")
        rp[block] = dict(zip(edited["Metric"], edited["Value"]))
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.right_panel = rp

# -----------------------------
# Helpers for grouped sub-sector layout
# -----------------------------
def grouped_block(groups: dict[str, list[str]], df_by_ticker: dict[str, dict]) -> pd.DataFrame:
    # Create a single dataframe with header rows (like Excel)
    out_rows = []
    for group_name, ticks in groups.items():
        # header row
        out_rows.append({
            "Ticker": group_name, "Name": "", "Price": np.nan,
            "Relative Strength 1mo": "", "RS STS %": np.nan, "RS 3M": np.nan, "RS 6M": np.nan, "RS 1Y": np.nan,
            "% 1D": np.nan, "% 1W": np.nan, "% 1M": np.nan, "% 3M": np.nan, "% 6M": np.nan, "% 1Y": np.nan,
        })
        for t in ticks:
            if t in df_by_ticker:
                out_rows.append(df_by_ticker[t])
    return pd.DataFrame(out_rows)

def style_grouped(df: pd.DataFrame):
    sty = style_df(df)
    # Make group header rows bold
    def _bold_headers(row):
        is_header = (row["Name"] == "") and isinstance(row["Ticker"], str) and (row["Ticker"] not in ALL_TICKERS_SET)
        return ["font-weight: 900;" if is_header else "" for _ in row.index]
    return sty.apply(_bold_headers, axis=1)

# -----------------------------
# UI
# -----------------------------
st.title("Market Overview Dashboard")
st.caption(f"As of: {_asof_ts()} • Auto data: Yahoo Finance • RS Benchmark: {BENCHMARK}")

with st.sidebar:
    st.subheader("Controls")
    refresh = st.button("Refresh Data")
    period = st.selectbox("History window", ["1y", "2y", "5y"], index=1)

if refresh:
    fetch_prices.clear()

ALL_TICKERS = parse_ticker_list(TICKERS_RAW)
ALL_TICKERS_SET = set(ALL_TICKERS)

# Excel structure
MAJOR = ALL_TICKERS[:10]
SECTORS = ALL_TICKERS[10:21]

# Pull prices
pull_list = list(dict.fromkeys(ALL_TICKERS + [BENCHMARK]))
st.success(f"Tickers loaded: {len(pull_list)} (unique) • Raw list: {len(ALL_TICKERS)}")

try:
    prices = fetch_prices(pull_list, period=period)
except Exception as e:
    st.error(f"Data pull failed: {e}")
    st.stop()

name_map = fetch_names(pull_list)

df_major = build_table(prices, MAJOR, name_map)
df_sectors = build_table(prices, SECTORS, name_map)

# Build sub-sector tables from group maps
# Create one “master” table to reuse per-row dictionaries
ALL_SUB_TICKS = []
for g in (SUBSECTOR_LEFT | SUBSECTOR_RIGHT).values():
    ALL_SUB_TICKS.extend(g)
ALL_SUB_TICKS = [t for t in ALL_SUB_TICKS if t in ALL_TICKERS_SET]

df_sub_master = build_table(prices, ALL_SUB_TICKS, name_map)

# Convert to dict-of-rows by ticker for fast grouping
df_by_ticker = {}
for _, r in df_sub_master.iterrows():
    df_by_ticker[r["Ticker"]] = r.to_dict()

df_left = grouped_block(SUBSECTOR_LEFT, df_by_ticker)
df_right = grouped_block(SUBSECTOR_RIGHT, df_by_ticker)

show_cols = [
    "Ticker","Name","Price","Relative Strength 1mo","RS STS %","RS 3M","RS 6M","RS 1Y",
    "% 1D","% 1W","% 1M","% 3M","% 6M","% 1Y"
]

left_col, right_col = st.columns([3.6, 1.4], gap="large")

with left_col:
    st.markdown('<div class="section-title">Major U.S. Indexes</div>', unsafe_allow_html=True)
    st.dataframe(style_df(df_major[show_cols]), use_container_width=True, height=310)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">U.S. Sectors</div>', unsafe_allow_html=True)
    st.dataframe(style_df(df_sectors[show_cols]), use_container_width=True, height=340)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">U.S. Sub-Sectors / Industry Groups</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.dataframe(style_grouped(df_left[show_cols]), use_container_width=True, height=820)
    with c2:
        st.dataframe(style_grouped(df_right[show_cols]), use_container_width=True, height=820)

with right_col:
    right_panel_ui()

