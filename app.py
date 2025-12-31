import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Market Dashboard", layout="wide")

DEFAULT_XLSX = "Market Dashboard 12.20.25.xlsx"

def _make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

@st.cache_data(show_spinner=False)
def load_excel(xlsx_bytes: bytes | None, xlsx_path: str):
    """
    Load the workbook either from an uploaded file (bytes) or from a path in the repo.
    Returns (dashboard_df, data_df, price_df).
    """
    if xlsx_bytes is not None:
        xl = pd.ExcelFile(xlsx_bytes)
        dash = pd.read_excel(xlsx_bytes, sheet_name="Dashboard")
        data = pd.read_excel(xlsx_bytes, sheet_name="Data")
        price = pd.read_excel(xlsx_bytes, sheet_name="Price")
    else:
        xl = pd.ExcelFile(xlsx_path)
        dash = pd.read_excel(xlsx_path, sheet_name="Dashboard")
        data = pd.read_excel(xlsx_path, sheet_name="Data")
        price = pd.read_excel(xlsx_path, sheet_name="Price")
    return dash, data, price

def parse_dashboard_tables(dash_df: pd.DataFrame):
    """
    Your Dashboard sheet contains 3 separate tables, each starting with a row where col0 == 'Ticker'.
    We split the sheet into those 3 tables and return a list of (section_name, table_df).
    """
    # Find header rows
    header_rows = dash_df.index[dash_df.iloc[:, 0].astype(str).str.strip().eq("Ticker")].tolist()

    tables = []
    for r in header_rows:
        # Section title = nearest previous non-null value in column 0 that isn't "Ticker"
        section = None
        for j in range(r - 1, -1, -1):
            v = dash_df.iloc[j, 0]
            if pd.notna(v) and str(v).strip() and str(v).strip().lower() != "ticker":
                section = str(v).strip()
                break
        if section is None:
            section = f"Table_{r}"

        # Build columns from the header row
        header_vals = dash_df.iloc[r, :].tolist()
        cols = []
        for i, h in enumerate(header_vals):
            if pd.isna(h) or str(h).strip().lower() in ("", "nan"):
                cols.append(f"col_{i}")
            else:
                cols.append(str(h).strip())
        cols = _make_unique(cols)

        # Table ends at the next fully blank row (all NaN)
        end = len(dash_df)
        for k in range(r + 1, len(dash_df)):
            if dash_df.iloc[k, :].isna().all():
                end = k
                break

        df = dash_df.iloc[r + 1 : end, :].copy()
        df.columns = cols
        df = df.dropna(axis=1, how="all")

        # Drop empty rows
        if "Ticker" in df.columns:
            df = df[df["Ticker"].notna()]

        tables.append((section, df.reset_index(drop=True)))

    return tables

def style_table(df: pd.DataFrame):
    """
    Nice formatting:
      - columns with '%' in the name (or starting with '%') are shown as percents
      - numeric columns get thousands separators where relevant
    """
    percent_cols = [c for c in df.columns if isinstance(c, str) and ("%" in c or c.strip().startswith("%"))]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    styler = df.style

    # Format percent columns (Excel stores as decimals like 0.006 -> 0.60%)
    for c in percent_cols:
        if c in numeric_cols:
            styler = styler.format({c: "{:.2%}"})

    # For other numeric columns, keep it readable
    other_numeric = [c for c in numeric_cols if c not in percent_cols]
    for c in other_numeric:
        styler = styler.format({c: "{:,.2f}"})

    return styler

# --- UI ---
st.title("Market Dashboard (Excel → Web App)")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload the Excel file", type=["xlsx"])
    use_uploaded = uploaded is not None

    if not use_uploaded:
        st.caption("Using local file from the app folder:")
        st.code(DEFAULT_XLSX)

    st.divider()
    view = st.radio("View", ["Dashboard", "Raw Sheets"], index=0)

# Load workbook
xlsx_bytes = uploaded.getvalue() if use_uploaded else None
xlsx_path = str(Path(DEFAULT_XLSX).resolve())

try:
    dash_df, data_df, price_df = load_excel(xlsx_bytes, xlsx_path)
except FileNotFoundError:
    st.error(
        f"Could not find '{DEFAULT_XLSX}' next to app.py.\n\n"
        "Either upload the file in the sidebar, or place the Excel file in the same folder as app.py."
    )
    st.stop()

if view == "Dashboard":
    tables = parse_dashboard_tables(dash_df)

    if not tables:
        st.warning("No dashboard tables found (no 'Ticker' header rows detected).")
        st.stop()

    tab_names = [name for name, _ in tables]
    tabs = st.tabs(tab_names)

    for (section_name, table_df), tab in zip(tables, tabs):
        with tab:
            st.subheader(section_name)

            # Optional quick filter by ticker/name if present
            search = st.text_input("Search (Ticker or Name)", value="", key=f"search_{section_name}")
            filtered = table_df.copy()
            if search.strip():
                s = search.strip().lower()
                cols_to_search = [c for c in filtered.columns if str(c).lower() in ("ticker", "name")]
                if cols_to_search:
                    mask = False
                    for c in cols_to_search:
                        mask = mask | filtered[c].astype(str).str.lower().str.contains(s, na=False)
                    filtered = filtered[mask]

            st.dataframe(style_table(filtered), use_container_width=True, height=650)

            # Download
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name=f"{section_name.replace(' ', '_')}.csv",
                mime="text/csv",
            )

else:
    sheet = st.selectbox("Choose sheet", ["Data", "Price", "Dashboard (raw)"], index=0)

    if sheet == "Data":
        df = data_df
    elif sheet == "Price":
        df = price_df
    else:
        df = dash_df

    st.subheader(sheet)
    st.dataframe(df, use_container_width=True, height=700)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"{sheet.replace(' ', '_')}.csv",
        mime="text/csv",
    )
