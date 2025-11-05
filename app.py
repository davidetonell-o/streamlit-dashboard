# app.py — Amazon Sales Dashboard (robust + user-select Amount/Category)
# Run locally:  streamlit run app.py

from pathlib import Path
import re
import calendar

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -----------------------------
# UI Config
# -----------------------------
st.set_page_config(page_title="Amazon Sales Dashboard", page_icon="📦", layout="wide")
sns.set(style="whitegrid")

# -----------------------------
# Helpers
# -----------------------------
def normalize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()

def clean_amount_series(s: pd.Series) -> pd.Series:
    """Convert textual amounts to float (handles currency symbols, thousands, decimals)."""
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)                   # keep digits, comma, dot, minus
    both = s.str.contains(",") & s.str.contains(r"\.")
    s.loc[both] = s.loc[both].str.replace(",", "", regex=False)       # "1,234.56" -> "1234.56"
    only_comma = s.str.contains(",") & ~s.str.contains(r"\.")
    s.loc[only_comma] = s.loc[only_comma].str.replace(",", ".", regex=False)  # "123,45" -> "123.45"
    return pd.to_numeric(s, errors="coerce")

def detect_amount_candidates(df: pd.DataFrame, max_cols: int = 12):
    """
    Scan ALL columns, clean as amount and compute sum/valid count.
    Returns a sorted list of (column, valid_count, total_sum) best candidates.
    """
    candidates = []
    for c in df.columns:
        try:
            s = clean_amount_series(df[c])
            valid = int(s.notna().sum())
            total = float(s.fillna(0).sum())
            if valid > 0 and total != 0.0:
                candidates.append((c, valid, total))
        except Exception:
            continue
    candidates.sort(key=lambda x: abs(x[2]), reverse=True)
    return candidates[:max_cols]

def top_n_with_other(counts: pd.Series, top_n: int = 8) -> pd.Series:
    counts = counts.dropna()
    if len(counts) <= top_n:
        return counts
    top = counts.nlargest(top_n)
    other_sum = counts.drop(top.index).sum()
    if other_sum > 0:
        top.loc["Other"] = other_sum
    return top

# -----------------------------
# Data loading & base prep
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    # Look for the CSV in common locations
    candidates = [
        Path("data") / "archive" / "Amazon Sale Report.csv",
        Path("data") / "Amazon Sale Report.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if not csv_path:
        st.error("CSV not found. Place the file under data/archive/ or data/.")
        st.stop()

    df = pd.read_csv(csv_path, low_memory=False)

    # Normalize column names for detection
    cols_norm = {c: re.sub(r"\s+", "", str(c).strip().lower()) for c in df.columns}

    # ---- Date (create Date/Year/MonthNum) ----
    date_col = None
    for c, n in cols_norm.items():
        if any(k in n for k in ["date", "orderdate", "invoicedate", "shipdate", "shippingdate"]):
            date_col = c
            break

    if date_col:
        dt = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
        if dt.notna().sum() == 0:
            dt = pd.to_datetime(df[date_col], errors="coerce")
        df["Date"] = dt
        df["Year"] = df["Date"].dt.year
        df["MonthNum"] = df["Date"].dt.month  # 1..12
    else:
        df["Date"] = pd.NaT
        df["Year"] = pd.NA
        df["MonthNum"] = pd.NA

    # ---- Status (optional) ----
    status_col = None
    for c, n in cols_norm.items():
        if "status" in n:
            status_col = c
            break
    if status_col:
        s = normalize_text(df[status_col])
        # Collapse verbose variants
        s = s.str.replace(r"^Pending\s*-\s*.*$", "Pending", regex=True)
        s = s.str.replace(r"^Shipped\s*-\s*.*$", "Shipped", regex=True)
        s = s.str.replace(r"^Returned\s*-\s*.*$", "Returned", regex=True)
        s = s.str.replace(r"^Cancelled\s*-\s*.*$", "Cancelled", regex=True)
        df["Status"] = s

    return df

df = load_data()

# -----------------------------
# Header
# -----------------------------
st.title("📦 Amazon Sales Dashboard")
st.caption("KPIs • Monthly Sales (MonthNum) • Status Distribution • Top Categories")

# -----------------------------
# Sidebar: filters + column selection (Amount/Category)
# -----------------------------
with st.sidebar:
    st.header("Filters")

    # Year
    if df["Year"].notna().any():
        years = sorted([int(y) for y in df["Year"].dropna().unique()])
        year_sel = st.multiselect("Year", years, default=years)
    else:
        year_sel = []

    # Status
    if "Status" in df.columns:
        status_vals = sorted(df["Status"].dropna().unique().tolist())
        status_sel = st.multiselect("Status", status_vals, default=status_vals[:8] if status_vals else [])
    else:
        status_sel = []

    st.divider()
    st.subheader("Column selection")

    # AMOUNT: autodetect + user choice
    amount_candidates = detect_amount_candidates(df)
    if amount_candidates:
        default_amount_col = amount_candidates[0][0]
        amount_choice = st.selectbox(
            "Amount column (autodetected):",
            options=[c for c, _, _ in amount_candidates],
            index=0,
            help="Pick the monetary amount column"
        )
    else:
        amount_choice = st.selectbox(
            "Amount column (no candidates found):",
            options=df.columns.tolist()
        )

    # CATEGORY: choose from textual columns (optional)
    text_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    category_choice = st.selectbox(
        "Category column (optional):",
        options=["<none>"] + text_cols,
        index=0,
        help="Select the product category column (e.g., Product Category)"
    )

    debug = st.checkbox("Show data diagnostics", value=False)

# Apply user choices (create clean Amount/Category)
df = df.copy()
df["Amount"] = clean_amount_series(df[amount_choice]) if amount_choice else pd.NA
if category_choice != "<none>":
    df["Category"] = normalize_text(df[category_choice])

# Apply filters
df_f = df.copy()
if year_sel:
    df_f = df_f[df_f["Year"].isin(year_sel)]
if "Status" in df_f.columns and status_sel:
    df_f = df_f[df_f["Status"].isin(status_sel)]

# -----------------------------
# KPIs
# -----------------------------
total_sales = float(df_f["Amount"].sum()) if "Amount" in df_f.columns else 0.0
n_orders = int(len(df_f))
avg_ticket = (total_sales / n_orders) if n_orders else 0.0

k1, k2, k3 = st.columns(3)
k1.metric("Total Sales", f"{total_sales:,.2f}")
k2.metric("Orders", f"{n_orders:,}")
k3.metric("Average Ticket", f"{avg_ticket:,.2f}")

if debug:
    st.info(
        f"Filtered rows: {len(df_f)} • Valid dates: {int(df_f['Date'].notna().sum())} • "
        f"Amount sum: {total_sales:,.2f}"
    )

st.divider()

# -----------------------------
# Chart 1: Monthly sales (safe aggregation on MonthNum 1..12)
# -----------------------------
st.subheader("Monthly Sales")
ok_month = ("MonthNum" in df_f.columns) and ("Amount" in df_f.columns) and df_f["MonthNum"].notna().any()
if ok_month:
    gf = df_f.dropna(subset=["MonthNum", "Amount"]).copy()
    monthly = gf.groupby("MonthNum")["Amount"].sum().reindex(range(1, 13))
    if monthly.notna().any() and monthly.fillna(0).sum() > 0:
        fig1, ax1 = plt.subplots(figsize=(9, 3.6))
        monthly.fillna(0).plot(kind="bar", ax=ax1, color="#4f46e5")
        ax1.set_xticklabels([calendar.month_name[i] for i in range(1, 13)], rotation=45, ha="right")
        ax1.set_xlabel("Month"); ax1.set_ylabel("Sales Sum")
        ax1.set_title("Total Sales by Month", pad=8)
        st.pyplot(fig1)
    else:
        st.info("No valid data for the monthly chart (check Amount/filters).")
else:
    st.info("Monthly analysis columns not available or all NaN.")

st.divider()

# -----------------------------
# Chart 2: Status distribution (Top + Other)
# -----------------------------
if "Status" in df_f.columns:
    st.subheader("Order Distribution by Status (Top + Other)")
    counts = df_f["Status"].value_counts()
    counts_plot = top_n_with_other(counts, top_n=8)
    if not counts_plot.empty:
        data = (
            counts_plot.rename_axis("Status")
                       .reset_index(name="Count")
                       .sort_values("Count", ascending=True)
        )
        total_counts = int(counts.sum()) if counts.sum() else 1
        palette = ["#94a3b8" if v == "Other" else "#4f46e5" for v in data["Status"]]

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        sns.barplot(
            data=data, y="Status", x="Count",
            hue="Status", palette=palette, dodge=False, legend=False, ax=ax2
        )
        for i, row in data.iterrows():
            ax2.text(row["Count"], i, f"  {row['Count']:,} ({row['Count']/total_counts*100:.1f}%)",
                     va="center", ha="left", fontsize=9)
        ax2.set_xlabel("Count"); ax2.set_ylabel("Status")
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("No data for Status (filters may be too restrictive).")

st.divider()

# -----------------------------
# Chart 3: Top categories by sales
# -----------------------------
if "Category" in df_f.columns and "Amount" in df_f.columns:
    st.subheader("Top Categories by Sales")
    df_cat = df_f.dropna(subset=["Category", "Amount"]).copy()
    top_cat = df_cat.groupby("Category")["Amount"].sum().sort_values(ascending=False).head(10)
    if not top_cat.empty and top_cat.fillna(0).sum() > 0:
        fig3, ax3 = plt.subplots(figsize=(9, 4))
        top_cat.sort_values().plot(kind="barh", ax=ax3, color="#4f46e5")
        ax3.set_xlabel("Sales Sum"); ax3.set_ylabel("Category")
        plt.tight_layout()
        st.pyplot(fig3)
    else:
        st.info("No data for categories (check Amount/filters).")
else:
    st.info("Category column not present/selected: choose the correct column in the sidebar.")

st.caption("Built with Streamlit • Robust handling for Date/Amount/Category • © Davide Tonello")
