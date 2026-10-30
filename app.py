import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from crime_utils import (
    normalize_name, find_column, read_uploaded_file, clean_data,
    detect_schema, apply_schema, aggregate_counts, kpi_summary,
    filter_data, build_time_series, train_models
)


warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------
# Page setup and theme (matched to the rainfall dashboard style)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Crime Pattern Analysis and Prediction",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* New crime-dashboard theme: charcoal, forest green and amber */
.stApp {
  background:
    radial-gradient(circle at 8% 12%, rgba(35, 180, 130, .18), transparent 30%),
    radial-gradient(circle at 92% 8%, rgba(235, 157, 65, .15), transparent 28%),
    linear-gradient(135deg, #101b1b 0%, #172b2a 48%, #26352d 78%, #171f20 100%);
  background-attachment: fixed;
  color: #f4f5ed;
}
[data-testid="stHeader"] {
  background: rgba(16, 27, 27, .55);
}
[data-testid="stSidebar"] {
  background: linear-gradient(165deg, #111b1b 0%, #1b2927 52%, #26352d 100%);
  border-right: 1px solid rgba(111, 190, 151, .28);
  box-shadow: 8px 0 28px rgba(0, 0, 0, .18);
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.4rem;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {
  color: #d8e5dc !important;
}
h1, h2, h3, h4, p, label, span {
  color: #f4f5ed;
}
h1 {
  background: linear-gradient(90deg, #9ce6c0, #f2c078, #f0a77d);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
div[data-testid="stMetric"] {
  background: linear-gradient(125deg, rgba(30, 111, 83, .92), rgba(91, 105, 59, .92));
  border: 1px solid rgba(174, 215, 174, .30);
  padding: 17px;
  border-radius: 15px;
  box-shadow: 0 8px 25px rgba(0, 0, 0, .22);
}
div[data-testid="stMetricLabel"] { color: #dceadd !important; }
div[data-testid="stMetricValue"] { color: #ffffff !important; }

/* Sidebar navigation and download buttons */
[data-testid="stSidebar"] .stButton > button {
  background: linear-gradient(105deg, #267b61, #52734b);
  color: #ffffff;
  border: 1px solid rgba(179, 220, 177, .28);
  border-radius: 9px;
  min-height: 46px;
  font-weight: 650;
  box-shadow: 0 4px 12px rgba(0, 0, 0, .16);
  transition: all .2s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: linear-gradient(105deg, #329879, #718b4e);
  border: 1px solid #c5e6ad;
  transform: translateY(-1px);
}
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(105deg, #267b61, #b47a3e);
  color: white;
  border: 1px solid rgba(210, 220, 190, .25);
  border-radius: 9px;
  padding: .55rem .8rem;
  font-weight: 650;
  transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border: 1px solid #c5e6ad;
  color: white;
  transform: translateY(-1px);
}
[data-testid="stFileUploader"] section {
  background: rgba(20, 37, 33, .72);
  border: 1px dashed rgba(150, 204, 164, .42);
  border-radius: 12px;
}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
  border: 1px solid rgba(133, 185, 151, .25);
  border-radius: 12px;
  overflow: hidden;
}
div[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: rgba(133, 185, 151, .25); }
.small-muted { color: #c0d1c4; font-size: 13px; }
.panel {
  background: rgba(19, 39, 34, .78);
  border: 1px solid rgba(133, 185, 151, .25);
  border-radius: 15px;
  padding: 14px 16px;
  margin: 4px 0 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# General helpers
# ------------------------------------------------------------
CHART_BG = "#172b28"
PANEL_BG = "#203a32"
TEXT_COLOR = "#edf4e9"
PALETTE = ["#68d6a5", "#e8b45d", "#8fc5a2", "#e88962", "#b6cf78",
           "#75b9b0", "#d9a078", "#a6c78d", "#f0d17c", "#83c9bd"]




@st.cache_data(show_spinner=False)
def read_local_file(path, modified_time):
    """Cache a local CSV/Excel file using its modified time as cache key."""
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)




def polish(ax, title, xlabel=None, ylabel=None, legend=False):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COLOR, labelpad=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_COLOR, labelpad=7)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#31517b")
    ax.grid(True, color="#7d9ac7", alpha=.15)
    ax.set_axisbelow(True)
    if legend:
        leg = ax.legend(facecolor=PANEL_BG, edgecolor="#31517b", fontsize=8)
        for text in leg.get_texts():
            text.set_color(TEXT_COLOR)

def finish_chart(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)




def create_trend_chart(data, title="Recorded crime-related records over time"):
    dated = data.dropna(subset=["_DATE"])
    if dated.empty:
        st.info("A usable date or year column was not found, so a time trend cannot be drawn.")
        return
    daily = dated.groupby(dated["_DATE"].dt.to_period("M"))["_COUNT"].sum()
    trend = daily.copy()
    trend.index = trend.index.astype(str)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(CHART_BG)
    ax.plot(trend.index, trend.values, color=PALETTE[0], linewidth=2.3,
            marker="o", markersize=3, label="Recorded count")
    ax.fill_between(np.arange(len(trend)), trend.values, color=PALETTE[0], alpha=.16)
    ax.set_xticks(np.arange(len(trend)))
    ax.set_xticklabels(trend.index, rotation=45, ha="right")
    polish(ax, title, "Month", "Recorded count")
    finish_chart(fig)

def create_bar_chart(series, title, xlabel, ylabel="Recorded count", horizontal=False):
    if series is None or len(series) == 0:
        st.info("There is not enough data for this chart.")
        return
    values = series.sort_values(ascending=horizontal)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(CHART_BG)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(values))]
    if horizontal:
        ax.barh(values.index.astype(str), values.values, color=colors, edgecolor="#d8e8ff", linewidth=.35)
    else:
        ax.bar(values.index.astype(str), values.values, color=colors, edgecolor="#d8e8ff", linewidth=.35)
        ax.tick_params(axis="x", rotation=35)
    polish(ax, title, xlabel, ylabel)
    finish_chart(fig)

# ------------------------------------------------------------
# Load data: local Crime_data.csv preferred; Excel also supported
# ------------------------------------------------------------
st.sidebar.markdown("## 🚔 Crime Analysis")
st.sidebar.caption("Patterns • Regions • Prediction")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload your crime dataset",
    type=["csv", "xlsx", "xls"],
    help="CSV is recommended. Excel files are also accepted."
)

raw_df = None
source_label = ""
try:
    if uploaded_file is not None:
        raw_df = read_uploaded_file(uploaded_file)
        source_label = uploaded_file.name
    else:
        local_candidates = ["Crime_data.csv", "Crime_data.xlsx", "Crime_data.xls"]
        local_path = next((p for p in local_candidates if os.path.exists(p)), None)
        if local_path:
            raw_df = read_local_file(local_path, os.path.getmtime(local_path))
            source_label = local_path
except Exception as exc:
    st.sidebar.error(f"Could not read the dataset: {exc}")

if raw_df is None:
    st.title("🚔 Crime Pattern Analysis and Prediction")
    st.caption("Explore Crime Trends | Analyze Crime Hotspots | Compare Machine Learning Models")
    st.warning(
        "No dataset was loaded. Place **Crime_data.csv** beside `app.py`, "
        "or upload a CSV/Excel file using the sidebar."
    )
    st.markdown("""
    <div class="panel">
    <b>Recommended columns</b><br>
    Date or Year • State/Region/District • Crime Category • Total Crimes (optional count field)
    <br><br>
    Each row can represent one incident/record, or a grouped record with a numeric count column.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if raw_df.empty:
    st.error("The selected dataset has no rows. Please choose a file containing records.")
    st.stop()

df = clean_data(raw_df)
detected = detect_schema(df)

# Automatically use the columns detected from the uploaded dataset.
# No manual column-mapping dropdowns are shown in the sidebar.
date_col = detected.get("date")
year_col = detected.get("year")
region_col = detected.get("region")
category_col = detected.get("category")
count_col = detected.get("count")
latitude_col = detected.get("latitude")
longitude_col = detected.get("longitude")

# Date can supply the year automatically; year-only datasets do not need a date.
if not date_col and not year_col:
    st.info("No date/year column was detected. Trend and time-based prediction features will be limited.")

schema = {
    "date": date_col,
    "year": year_col,
    "region": region_col,
    "category": category_col,
    "count": count_col,
    "latitude": latitude_col,
    "longitude": longitude_col
}
df = apply_schema(df, schema)

st.title("🚔 Crime Pattern Analysis and Prediction")
st.caption("Explore Crime Trends | Analyze Crime Hotspots | Compare Machine Learning Models")
st.markdown(f"<div class='small-muted'>Data source: <b>{source_label}</b> · {len(df):,} rows loaded</div>",
            unsafe_allow_html=True)

if not region_col:
    st.info("No region column was detected. Region-level comparisons will use a single 'All records' group.")
if not category_col:
    st.info("No crime category column was detected. Category charts will show a single 'Uncategorized' group.")
if not count_col:
    st.caption("No numeric count column was detected: each row is treated as one recorded record. This is suitable for incident-level datasets.")

# ------------------------------------------------------------
# Four-button navigation: no radio or checkbox navigation
# ------------------------------------------------------------
pages = {
    "📊  Dashboard": "Dashboard",
    "🔎  Crime Explorer": "Crime Explorer",
    "🧠  Prediction Lab": "Prediction Lab",
    "🚨  Hotspot & Anomalies": "Hotspot & Anomalies",
}
if "crime_page" not in st.session_state:
    st.session_state["crime_page"] = "Dashboard"

for label, page_name in pages.items():
    is_selected = st.session_state["crime_page"] == page_name
    button_label = ("▶ " if is_selected else "") + label
    if st.sidebar.button(button_label, key=f"nav_{page_name}", use_container_width=True):
        st.session_state["crime_page"] = page_name
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Academic analysis dashboard")
st.sidebar.caption("Recorded counts are not the same as actual crime prevalence.")

page = st.session_state["crime_page"]

# ------------------------------------------------------------
# Section 1: Dashboard
# ------------------------------------------------------------
if page == "Dashboard":
    st.subheader("📊 Dashboard Overview")
    total, regions_count, categories_count, highest, highest_label = kpi_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total recorded count", f"{total:,.0f}")
    c2.metric("Regions", f"{regions_count:,}")
    c3.metric("Crime categories", f"{categories_count:,}")
    c4.metric(highest_label, str(highest))

    left, right = st.columns([1.15, .95], gap="medium")
    with left:
        st.markdown("### 📈 Crime Trend")
        if df["_DATE"].notna().any():
            create_trend_chart(df, "Monthly recorded count")
        elif df["_YEAR"].notna().any():
            yearly = df.dropna(subset=["_YEAR"]).groupby("_YEAR")["_COUNT"].sum().sort_index()
            fig, ax = plt.subplots(figsize=(8, 3.5))
            fig.patch.set_facecolor(CHART_BG)
            ax.plot(yearly.index, yearly.values, color=PALETTE[0], marker="o", linewidth=2.3)
            ax.fill_between(yearly.index, yearly.values, color=PALETTE[0], alpha=.18)
            polish(ax, "Recorded count by year", "Year", "Recorded count")
            finish_chart(fig)
        else:
            st.info("A date/year column is needed for a time trend.")
    with right:
        st.markdown("### 🗺️ Top 10 Regions")
        region_totals = df.groupby("_REGION")["_COUNT"].sum().sort_values(ascending=False).head(10)
        create_bar_chart(region_totals.sort_values(), "Top 10 regions by recorded count",
                         "Region", "Recorded count", horizontal=True)

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("### 🧩 Crime Category Distribution")
        cat_totals = df.groupby("_CATEGORY")["_COUNT"].sum().sort_values(ascending=False).head(10)
        create_bar_chart(cat_totals, "Top crime categories", "Category", "Recorded count")
    with right:
        st.markdown("### 🗓️ Monthly Pattern")
        if df["_MONTH"].notna().any():
            month_order = list(range(1, 13))
            month_totals = df.dropna(subset=["_MONTH"]).groupby("_MONTH")["_COUNT"].sum().reindex(month_order, fill_value=0)
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            fig, ax = plt.subplots(figsize=(8, 3.5))
            fig.patch.set_facecolor(CHART_BG)
            ax.bar(month_names, month_totals.values,
                   color=[PALETTE[i % len(PALETTE)] for i in range(12)])
            polish(ax, "Recorded count by month", "Month", "Recorded count")
            finish_chart(fig)
        else:
            st.info("Monthly patterns require a valid date column.")

    st.markdown("""
    <div class="panel"><b>How to read this dashboard</b><br>
    These charts summarize the records in the selected file. They do not automatically represent
    all crimes that occurred in a region; coverage depends on the source dataset and its definitions.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# Section 2: Crime Explorer
# ------------------------------------------------------------
elif page == "Crime Explorer":
    st.subheader("🔎 Crime Explorer")
    st.write("Filter the dataset and explore how the displayed counts change.")
    f1, f2, f3 = st.columns([1, 1, 1])
    region_options = ["All regions"] + sorted(df["_REGION"].dropna().astype(str).unique().tolist())
    category_options = ["All categories"] + sorted(df["_CATEGORY"].dropna().astype(str).unique().tolist())
    with f1:
        selected_regions = st.multiselect("Region / state / district", region_options, default=["All regions"])
    with f2:
        selected_categories = st.multiselect("Crime category", category_options, default=["All categories"])
    with f3:
        valid_years = df["_YEAR"].dropna()
        year_range = None
        if len(valid_years):
            min_year, max_year = int(valid_years.min()), int(valid_years.max())
            if min_year < max_year:
                year_range = st.slider("Year range", min_year, max_year, (min_year, max_year))
            else:
                st.caption(f"Available year: {min_year}")
                year_range = (min_year, max_year)

    filtered = filter_data(
        df,
        regions=selected_regions,
        categories=selected_categories,
        years=year_range
    )
    st.metric("Filtered recorded count", f"{filtered['_COUNT'].sum():,.0f}",
              delta=f"{len(filtered):,} rows")
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("### 📈 Filtered trend")
        if filtered["_DATE"].notna().any():
            create_trend_chart(filtered, "Monthly count for current filters")
        elif filtered["_YEAR"].notna().any():
            yearly = filtered.dropna(subset=["_YEAR"]).groupby("_YEAR")["_COUNT"].sum().sort_index()
            fig, ax = plt.subplots(figsize=(7, 3.4))
            fig.patch.set_facecolor(CHART_BG)
            ax.plot(yearly.index, yearly.values, marker="o", color=PALETTE[0], linewidth=2.2)
            polish(ax, "Recorded count by year", "Year", "Recorded count")
            finish_chart(fig)
        else:
            st.info("Date/year data is required for the trend chart.")
    with right:
        st.markdown("### 📊 Category / region comparison")
        group_choice = st.selectbox("Compare by", ["Crime category", "Region"], key="explorer_group")
        group_col = "_CATEGORY" if group_choice == "Crime category" else "_REGION"
        values = filtered.groupby(group_col)["_COUNT"].sum().sort_values(ascending=False).head(12)
        create_bar_chart(values, f"Top groups by {group_choice.lower()}", group_choice, "Recorded count")

    st.markdown("### 📋 Filtered records")
    display_df = filtered.drop(columns=["_DATE", "_YEAR", "_REGION", "_CATEGORY", "_COUNT",
                                        "_MONTH", "_MONTH_NAME", "_TIME"], errors="ignore")
    st.dataframe(display_df, use_container_width=True, height=330)
    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_crime_data.csv",
        mime="text/csv",
        use_container_width=False
    )
    if filtered.empty:
        st.warning("No rows match the current filters. Change one or more filters.")

# ------------------------------------------------------------
# Model training helpers
# ------------------------------------------------------------


# ------------------------------------------------------------
# Section 3: Prediction Lab
# ------------------------------------------------------------
if page == "Prediction Lab":
    st.subheader("🧠 Prediction Lab")
    st.markdown("""
    <div class="panel"><b>Prediction target</b><br>
    The models estimate the future <b>recorded count per time period</b> (daily, monthly, or yearly,
    depending on available data). This is a learning exercise, not an official crime forecast.
    </div>
    """, unsafe_allow_html=True)

    selected_regions = st.multiselect(
        "Choose regions for this model",
        sorted(df["_REGION"].dropna().astype(str).unique().tolist()),
        default=sorted(df["_REGION"].dropna().astype(str).unique().tolist())[:min(3, df["_REGION"].nunique())],
        key="prediction_regions"
    )
    selected_categories = st.multiselect(
        "Choose crime categories",
        sorted(df["_CATEGORY"].dropna().astype(str).unique().tolist()),
        default=sorted(df["_CATEGORY"].dropna().astype(str).unique().tolist())[:min(5, df["_CATEGORY"].nunique())],
        key="prediction_categories"
    )
    model_data = df.copy()
    if selected_regions:
        model_data = model_data[model_data["_REGION"].isin(selected_regions)]
    if selected_categories:
        model_data = model_data[model_data["_CATEGORY"].isin(selected_categories)]

    series_df, target_name = build_time_series(model_data)
    if series_df is None:
        st.warning(
            "This dataset does not contain enough ordered date/year observations for a responsible "
            "time-based model comparison. Try a dataset with more dates or multiple years."
        )
    else:
        st.info(f"Target: **{target_name}**. The last 20% of time points are held out for testing; "
                "the models are trained only on earlier periods.")
        results, fitted_models, error = train_models(series_df)
        if error:
            st.warning(error)
        else:
            best = results.iloc[0]
            st.metric("Lowest test RMSE", f"{best['RMSE']:.3f}", delta=str(best["Model"]))
            st.markdown("### 📊 Model comparison")
            fig, ax = plt.subplots(figsize=(9, 4))
            fig.patch.set_facecolor(CHART_BG)
            ax.bar(results["Model"], results["RMSE"],
                   color=[PALETTE[i % len(PALETTE)] for i in range(len(results))])
            ax.tick_params(axis="x", rotation=25)
            polish(ax, "Test-set RMSE (lower is better)", "Model", "RMSE")
            finish_chart(fig)
            st.dataframe(results.round(3), use_container_width=True)

            st.markdown("### 🔮 Optional future-period estimate")
            st.caption("Only use this for exploratory learning. A short or incomplete history may not generalize.")
            chosen_model = st.selectbox("Model for estimate", list(fitted_models.keys()), key="future_model")
            future_steps = st.slider("Number of future periods", 1, 12, 3)
            full_X = series_df[["time"]].to_numpy(dtype=float)
            full_y = series_df["actual"].to_numpy(dtype=float)
            final_model = fitted_models[chosen_model]
            final_model.fit(full_X, full_y)
            last_time = float(full_X[-1, 0])
            future_x = np.arange(last_time + 1, last_time + future_steps + 1, dtype=float).reshape(-1, 1)
            future_values = np.maximum(final_model.predict(future_x), 0)
            if target_name == "yearly recorded count":
                last_year = int(series_df["period"].iloc[-1])
                future_labels = [str(last_year + i) for i in range(1, future_steps + 1)]
            else:
                future_labels = [f"Future period {i}" for i in range(1, future_steps + 1)]
            future_df = pd.DataFrame({"Period": future_labels, "Estimated recorded count": np.round(future_values, 2)})
            st.dataframe(future_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 3.4))
            fig.patch.set_facecolor(CHART_BG)
            recent_n = min(18, len(series_df))
            recent = series_df.tail(recent_n)
            ax.plot(np.arange(recent_n), recent["actual"].values, color=PALETTE[0],
                    marker="o", linewidth=2, label="Observed")
            ax.plot(np.arange(recent_n, recent_n + future_steps), future_values,
                    color=PALETTE[2], marker="D", linestyle="--", linewidth=2, label="Estimate")
            polish(ax, "Recent observations and exploratory estimates", "Ordered time period", "Recorded count", legend=True)
            finish_chart(fig)
            st.caption("Estimates are model outputs, not verified future events or official forecasts.")

# ------------------------------------------------------------
# Section 4: Hotspot and anomaly analysis
# ------------------------------------------------------------
if page == "Hotspot & Anomalies":
    st.subheader("🚨 Crime Hotspot and Anomaly Analysis")
    st.markdown("""
    <div class="panel"><b>Important interpretation note</b><br>
    A high recorded count is a data hotspot, not proof that a place is inherently dangerous.
    Counts can be affected by population, reporting practices, enforcement and dataset coverage.
    </div>
    """, unsafe_allow_html=True)

    region_counts = df.groupby("_REGION")["_COUNT"].sum().sort_values(ascending=False)
    top_regions = region_counts.head(15)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("### 🏙️ Highest recorded counts")
        create_bar_chart(top_regions.sort_values(), "Top regions by recorded count",
                         "Region", "Recorded count", horizontal=True)
    with right:
        st.markdown("### 📦 Count distribution")
        values = df["_COUNT"].dropna().to_numpy()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(CHART_BG)
        if len(values):
            box = ax.boxplot(values, patch_artist=True, vert=True,
                             boxprops=dict(facecolor=PALETTE[1], color="#d8e8ff"),
                             medianprops=dict(color=PALETTE[3], linewidth=2),
                             whiskerprops=dict(color="#d8e8ff"),
                             capprops=dict(color="#d8e8ff"),
                             flierprops=dict(marker="o", markerfacecolor=PALETTE[2],
                                             markeredgecolor="white", markersize=4))
            polish(ax, "IQR boxplot of row-level recorded counts", "", "Recorded count")
            ax.set_xticks([1])
            ax.set_xticklabels(["All records"], color=TEXT_COLOR)
        else:
            polish(ax, "IQR boxplot", "", "Recorded count")
        finish_chart(fig)

    # IQR anomaly detection on aggregate count by region, suitable for grouped records.
    region_table = region_counts.reset_index()
    region_table.columns = ["Region", "Recorded count"]
    if len(region_table) >= 4:
        q1 = region_table["Recorded count"].quantile(0.25)
        q3 = region_table["Recorded count"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        region_table["Anomaly"] = (region_table["Recorded count"] < lower) | (region_table["Recorded count"] > upper)
        flagged_regions = region_table[region_table["Anomaly"]].sort_values("Recorded count", ascending=False)
    else:
        lower, upper = np.nan, np.nan
        region_table["Anomaly"] = False
        flagged_regions = region_table.iloc[0:0]

    st.markdown("### 🎯 Region-level IQR anomalies")
    if len(region_table) >= 4:
        st.caption(f"Flagging rule: below {lower:,.2f} or above {upper:,.2f} recorded counts (1.5 × IQR).")
        fig, ax = plt.subplots(figsize=(9, 3.8))
        fig.patch.set_facecolor(CHART_BG)
        plot_table = region_table.sort_values("Recorded count", ascending=False).head(25).copy()
        colors = [PALETTE[2] if flag else PALETTE[0] for flag in plot_table["Anomaly"]]
        ax.scatter(plot_table["Region"].astype(str), plot_table["Recorded count"],
                   c=colors, s=70, edgecolors="#eaf2ff", linewidths=.5)
        ax.tick_params(axis="x", rotation=40)
        polish(ax, "Region counts and flagged outliers", "Region", "Recorded count")
        finish_chart(fig)
        st.dataframe(flagged_regions, use_container_width=True)
        st.download_button(
            "⬇️ Download flagged regions (CSV)",
            data=flagged_regions.to_csv(index=False).encode("utf-8"),
            file_name="crime_anomaly_regions.csv",
            mime="text/csv"
        )
        if flagged_regions.empty:
            st.success("No region-level values crossed the IQR threshold in this dataset.")
    else:
        st.info("At least four distinct regions are needed for a useful region-level IQR check.")

    # Optional coordinate plot only if real latitude/longitude columns are mapped.
    if latitude_col and longitude_col:
        st.markdown("### 📍 Coordinate view (optional)")
        geo = df.copy()
        geo["_LAT"] = pd.to_numeric(geo[latitude_col], errors="coerce")
        geo["_LON"] = pd.to_numeric(geo[longitude_col], errors="coerce")
        geo = geo.dropna(subset=["_LAT", "_LON"])
        geo = geo[geo["_LAT"].between(-90, 90) & geo["_LON"].between(-180, 180)]
        if not geo.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor(CHART_BG)
            points = ax.scatter(geo["_LON"], geo["_LAT"], c=geo["_COUNT"], cmap="plasma",
                                s=18, alpha=.75, edgecolors="none")
            polish(ax, "Recorded events at supplied coordinates", "Longitude", "Latitude")
            cbar = fig.colorbar(points, ax=ax)
            cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
            plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR)
            cbar.set_label("Recorded count", color=TEXT_COLOR)
            finish_chart(fig)
        else:
            st.info("The selected coordinate columns do not contain valid latitude/longitude pairs.")

    st.markdown("### 📋 Highest-count region ranking")
    st.dataframe(region_table.sort_values("Recorded count", ascending=False).head(25),
                 use_container_width=True)


