
import re
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Normalize column names
def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# 2. Find a column automatically
def find_column(df, keywords):
    for col in df.columns:
        col_name = normalize_name(col)
        for word in keywords:
            word_name = normalize_name(word)
            if word_name and word_name in col_name:
                return col
    return None


# 3. Read CSV or Excel file
def read_uploaded_file(file):
    name = file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(file, low_memory=False)

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    raise ValueError("Please upload a CSV or Excel file.")


# 4. Clean dataset
def clean_data(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(how="all")
    df = df.drop_duplicates()
    return df


# 5. Detect dataset columns
def detect_schema(df):
    date_col = find_column(
        df, ["ARREST_DATE", "OCCUR_DATE", "CRIME_DATE",
             "INCIDENT_DATE", "OFFENSE_DATE", "DATE"]
    )

    year_col = find_column(
        df, ["CRIME_YEAR", "INCIDENT_YEAR", "YEAR"]
    )

    region_col = find_column(
        df, ["ARREST_BORO", "BOROUGH", "BORO", "REGION",
             "DISTRICT", "STATE", "SUBDIVISION", "COUNTY", "CITY"]
    )

    category_col = find_column(
        df, ["OFNS_DESC", "CRIME_CATEGORY", "CRIME_TYPE",
             "OFFENSE", "OFFENCE", "CATEGORY", "DESCRIPTION"]
    )

    count_col = find_column(
        df, ["CRIME_COUNT", "TOTAL_CRIMES", "TOTAL_COUNT",
             "NUMBER_OF_CRIMES", "COUNT", "TOTAL"]
    )

    latitude_col = find_column(df, ["LATITUDE", "LAT"])
    longitude_col = find_column(df, ["LONGITUDE", "LON", "LONG"])

    if count_col and normalize_name(count_col) in [
        "id", "crimeid", "arrestkey", "incidentid", "recordid"
    ]:
        count_col = None

    return {
        "date": date_col,
        "year": year_col,
        "region": region_col,
        "category": category_col,
        "count": count_col,
        "latitude": latitude_col,
        "longitude": longitude_col
    }


# 6. Prepare standard columns for dashboard
def apply_schema(df, schema):
    data = df.copy()

    date_col = schema.get("date")
    year_col = schema.get("year")
    region_col = schema.get("region")
    category_col = schema.get("category")
    count_col = schema.get("count")

    # Date
    if date_col and date_col in data.columns:
        data["_DATE"] = pd.to_datetime(
            data[date_col], errors="coerce"
        )
    else:
        data["_DATE"] = pd.NaT

    # Year
    if year_col and year_col in data.columns:
        data["_YEAR"] = pd.to_numeric(
            data[year_col], errors="coerce"
        )
    else:
        data["_YEAR"] = np.nan

    data["_YEAR"] = data["_YEAR"].fillna(
        data["_DATE"].dt.year
    )

    # Region
    if region_col and region_col in data.columns:
        data["_REGION"] = (
            data[region_col].fillna("Unknown").astype(str).str.strip()
        )
        data.loc[data["_REGION"] == "", "_REGION"] = "Unknown"
    else:
        data["_REGION"] = "All records"

    # Crime category
    if category_col and category_col in data.columns:
        data["_CATEGORY"] = (
            data[category_col]
            .fillna("Uncategorized")
            .astype(str)
            .str.strip()
        )
        data.loc[data["_CATEGORY"] == "", "_CATEGORY"] = "Uncategorized"
    else:
        data["_CATEGORY"] = "Uncategorized"

    # Crime count
    if count_col and count_col in data.columns:
        data["_COUNT"] = pd.to_numeric(
            data[count_col], errors="coerce"
        ).fillna(0).clip(lower=0)
    else:
        # Each row represents one recorded incident
        data["_COUNT"] = 1.0

    data["_MONTH"] = data["_DATE"].dt.month
    data["_MONTH_NAME"] = data["_DATE"].dt.month_name()
    data["_TIME"] = np.arange(len(data), dtype=float)

    return data


# 7. Aggregate recorded counts
def aggregate_counts(df, column):
    if column not in df.columns:
        return pd.Series(dtype=float)

    return df.groupby(column)["_COUNT"].sum().sort_values(
        ascending=False
    )


# 8. Calculate dashboard KPIs
def kpi_summary(df):
    total = (
        float(df["_COUNT"].sum())
        if "_COUNT" in df.columns else float(len(df))
    )

    regions_count = (
        int(df["_REGION"].nunique())
        if "_REGION" in df.columns else 0
    )

    categories_count = (
        int(df["_CATEGORY"].nunique())
        if "_CATEGORY" in df.columns else 0
    )

    if not df.empty and "_REGION" in df.columns:
        region_totals = df.groupby("_REGION")["_COUNT"].sum()
        highest = float(region_totals.max()) if len(region_totals) else 0
        highest_label = "Highest region count"
    else:
        highest = total
        highest_label = "Total recorded count"

    return total, regions_count, categories_count, highest, highest_label


# 9. Filter data for Crime Explorer
def filter_data(df, regions=None, categories=None, years=None):
    result = df.copy()

    if regions and "All regions" not in regions:
        result = result[result["_REGION"].isin(regions)]

    if categories and "All categories" not in categories:
        result = result[result["_CATEGORY"].isin(categories)]

    if years is not None:
        start_year, end_year = years
        result = result[
            result["_YEAR"].notna()
            & (result["_YEAR"] >= start_year)
            & (result["_YEAR"] <= end_year)
        ]

    return result


# 10. Build time series for prediction
def build_time_series(df):
    if df is None or df.empty or "_COUNT" not in df.columns:
        return None, None

    data = df.copy()

    if "_DATE" in data.columns and data["_DATE"].notna().any():
        data = data.dropna(subset=["_DATE"]).copy()
        data["_PERIOD"] = data["_DATE"].dt.to_period("M").astype(str)

        grouped = (
            data.groupby("_PERIOD")["_COUNT"]
            .sum().reset_index()
        )

        grouped.columns = ["period", "actual"]
        target_name = "monthly recorded count"

    elif "_YEAR" in data.columns and data["_YEAR"].notna().any():
        data = data.dropna(subset=["_YEAR"]).copy()
        data["_PERIOD"] = data["_YEAR"].astype(int)

        grouped = (
            data.groupby("_PERIOD")["_COUNT"]
            .sum().reset_index()
        )

        grouped.columns = ["period", "actual"]
        target_name = "yearly recorded count"

    else:
        return None, None

    grouped = grouped.sort_values("period").reset_index(drop=True)

    if len(grouped) < 10:
        return None, None

    grouped["time"] = np.arange(len(grouped), dtype=float)

    return grouped[["period", "time", "actual"]], target_name


# 11. Train and compare five models
def train_models(series_df):
    if series_df is None or len(series_df) < 10:
        return (
            pd.DataFrame(),
            {},
            "At least 10 time periods are required."
        )

    data = series_df.sort_values("time").reset_index(drop=True)

    X = data[["time"]].to_numpy(dtype=float)
    y = pd.to_numeric(
        data["actual"], errors="coerce"
    ).to_numpy(dtype=float)

    valid = np.isfinite(X[:, 0]) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    if len(y) < 10:
        return pd.DataFrame(), {}, "Not enough valid time periods."

    split = int(len(y) * 0.8)
    split = max(1, min(split, len(y) - 1))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    models = {
        "Linear Regression": LinearRegression(),

        "Polynomial Regression": make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        ),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=3, random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    results = []
    fitted_models = {}

    try:
        for name, model in models.items():
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)

            mse = mean_squared_error(y_test, predicted)

            r2 = (
                r2_score(y_test, predicted)
                if len(y_test) > 1 else np.nan
            )

            results.append({
                "Model": name,
                "MAE": mean_absolute_error(y_test, predicted),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2": r2
            })

            fitted_models[name] = model

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            "RMSE"
        ).reset_index(drop=True)

        return results_df, fitted_models, None

    except Exception as error:
        return pd.DataFrame(), {}, str(error)
