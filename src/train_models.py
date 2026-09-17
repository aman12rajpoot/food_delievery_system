from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

DATA = Path("data/processed/clean_orders.csv")
MODELS = Path("models")
REPORTS = Path("reports")

FEATURES = [
    "distance_km","age","rating","vehicle_condition",
    "multiple_deliveries","pickup_hour","day_of_week",
    "is_peak","prep_time_min"
]

def prepare_features(df):
    x = df.copy()
    for c in FEATURES:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    # Drop all-NaN feature columns rather than allowing median to remain NaN.
    usable = [c for c in FEATURES if x[c].notna().any()]
    X = x[usable].replace([np.inf,-np.inf],np.nan)
    return X, usable

def train_eta(df):
    x = df.copy()
    X, features = prepare_features(x)
    y = pd.to_numeric(x["delivery_time"], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    # Chronological split when dates are available; otherwise preserve row order.
    if "date" in x.columns:
        dates = pd.to_datetime(x.loc[valid,"date"], errors="coerce")
        order = dates.fillna(pd.Timestamp("1900-01-01")).sort_values().index
        X, y = X.loc[order], y.loc[order]
    cut = max(1, int(len(X)*.8))
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y.iloc[:cut], y.iloc[cut:]

    if len(Xte) == 0:
        raise ValueError("Not enough rows for an ETA test split.")

    models = {
        "linear": Pipeline([("imputer",SimpleImputer(strategy="median")),
                            ("model",LinearRegression())]),
        "random_forest": Pipeline([("imputer",SimpleImputer(strategy="median")),
                                   ("model",RandomForestRegressor(
                                       n_estimators=250,random_state=42,n_jobs=-1,min_samples_leaf=3))]),
        "xgboost": Pipeline([("imputer",SimpleImputer(strategy="median")),
                             ("model",XGBRegressor(
                                 n_estimators=350,max_depth=6,learning_rate=.05,
                                 subsample=.85,colsample_bytree=.85,random_state=42,
                                 objective="reg:squarederror",n_jobs=4))])
    }

    results, fitted = {}, {}
    for name,m in models.items():
        m.fit(Xtr,ytr)
        p=m.predict(Xte)
        train_pred=m.predict(Xtr)
        results[name]={
            "MAE":float(mean_absolute_error(yte,p)),
            "RMSE":float(mean_squared_error(yte,p)**.5),
            "R2":float(r2_score(yte,p)),
            "train_R2":float(r2_score(ytr,train_pred)),
            "r2_gap":float(r2_score(ytr,train_pred)-r2_score(yte,p))
        }
        fitted[name]=m

    best=min(results,key=lambda k:results[k]["MAE"])
    summary={"best_model":best,"features":features,"models":results}
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump(fitted[best],MODELS/"eta_model.joblib")
    (MODELS/"eta_features.json").write_text(json.dumps(features,indent=2))
    (REPORTS/"eta_metrics.json").write_text(json.dumps(summary,indent=2))
    return summary

def train_demand(df):
    x = df.copy()
    if "date" not in x.columns and "order_date" in x.columns:
        x["date"] = x["order_date"]
    if "pickup_hour" not in x.columns and "pickup_time" in x.columns:
        x["pickup_hour"] = pd.to_datetime(x["pickup_time"], format="%I:%M:%S %p", errors="coerce")
    x = x.dropna(subset=["date","pickup_hour"]).copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce", dayfirst=True)
    x["pickup_hour"] = pd.to_numeric(x["pickup_hour"], errors="coerce")
    x = x.dropna(subset=["date","pickup_hour"])
    if x.empty:
        raise ValueError("Insufficient demand observations after parsing order dates and pickup times. Check the raw time columns and date format.")

    hourly = x.groupby(["date","pickup_hour"]).size().reset_index(name="orders")
    hourly["timestamp"] = hourly["date"] + pd.to_timedelta(hourly["pickup_hour"], unit="h")
    hourly = hourly.sort_values("timestamp").set_index("timestamp")
    full = hourly[["orders"]].resample("h").sum().fillna(0)

    for lag in [1,2,3,24,48,168]:
        full[f"lag_{lag}"] = full.orders.shift(lag)
    full["roll_24"] = full.orders.shift(1).rolling(24).mean()
    full["hour"] = full.index.hour
    full["dow"] = full.index.dayofweek
    full = full.dropna()
    if len(full) < 48:
        raise ValueError(
            f"Not enough temporal demand observations after building the hourly series ({len(full)} rows). "
            "The dataset does not provide enough valid hourly coverage for the configured lag features."
        )

    feats = [c for c in full.columns if c != "orders"]
    cut = max(1, int(len(full) * 0.8))
    if cut >= len(full):
        raise ValueError(f"Demand training requires at least 2 rows in the test split; observed {len(full)} rows.")
    tr, te = full.iloc[:cut], full.iloc[cut:]
    m = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=.05,
        max_leaf_nodes=31,
        random_state=42,
        loss="squared_error"
    )

    m.fit(tr[feats], tr.orders)

    p = np.clip(
        m.predict(te[feats]),
        0,
        None
    )
    naive = te["lag_24"].values
    out = {
        "demand_observations": int(len(full)),
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "granularity": "hour",
        "model_MAE": float(mean_absolute_error(te.orders, p)),
        "model_RMSE": float(mean_squared_error(te.orders, p) ** 0.5),
        "baseline_MAE": float(mean_absolute_error(te.orders, naive)),
        "baseline_RMSE": float(mean_squared_error(te.orders, naive) ** 0.5),
        "baseline_model": "lag_24",
        "features": feats,
    }
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    joblib.dump(m, MODELS/"demand_model.joblib")
    (MODELS/"demand_features.json").write_text(json.dumps(feats, indent=2))
    (REPORTS/"demand_metrics.json").write_text(json.dumps(out, indent=2))
    return out

def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    df=pd.read_csv(DATA)
    return train_eta(df),train_demand(df)

if __name__=="__main__":
    print(main())
