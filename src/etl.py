from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

RAW = Path("data/raw/Zomato Dataset.csv")
PROC = Path("data/processed")
DB = Path("data/food_delivery.db")

def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2-lat1)
    dl = np.radians(lon2-lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arcsin(np.sqrt(a))

def clean():
    if not RAW.exists():
        raise FileNotFoundError(f"Put the public CSV at {RAW}")

    df = pd.read_csv(RAW)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {
        "delivery_person_age": "age",
        "delivery_person_ratings": "rating",
        "delivery_location_latitude": "delivery_lat",
        "delivery_location_longitude": "delivery_lon",
        "restaurant_latitude": "restaurant_lat",
        "restaurant_longitude": "restaurant_lon",
        "time_taken_(min)": "delivery_time",
        "time_order_picked": "pickup_time",
        "time_orderd": "order_time",
        "order_date": "order_date",
        "weatherconditions": "weather",
        "weather_conditions": "weather",
        "road_traffic_density": "traffic",
        "multiple_deliveries": "multiple_deliveries",
        "vehicle_condition": "vehicle_condition",
        "festival": "festival",
        "type_of_order": "order_type",
        "type_of_vehicle": "vehicle_type",
        "city": "city",
        "delivery_person_id": "delivery_person_id",
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

    # Numeric conversion. Invalid strings become NaN and are handled explicitly below.
    numeric_cols = [
        "age","rating","delivery_lat","delivery_lon","restaurant_lat",
        "restaurant_lon","vehicle_condition","multiple_deliveries","delivery_time"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=["delivery_time"])
    # Remove extreme target outliers while retaining realistic observations.
    upper = df["delivery_time"].quantile(.99)
    df = df[df["delivery_time"].between(5, upper)].copy()

    if {"restaurant_lat","restaurant_lon","delivery_lat","delivery_lon"}.issubset(df.columns):
        df["distance_km"] = haversine(
            df["restaurant_lat"], df["restaurant_lon"],
            df["delivery_lat"], df["delivery_lon"]
        )
        df["distance_km"] = df["distance_km"].replace([np.inf,-np.inf],np.nan)
        df.loc[df["distance_km"] < 0, "distance_km"] = np.nan
    else:
        df["distance_km"] = np.nan

    # Parse actual order/pickup time robustly. The public dataset mixes AM/PM and 24-hour values.
    def parse_hour(series):
        s = series.astype("string").str.strip().replace({"NaN": pd.NA, "nan": pd.NA, "": pd.NA})
        parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        for fmt in ["%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"]:
            parsed = parsed.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
        return parsed.dt.hour

    df["pickup_hour"] = parse_hour(df["pickup_time"]) if "pickup_time" in df.columns else np.nan
    df["order_hour"] = parse_hour(df["order_time"]) if "order_time" in df.columns else df["pickup_hour"]

    if "order_date" in df.columns:
        dt = pd.to_datetime(df["order_date"], errors="coerce", dayfirst=True)
        df["date"] = dt.dt.date.astype("string")
        df["day_of_week"] = dt.dt.dayofweek
    else:
        df["date"] = pd.Series(pd.NA, index=df.index, dtype="string")
        df["day_of_week"] = np.nan

    # Some rows may still have missing date/hour values after parsing; keep the record if date is valid
    # and let downstream training validate the temporal coverage explicitly.

    # Practical feature: preparation/pickup delay when both timestamps are parseable.
    if "order_time" in df.columns and "pickup_time" in df.columns:
        order_dt = pd.to_datetime(
            df["order_time"].astype("string").str.strip(),
            format="%I:%M:%S %p", errors="coerce"
        )
        pickup_dt = pd.to_datetime(
            df["pickup_time"].astype("string").str.strip(),
            format="%I:%M:%S %p", errors="coerce"
        )
        prep = (pickup_dt - order_dt).dt.total_seconds()/60
        # If pickup crossed midnight, add one day.
        prep = prep.where(prep >= 0, prep + 1440)
        df["prep_time_min"] = prep.clip(lower=0, upper=180)
    else:
        df["prep_time_min"] = np.nan

    df["is_peak"] = df["pickup_hour"].isin([12,13,19,20,21]).astype(int)
    df["demand_key"] = (
        df["date"].astype("string").fillna("unknown") + "_" +
        df["pickup_hour"].fillna(-1).astype(int).astype(str)
    )

    # Persist a clean analytical dataset.
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROC/"clean_orders.csv", index=False)

    with sqlite3.connect(DB) as con:
        df.to_sql("orders", con, if_exists="replace", index=False)
        hourly = (
            df.dropna(subset=["date","pickup_hour"])
              .groupby(["date","pickup_hour"])
              .size().reset_index(name="orders")
        )
        hourly.to_sql("hourly_demand", con, if_exists="replace", index=False)

    print(f"Cleaned rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    return df

if __name__ == "__main__":
    clean()
