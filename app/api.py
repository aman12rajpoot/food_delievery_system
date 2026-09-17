from pathlib import Path

import json

import joblib

import numpy as np

import pandas as pd

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

from src.recommender import recommend

from src.optimization import optimize_assignments

app = FastAPI(
title="Food Delivery Intelligence API",
description=(
"End-to-end food delivery analytics API covering ETA prediction, "
"demand forecasting, context-aware restaurant recommendation, "
"ranking evaluation and delivery assignment optimization."
),
version="2.1",
)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
DATA_PATH = Path("data/processed/clean_orders.csv")

# ---------------------------------------------------------

# Request Models

# ---------------------------------------------------------

class ETARequest(BaseModel):
    distance_km: float
    age: float = 30
    rating: float = 4.5
    vehicle_condition: float = 1
    multiple_deliveries: float = 0
    pickup_hour: float = 19
    day_of_week: float = 4
    is_peak: int = 1

    # Kept for backward compatibility.
    # Current trained ETA model may not use this feature.
    prep_time_min: float = 10


class OptimizationOrder(BaseModel):
    distance_km: float
    delivery_time: float


class OptimizationPartner(BaseModel):
    partner_id: str
    base_distance: float
    capacity: int = 4


class OptimizationRequest(BaseModel):
    orders: list[OptimizationOrder]
    partners: list[OptimizationPartner]

# ---------------------------------------------------------

# Health

# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "eta_model_available": (MODEL_DIR / "eta_model.joblib").exists(),
        "demand_model_available": (MODEL_DIR / "demand_model.joblib").exists(),
        "recommendation_available": (MODEL_DIR / "restaurant_catalog.csv").exists(),
        "optimization_available": True,
        "ab_test_available": (REPORT_DIR / "ab_test_results.json").exists(),
    }

# ---------------------------------------------------------

# ETA Prediction

# ---------------------------------------------------------

@app.post("/predict-eta")
def predict_eta(req: ETARequest):
    model_path = MODEL_DIR / "eta_model.joblib"
    feature_path = MODEL_DIR / "eta_features.json"

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "ETA model not found. "
                "Run python run_pipeline.py first."
            ),
        )

    if not feature_path.exists():
        raise HTTPException(
            status_code=404,
            detail="ETA feature file not found.",
        )

    try:
        model = joblib.load(model_path)
        features = json.loads(feature_path.read_text())

        row = {feature: getattr(req, feature, 0) for feature in features}
        x = pd.DataFrame([row], columns=features)

        prediction = float(model.predict(x)[0])
        prediction = max(0, prediction)

        return {
            "predicted_delivery_minutes": round(prediction, 2),
            "model": "XGBoost",
            "features_used": features,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ETA prediction failed: {str(e)}",
        )

# ---------------------------------------------------------

# Demand Forecasting

# ---------------------------------------------------------

def build_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the exact hourly demand feature engineering
    used during model training.
    """

    x = df.copy()

    if "date" not in x.columns and "order_date" in x.columns:
        x["date"] = x["order_date"]

    if "pickup_hour" not in x.columns and "pickup_time" in x.columns:
        parsed_time = pd.to_datetime(
            x["pickup_time"],
            format="%I:%M:%S %p",
            errors="coerce",
        )
        x["pickup_hour"] = parsed_time.dt.hour

    x = x.dropna(subset=["date", "pickup_hour"]).copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce", dayfirst=True)
    x["pickup_hour"] = pd.to_numeric(x["pickup_hour"], errors="coerce")
    x = x.dropna(subset=["date", "pickup_hour"])

    if x.empty:
        raise ValueError(
            "No valid demand observations after parsing dates and pickup times."
        )

    hourly = (
        x.groupby(["date", "pickup_hour"])
        .size()
        .reset_index(name="orders")
    )

    hourly["timestamp"] = hourly["date"] + pd.to_timedelta(hourly["pickup_hour"], unit="h")
    hourly = hourly.sort_values("timestamp").set_index("timestamp")

    full = hourly[["orders"]].resample("h").sum().fillna(0)

    for lag in [1, 2, 3, 24, 48, 168]:
        full[f"lag_{lag}"] = full["orders"].shift(lag)

    full["roll_24"] = full["orders"].shift(1).rolling(24).mean()
    full["hour"] = full.index.hour
    full["dow"] = full.index.dayofweek
    full = full.dropna()

    return full


@app.get("/forecast-demand")
def forecast_demand():
    model_path = MODEL_DIR / "demand_model.joblib"
    feature_path = MODEL_DIR / "demand_features.json"

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Demand model not found. "
                "Run python run_pipeline.py first."
            ),
        )

    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Processed dataset not found.",
        )

    try:
        model = joblib.load(model_path)
        features = json.loads(feature_path.read_text())

        df = pd.read_csv(DATA_PATH)
        demand = build_demand_features(df)

        if demand.empty:
            raise ValueError("Demand feature table is empty.")

        latest = demand.iloc[-1]
        x = pd.DataFrame([[latest[feature] for feature in features]], columns=features)

        prediction = float(model.predict(x)[0])
        prediction = max(0, prediction)

        return {
            "predicted_demand": round(prediction, 2),
            "forecast_granularity": "hour",
            "forecast_for": (demand.index[-1] + pd.Timedelta(hours=1)).isoformat(),
            "latest_observed_orders": int(latest["orders"]),
            "features_used": features,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Demand forecast failed: {str(e)}",
        )

# ---------------------------------------------------------

# Context-Aware Restaurant Recommendation

# ---------------------------------------------------------

@app.get("/recommend")
def get_recommend(
    query: str = "",
    top_k: int = 10,
    user_lat: float | None = None,
    user_lon: float | None = None,
    min_rating: float | None = None,
    max_distance_km: float | None = None,
    pickup_hour: float = 19,
    day_of_week: float = 4,
    is_peak: int = 1,
    vehicle_condition: float = 1,
    multiple_deliveries: float = 0,
    delivery_person_age: float = 30,
):
    """
    Context-aware restaurant recommendation endpoint.

    Supports:
    - food/order-type query
    - customer location
    - minimum rating
    - maximum distance
    - order hour
    - day of week
    - peak-hour context
    - delivery vehicle condition
    - multiple deliveries
    - delivery partner age

    Returns explainable ranking components.
    """

    required_files = [
        MODEL_DIR / "restaurant_catalog.csv",
        MODEL_DIR / "restaurant_vectorizer.joblib",
        MODEL_DIR / "restaurant_matrix.joblib",
        MODEL_DIR / "eta_model.joblib",
        MODEL_DIR / "eta_features.json",
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Recommendation artifacts are incomplete.",
                "missing_files": missing_files,
            },
        )

    if top_k < 1 or top_k > 50:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 50.",
        )

    if min_rating is not None and (min_rating < 0 or min_rating > 5):
        raise HTTPException(
            status_code=400,
            detail="min_rating must be between 0 and 5.",
        )

    if max_distance_km is not None and max_distance_km < 0:
        raise HTTPException(
            status_code=400,
            detail="max_distance_km cannot be negative.",
        )

    if (user_lat is None) != (user_lon is None):
        raise HTTPException(
            status_code=400,
            detail="user_lat and user_lon must be provided together.",
        )

    if user_lat is not None and user_lon is not None:
        if not (6 <= user_lat <= 37 and 68 <= user_lon <= 97):
            raise HTTPException(
                status_code=400,
                detail="Customer coordinates are outside supported India bounds.",
            )

    if not (0 <= day_of_week <= 6):
        raise HTTPException(
            status_code=400,
            detail="day_of_week must be between 0 and 6.",
        )

    if not (0 <= pickup_hour <= 23):
        raise HTTPException(
            status_code=400,
            detail="pickup_hour must be between 0 and 23.",
        )

    if is_peak not in [0, 1]:
        raise HTTPException(
            status_code=400,
            detail="is_peak must be either 0 or 1.",
        )

    try:
        results = recommend(
            query=query,
            top_k=top_k,
            user_lat=user_lat,
            user_lon=user_lon,
            min_rating=min_rating,
            max_distance_km=max_distance_km,
            pickup_hour=pickup_hour,
            day_of_week=day_of_week,
            is_peak=is_peak,
            vehicle_condition=vehicle_condition,
            multiple_deliveries=multiple_deliveries,
            delivery_person_age=delivery_person_age,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}") from e

    results = results.replace({np.nan: None})
    records = results.to_dict(orient="records")

    return {
        "query": query,
        "top_k": top_k,
        "context": {
            "user_location": (
                {"latitude": user_lat, "longitude": user_lon}
                if user_lat is not None and user_lon is not None
                else None
            ),
            "min_rating": min_rating,
            "max_distance_km": max_distance_km,
            "pickup_hour": pickup_hour,
            "day_of_week": day_of_week,
            "is_peak": is_peak,
            "vehicle_condition": vehicle_condition,
            "multiple_deliveries": multiple_deliveries,
            "delivery_person_age": delivery_person_age,
        },
        "ranking_method": (
            "Hybrid context-aware ranking using relevance, rating, popularity, distance "
            "and predicted ETA."
        ),
        "results": records,
    }

# ---------------------------------------------------------

# Ranking Metrics

# ---------------------------------------------------------

@app.get("/ranking-metrics")
def ranking_metrics():
    path = REPORT_DIR / "ranking_metrics.json"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Ranking evaluation not found.",
        )

    return json.loads(path.read_text())

# ---------------------------------------------------------

# Delivery Optimization

# ---------------------------------------------------------

@app.post("/optimize")
def optimize(request: OptimizationRequest):
    orders = pd.DataFrame(
        [
            {
                "distance_km": o.distance_km,
                "delivery_time": o.delivery_time,
            }
            for o in request.orders
        ]
    )

    partners = pd.DataFrame(
        [
            {
                "partner_id": p.partner_id,
                "base_distance": p.base_distance,
                "capacity": p.capacity,
            }
            for p in request.partners
        ]
    )

    if orders.empty:
        raise HTTPException(
            status_code=400,
            detail="At least one order is required.",
        )

    if partners.empty:
        raise HTTPException(
            status_code=400,
            detail="At least one delivery partner is required.",
        )

    try:
        result = optimize_assignments(orders, partners)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}") from e

    return {
        "assigned_orders": len(result),
        "assignments": result.to_dict(orient="records"),
    }

# ---------------------------------------------------------

# A/B Test

# ---------------------------------------------------------

@app.get("/ab-test")
def ab_test():
    path = REPORT_DIR / "ab_test_results.json"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="A/B experiment results not found.",
        )

    return json.loads(path.read_text())

