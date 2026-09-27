from pathlib import Path
import hashlib
import json
import re
import secrets
import sqlite3

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.recommender import recommend
from src.optimization import optimize_assignments


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _normalize(series):
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    minimum = series.min()
    maximum = series.max()
    if maximum <= minimum:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - minimum) / (maximum - minimum)


def _load_eta_model():
    model_path = MODEL_DIR / "eta_model.joblib"
    feature_path = MODEL_DIR / "eta_features.json"
    if not model_path.exists() or not feature_path.exists():
        raise FileNotFoundError("ETA model artifacts not found. Run: python run_pipeline.py")
    model = joblib.load(model_path)
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    return model, features


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="Food Delivery Intelligence API",
    description=(
        "End-to-end backend for a data-driven food-delivery platform. "
        "Serves restaurant recommendations, ETA prediction, demand forecasting, "
        "ranking evaluation, and delivery assignment optimization."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_orders.csv"
AUTH_DB = BASE_DIR / "data" / "food_delivery.db"
AUTH_DB.parent.mkdir(parents=True, exist_ok=True)

PASSWORD_ITERATIONS = 220_000


def init_auth_db():
    with sqlite3.connect(AUTH_DB) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS app_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                mobile TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        con.commit()


def normalize_mobile(mobile: str) -> str:
    digits = re.sub(r"\D", "", mobile or "")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    return digits


def validate_user_id(user_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{3,30}", user_id or ""))


def validate_mobile(mobile: str) -> bool:
    return bool(re.fullmatch(r"[6-9]\d{9}", mobile or ""))


def hash_password(password: str, salt_hex: str | None = None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    _, actual_hash = hash_password(password, salt_hex)
    return secrets.compare_digest(actual_hash, expected_hash)


init_auth_db()


# ============================================================
# Request models
# ============================================================

class ETARequest(BaseModel):
    distance_km: float = Field(..., ge=0)
    age: float = Field(30, ge=18, le=80)
    rating: float = Field(4.5, ge=0, le=5)
    vehicle_condition: float = Field(1, ge=0)
    multiple_deliveries: float = Field(0, ge=0)
    pickup_hour: float = Field(19, ge=0, le=23)
    day_of_week: float = Field(4, ge=0, le=6)
    is_peak: int = Field(1, ge=0, le=1)
    prep_time_min: float = Field(10, ge=0)


class OptimizationOrder(BaseModel):
    distance_km: float = Field(..., ge=0)
    delivery_time: float = Field(..., ge=0)


class OptimizationPartner(BaseModel):
    partner_id: str
    base_distance: float = Field(..., ge=0)
    capacity: int = Field(4, ge=1)


class OptimizationRequest(BaseModel):
    orders: list[OptimizationOrder]
    partners: list[OptimizationPartner]


class RegisterRequest(BaseModel):
    user_id: str = Field(..., min_length=3, max_length=30)
    mobile: str
    password: str = Field(..., min_length=6, max_length=128)
    name: str = Field(..., min_length=1, max_length=80)


class LoginRequest(BaseModel):
    identifier: str = Field(..., min_length=1, max_length=80)
    password: str = Field(..., min_length=1, max_length=128)


# ============================================================
# Authentication
# ============================================================

@app.post("/auth/register")
def register_user(req: RegisterRequest):
    user_id = req.user_id.strip()
    name = req.name.strip()
    mobile = normalize_mobile(req.mobile)

    if not validate_user_id(user_id):
        raise HTTPException(
            status_code=400,
            detail="User ID must be 3-30 characters using letters, numbers, _, -, or .",
        )

    if not validate_mobile(mobile):
        raise HTTPException(
            status_code=400,
            detail="Enter a valid 10-digit Indian mobile number.",
        )

    salt_hex, password_hash = hash_password(req.password)

    try:
        with sqlite3.connect(AUTH_DB) as con:
            con.execute(
                """
                INSERT INTO app_users
                (user_id, mobile, name, password_hash, password_salt)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, mobile, name, password_hash, salt_hex),
            )
            con.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="User ID or mobile number is already registered.",
        )

    return {
        "authenticated": True,
        "user": {
            "user_id": user_id,
            "mobile": mobile,
            "name": name,
        },
        "demo_note": "Demo authentication only. Production apps should use a proper identity provider and secure session management.",
    }


@app.post("/auth/login")
def login_user(req: LoginRequest):
    identifier = req.identifier.strip()
    mobile = normalize_mobile(identifier)

    with sqlite3.connect(AUTH_DB) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT user_id, mobile, name, password_hash, password_salt
            FROM app_users
            WHERE lower(user_id) = lower(?) OR mobile = ?
            LIMIT 1
            """,
            (identifier, mobile),
        ).fetchone()

    if row is None or not verify_password(
        req.password,
        row["password_salt"],
        row["password_hash"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid User ID/mobile or password.",
        )

    return {
        "authenticated": True,
        "user": {
            "user_id": row["user_id"],
            "mobile": row["mobile"],
            "name": row["name"],
        },
    }


@app.get("/auth/user/{user_id}")
def get_auth_user(user_id: str):
    with sqlite3.connect(AUTH_DB) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT user_id, mobile, name, created_at
            FROM app_users
            WHERE lower(user_id) = lower(?)
            """,
            (user_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="User not found.")

    return {
        "user": {
            "user_id": row["user_id"],
            "mobile": row["mobile"],
            "name": row["name"],
            "created_at": row["created_at"],
        }
    }


# ============================================================
# Basic routes
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Food Delivery Intelligence API",
        "version": "3.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "eta_model_available": (MODEL_DIR / "eta_model.joblib").exists(),
        "eta_features_available": (MODEL_DIR / "eta_features.json").exists(),
        "demand_model_available": (MODEL_DIR / "demand_model.joblib").exists(),
        "demand_features_available": (MODEL_DIR / "demand_features.json").exists(),
        "recommendation_catalog_available": (MODEL_DIR / "restaurant_catalog.csv").exists(),
        "recommendation_vectorizer_available": (MODEL_DIR / "restaurant_vectorizer.joblib").exists(),
        "recommendation_matrix_available": (MODEL_DIR / "restaurant_matrix.joblib").exists(),
        "processed_data_available": DATA_PATH.exists(),
        "optimization_available": True,
        "ranking_report_available": (REPORT_DIR / "ranking_metrics.json").exists(),
        "ab_test_report_available": (REPORT_DIR / "ab_test_results.json").exists(),
    }


# ============================================================
# ETA prediction
# ============================================================

@app.post("/predict-eta")
def predict_eta(req: ETARequest):
    model_path = MODEL_DIR / "eta_model.joblib"
    feature_path = MODEL_DIR / "eta_features.json"

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail="ETA model not found. Run: python run_pipeline.py",
        )

    if not feature_path.exists():
        raise HTTPException(
            status_code=404,
            detail="ETA feature definition not found. Run: python run_pipeline.py",
        )

    try:
        model = joblib.load(model_path)
        features = json.loads(feature_path.read_text(encoding="utf-8"))

        row = {feature: getattr(req, feature, 0) for feature in features}
        x = pd.DataFrame([row], columns=features)
        prediction = float(model.predict(x)[0])

        return {
            "predicted_delivery_minutes": round(max(0.0, prediction), 2),
            "model": "XGBoost",
            "features_used": features,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ETA prediction failed: {exc}") from exc


# ============================================================
# Demand forecasting
# ============================================================

def build_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    if "date" not in x.columns and "order_date" in x.columns:
        x["date"] = x["order_date"]

    if "pickup_hour" not in x.columns and "pickup_time" in x.columns:
        parsed_time = pd.to_datetime(
            x["pickup_time"].astype("string"),
            format="%I:%M:%S %p",
            errors="coerce",
        )
        x["pickup_hour"] = parsed_time.dt.hour

    if "date" not in x.columns or "pickup_hour" not in x.columns:
        raise ValueError("Processed data must contain date/order_date and pickup_hour/pickup_time.")

    x = x.dropna(subset=["date", "pickup_hour"]).copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce", dayfirst=True)
    x["pickup_hour"] = pd.to_numeric(x["pickup_hour"], errors="coerce")
    x = x.dropna(subset=["date", "pickup_hour"])

    if x.empty:
        raise ValueError("No valid demand observations after parsing dates and pickup times.")

    hourly = (
        x.groupby(["date", "pickup_hour"])
        .size()
        .reset_index(name="orders")
    )

    hourly["timestamp"] = hourly["date"] + pd.to_timedelta(
        hourly["pickup_hour"], unit="h"
    )

    full = (
        hourly.sort_values("timestamp")
        .set_index("timestamp")[["orders"]]
        .resample("h")
        .sum()
        .fillna(0)
    )

    for lag in [1, 2, 3, 24, 48, 168]:
        full[f"lag_{lag}"] = full["orders"].shift(lag)

    full["roll_24"] = full["orders"].shift(1).rolling(24).mean()
    full["hour"] = full.index.hour
    full["dow"] = full.index.dayofweek

    return full.dropna()


@app.get("/forecast-demand")
def forecast_demand():
    model_path = MODEL_DIR / "demand_model.joblib"
    feature_path = MODEL_DIR / "demand_features.json"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Demand model not found. Run: python run_pipeline.py")
    if not feature_path.exists():
        raise HTTPException(status_code=404, detail="Demand feature definition not found.")
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Processed dataset not found. Run: python run_pipeline.py")

    try:
        model = joblib.load(model_path)
        features = json.loads(feature_path.read_text(encoding="utf-8"))
        demand = build_demand_features(pd.read_csv(DATA_PATH))

        if demand.empty:
            raise ValueError("Demand feature table is empty.")

        latest = demand.iloc[-1]
        x = pd.DataFrame([[latest[feature] for feature in features]], columns=features)
        prediction = float(model.predict(x)[0])

        return {
            "predicted_demand": round(max(0.0, prediction), 2),
            "forecast_granularity": "hour",
            "forecast_for": (demand.index[-1] + pd.Timedelta(hours=1)).isoformat(),
            "latest_observed_orders": int(latest["orders"]),
            "features_used": features,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Demand forecast failed: {exc}") from exc


# ============================================================
# Restaurant discovery / recommendation
# ============================================================


def browse_catalog(
    top_k: int,
    user_lat: float,
    user_lon: float,
    min_rating: float | None,
    max_distance_km: float | None,
    pickup_hour: float,
    day_of_week: float,
    is_peak: int,
    vehicle_condition: float,
    multiple_deliveries: float,
    delivery_person_age: float,
):
    """Browse/rank the actual generated restaurant catalog for a blank query.

    This is intentionally used only for the generic Home-page browse state.
    It uses the same catalog, location calculation and trained ETA model as the
    recommendation engine, but does not require a text query.
    """
    catalog_path = MODEL_DIR / "restaurant_catalog.csv"
    df = pd.read_csv(catalog_path)

    if df.empty:
        return pd.DataFrame()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["orders"] = pd.to_numeric(df["orders"], errors="coerce").fillna(0)
    df["restaurant_lat"] = pd.to_numeric(df["restaurant_lat"], errors="coerce")
    df["restaurant_lon"] = pd.to_numeric(df["restaurant_lon"], errors="coerce")

    df["customer_distance_km"] = _haversine_km(
        float(user_lat),
        float(user_lon),
        df["restaurant_lat"],
        df["restaurant_lon"],
    )

    if max_distance_km is not None:
        df = df[df["customer_distance_km"] <= float(max_distance_km)].copy()

    if min_rating is not None:
        df = df[df["rating"] >= float(min_rating)].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "restaurant_name", "restaurant_proxy_id", "city", "order_type",
            "score", "rating", "orders", "customer_distance_km",
            "predicted_eta_min", "distance_km"
        ])

    # Rating and popularity come directly from the generated catalog.
    df["rating_score"] = _normalize(df["rating"])
    df["popularity_score"] = _normalize(np.log1p(df["orders"]))

    # Prefer nearby restaurants.
    df["distance_score"] = _normalize(1.0 / (1.0 + df["customer_distance_km"]))

    # Use the same trained ETA model used by the recommender.
    try:
        model, features = _load_eta_model()
        eta_input = pd.DataFrame(index=df.index)
        for feature in features:
            eta_input[feature] = 0.0

        if "distance_km" in features:
            eta_input["distance_km"] = df["customer_distance_km"].fillna(df["distance_km"])
        if "age" in features:
            eta_input["age"] = float(delivery_person_age)
        if "rating" in features:
            eta_input["rating"] = df["rating"]
        if "vehicle_condition" in features:
            eta_input["vehicle_condition"] = float(vehicle_condition)
        if "multiple_deliveries" in features:
            eta_input["multiple_deliveries"] = float(multiple_deliveries)
        if "pickup_hour" in features:
            eta_input["pickup_hour"] = float(pickup_hour)
        if "day_of_week" in features:
            eta_input["day_of_week"] = float(day_of_week)
        if "is_peak" in features:
            eta_input["is_peak"] = int(is_peak)

        eta_input = eta_input[features]
        df["predicted_eta_min"] = np.clip(model.predict(eta_input), 0, None)
        df["eta_score"] = _normalize(1.0 / (1.0 + df["predicted_eta_min"]))
        eta_available = True
    except Exception:
        # Home browsing remains usable even if the optional ETA artifact is broken.
        # The API health endpoint still exposes ETA artifact availability.
        df["predicted_eta_min"] = np.nan
        df["eta_score"] = 0.0
        eta_available = False

    # Blank-query browse: retain the recommender's same business signals,
    # excluding text relevance because there is no text query.
    if eta_available:
        df["score"] = (
            0.20 * df["rating_score"]
            + 0.10 * df["popularity_score"]
            + 0.20 * df["distance_score"]
            + 0.15 * df["eta_score"]
        )
    else:
        df["score"] = (
            0.20 * df["rating_score"]
            + 0.10 * df["popularity_score"]
            + 0.20 * df["distance_score"]
        )

    cols = [
        "restaurant_name", "restaurant_proxy_id", "city", "order_type",
        "score", "rating", "orders", "customer_distance_km",
        "predicted_eta_min", "distance_km"
    ]
    return df.sort_values(["score", "rating", "orders"], ascending=[False, False, False]).head(top_k)[cols].reset_index(drop=True), eta_available


def fallback_recommend(
    query,
    top_k,
    user_lat,
    user_lon,
    min_rating,
    max_distance_km,
    pickup_hour,
    day_of_week,
    is_peak,
    vehicle_condition,
    multiple_deliveries,
    delivery_person_age,
):
    """
    Resilient catalog-based fallback.

    Primary path remains src.recommender.recommend().
    This fallback uses the actual generated restaurant catalog and the
    trained ETA model when available, so the customer UI never invents
    restaurants just because an optional recommender artifact fails.
    """

    path = MODEL_DIR / "restaurant_catalog.csv"
    df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame(), False, "The restaurant catalog is empty."

    for col in ["rating", "orders", "restaurant_lat", "restaurant_lon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rating"] = df["rating"].fillna(0)
    df["orders"] = df["orders"].fillna(0)

    # Query relevance from the real catalog.
    q = (query or "").strip().lower()
    if q:
        order_type = df["order_type"].fillna("").astype(str).str.lower()
        city = df["city"].fillna("").astype(str).str.lower()

        exact = order_type.eq(q)
        contains = order_type.str.contains(q, na=False) | city.str.contains(q, na=False)

        df["relevance"] = 0.0
        df.loc[contains, "relevance"] = 0.5
        df.loc[exact, "relevance"] = 1.0

        # If there are no textual matches, return the closest/highest-rated
        # available catalog records rather than fabricating results.
        if not contains.any():
            df["relevance"] = 0.0
    else:
        df["relevance"] = 0.0

    # Customer distance from actual restaurant coordinates.
    if user_lat is not None and user_lon is not None:
        df["customer_distance_km"] = _haversine_km(
            float(user_lat),
            float(user_lon),
            df["restaurant_lat"].astype(float),
            df["restaurant_lon"].astype(float),
        )
    else:
        df["customer_distance_km"] = df["distance_km"].astype(float)

    if max_distance_km is not None:
        df = df[df["customer_distance_km"] <= float(max_distance_km)].copy()

    if min_rating is not None:
        df = df[df["rating"] >= float(min_rating)].copy()

    if df.empty:
        return pd.DataFrame(), False, "No catalog restaurants matched the selected filters."

    df["rating_score"] = _normalize(df["rating"])
    df["popularity_score"] = _normalize(np.log1p(df["orders"]))
    df["distance_score"] = _normalize(
        1.0 / (1.0 + df["customer_distance_km"].clip(lower=0))
    )

    # Try the project's trained ETA model.
    eta_available = False
    try:
        model, features = _load_eta_model()

        eta_input = pd.DataFrame(index=df.index)
        for feature in features:
            eta_input[feature] = 0.0

        if "distance_km" in features:
            eta_input["distance_km"] = df["customer_distance_km"].fillna(
                df["distance_km"]
            )
        if "age" in features:
            eta_input["age"] = float(delivery_person_age)
        if "rating" in features:
            eta_input["rating"] = df["rating"]
        if "vehicle_condition" in features:
            eta_input["vehicle_condition"] = float(vehicle_condition)
        if "multiple_deliveries" in features:
            eta_input["multiple_deliveries"] = float(multiple_deliveries)
        if "pickup_hour" in features:
            eta_input["pickup_hour"] = float(pickup_hour)
        if "day_of_week" in features:
            eta_input["day_of_week"] = float(day_of_week)
        if "is_peak" in features:
            eta_input["is_peak"] = int(is_peak)

        eta_input = eta_input[features]
        df["predicted_eta_min"] = np.clip(model.predict(eta_input), 0, None)
        df["eta_score"] = _normalize(
            1.0 / (1.0 + df["predicted_eta_min"])
        )
        eta_available = True
    except Exception:
        df["predicted_eta_min"] = np.nan
        df["eta_score"] = 0.0

    # Same general business signals as the main recommender.
    if user_lat is not None and user_lon is not None:
        df["score"] = (
            0.35 * df["relevance"]
            + 0.20 * df["rating_score"]
            + 0.10 * df["popularity_score"]
            + 0.20 * df["distance_score"]
            + 0.15 * df["eta_score"]
        )
    else:
        df["score"] = (
            0.45 * df["relevance"]
            + 0.25 * df["rating_score"]
            + 0.15 * df["popularity_score"]
            + 0.15 * df["eta_score"]
        )

    columns = [
        "restaurant_name",
        "restaurant_proxy_id",
        "city",
        "order_type",
        "score",
        "relevance",
        "rating_score",
        "popularity_score",
        "distance_score",
        "eta_score",
        "rating",
        "orders",
        "customer_distance_km",
        "predicted_eta_min",
        "distance_km",
    ]

    return (
        df.sort_values(
            ["score", "rating", "orders"],
            ascending=[False, False, False],
        )
        .head(top_k)
        .reset_index(drop=True)[columns],
        eta_available,
        "Primary recommender failed; catalog fallback was used.",
    )


@app.get("/recommend")
@app.get("/search")
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
    required_files = [
        MODEL_DIR / "restaurant_catalog.csv",
        MODEL_DIR / "eta_model.joblib",
        MODEL_DIR / "eta_features.json",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Required recommendation artifacts are missing.",
                "missing_files": missing,
            },
        )

    if not 1 <= top_k <= 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50.")
    if min_rating is not None and not 0 <= min_rating <= 5:
        raise HTTPException(status_code=400, detail="min_rating must be between 0 and 5.")
    if max_distance_km is not None and max_distance_km < 0:
        raise HTTPException(status_code=400, detail="max_distance_km cannot be negative.")
    if (user_lat is None) != (user_lon is None):
        raise HTTPException(
            status_code=400,
            detail="user_lat and user_lon must be provided together.",
        )
    if user_lat is not None and not (6 <= user_lat <= 37 and 68 <= user_lon <= 97):
        raise HTTPException(
            status_code=400,
            detail="Customer coordinates are outside supported India bounds.",
        )
    if not 0 <= pickup_hour <= 23:
        raise HTTPException(status_code=400, detail="pickup_hour must be between 0 and 23.")
    if not 0 <= day_of_week <= 6:
        raise HTTPException(status_code=400, detail="day_of_week must be between 0 and 6.")
    if is_peak not in [0, 1]:
        raise HTTPException(status_code=400, detail="is_peak must be either 0 or 1.")

    query = (query or "").strip()

    # Blank query = Home browse. Use the actual catalog directly.
    if not query:
        try:
            results, eta_available, note = fallback_recommend(
                query="",
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
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Home restaurant browse failed.",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            ) from exc

        return {
            "query": "",
            "top_k": top_k,
            "mode": "browse",
            "ranking_method": (
                "Actual catalog ranked by rating, popularity, distance "
                "and trained ETA where available."
            ),
            "eta_available": bool(eta_available),
            "note": note,
            "results": results.replace({np.nan: None}).to_dict(orient="records"),
        }

    # Search / For You: primary project recommender first.
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
        results = results.replace({np.nan: None})

        return {
            "query": query,
            "top_k": top_k,
            "mode": "recommendation",
            "engine": "src.recommender.recommend",
            "ranking_method": (
                "TF-IDF relevance + rating + popularity + distance + "
                "trained ETA hybrid ranking."
            ),
            "results": results.to_dict(orient="records"),
        }

    except Exception as primary_exc:
        # Never replace real restaurants with fake data. Fall back only to
        # the actual generated catalog.
        try:
            results, eta_available, note = fallback_recommend(
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

            return {
                "query": query,
                "top_k": top_k,
                "mode": "fallback",
                "engine": "catalog_fallback",
                "primary_recommender_error": {
                    "type": type(primary_exc).__name__,
                    "message": str(primary_exc),
                },
                "eta_available": bool(eta_available),
                "note": note,
                "results": results.replace({np.nan: None}).to_dict(orient="records"),
            }

        except Exception as fallback_exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Both the primary recommender and catalog fallback failed.",
                    "primary_error": {
                        "type": type(primary_exc).__name__,
                        "message": str(primary_exc),
                    },
                    "fallback_error": {
                        "type": type(fallback_exc).__name__,
                        "message": str(fallback_exc),
                    },
                },
            ) from fallback_exc


@app.get("/recommend/{user_id}")
def get_user_recommendations(user_id: str, query: str = "", top_k: int = 10):
    if not 1 <= top_k <= 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50.")

    try:
        results = recommend(query=query or "", top_k=top_k)
        results = results.replace({np.nan: None})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"User recommendation failed: {exc}") from exc

    return {
        "user_id": user_id,
        "query": query,
        "data_note": "Recommendations are generated from the available public-dataset catalog and current context; no fabricated user history is used.",
        "results": results.to_dict(orient="records"),
    }


@app.get("/catalog")
def catalog(limit: int = 50):
    """Return actual restaurant proxy records from the generated catalog."""
    path = MODEL_DIR / "restaurant_catalog.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Restaurant catalog not found. Run: python run_pipeline.py")

    if not 1 <= limit <= 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500.")

    df = pd.read_csv(path).replace({np.nan: None})
    return {
        "count": min(limit, len(df)),
        "total_catalog_records": int(len(df)),
        "data_note": "These restaurant records are coordinate-based proxies generated from the public delivery dataset.",
        "restaurants": df.head(limit).to_dict(orient="records"),
    }


@app.get("/restaurant/{restaurant_name}")
def get_restaurant_details(restaurant_name: str):
    path = MODEL_DIR / "restaurant_catalog.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Restaurant catalog not found. Run: python run_pipeline.py")

    df = pd.read_csv(path)
    if "restaurant_name" not in df.columns:
        raise HTTPException(status_code=500, detail="Restaurant catalog is missing restaurant_name.")

    target = restaurant_name.strip().lower()
    match = df[df["restaurant_name"].astype(str).str.lower() == target]

    if match.empty:
        # Also support partial lookup.
        match = df[df["restaurant_name"].astype(str).str.lower().str.contains(target, na=False)]

    if match.empty:
        raise HTTPException(status_code=404, detail="Restaurant not found in the generated catalog.")

    restaurant = match.iloc[0].replace({np.nan: None}).to_dict()
    order_type = str(restaurant.get("order_type") or "")

    try:
        similar = recommend(query=order_type, top_k=6).replace({np.nan: None})
        similar_records = similar.to_dict(orient="records")
    except Exception:
        similar_records = []

    return {
        "restaurant": restaurant,
        "similar": similar_records,
        "menu_available": False,
        "menu_note": "The source dataset does not contain reliable item-level menu data, so no fake menu is generated.",
    }


# ============================================================
# Demo profile / evaluation endpoints
# ============================================================

@app.get("/user/{user_id}/profile")
def get_user_profile(user_id: str):
    return {
        "user_id": user_id,
        "profile_type": "cold_start_demo",
        "note": "The public dataset does not provide trustworthy production user histories.",
        "preferred_contexts": ["Meal", "Snack", "Drinks", "Buffet"],
    }


@app.get("/ranking-metrics")
@app.get("/metrics")
def ranking_metrics():
    path = REPORT_DIR / "ranking_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Ranking evaluation not found.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/ab-test")
def ab_test():
    path = REPORT_DIR / "ab_test_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="A/B experiment results not found.")
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# Delivery assignment optimization
# ============================================================

@app.post("/optimize")
def optimize(request: OptimizationRequest):
    orders = pd.DataFrame(
        [{"distance_km": o.distance_km, "delivery_time": o.delivery_time} for o in request.orders]
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
        raise HTTPException(status_code=400, detail="At least one order is required.")
    if partners.empty:
        raise HTTPException(status_code=400, detail="At least one delivery partner is required.")

    try:
        result = optimize_assignments(orders, partners)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc

    return {
        "assigned_orders": int(len(result)),
        "assignments": result.to_dict(orient="records"),
    }
