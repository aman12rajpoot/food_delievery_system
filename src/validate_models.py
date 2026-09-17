import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.train_models import prepare_features
from src.optimization import optimize_assignments


DATA = Path("data/processed/clean_orders.csv")
MODELS = Path("models")
REPORTS = Path("reports")

ETA_FEATURES_FILE = MODELS / "eta_features.json"
DEMAND_FEATURES_FILE = MODELS / "demand_features.json"


# ============================================================
# ETA VALIDATION
# ============================================================

def validate_eta(df):
    print("\n" + "=" * 70)
    print("1. ETA MODEL VALIDATION")
    print("=" * 70)

    model_path = MODELS / "eta_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = joblib.load(model_path)

    X, features = prepare_features(df)

    y = pd.to_numeric(
        df["delivery_time"],
        errors="coerce"
    )

    valid = y.notna()

    X = X.loc[valid]
    y = y.loc[valid]

    # Reproduce exact chronological split used in training
    dates = pd.to_datetime(
        df.loc[valid, "date"],
        errors="coerce"
    )

    order = (
        dates
        .fillna(pd.Timestamp("1900-01-01"))
        .sort_values()
        .index
    )

    X = X.loc[order]
    y = y.loc[order]

    cut = max(
        1,
        int(len(X) * 0.8)
    )

    X_train = X.iloc[:cut]
    X_test = X.iloc[cut:]

    y_train = y.iloc[:cut]
    y_test = y.iloc[cut:]

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )

    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )

    train_rmse = mean_squared_error(
        y_train,
        train_pred
    ) ** 0.5

    test_rmse = mean_squared_error(
        y_test,
        test_pred
    ) ** 0.5

    train_r2 = r2_score(
        y_train,
        train_pred
    )

    test_r2 = r2_score(
        y_test,
        test_pred
    )

    residuals = y_test.values - test_pred

    residual_summary = {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "median": float(np.median(residuals)),
        "q05": float(np.percentile(residuals, 5)),
        "q25": float(np.percentile(residuals, 25)),
        "q75": float(np.percentile(residuals, 75)),
        "q95": float(np.percentile(residuals, 95)),
    }

    errors = pd.DataFrame({
        "actual": y_test.values,
        "predicted": test_pred,
    })

    errors["error"] = (
        errors["actual"] -
        errors["predicted"]
    )

    errors["absolute_error"] = (
        errors["error"].abs()
    )

    extreme_errors = (
        errors
        .sort_values(
            "absolute_error",
            ascending=False
        )
        .head(10)
    )

    # Sensibility checks
    negative_predictions = int(
        (test_pred < 0).sum()
    )

    unrealistic_predictions = int(
        ((test_pred < 5) | (test_pred > 120)).sum()
    )

    result = {
        "model": "XGBoost",
        "features": features,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train": {
            "MAE": float(train_mae),
            "RMSE": float(train_rmse),
            "R2": float(train_r2),
        },
        "test": {
            "MAE": float(test_mae),
            "RMSE": float(test_rmse),
            "R2": float(test_r2),
        },
        "generalization": {
            "r2_gap": float(train_r2 - test_r2),
            "mae_gap": float(test_mae - train_mae),
        },
        "residual_summary": residual_summary,
        "prediction_sanity": {
            "minimum_prediction": float(np.min(test_pred)),
            "maximum_prediction": float(np.max(test_pred)),
            "mean_prediction": float(np.mean(test_pred)),
            "negative_predictions": negative_predictions,
            "predictions_outside_5_to_120_min": unrealistic_predictions,
        },
        "extreme_errors": extreme_errors.to_dict(
            orient="records"
        ),
    }

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        REPORTS / "eta_validation.json"
    ).write_text(
        json.dumps(
            result,
            indent=2
        )
    )

    errors.to_csv(
        REPORTS / "eta_test_predictions.csv",
        index=False
    )

    print(f"Model: XGBoost")
    print(f"Features: {features}")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")

    print("\nTRAIN PERFORMANCE")
    print(f"MAE  : {train_mae:.4f}")
    print(f"RMSE : {train_rmse:.4f}")
    print(f"R²   : {train_r2:.4f}")

    print("\nTEST PERFORMANCE")
    print(f"MAE  : {test_mae:.4f}")
    print(f"RMSE : {test_rmse:.4f}")
    print(f"R²   : {test_r2:.4f}")

    print("\nGENERALIZATION")
    print(f"R² gap  : {train_r2 - test_r2:.4f}")
    print(f"MAE gap : {test_mae - train_mae:.4f}")

    print("\nRESIDUALS")
    for key, value in residual_summary.items():
        print(f"{key:>8}: {value:.4f}")

    print("\nPREDICTION SANITY")
    print(
        f"Min prediction: {np.min(test_pred):.2f} min"
    )
    print(
        f"Max prediction: {np.max(test_pred):.2f} min"
    )
    print(
        f"Mean prediction: {np.mean(test_pred):.2f} min"
    )
    print(
        f"Negative predictions: {negative_predictions}"
    )
    print(
        "Predictions outside 5-120 min:",
        unrealistic_predictions
    )

    print("\nTOP 10 EXTREME ERRORS")
    print(
        extreme_errors.to_string(
            index=False
        )
    )

    return result


# ============================================================
# DEMAND VALIDATION
# ============================================================

def build_demand_features(df):
    x = df.copy()

    if (
        "date" not in x.columns
        and "order_date" in x.columns
    ):
        x["date"] = x["order_date"]

    if (
        "pickup_hour" not in x.columns
        and "pickup_time" in x.columns
    ):
        parsed = pd.to_datetime(
            x["pickup_time"],
            format="%I:%M:%S %p",
            errors="coerce"
        )

        x["pickup_hour"] = parsed.dt.hour

    x = x.dropna(
        subset=["date", "pickup_hour"]
    ).copy()

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
        dayfirst=True
    )

    x["pickup_hour"] = pd.to_numeric(
        x["pickup_hour"],
        errors="coerce"
    )

    x = x.dropna(
        subset=["date", "pickup_hour"]
    )

    hourly = (
        x.groupby(
            ["date", "pickup_hour"]
        )
        .size()
        .reset_index(name="orders")
    )

    hourly["timestamp"] = (
        hourly["date"]
        + pd.to_timedelta(
            hourly["pickup_hour"],
            unit="h"
        )
    )

    hourly = (
        hourly
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    full = (
        hourly[["orders"]]
        .resample("h")
        .sum()
        .fillna(0)
    )

    for lag in [1, 2, 3, 24, 48, 168]:
        full[f"lag_{lag}"] = (
            full["orders"].shift(lag)
        )

    full["roll_24"] = (
        full["orders"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    full["hour"] = full.index.hour
    full["dow"] = full.index.dayofweek

    return full.dropna()


def validate_demand(df):
    print("\n" + "=" * 70)
    print("2. DEMAND FORECAST MODEL VALIDATION")
    print("=" * 70)

    model_path = MODELS / "demand_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = joblib.load(model_path)

    features = json.loads(
        DEMAND_FEATURES_FILE.read_text()
    )

    full = build_demand_features(df)

    cut = int(
        len(full) * 0.8
    )

    train = full.iloc[:cut]
    test = full.iloc[cut:]

    predictions = np.clip(
    model.predict(test[features]),
    0,
    None)


   

    baseline = test["lag_24"].values

    model_mae = mean_absolute_error(
        test["orders"],
        predictions
    )

    model_rmse = mean_squared_error(
        test["orders"],
        predictions
    ) ** 0.5

    baseline_mae = mean_absolute_error(
        test["orders"],
        baseline
    )

    baseline_rmse = mean_squared_error(
        test["orders"],
        baseline
    ) ** 0.5

    prediction_df = pd.DataFrame({
        "timestamp": test.index,
        "actual": test["orders"].values,
        "predicted": predictions,
        "lag_24_baseline": baseline,
    })

    prediction_df["model_error"] = (
        prediction_df["actual"]
        - prediction_df["predicted"]
    )

    prediction_df["baseline_error"] = (
        prediction_df["actual"]
        - prediction_df["lag_24_baseline"]
    )

    result = {
        "granularity": "hour",
        "total_observations": int(len(full)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "chronological_split": True,
        "model": {
            "MAE": float(model_mae),
            "RMSE": float(model_rmse),
        },
        "baseline_lag_24": {
            "MAE": float(baseline_mae),
            "RMSE": float(baseline_rmse),
        },
        "improvement_vs_baseline": {
            "MAE_reduction": float(
                baseline_mae - model_mae
            ),
            "MAE_reduction_percent": float(
                (baseline_mae - model_mae)
                / baseline_mae
                * 100
            ),
            "RMSE_reduction": float(
                baseline_rmse - model_rmse
            ),
            "RMSE_reduction_percent": float(
                (baseline_rmse - model_rmse)
                / baseline_rmse
                * 100
            ),
        },
        "prediction_sanity": {
            "min_prediction": float(
                np.min(predictions)
            ),
            "max_prediction": float(
                np.max(predictions)
            ),
            "mean_prediction": float(
                np.mean(predictions)
            ),
            "negative_predictions": int(
                (predictions < 0).sum()
            ),
        },
    }

    (
        REPORTS / "demand_validation.json"
    ).write_text(
        json.dumps(
            result,
            indent=2
        )
    )

    prediction_df.to_csv(
        REPORTS / "demand_test_predictions.csv",
        index=False
    )

    print(
        f"Total observations: {len(full):,}"
    )
    print(
        f"Train rows: {len(train):,}"
    )
    print(
        f"Test rows: {len(test):,}"
    )

    print("\nMODEL")
    print(f"MAE  : {model_mae:.4f}")
    print(f"RMSE : {model_rmse:.4f}")

    print("\nLAG-24 BASELINE")
    print(f"MAE  : {baseline_mae:.4f}")
    print(f"RMSE : {baseline_rmse:.4f}")

    print("\nIMPROVEMENT VS BASELINE")
    print(
        f"MAE reduction : "
        f"{baseline_mae - model_mae:.4f}"
    )
    print(
        f"MAE reduction % : "
        f"{(baseline_mae - model_mae) / baseline_mae * 100:.2f}%"
    )
    print(
        f"RMSE reduction : "
        f"{baseline_rmse - model_rmse:.4f}"
    )
    print(
        f"RMSE reduction % : "
        f"{(baseline_rmse - model_rmse) / baseline_rmse * 100:.2f}%"
    )

    print("\nPREDICTION SANITY")
    print(
        f"Min prediction : {np.min(predictions):.2f}"
    )
    print(
        f"Max prediction : {np.max(predictions):.2f}"
    )
    print(
        f"Mean prediction: {np.mean(predictions):.2f}"
    )
    print(
        f"Negative predictions: "
        f"{(predictions < 0).sum()}"
    )

    return result


# ============================================================
# RECOMMENDATION VALIDATION
# ============================================================

def validate_recommendation():
    print("\n" + "=" * 70)
    print("3. RECOMMENDATION VALIDATION")
    print("=" * 70)

    path = REPORTS / "ranking_metrics.json"

    if not path.exists():
        print("Ranking report not found.")
        return None

    metrics = json.loads(
        path.read_text()
    )

    print(
        "Evaluation type:",
        metrics.get("evaluation")
    )

    print(
        "Queries evaluated:",
        metrics.get("queries_evaluated")
    )

    print(
        f"Mean Precision@10: "
        f"{metrics.get('mean_precision_at_k', 0):.4f}"
    )

    print(
        f"Mean Recall@10: "
        f"{metrics.get('mean_recall_at_k', 0):.4f}"
    )

    print(
        f"Mean NDCG@10: "
        f"{metrics.get('mean_ndcg_at_k', 0):.4f}"
    )

    print("\nIMPORTANT:")
    print(
        "Recommendation evaluation is an OFFLINE PROXY evaluation."
    )
    print(
        "The public dataset does not contain genuine "
        "user-restaurant interaction history."
    )
    print(
        "Therefore these metrics must not be presented "
        "as production personalized recommendation performance."
    )

    return metrics


# ============================================================
# OPTIMIZATION VALIDATION
# ============================================================

def validate_optimization(df):
    print("\n" + "=" * 70)
    print("4. DELIVERY OPTIMIZATION VALIDATION")
    print("=" * 70)

    sample = df[
        ["distance_km", "delivery_time"]
    ].dropna().head(20).copy()

    sample = sample.reset_index(
        drop=True
    )

    rng = np.random.default_rng(42)

    partners = pd.DataFrame({
        "partner_id": [
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
        ],
        "base_distance": rng.uniform(
            1,
            6,
            6
        ),
        "capacity": [
            4,
            4,
            4,
            4,
            4,
            4,
        ],
    })

    result = optimize_assignments(
        sample,
        partners
    )

    assigned_orders = len(result)
    total_orders = len(sample)

    unique_orders = (
        result["order_id"]
        .nunique()
    )

    duplicate_orders = (
        result["order_id"]
        .duplicated()
        .sum()
    )

    partner_load = (
        result
        .groupby("partner_id")
        .size()
    )

    capacity_lookup = (
        partners
        .set_index("partner_id")["capacity"]
    )

    capacity_violations = []

    for partner, load in partner_load.items():
        capacity = capacity_lookup.loc[
            partner
        ]

        if load > capacity:
            capacity_violations.append({
                "partner_id": partner,
                "assigned": int(load),
                "capacity": int(capacity),
            })

    average_cost = float(
        result["assignment_cost"].mean()
    )

    result.to_csv(
        REPORTS / "optimization_validation.csv",
        index=False
    )

    validation = {
        "total_orders": int(total_orders),
        "assigned_orders": int(assigned_orders),
        "unique_orders_assigned": int(unique_orders),
        "duplicate_order_assignments": int(
            duplicate_orders
        ),
        "all_orders_assigned": bool(
    assigned_orders == total_orders
    ),
    "exactly_one_assignment_per_order": bool(
        unique_orders == total_orders
        and duplicate_orders == 0
    ),
        "capacity_violations": capacity_violations,
        "capacity_constraints_satisfied": bool(
            len(capacity_violations) == 0
        
),
       
        "average_assignment_cost": average_cost,
    }

    (
        REPORTS / "optimization_validation.json"
    ).write_text(
        json.dumps(
            validation,
            indent=2
        )
    )

    print(
        f"Total orders: {total_orders}"
    )

    print(
        f"Assigned orders: {assigned_orders}"
    )

    print(
        f"Unique orders: {unique_orders}"
    )

    print(
        f"Duplicate assignments: {duplicate_orders}"
    )

    print(
        "All orders assigned:",
        validation["all_orders_assigned"]
    )

    print(
        "Exactly one assignment/order:",
        validation[
            "exactly_one_assignment_per_order"
        ]
    )

    print(
        "Capacity violations:",
        len(capacity_violations)
    )

    print(
        f"Average assignment cost: "
        f"{average_cost:.4f}"
    )

    print("\nPARTNER UTILIZATION")

    utilization = (
        result
        .groupby("partner_id")
        .size()
        .reset_index(name="assigned_orders")
    )

    utilization["capacity"] = (
        utilization["partner_id"]
        .map(capacity_lookup)
    )

    utilization["utilization_pct"] = (
        utilization["assigned_orders"]
        / utilization["capacity"]
        * 100
    )

    print(
        utilization.to_string(
            index=False
        )
    )

    return validation


# ============================================================
# MAIN
# ============================================================

def main():

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    df = pd.read_csv(DATA)

    REPORTS.mkdir(
        parents=True,
        exist_ok=True
    )

    eta = validate_eta(df)

    demand = validate_demand(df)

    recommendation = (
        validate_recommendation()
    )

    optimization = (
        validate_optimization(df)
    )

    final = {
        "eta": eta,
        "demand": demand,
        "recommendation": recommendation,
        "optimization": optimization,
    }

    (
        REPORTS / "full_model_validation.json"
    ).write_text(
        json.dumps(
            final,
            indent=2
        )
    )

    print("\n" + "=" * 70)
    print("FULL MODEL VALIDATION COMPLETE")
    print("=" * 70)

    print(
        "\nReports created in:"
        "\nreports/"
    )

    print(
        "\nKey files:"
        "\n- eta_validation.json"
        "\n- eta_test_predictions.csv"
        "\n- demand_validation.json"
        "\n- demand_test_predictions.csv"
        "\n- optimization_validation.json"
        "\n- optimization_validation.csv"
        "\n- full_model_validation.json"
    )


if __name__ == "__main__":
    main()