from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_DIR = Path("models")
PROC = Path("data/processed/clean_orders.csv")

ETA_MODEL_PATH = MODEL_DIR / "eta_model.joblib"
ETA_FEATURES_PATH = MODEL_DIR / "eta_features.json"


def _mode_value(series, default="unknown"):
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else default


def _haversine_km(lat1, lon1, lat2, lon2):
    """Calculate straight-line geographic distance in kilometers."""

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _normalize(series):
    """Min-max normalization."""

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)

    minimum = series.min()
    maximum = series.max()

    if maximum <= minimum:
        return pd.Series(
            np.ones(len(series)),
            index=series.index,
        )

    return (series - minimum) / (maximum - minimum)


def _load_eta_model():
    """Load the existing trained ETA model and feature list."""

    if not ETA_MODEL_PATH.exists():
        raise FileNotFoundError(
            "ETA model not found. Run python src\\train_models.py first."
        )

    if not ETA_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "ETA feature file not found. Run python src\\train_models.py first."
        )

    model = joblib.load(ETA_MODEL_PATH)

    features = json.loads(
        ETA_FEATURES_PATH.read_text()
    )

    return model, features


def build_catalog(df):
    """
    Build a restaurant proxy catalog from the public Zomato dataset.

    The dataset does not contain real restaurant IDs/names.
    Therefore, coordinate-based restaurant proxies are used.

    The catalog contains:
    - restaurant location
    - city
    - order type
    - rating
    - popularity
    - historical delivery distance
    """

    df = df.copy()

    required = [
        "restaurant_lat",
        "restaurant_lon",
        "city",
        "order_type",
        "rating",
        "distance_km",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns for recommender: {missing}"
        )

    # Keep only geographically valid restaurant coordinates.
    df = df[
        df["restaurant_lat"].between(6, 37)
        & df["restaurant_lon"].between(68, 97)
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid restaurant coordinates available "
            "for recommendation catalog."
        )

    # Create coordinate-based restaurant proxy.
    df["restaurant_proxy_id"] = (
        df["restaurant_lat"].round(4).astype(str)
        + "_"
        + df["restaurant_lon"].round(4).astype(str)
    )

    agg = (
        df.groupby(
            "restaurant_proxy_id",
            as_index=False,
        )
        .agg(
            rating=("rating", "mean"),
            orders=("restaurant_proxy_id", "size"),
            distance_km=("distance_km", "mean"),
            city=("city", _mode_value),
            order_type=("order_type", _mode_value),
            restaurant_lat=(
                "restaurant_lat",
                "mean",
            ),
            restaurant_lon=(
                "restaurant_lon",
                "mean",
            ),
        )
    )

    agg["restaurant_name"] = (
        "RestaurantProxy_"
        + agg["restaurant_proxy_id"].astype(str)
    )

    # Text used for food/order-type relevance.
    agg["text"] = (
        agg["city"].fillna("").astype(str)
        + " "
        + agg["order_type"].fillna("").astype(str)
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
    )

    matrix = vectorizer.fit_transform(
        agg["text"]
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "restaurant_vectorizer.joblib",
    )

    joblib.dump(
        matrix,
        MODEL_DIR / "restaurant_matrix.joblib",
    )

    agg.to_csv(
        MODEL_DIR / "restaurant_catalog.csv",
        index=False,
    )

    return agg, vectorizer, matrix


def _predict_eta(
    cat,
    pickup_hour,
    day_of_week,
    is_peak,
    vehicle_condition,
    multiple_deliveries,
    delivery_person_age,
):
    """
    Predict ETA for each restaurant candidate using
    the existing trained ETA model.

    Customer-to-restaurant distance is used as the
    distance feature for recommendation-time scoring.

    Note:
    This is straight-line geographic distance, not
    actual road-network distance.
    """

    model, features = _load_eta_model()

    # Create ETA input using the actual model feature list.
    eta_input = pd.DataFrame(index=cat.index)

    for feature in features:
        eta_input[feature] = 0.0

    # Distance comes from the customer's location.
    if "distance_km" in features:
        eta_input["distance_km"] = (
            cat["customer_distance_km"]
            .fillna(cat["distance_km"])
        )

    if "age" in features:
        eta_input["age"] = float(
            delivery_person_age
        )

    if "rating" in features:
        eta_input["rating"] = cat["rating"]

    if "vehicle_condition" in features:
        eta_input["vehicle_condition"] = float(
            vehicle_condition
        )

    if "multiple_deliveries" in features:
        eta_input["multiple_deliveries"] = float(
            multiple_deliveries
        )

    if "pickup_hour" in features:
        eta_input["pickup_hour"] = float(
            pickup_hour
        )

    if "day_of_week" in features:
        eta_input["day_of_week"] = float(
            day_of_week
        )

    if "is_peak" in features:
        eta_input["is_peak"] = int(
            is_peak
        )

    eta_input = eta_input[
        features
    ]

    predictions = model.predict(
        eta_input
    )

    # ETA cannot be negative.
    predictions = np.clip(
        predictions,
        0,
        None,
    )

    return predictions


def recommend(
    query="",
    top_k=10,
    user_lat=None,
    user_lon=None,
    min_rating=None,
    max_distance_km=None,
    pickup_hour=19,
    day_of_week=4,
    is_peak=1,
    vehicle_condition=1,
    multiple_deliveries=0,
    delivery_person_age=30,
):
    """
    Context-aware restaurant recommendation.

    Handles cold-start users through explicit context.

    Parameters
    ----------
    query:
        Food/order type/city preference.

    top_k:
        Number of recommendations.

    user_lat, user_lon:
        Customer location.

    min_rating:
        Optional minimum restaurant rating.

    max_distance_km:
        Hard maximum customer-to-restaurant distance.

    pickup_hour:
        Expected pickup/order hour.

    day_of_week:
        Monday=0 ... Sunday=6.

    is_peak:
        1 for peak period, 0 otherwise.

    vehicle_condition:
        Delivery vehicle condition used by ETA model.

    multiple_deliveries:
        Current number of multiple deliveries.

    delivery_person_age:
        Delivery partner age used by ETA model.
    """

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    try:
        top_k = max(1, int(top_k))
    except (TypeError, ValueError):
        top_k = 10

    cat = pd.read_csv(
        MODEL_DIR / "restaurant_catalog.csv"
    )

    vectorizer = joblib.load(
        MODEL_DIR / "restaurant_vectorizer.joblib"
    )

    matrix = joblib.load(
        MODEL_DIR / "restaurant_matrix.joblib"
    )

    # ---------------------------------------------------------
    # 1. Food/order relevance
    # ---------------------------------------------------------

    query = str(query or "").strip()

    if query:
        query_vector = vectorizer.transform(
            [query]
        )

        similarity = cosine_similarity(
            query_vector,
            matrix,
        ).ravel()
    else:
        similarity = np.zeros(len(cat))

    cat["relevance"] = similarity

    # ---------------------------------------------------------
    # 2. Candidate retrieval
    # ---------------------------------------------------------
    # If an exact order type has enough candidates,
    # recommend from that category first.
    #
    # Example:
    # "Meal" -> Meal restaurants
    # "Snack" -> Snack restaurants
    # "Drinks" -> Drinks restaurants
    # "Buffet" -> Buffet restaurants
    #
    # If fewer than top_k exact matches exist,
    # retain broader candidate pool.

    if query:
        normalized_query = query.strip().lower()

        exact_match = (
            cat["order_type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            == normalized_query
        )

        if exact_match.sum() >= top_k:
            cat = cat[exact_match].copy()
            similarity = similarity[
                exact_match.to_numpy()
            ]

            cat["relevance"] = similarity

    # ---------------------------------------------------------
    # 3. Rating
    # ---------------------------------------------------------

    cat["rating"] = pd.to_numeric(
        cat["rating"],
        errors="coerce",
    ).fillna(0)

    cat["rating_score"] = _normalize(
        cat["rating"]
    )

    # ---------------------------------------------------------
    # 4. Popularity
    # ---------------------------------------------------------

    cat["orders"] = pd.to_numeric(
        cat["orders"],
        errors="coerce",
    ).fillna(0)

    popularity = np.log1p(
        cat["orders"]
    )

    cat["popularity_score"] = _normalize(
        popularity
    )

    # ---------------------------------------------------------
    # 5. Customer location
    # ---------------------------------------------------------

    has_location = (
        user_lat is not None
        and user_lon is not None
    )

    if has_location:

        try:
            user_lat = float(user_lat)
            user_lon = float(user_lon)

            # Validate customer coordinates.
            if not (
                6 <= user_lat <= 37
                and 68 <= user_lon <= 97
            ):
                has_location = False

        except (TypeError, ValueError):
            has_location = False

    if has_location:

        cat["customer_distance_km"] = (
            _haversine_km(
                user_lat,
                user_lon,
                cat["restaurant_lat"].astype(float),
                cat["restaurant_lon"].astype(float),
            )
        )

    else:

        cat["customer_distance_km"] = np.nan

    # ---------------------------------------------------------
    # 6. Hard distance filter
    # ---------------------------------------------------------

    if (
        has_location
        and max_distance_km is not None
    ):

        try:
            max_distance_km = float(
                max_distance_km
            )
        except (TypeError, ValueError):
            max_distance_km = None

        if (
            max_distance_km is not None
            and max_distance_km >= 0
        ):

            cat = cat[
                cat["customer_distance_km"]
                <= max_distance_km
            ].copy()

            # Hard constraint:
            # never return restaurants outside radius.
            if cat.empty:
                return pd.DataFrame(
                    columns=[
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
                )

    # ---------------------------------------------------------
    # 7. Distance score
    # ---------------------------------------------------------

    if has_location:

        distance = cat[
            "customer_distance_km"
        ].clip(lower=0)

        # Higher score = closer restaurant.
        cat["distance_score"] = (
            1 / (1 + distance)
        )

    else:

        cat["distance_score"] = 0.0

    # ---------------------------------------------------------
    # 8. ETA prediction
    # ---------------------------------------------------------

    cat["predicted_eta_min"] = _predict_eta(
        cat=cat,
        pickup_hour=pickup_hour,
        day_of_week=day_of_week,
        is_peak=is_peak,
        vehicle_condition=vehicle_condition,
        multiple_deliveries=multiple_deliveries,
        delivery_person_age=delivery_person_age,
    )

    # Lower ETA is better.
    cat["eta_score"] = (
        1 / (
            1 + cat["predicted_eta_min"]
        )
    )

    # ---------------------------------------------------------
    # 9. Minimum rating filter
    # ---------------------------------------------------------

    if min_rating is not None:

        try:
            min_rating = float(
                min_rating
            )
        except (TypeError, ValueError):
            min_rating = None

        if min_rating is not None:

            cat = cat[
                cat["rating"] >= min_rating
            ].copy()

            # No restaurants satisfy rating requirement.
            if cat.empty:
                return pd.DataFrame(
                    columns=[
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
                )

    # ---------------------------------------------------------
    # 10. Hybrid ranking
    # ---------------------------------------------------------
    #
    # Location-aware ranking:
    #
    # Relevance       = 35%
    # Rating          = 20%
    # Popularity      = 10%
    # Distance        = 20%
    # ETA             = 15%
    #
    # Without location:
    #
    # Relevance       = 45%
    # Rating          = 25%
    # Popularity      = 15%
    # ETA             = 15%

    if has_location:

        cat["distance_score"] = _normalize(
            cat["distance_score"]
        )

        cat["eta_score"] = _normalize(
            cat["eta_score"]
        )

        cat["score"] = (
            0.35 * cat["relevance"]
            + 0.20 * cat["rating_score"]
            + 0.10 * cat["popularity_score"]
            + 0.20 * cat["distance_score"]
            + 0.15 * cat["eta_score"]
        )

    else:

        cat["eta_score"] = _normalize(
            cat["eta_score"]
        )

        cat["score"] = (
            0.45 * cat["relevance"]
            + 0.25 * cat["rating_score"]
            + 0.15 * cat["popularity_score"]
            + 0.15 * cat["eta_score"]
        )

    # ---------------------------------------------------------
    # 11. Final ranking
    # ---------------------------------------------------------

    cat = (
        cat.sort_values(
            [
                "score",
                "rating",
                "orders",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 12. Explainable recommendation output
    # ---------------------------------------------------------
    #
    # These components allow us to explain WHY a restaurant
    # received its final ranking score.

    return cat[
        [
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
    ]


def ranking_metrics(
    recs,
    relevant_names,
    k=10,
):
    """
    Offline ranking evaluation.

    This requires externally defined relevant items.
    It does not represent actual user behavior unless
    real user interaction data is available.
    """

    if recs is None or recs.empty:
        return {
            "Precision@K": 0,
            "Recall@K": 0,
            "NDCG@K": 0,
        }

    k = min(
        int(k),
        len(recs),
    )

    ranked = list(
        recs["restaurant_name"].head(k)
    )

    relevant = set(
        relevant_names
    )

    if not relevant:
        return {
            "Precision@K": 0,
            "Recall@K": 0,
            "NDCG@K": 0,
        }

    hits = [
        1 if x in relevant else 0
        for x in ranked
    ]

    precision = (
        sum(hits) / k
        if k > 0
        else 0
    )

    recall = (
        sum(hits) / len(relevant)
        if relevant
        else 0
    )

    dcg = sum(
        h / np.log2(i + 2)
        for i, h in enumerate(hits)
    )

    ideal = sum(
        1 / np.log2(i + 2)
        for i in range(
            min(len(relevant), k)
        )
    )

    ndcg = (
        dcg / ideal
        if ideal
        else 0
    )

    return {
        "Precision@K": precision,
        "Recall@K": recall,
        "NDCG@K": ndcg,
    }


if __name__ == "__main__":

    df = pd.read_csv(
        PROC
    )

    build_catalog(
        df
    )

    print(
        "\n--- DEFAULT ETA-AWARE RECOMMENDATION ---"
    )

    print(
        recommend(
            query="Meal",
            top_k=10,
        ).to_string(
            index=False
        )
    )

    print(
        "\n--- LOCATION + ETA-AWARE RECOMMENDATION ---"
    )

    print(
        recommend(
            query="Meal",
            top_k=10,
            user_lat=23.3538,
            user_lon=85.3270,
            min_rating=4.0,
            max_distance_km=20,
            pickup_hour=19,
            day_of_week=4,
            is_peak=1,
        ).to_string(
            index=False
        )
    )