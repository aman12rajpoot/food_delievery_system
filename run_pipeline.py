from pathlib import Path
import json
import pandas as pd

from src.download_data import main as download
from src.etl import clean
from src.train_models import main as train
from src.recommender import (
    build_catalog,
    recommend,
    ranking_metrics,
)
from src.optimization import demo
from src.ab_test import run as ab_run


REPORT_DIR = Path("reports")


def evaluate_ranking(df, top_k=10):
    """
    Offline ranking evaluation.

    Because the public dataset does not contain user-item interaction
    history, relevance is approximated using observed restaurant
    location proxies for each order type.

    This should be described as an offline proxy evaluation,
    not as production recommendation performance.
    """

    catalog, _, _ = build_catalog(df)

    results = []

    # Evaluate several order-type queries.
    order_types = (
        df["order_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    for order_type in order_types:

        query = str(order_type)

        # Get recommendations.
        recs = recommend(
            query,
            top_k=top_k,
        )

        # Determine relevant restaurant proxies for this category.
        relevant = (
            catalog[
                catalog["order_type"]
                .astype(str)
                .str.lower()
                == query.lower()
            ]["restaurant_name"]
            .head(top_k)
            .tolist()
        )

        metrics = ranking_metrics(
            recs,
            relevant,
            k=top_k,
        )

        metrics["query"] = query
        results.append(metrics)

    result_df = pd.DataFrame(results)

    summary = {
        "evaluation": "offline_proxy",
        "k": top_k,
        "queries_evaluated": len(result_df),
        "metrics_by_query": results,
        "mean_precision_at_k": (
            float(
                result_df["Precision@K"].mean()
            )
            if not result_df.empty
            else 0.0
        ),
        "mean_recall_at_k": (
            float(
                result_df["Recall@K"].mean()
            )
            if not result_df.empty
            else 0.0
        ),
        "mean_ndcg_at_k": (
            float(
                result_df["NDCG@K"].mean()
            )
            if not result_df.empty
            else 0.0
        ),
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_DIR / "ranking_metrics.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    return summary


def main():

    try:
        download()

    except Exception as e:
        print(
            "Automatic download failed:",
            e,
        )
        print(
            "Place the public CSV at "
            "data/raw/Zomato Dataset.csv "
            "and rerun."
        )
        return

    # -------------------------
    # 1. ETL
    # -------------------------

    df = clean()

    # -------------------------
    # 2. ETA + Demand
    # -------------------------

    eta, demand = train()

    # -------------------------
    # 3. Recommendation
    # -------------------------

    build_catalog(df)

    ranking = evaluate_ranking(
        df,
        top_k=10,
    )

    # -------------------------
    # 4. Optimization
    # -------------------------

    optimization = demo(df)

    # -------------------------
    # 5. A/B experiment
    # -------------------------

    ab = ab_run()

    # -------------------------
    # Final output
    # -------------------------

    print(
        f"Best ETA model: "
        f"{eta['best_model']}"
    )

    print(
        "Demand observations:",
        demand.get(
            "demand_observations",
            "n/a",
        ),
    )

    print(
        "Ranking:",
        ranking,
    )

    print(
        "Optimization assignments:",
        len(optimization),
    )

    print(
        "\nPIPELINE COMPLETE"
    )

    print(
        "ETA:",
        eta,
    )

    print(
        "Demand:",
        demand,
    )

    print(
        "Ranking:",
        ranking,
    )

    print(
        "A/B:",
        ab,
    )


if __name__ == "__main__":
    main()