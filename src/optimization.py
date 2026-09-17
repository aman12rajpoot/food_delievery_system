from pathlib import Path
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp


def optimize_assignments(orders, partners):
    """
    Assign every order to exactly one delivery partner,
    subject to partner capacity.

    Partner data is synthetic and used only to demonstrate
    dispatch optimization at portfolio scale.
    """

    orders = orders.reset_index(drop=True).copy()
    partners = partners.reset_index(drop=True).copy()

    n = len(orders)
    m = len(partners)

    if n == 0 or m == 0:
        return pd.DataFrame()

    # Check total capacity.
    total_capacity = partners["capacity"].sum()

    if total_capacity < n:
        raise ValueError(
            f"Partner capacity ({total_capacity}) is lower "
            f"than number of orders ({n})."
        )

    costs = np.zeros((n, m))

    for i in range(n):
        for j in range(m):

            distance = abs(
                float(orders.loc[i, "distance_km"])
                - float(partners.loc[j, "base_distance"])
            )

            predicted_delay = (
                float(orders.loc[i, "delivery_time"])
                + 2 * distance
            )

            costs[i, j] = (
                distance
                + 0.15 * predicted_delay
            )

    solver = pywraplp.Solver.CreateSolver("SCIP")

    if solver is None:
        raise RuntimeError(
            "OR-Tools SCIP solver unavailable."
        )

    x = {
        (i, j): solver.BoolVar(f"x_{i}_{j}")
        for i in range(n)
        for j in range(m)
    }

    # Every order MUST be assigned exactly once.
    for i in range(n):
        solver.Add(
            sum(x[i, j] for j in range(m)) == 1
        )

    # Partner capacity.
    for j in range(m):
        capacity = int(
            partners.loc[j, "capacity"]
        )

        solver.Add(
            sum(x[i, j] for i in range(n))
            <= capacity
        )

    solver.Minimize(
        sum(
            costs[i, j] * x[i, j]
            for i in range(n)
            for j in range(m)
        )
    )

    status = solver.Solve()

    if status not in (
        pywraplp.Solver.OPTIMAL,
        pywraplp.Solver.FEASIBLE,
    ):
        return pd.DataFrame()

    rows = []

    for i in range(n):
        for j in range(m):

            if x[i, j].solution_value() > 0.5:

                rows.append(
                    {
                        "order_id": i,
                        "partner_id": partners.loc[
                            j, "partner_id"
                        ],
                        "assignment_cost": round(
                            costs[i, j], 4
                        ),
                        "order_distance_km": float(
                            orders.loc[
                                i, "distance_km"
                            ]
                        ),
                        "predicted_delivery_time": float(
                            orders.loc[
                                i, "delivery_time"
                            ]
                        ),
                    }
                )

    return pd.DataFrame(rows)


def demo(df, n_orders=20, n_partners=6):

    orders = (
        df.dropna(
            subset=[
                "distance_km",
                "delivery_time",
            ]
        )
        .sample(
            min(n_orders, len(df)),
            random_state=42,
        )
        .copy()
    )

    partners = pd.DataFrame(
        {
            "partner_id": [
                f"P{i+1}"
                for i in range(n_partners)
            ],
            "base_distance": np.linspace(
                0.5,
                8,
                n_partners,
            ),
            "capacity": [
                max(
                    4,
                    int(
                        np.ceil(
                            len(orders)
                            / n_partners
                        )
                    ),
                )
                for _ in range(n_partners)
            ],
        }
    )

    result = optimize_assignments(
        orders[
            [
                "distance_km",
                "delivery_time",
            ]
        ],
        partners,
    )

    Path("reports").mkdir(
        exist_ok=True
    )

    result.to_csv(
        "reports/optimized_assignments.csv",
        index=False,
    )

    return result


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/clean_orders.csv"
    )

    result = demo(df)

    print(
        result.to_string(index=False)
    )