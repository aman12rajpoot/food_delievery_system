from pathlib import Path
import json

import numpy as np
from scipy.stats import norm


def run(n=12000, seed=42):
    """
    Run an offline/simulated A/B experiment comparing:

    Control:
        Popularity-based baseline ranking

    Treatment:
        Context-aware hybrid ranking using relevance,
        rating, popularity, distance and predicted ETA.

    Important:
        This is a simulated experiment. It does not represent
        causal results from real production users.
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 1. Create experiment groups
    # ---------------------------------------------------------
    control_n = n // 2
    treatment_n = n - control_n

    group = np.array(["control"] * control_n + ["treatment"] * treatment_n)
    rng.shuffle(group)

    # ---------------------------------------------------------
    # 2. Simulated behavioral assumptions
    # ---------------------------------------------------------
    # These are assumed probabilities for an offline
    # demonstration of experimentation methodology.
    # They are NOT measured results from real users.
    p_control_conversion = 0.085
    p_treatment_conversion = 0.098

    p_control_ctr = 0.210
    p_treatment_ctr = 0.235

    # ---------------------------------------------------------
    # 3. Simulate conversion
    # ---------------------------------------------------------
    conversion_random = rng.random(n)
    conversion = np.where(
        group == "control",
        conversion_random < p_control_conversion,
        conversion_random < p_treatment_conversion,
    )

    # ---------------------------------------------------------
    # 4. Simulate click-through rate
    # ---------------------------------------------------------
    ctr_random = rng.random(n)
    clicked = np.where(
        group == "control",
        ctr_random < p_control_ctr,
        ctr_random < p_treatment_ctr,
    )

    # ---------------------------------------------------------
    # 5. Simulate revenue
    # ---------------------------------------------------------
    # Revenue is generated only for converted users.
    # A log-normal distribution gives a realistic
    # right-skewed order-value distribution.
    order_value = rng.lognormal(mean=np.log(320), sigma=0.45, size=n)
    revenue = np.where(conversion, order_value, 0.0)

    # ---------------------------------------------------------
    # 6. Split groups
    # ---------------------------------------------------------
    control = group == "control"
    treatment = group == "treatment"

    # ---------------------------------------------------------
    # 7. Helper functions
    # ---------------------------------------------------------
    def rate(values):
        if len(values) == 0:
            return 0.0
        return float(values.mean())

    def safe_relative_lift(treatment_value, control_value):
        if control_value == 0:
            return 0.0
        return (treatment_value - control_value) / control_value

    # ---------------------------------------------------------
    # 8. Conversion metrics
    # ---------------------------------------------------------
    control_conversion = rate(conversion[control])
    treatment_conversion = rate(conversion[treatment])
    absolute_conversion_lift = treatment_conversion - control_conversion
    relative_conversion_lift = safe_relative_lift(treatment_conversion, control_conversion)

    # ---------------------------------------------------------
    # 9. Statistical significance for conversion
    # ---------------------------------------------------------
    control_size = int(control.sum())
    treatment_size = int(treatment.sum())

    standard_error = np.sqrt(
        (control_conversion * (1 - control_conversion) / control_size)
        + (treatment_conversion * (1 - treatment_conversion) / treatment_size)
    )

    z_score = absolute_conversion_lift / standard_error if standard_error > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    # ---------------------------------------------------------
    # 10. 95% confidence interval
    # ---------------------------------------------------------
    ci_lower = absolute_conversion_lift - 1.96 * standard_error
    ci_upper = absolute_conversion_lift + 1.96 * standard_error

    # ---------------------------------------------------------
    # 11. CTR metrics
    # ---------------------------------------------------------
    control_ctr = rate(clicked[control])
    treatment_ctr = rate(clicked[treatment])
    absolute_ctr_lift = treatment_ctr - control_ctr
    relative_ctr_lift = safe_relative_lift(treatment_ctr, control_ctr)

    # ---------------------------------------------------------
    # 12. AOV
    # ---------------------------------------------------------
    control_converted = conversion[control]
    treatment_converted = conversion[treatment]
    control_revenue = revenue[control]
    treatment_revenue = revenue[treatment]

    control_orders = int(control_converted.sum())
    treatment_orders = int(treatment_converted.sum())

    control_aov = control_revenue.sum() / max(control_orders, 1)
    treatment_aov = treatment_revenue.sum() / max(treatment_orders, 1)

    # ---------------------------------------------------------
    # 13. Revenue per user
    # ---------------------------------------------------------
    control_revenue_per_user = control_revenue.mean()
    treatment_revenue_per_user = treatment_revenue.mean()
    absolute_revenue_per_user_lift = treatment_revenue_per_user - control_revenue_per_user
    relative_revenue_per_user_lift = safe_relative_lift(
        treatment_revenue_per_user,
        control_revenue_per_user,
    )

    # ---------------------------------------------------------
    # 14. Statistical interpretation
    # ---------------------------------------------------------
    # This describes the simulated statistical result.
    # It should NOT be interpreted as a production claim.
    statistically_significant = p_value < 0.05

    # ---------------------------------------------------------
    # 15. Final experiment result
    # ---------------------------------------------------------
    out = {
        "experiment": "Context-aware hybrid ranking vs popularity baseline",
        "design": "simulated/offline",
        "note": (
            "Behavioral probabilities and revenue are simulated assumptions for "
            "demonstrating A/B testing methodology. Results are not production causal estimates."
        ),
        "control_strategy": "Popularity-based baseline ranking",
        "treatment_strategy": (
            "Hybrid ranking using relevance, rating, popularity, customer distance and predicted ETA"
        ),
        "control_users": control_size,
        "treatment_users": treatment_size,
        "control_conversion": float(control_conversion),
        "treatment_conversion": float(treatment_conversion),
        "absolute_conversion_lift": float(absolute_conversion_lift),
        "relative_conversion_lift": float(relative_conversion_lift),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "statistically_significant_at_5pct": bool(statistically_significant),
        "conversion_lift_95ci": [float(ci_lower), float(ci_upper)],
        "control_CTR": float(control_ctr),
        "treatment_CTR": float(treatment_ctr),
        "absolute_CTR_lift": float(absolute_ctr_lift),
        "relative_CTR_lift": float(relative_ctr_lift),
        "control_AOV": float(control_aov),
        "treatment_AOV": float(treatment_aov),
        "control_revenue_per_user": float(control_revenue_per_user),
        "treatment_revenue_per_user": float(treatment_revenue_per_user),
        "absolute_revenue_per_user_lift": float(absolute_revenue_per_user_lift),
        "relative_revenue_per_user_lift": float(relative_revenue_per_user_lift),
    }

    # ---------------------------------------------------------
    # 16. Save result
    # ---------------------------------------------------------
    Path("reports").mkdir(exist_ok=True)
    Path("reports/ab_test_results.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(run())

