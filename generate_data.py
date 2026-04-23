"""
Synthetic credit risk data generator.

Produces two datasets:
  - reference.csv  : clean training data (model was trained on this)
  - production.csv : drifted production data (simulates regime change)

Features mimic a typical credit scorecard:
  age, income, debt_ratio, credit_score, num_accounts, default (target)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def make_batch(n: int, drift: bool = False) -> pd.DataFrame:
    if not drift:
        age          = RNG.integers(22, 65, n)
        income       = RNG.normal(55_000, 15_000, n).clip(15_000, 200_000)
        debt_ratio   = RNG.beta(2, 5, n)                  # mostly low
        credit_score = RNG.normal(680, 60, n).clip(300, 850)
        num_accounts = RNG.integers(1, 10, n)
    else:
        # Simulate recession drift: lower income, higher debt, worse scores
        age          = RNG.integers(22, 65, n)
        income       = RNG.normal(40_000, 18_000, n).clip(10_000, 200_000)
        debt_ratio   = RNG.beta(5, 3, n)                  # skewed high
        credit_score = RNG.normal(620, 80, n).clip(300, 850)
        num_accounts = RNG.integers(1, 15, n)

    # Default probability driven by features
    log_odds = (
        -3.5
        + 0.01  * (30 - age)
        - 0.000015 * income
        + 3.0   * debt_ratio
        - 0.005 * (credit_score - 650)
        + 0.05  * num_accounts
    )
    prob    = 1 / (1 + np.exp(-log_odds))
    default = RNG.binomial(1, prob, n)

    return pd.DataFrame({
        "age":          age,
        "income":       income.round(0).astype(int),
        "debt_ratio":   debt_ratio.round(4),
        "credit_score": credit_score.round(0).astype(int),
        "num_accounts": num_accounts,
        "default":      default,
    })


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    ref  = make_batch(1000, drift=False)
    prod = make_batch(500,  drift=True)

    ref.to_csv("data/reference.csv",  index=False)
    prod.to_csv("data/production.csv", index=False)

    print(f"Reference  : {len(ref)} rows  | default rate: {ref['default'].mean():.2%}")
    print(f"Production : {len(prod)} rows | default rate: {prod['default'].mean():.2%}")
    print("Saved → data/reference.csv, data/production.csv")
