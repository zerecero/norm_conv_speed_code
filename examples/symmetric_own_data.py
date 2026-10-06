"""Application template; see docs/own_data.md for inputs and solve options."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

from blp_convergence import PyBLPShareMatchingIterator

# Set the product-data path (relative to the working directory, or absolute).
DATA_PATH = Path("data/products.csv")
product_data = pd.read_csv(DATA_PATH)

# Validate required fields and positive inside/outside shares.
required = {"market_ids", "shares", "prices", "quality", "demand_instruments0"}
missing = required.difference(product_data.columns)
if missing:
    raise ValueError(f"Missing columns for this example: {sorted(missing)}")
if product_data["market_ids"].isna().any():
    raise ValueError("Each product must have a market_ids value.")
if not np.all(np.isfinite(product_data["shares"])) or (
    product_data["shares"] <= 0
).any():
    raise ValueError("All inside shares must be finite and positive.")
outside_shares = 1 - product_data.groupby("market_ids")["shares"].sum()
if (outside_shares <= 0).any():
    raise ValueError("Inside shares must sum to less than one in each market.")

# Replace quality, the formulations, and excluded instruments for your model.
# PyBLP adds exogenous X1 characteristics to the instrument matrix by default.
formulations = (
    pyblp.Formulation("1 + prices + quality"),
    pyblp.Formulation("0 + prices"),  # one random price coefficient
)
integration = pyblp.Integration(
    "halton", size=500, specification_options={"seed": 0}
)
problem = pyblp.Problem(formulations, product_data, integration=integration)

iterator = PyBLPShareMatchingIterator("symmetric")
results = problem.solve(
    sigma=np.array([[0.5]]),  # Starting SD of the random price coefficient.
    method="2s",             # Two-step GMM.
    optimization=pyblp.Optimization("l-bfgs-b", {"gtol": 1e-6, "ftol": 0}),
    iteration=iterator.configuration(),
    fp_type="safe_linear",
    delta_behavior="logit",
    shares_bounds=(None, None),
    error_behavior="raise",
)

print(results)
print(f"Optimizer converged: {results.converged}")
print(f"Failed inner inversions: {iterator.failures}")
print(f"Inner share evaluations: {iterator.total_share_evaluator_calls:,}")
# results.beta, results.sigma, results.beta_se, results.sigma_se, results.delta
# results.compute_shares(), results.compute_elasticities(), ... work as usual.

