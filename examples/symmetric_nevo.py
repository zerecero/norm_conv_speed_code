"""Estimate a random-coefficients demand model with the symmetric iterator.

Uses PyBLP's Nevo product data, no demographics, and 100 Halton draws.
See docs/example.md for the specification and a walkthrough of the code.
"""

import numpy as np
import pandas as pd
import pyblp

from blp_convergence import PyBLPShareMatchingIterator

# Load products and excluded demand instruments packaged with PyBLP.
product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)

# X1: price and product fixed effects. X2: random coefficients on all four terms.
formulations = (
    pyblp.Formulation("0 + prices", absorb="C(product_ids)"),
    pyblp.Formulation("1 + prices + sugar + mushy"),
)
# Fix the integration draws across runs.
integration = pyblp.Integration(
    "halton", size=100, specification_options={"seed": 0}
)
problem = pyblp.Problem(formulations, product_data, integration=integration)

# Configure symmetric demand inversion through PyBLP's custom iteration API.
iterator = PyBLPShareMatchingIterator("symmetric")

# One-step GMM. Starting standard deviations follow X2 column order.
# The adapter requires safe_linear, logit starts, and no share clipping.
results = problem.solve(
    sigma=np.diag([0.5, 2.0, 0.05, 0.5]),
    method="1s",
    optimization=pyblp.Optimization("l-bfgs-b", {"gtol": 1e-5, "ftol": 0}),
    iteration=iterator.configuration(),
    fp_type="safe_linear",
    delta_behavior="logit",
    shares_bounds=(None, None),
    error_behavior="raise",
)

# Validate the share fit using the standard ProblemResults interface.
predicted_shares = results.compute_shares().ravel()
max_log_share_residual = np.max(
    np.abs(np.log(product_data["shares"].to_numpy()) - np.log(predicted_shares))
)
print(f"Optimizer converged: {results.converged}")
print(f"Inner share evaluations: {iterator.total_share_evaluator_calls:,}")
print(f"Failed inner inversions: {iterator.failures}")
print(f"Maximum inside log-share residual: {max_log_share_residual:.3e}")
if not results.converged or iterator.failures:
    raise RuntimeError("Review the optimizer/inner-loop diagnostics above.")

# Post-estimation: results.beta, results.sigma, results.compute_elasticities(), ...

