# Nevo example

[examples/symmetric_nevo.py](../examples/symmetric_nevo.py) estimates a
random-coefficients demand model using the symmetric iterator. It uses Nevo
product data supplied with PyBLP, without demographics. The specification and
integration settings are chosen for a compact example and differ from the
paper's empirical benchmark.

Run the script in full, or execute its blocks in order. In VS Code, use
**Run Python File** or select a complete statement or block and press
**Shift+Enter**.

## Data and specification

```python
product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
formulations = (
    pyblp.Formulation("0 + prices", absorb="C(product_ids)"),
    pyblp.Formulation("1 + prices + sugar + mushy"),
)
```

The product file includes market shares, prices, characteristics, identifiers,
and excluded demand instruments. `X1` contains the mean price coefficient and
absorbs product fixed effects. `X2` allows random coefficients on the constant,
price, sugar, and mushy. Sugar and mushy do not enter `X1` separately because
they are absorbed by the product fixed effects.

```python
integration = pyblp.Integration(
    "halton", size=100, specification_options={"seed": 0}
)
problem = pyblp.Problem(formulations, product_data, integration=integration)
```

PyBLP generates 100 Halton draws and their weights per market. The seed fixes
the integration draws across runs. Increase integration accuracy as appropriate
for an empirical application. Models with demographics can instead supply
`agent_formulation` and `agent_data` through the standard PyBLP interface.

## Estimation with the symmetric iterator

```python
iterator = PyBLPShareMatchingIterator("symmetric")
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
```

`sigma` provides starting standard deviations in `X2` column order; its zero
off-diagonal entries remain fixed. The example uses one-step GMM and L-BFGS-B.
The adapter-specific arguments are:

| Argument | Role |
| --- | --- |
| `iteration=iterator.configuration()` | Select the symmetric inner iteration |
| `fp_type="safe_linear"` | Use PyBLP's numerically stabilized linear contraction |
| `delta_behavior="logit"` | Start each inversion at logit utilities, from which this adapter recovers observed shares |
| `shares_bounds=(None, None)` | Disable share clipping so the contraction increment identifies the log-share mismatch |

`error_behavior="raise"` exposes numerical failures during estimation. Other
optimizer, GMM, and standard-error settings remain ordinary PyBLP choices.

Each symmetric step updates all alternatives and then normalizes the outside
utility to zero. The default stopping test is
`max(abs(log(observed_inside_shares) - log(predicted_inside_shares))) <= 1e-12`.
See [compatibility.md](compatibility.md) for implementation details and limitations.

## Results and diagnostics

The returned `results` supports the usual PyBLP attributes and methods:

```python
results.beta
results.sigma
results.beta_se
results.sigma_se
results.delta
results.compute_shares()
results.compute_elasticities()
```

The script checks the final inside-share fit with `compute_shares()` and prints
optimizer convergence, inner failures, and total inner share evaluations.
For serial estimation, `iterator.records` stores the evaluation count,
convergence flag, and final residual of each market inversion. The aggregate
counter also includes inner inversions used in final derivative computations.

The model setup follows the no-demographics section of
[PyBLP's Nevo tutorial](https://pyblp.readthedocs.io/en/stable/_notebooks/tutorial/nevo.html).

