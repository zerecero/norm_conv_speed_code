# Adapting the example to your application

[examples/symmetric_own_data.py](../examples/symmetric_own_data.py) follows the
standard PyBLP workflow: load product data, specify `X1` and `X2`, configure
integration, construct a `Problem`, and call `solve`. Adapt the data path,
formulations, instruments, parameter starts, and estimation settings to your
application; retain the iterator configuration shown below.

## Product data and model

The template reads `data/products.csv` relative to the working directory.
Set `DATA_PATH` to your file. It assumes one row per inside product and market,
with `market_ids`, `shares`, `prices`, an exogenous characteristic named
`quality`, and excluded instruments `demand_instruments0`,
`demand_instruments1`, etc. The outside share is `1 - sum(shares)` within each
market and must be positive; the outside option is not a row in the product data.

```python
formulations = (
    pyblp.Formulation("1 + prices + quality"),
    pyblp.Formulation("0 + prices"),
)
```

This specification has linear coefficients on the constant, price, and quality,
and one random price coefficient. Hence `sigma=np.array([[0.5]])` is a starting
standard deviation. Update its dimensions and entries when changing `X2`.
PyBLP combines the excluded instruments with exogenous `X1` characteristics
by default. Fixed effects, additional controls, supply, and demographics use
the usual PyBLP specifications; check the adapter's
[supported settings](compatibility.md).

## Integration

The template generates 500 Halton draws with seed 0 using `pyblp.Integration`.
For demographics or custom nodes and weights, pass `agent_formulation`,
`agent_data`, and starting `pi` as appropriate. Integration and identification
choices are application-specific and do not change the iterator interface.

## Solve

```python
iterator = PyBLPShareMatchingIterator("symmetric")
results = problem.solve(
    sigma=initial_sigma,
    iteration=iterator.configuration(),
    fp_type="safe_linear",
    delta_behavior="logit",
    shares_bounds=(None, None),
    # Add the optimizer, GMM, bounds, and other options for your application.
)
```

The template uses two-step GMM with L-BFGS-B. Parameter bounds, weighting
matrices, standard errors, and post-estimation methods are configured as in
any other PyBLP application. Inspect `results.converged`, the share fit, and
inner failures before interpreting estimates. The three settings accompanying
`iteration` above are required for the recommended adapter path; warm starts
need the handling described in [compatibility.md](compatibility.md).

For the full API, see PyBLP's
[Problem](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.html),
[Problem.solve](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html),
and [Integration](https://pyblp.readthedocs.io/en/stable/_api/pyblp.Integration.html)
documentation.

