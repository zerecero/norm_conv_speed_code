# Iterator settings and compatibility

The recommended interface is:

```python
iterator = PyBLPShareMatchingIterator("symmetric", tolerance=1e-12,
                                    max_evaluations=10_000)
results = problem.solve(
    sigma=initial_sigma,
    iteration=iterator.configuration(),
    fp_type="safe_linear",
    delta_behavior="logit",
    shares_bounds=(None, None),
)
```

This path is for non-nested random-coefficients logit with the ordinary
`epsilon_scale=1`. Leave `delta` unspecified. Use strictly positive observed
shares and positive integration weights that sum to one within each market.
The examples use equal-weight Halton draws. Nested logit, nonlinear fixed
points, changed epsilon scaling, share clipping, and signed quadrature weights
are outside the supported example workflow. The callable does not receive all
`Problem.solve` options and cannot check every incompatible setting itself.

The adapter requests PyBLP's safe-linear contraction once per evaluation.
Its increment gives each inside log-share mismatch; from that identity the
adapter reconstructs predicted shares and computes the outside share as residual
mass. It subtracts the outside mismatch from every updated inside utility.
There is no second probability calculation for the stopping check. A checked
current iterate is returned when the inside log-share residual is at most the
tolerance. This uses the public `pyblp.Iteration` callback API.

All reported symmetric iterations use a full update of every alternative
followed by outside-zero normalization.

The outside share is computed as `1 - sum(inside shares)`. Extremely small
outside shares can cause cancellation; a nonpositive computed outside share
is an inner failure, not silently clipped. Review numerical errors and verify
the final share fit. `error_behavior="raise"` in the examples exposes failures.

## What PyBLP still does

PyBLP performs market-share evaluation, derivatives, GMM optimization, and
standard errors. Its installed source is never patched. The iterator occupies
the iteration layer, so it cannot be combined directly with SQUAREM, Anderson,
or another built-in accelerator. The optimizer (e.g. L-BFGS-B) remains separate.

The recommended logit-start path reads no private PyBLP state. The extracted
module also retains the research package's optional
`use_contraction_target_shares=True` warm-start path. That path reads named
Python closure variables inside PyBLP 1.2.0, is version-dependent, and is not
needed by either example. If using `delta_behavior="last"`, enable that option;
otherwise the default target reconstruction would be wrong. The default
`"first"` behavior should not be substituted for the documented `"logit"` path.
An optional `warm_start_logit_distance_limit` is also retained for research use.

Serial `iterator.records` provide total calls and failures. With PyBLP worker
processes, records on workers are not collected in the parent iterator; do not
interpret its parent counters as aggregate parallel statistics. Use PyBLP's
results statistics for parallel work. The supplied examples run serially.

The package pins PyBLP 1.2.0 and retains the `reference_fixed` mode for existing
code and tests. Both public examples estimate only with the symmetric mode.

