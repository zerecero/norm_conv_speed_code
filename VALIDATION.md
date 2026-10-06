# Package preparation checks

Validated on Windows with Python 3.12.14, PyBLP 1.2.0, NumPy 2.5.3,
SciPy 1.18.1, pandas 3.0.6, and pytest 9.1.1. The package was installed into
its own newly created virtual environment with:

```text
python -m pip install -e ".[examples,test]"
python -m pytest -q
python examples/symmetric_nevo.py
```

## Results

- All **14 tests passed**. They cover the update formula, exact evaluation
  accounting, stopping criterion, failure safeguard, reproducible integration,
  and public PyBLP solve/share calls with both logit and optional warm starts.
- The runnable symmetric Nevo example estimated all 94 markets (2,256
  products), with 100 generated draws per market. The optimizer converged;
  there were no failed inner inversions. The iterator counted 6,682 inner share
  evaluations, including checks and final derivative computations. Maximum
  post-estimation inside log-share residual: `9.117e-13`.
- The own-data template was also executed by supplying the packaged Nevo
  product DataFrame in memory and renaming `sugar` to `quality`. The specified
  two-step estimation converged, with zero inner failures and 1,961 inner share
  evaluations. Maximum inside log-share residual: `4.441e-15`. This checks the
  template's code path; users must supply and validate their own data.

The example Nevo fit places three random-coefficient standard deviations at
zero. Its estimates illustrate the interface; they are not the paper's Nevo
estimates or a claim that this simplified specification is empirically preferred.
The own-data smoke input also gives an imprecisely estimated random coefficient.

Last digits and optimizer paths can change with platform and dependency
versions. These measurements are software checks, not performance benchmarks.

## Provenance check

The extracted `src/blp_convergence/pyblp_iteration.py` is byte-for-byte identical
to the research module. Both SHA-256 hashes were:

```text
f0e732434d7d315acced09b871d777d5f5f26271d38aaa86386eba5ed66deb29
```

Neither installation nor the tests patch the PyBLP installation. The package
has no runtime dependency on the paper or replication folders.

