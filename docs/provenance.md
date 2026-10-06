# Source and verification

This software package extracts the reusable `pyblp_iteration.py` module from
the paper project *Normalization, Convergence, and Speed in Demand and Gravity
Inversion*. The module was copied without changes from paper-project commit
`12a9b39be743b1f5eacbc908c48e69629bdbf79a`. It has no dependencies on the paper's
simulation, benchmark, plotting, manuscript, or saved-result modules.

Version `0.2.0` identifies this separately installable software distribution;
the replication package used version `0.1.0`. Both use the import name
`blp_convergence`. Use separate virtual environments so installing one does
not replace the other in an existing research environment.

The stopping rule follows the current manuscript and numerical plan: maximum
inside log-share residual at most `1e-12`. Earlier development versions used an
all-alternative stopping rule; those are not the source for this extraction.

`symmetric_nevo.py` uses the packaged Nevo products and the no-demographics
formulations from the official PyBLP tutorial. It selects diagonal random
coefficients, 100 Halton draws with seed 0, one-step GMM, and L-BFGS-B. These
are example settings and differ from the paper's empirical specification. No example
consumer or product dataset is copied into this repository.

Tests cover normalization against an independent full-all-alternative update,
share recovery, the stopping residual, evaluation counting and the safeguard,
and a real public `Problem.solve` call using packaged products. See
`VALIDATION.md` for the verification performed when this folder was prepared.

