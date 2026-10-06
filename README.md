# Symmetric demand inversion for PyBLP

Use the symmetric outside-zero iteration in an existing `pyblp.Problem.solve`
call. The package supplies a custom `pyblp.Iteration`; model specification,
instruments, optimization, GMM, and post-estimation use the usual PyBLP API.

This package accompanies Miguel Zerecero's *Normalization, Convergence, and
Speed in Demand and Gravity Inversion*. The paper's replication package is
distributed separately.

## Installation

From the cloned or downloaded repository, install into your Python environment:

```sh
python -m pip install ".[examples]"
```

Requires Python 3.10+ and PyBLP 1.2.0 (pinned). The `examples` extra installs
pandas for loading CSV data. Use `python -m pip install .` for the iterator alone.

## Usage

Given an existing PyBLP `problem` and starting parameters:

```python
from blp_convergence import PyBLPShareMatchingIterator

iterator = PyBLPShareMatchingIterator("symmetric")
results = problem.solve(
    sigma=initial_sigma,
    iteration=iterator.configuration(),
    fp_type="safe_linear",
    delta_behavior="logit",
    shares_bounds=(None, None),
    # Include the other solve arguments for your specification.
)
```

`iteration` selects the symmetric map. The other three settings specify the
linear contraction, logit starting utilities, and unclipped shares required
by this adapter. Keep them together. `results` is a standard PyBLP
`ProblemResults` object.

The default tolerance is `1e-12`, with a limit of 10,000 share evaluations per
market inversion. These can be changed with
`PyBLPShareMatchingIterator("symmetric", tolerance=..., max_evaluations=...)`.
See [compatibility and implementation details](docs/compatibility.md) for
supported specifications and warm starts.

## Examples

- [symmetric_nevo.py](examples/symmetric_nevo.py): complete estimation using
  PyBLP's packaged Nevo products, generated integration draws, and no demographics.
  The [example guide](docs/example.md) explains the specification, iterator
  settings, and returned results.
- [symmetric_own_data.py](examples/symmetric_own_data.py): template for a product
  CSV and an application-specific model. See [adapting the example](docs/own_data.md).

Run the complete example with:

```sh
python examples/symmetric_nevo.py
```

In VS Code, select the environment where the package is installed and use
**Run Python File**, or execute successive statements or selected blocks with
**Shift+Enter**.

## Iteration

The symmetric update applies the log-share correction to every alternative,
including the outside good, then subtracts the updated outside coordinate:

$$
\delta_j^{\mathrm{new}}=\delta_j+\log\hat s_j-\log S_j(\delta)
-[\log\hat s_0-\log S_0(\delta)].
$$

Iteration is unaccelerated. Convergence is checked using the maximum absolute
inside-good log-share residual. The outside mismatch is included in the update
but omitted from the stopping criterion, following the paper. PyBLP evaluates
shares through its public custom-iteration interface; its source is not modified.

## Development and citation

```sh
python -m pip install -e ".[examples,test]"
python -m pytest
```

See [CITATION.cff](CITATION.cff), [LICENSE](LICENSE),
[source provenance](docs/provenance.md), and [validation](VALIDATION.md).

