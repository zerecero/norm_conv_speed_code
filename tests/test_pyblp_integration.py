"""Exercise PyBLP's public solve/share API with generated integration draws."""

import numpy as np
import pyblp
import pytest

from blp_convergence import PyBLPShareMatchingIterator


@pytest.fixture
def quiet_pyblp():
    previous = pyblp.options.verbose
    pyblp.options.verbose = False
    yield
    pyblp.options.verbose = previous


def build_problem():
    products = np.genfromtxt(pyblp.data.NEVO_PRODUCTS_LOCATION, delimiter=",",
                             names=True, dtype=None, encoding="utf-8")
    # Five complete markets; no demographics, agent CSV, or private API.
    products = products[np.isin(products["market_ids"], np.unique(products["market_ids"])[:5])]
    return pyblp.Problem(
        (pyblp.Formulation("1 + prices + sugar + mushy"),
         pyblp.Formulation("0 + prices")),
        products,
        integration=pyblp.Integration("halton", 20, {"seed": 0}),
    )


@pytest.mark.parametrize("warm_start", [False, True])
def test_public_problem_solve_matches_shares(quiet_pyblp, warm_start):
    problem = build_problem()
    iterator = PyBLPShareMatchingIterator("symmetric", use_contraction_target_shares=warm_start)
    # Fixed nonzero sigma isolates demand inversion without an expensive fit.
    result = problem.solve(
        sigma=np.array([[0.5]]), method="2s",
        optimization=pyblp.Optimization("return"),
        iteration=iterator.configuration(), fp_type="safe_linear",
        delta_behavior="last" if warm_start else "logit",
        shares_bounds=(None, None), error_behavior="raise",
    )
    predicted = result.compute_shares().ravel()
    target = problem.products.shares.ravel()
    assert np.max(np.abs(np.log(target) - np.log(predicted))) <= 1.1e-12
    assert iterator.failures == 0
    assert iterator.total_share_evaluator_calls > 0
    assert iterator.maximum_final_residual <= 1e-12
    for market in problem.unique_market_ids:
        mask = problem.products.market_ids.ravel() == market
        assert abs((1 - predicted[mask].sum()) - (1 - target[mask].sum())) < 1e-12


def test_fixed_integration_is_reproducible(quiet_pyblp):
    first, second = build_problem(), build_problem()
    np.testing.assert_array_equal(first.agents.nodes, second.agents.nodes)
    np.testing.assert_array_equal(first.agents.weights, second.agents.weights)
    for market in first.unique_market_ids:
        mask = first.agents.market_ids.ravel() == market
        assert np.all(first.agents.weights[mask] > 0)
        assert np.isclose(first.agents.weights[mask].sum(), 1)

