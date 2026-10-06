"""Test the extracted iterator independently of the paper's code and outputs."""

import numpy as np
import pyblp
import pytest

from blp_convergence import PyBLPShareMatchingIterator
from blp_convergence.pyblp_iteration import (
    pyblp_share_matching_step,
    shares_from_logit_delta,
)


def shares(delta, mu, weights):
    """Independent all-alternative evaluator, with outside in column zero."""
    utilities = delta[None, :] + mu
    exponentials = np.exp(utilities - utilities.max(axis=1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    return weights @ probabilities


def market():
    mu = np.array([[0, -0.2, 0.35, 0.1], [0, 0.45, -0.15, 0.25],
                   [0, 0.05, 0.2, -0.3]])
    weights = np.array([0.2, 0.5, 0.3])
    delta = np.array([0.0, 0.4, -0.3, 0.2])
    return delta, mu, weights, shares(delta, mu, weights)


def test_logit_start_recovers_target_shares():
    target = np.array([0.17, 0.11, 0.29, 0.43])
    np.testing.assert_allclose(
        shares_from_logit_delta(np.log(target[1:] / target[0])),
        target, rtol=0, atol=2e-16,
    )


def test_symmetric_step_matches_full_update_then_outside_zero():
    _, mu, weights, target = market()
    current = np.array([0.0, -0.1, 0.25, -0.35])
    predicted = shares(current, mu, weights)
    mismatch = np.log(target) - np.log(predicted)
    full_update = current + mismatch
    expected = full_update - full_update[0]
    step = pyblp_share_matching_step(
        current[1:], current[1:] + mismatch[1:], target, "symmetric"
    )
    np.testing.assert_allclose(step.next_inside_delta, expected[1:], atol=3e-15, rtol=0)
    np.testing.assert_allclose(step.predicted_all_shares, predicted, atol=3e-16, rtol=0)


@pytest.mark.parametrize("algorithm", ["symmetric", "reference_fixed"])
def test_recovers_shares_and_counts_each_check(algorithm):
    true_delta, mu, weights, target = market()
    initial = np.log(target[1:] / target[0])
    evaluated = []
    callbacks = []

    def contraction(delta):
        evaluated.append(delta.copy())
        predicted = shares(np.r_[0.0, delta], mu, weights)
        return delta + np.log(target[1:]) - np.log(predicted[1:]), None, None

    iterator = PyBLPShareMatchingIterator(algorithm)
    recovered, converged = iterator(initial, contraction, lambda: callbacks.append(1))
    assert converged
    assert iterator.failures == 0
    assert len(evaluated) == len(callbacks) == iterator.total_share_evaluator_calls
    np.testing.assert_array_equal(recovered, evaluated[-1])
    np.testing.assert_allclose(recovered, true_delta[1:], atol=2e-11, rtol=0)
    np.testing.assert_allclose(shares(np.r_[0.0, recovered], mu, weights), target,
                               atol=1e-12, rtol=0)
    assert iterator.maximum_final_residual <= 1e-12


def test_evaluation_limit_reports_failure():
    _, mu, weights, target = market()

    def contraction(delta):
        predicted = shares(np.r_[0.0, delta], mu, weights)
        return delta + np.log(target[1:] / predicted[1:]), None, None

    iterator = PyBLPShareMatchingIterator("symmetric", max_evaluations=1)
    _, converged = iterator(np.log(target[1:] / target[0]), contraction, lambda: None)
    assert not converged
    assert iterator.failures == iterator.total_share_evaluator_calls == 1


def test_stopping_omits_outside_but_update_includes_it():
    target = np.array([1e-8, 0.4, 0.59999999])
    current = np.zeros(2)
    reference_update = np.full(2, 1e-14)
    step = pyblp_share_matching_step(current, reference_update, target, "symmetric")
    assert step.residual < 1e-12
    assert abs(step.log_share_mismatch[0]) > 1e-8
    np.testing.assert_allclose(step.next_inside_delta,
                               reference_update - step.log_share_mismatch[0])


@pytest.mark.parametrize("options", [
    {"tolerance": 0}, {"tolerance": float("nan")},
    {"max_evaluations": 0}, {"inversion_map": "unknown"},
])
def test_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        PyBLPShareMatchingIterator(**{"inversion_map": "symmetric", **options})


def test_uses_public_iteration_configuration():
    assert isinstance(PyBLPShareMatchingIterator("symmetric").configuration(), pyblp.Iteration)

