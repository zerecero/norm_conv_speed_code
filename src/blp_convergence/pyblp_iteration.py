"""Reusable unaccelerated share-matching iterations for PyBLP.

The public ``pyblp.Iteration`` interface supplies PyBLP's standard linear BLP
update to a custom iterator. The iterator can retain that reference-fixed
update or convert it into a full all-alternative update followed by outside-zero
normalization. Both modes stop on the same non-reference log-share residual.

This adapter is designed for non-nested linear fixed points with logit starting
utilities and no share clipping. Because it occupies PyBLP's iteration layer,
it does not compose with PyBLP's built-in acceleration methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pyblp


FloatArray = NDArray[np.float64]
InversionMap = Literal["reference_fixed", "symmetric"]

DEFAULT_PYBLP_ITERATION_TOLERANCE = 1e-12
DEFAULT_PYBLP_MAX_EVALUATIONS = 10_000


def shares_from_logit_delta(initial_delta: ArrayLike) -> FloatArray:
    """Recover observed all-alternative shares from PyBLP's logit start."""

    inside_delta = np.asarray(initial_delta, dtype=np.float64)
    if inside_delta.ndim != 1 or inside_delta.size == 0:
        raise ValueError("initial_delta must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(inside_delta)):
        raise ValueError("initial_delta must contain only finite values")

    all_delta = np.concatenate(([0.0], inside_delta))
    maximum = float(np.max(all_delta))
    exponentials = np.exp(all_delta - maximum)
    shares = exponentials / exponentials.sum()
    return np.asarray(shares, dtype=np.float64)


def _closed_value(function: Callable[..., Any], name: str) -> Any:
    """Retrieve one named value captured by a Python closure."""

    closure = function.__closure__
    if closure is None:
        raise RuntimeError(f"callable does not capture {name!r}")
    values = dict(zip(function.__code__.co_freevars, closure, strict=True))
    if name not in values:
        raise RuntimeError(f"callable does not capture {name!r}")
    return values[name].cell_contents


def shares_from_pyblp_contraction(contraction: Callable[..., Any]) -> FloatArray:
    """Recover fixed observed shares from PyBLP's linear contraction closure.

    PyBLP's public custom-iteration interface does not pass observed shares to
    the iterator. Its linear contraction does capture their logarithms. Reading
    that fixed input lets the iterator support ``delta_behavior='last'``, for
    which the initial utilities generally are not the logit solution.
    """

    try:
        pyblp_contraction = _closed_value(contraction, "contraction")
        log_inside_shares = np.asarray(
            _closed_value(pyblp_contraction, "log_shares"), dtype=np.float64
        ).reshape(-1)
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "the symmetric iterator requires PyBLP's safe linear contraction"
        ) from error

    inside_shares = np.exp(log_inside_shares)
    outside_share = 1.0 - float(inside_shares.sum())
    shares = np.concatenate(([outside_share], inside_shares))
    if outside_share <= 0.0 or not np.all(np.isfinite(shares)):
        raise RuntimeError("PyBLP contraction contains invalid observed shares")
    return np.asarray(shares, dtype=np.float64)


@dataclass(frozen=True)
class PyBLPShareMatchingStep:
    """One checked PyBLP-compatible inner-loop step."""

    next_inside_delta: FloatArray
    predicted_all_shares: FloatArray
    log_share_mismatch: FloatArray
    residual: float


def pyblp_share_matching_step(
    current_inside_delta: ArrayLike,
    reference_updated_inside_delta: ArrayLike,
    target_all_shares: ArrayLike,
    inversion_map: InversionMap,
) -> PyBLPShareMatchingStep:
    """Transform PyBLP's standard inside update and evaluate the common residual.

    ``reference_updated_inside_delta`` must be the standard PyBLP linear BLP
    update evaluated at ``current_inside_delta``. The predicted inside shares
    are recovered from this update identity, so both maps use the same PyBLP
    share evaluation. Their residual mass is the predicted outside share.
    """

    if inversion_map not in {"reference_fixed", "symmetric"}:
        raise ValueError(f"unknown inversion map: {inversion_map}")

    current = np.asarray(current_inside_delta, dtype=np.float64)
    reference_updated = np.asarray(
        reference_updated_inside_delta, dtype=np.float64
    )
    target = np.asarray(target_all_shares, dtype=np.float64)
    if current.ndim != 1 or reference_updated.shape != current.shape:
        raise ValueError("inside utility arrays must have the same vector shape")
    if target.shape != (current.size + 1,):
        raise ValueError("target_all_shares must include one outside share")
    if np.any(target <= 0.0) or not np.isclose(
        target.sum(), 1.0, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("target_all_shares must be positive and sum to one")

    inside_mismatch = reference_updated - current
    predicted_inside = target[1:] * np.exp(-inside_mismatch)
    predicted_outside = 1.0 - float(predicted_inside.sum())
    if predicted_outside <= 0.0 or not np.isfinite(predicted_outside):
        raise FloatingPointError("computed outside share is not finite and positive")

    predicted = np.concatenate(([predicted_outside], predicted_inside))
    mismatch = np.log(target) - np.log(predicted)
    # The outside share equation is redundant and can be numerically unstable
    # when recovered as residual mass. It still recenters the symmetric update.
    residual = float(np.max(np.abs(mismatch[1:])))
    next_inside = reference_updated
    if inversion_map == "symmetric":
        # PyBLP omits the outside coordinate, so outside-zero normalization
        # subtracts its update from every updated inside coordinate.
        next_inside = reference_updated - mismatch[0]

    return PyBLPShareMatchingStep(
        next_inside_delta=np.asarray(next_inside, dtype=np.float64),
        predicted_all_shares=np.asarray(predicted, dtype=np.float64),
        log_share_mismatch=np.asarray(mismatch, dtype=np.float64),
        residual=residual,
    )


@dataclass(frozen=True)
class PyBLPInnerSolveRecord:
    """Diagnostics for one market inversion within ``Problem.solve``."""

    evaluations: int
    final_residual: float
    converged: bool


@dataclass
class PyBLPShareMatchingIterator:
    """Public PyBLP custom iterator with a common non-reference stopping rule."""

    inversion_map: InversionMap
    tolerance: float = DEFAULT_PYBLP_ITERATION_TOLERANCE
    max_evaluations: int = DEFAULT_PYBLP_MAX_EVALUATIONS
    use_contraction_target_shares: bool = False
    warm_start_logit_distance_limit: float | None = None
    records: list[PyBLPInnerSolveRecord] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.inversion_map not in {"reference_fixed", "symmetric"}:
            raise ValueError(f"unknown inversion map: {self.inversion_map}")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and strictly positive")
        if not isinstance(self.max_evaluations, int) or self.max_evaluations < 1:
            raise ValueError("max_evaluations must be a positive integer")
        if (
            self.warm_start_logit_distance_limit is not None
            and (
                not np.isfinite(self.warm_start_logit_distance_limit)
                or self.warm_start_logit_distance_limit <= 0.0
            )
        ):
            raise ValueError(
                "warm_start_logit_distance_limit must be finite and positive"
            )

    @property
    def total_share_evaluator_calls(self) -> int:
        """Count every PyBLP contraction/share evaluation made by this object."""

        return sum(record.evaluations for record in self.records)

    @property
    def failures(self) -> int:
        return sum(not record.converged for record in self.records)

    @property
    def maximum_final_residual(self) -> float:
        if not self.records:
            return float("nan")
        return max(record.final_residual for record in self.records)

    def __call__(
        self,
        initial: FloatArray,
        contraction: Callable,
        callback: Callable[[], None],
    ) -> tuple[FloatArray, bool]:
        """Run simple iteration and stop only at a checked current iterate."""

        current = np.asarray(initial, dtype=np.float64).copy()
        target = (
            shares_from_pyblp_contraction(contraction)
            if self.use_contraction_target_shares
            else shares_from_logit_delta(current)
        )
        if self.warm_start_logit_distance_limit is not None:
            logit_delta = np.log(target[1:]) - np.log(target[0])
            if np.max(np.abs(current - logit_delta)) > (
                self.warm_start_logit_distance_limit
            ):
                current = logit_delta
        final_residual = float("inf")

        for evaluation in range(1, self.max_evaluations + 1):
            reference_updated = np.asarray(contraction(current)[0], dtype=np.float64)
            try:
                step = pyblp_share_matching_step(
                    current,
                    reference_updated,
                    target,
                    self.inversion_map,
                )
            except (FloatingPointError, ValueError):
                callback()
                self.records.append(
                    PyBLPInnerSolveRecord(evaluation, final_residual, False)
                )
                return current, False

            callback()
            final_residual = step.residual
            if final_residual <= self.tolerance:
                self.records.append(
                    PyBLPInnerSolveRecord(evaluation, final_residual, True)
                )
                return current, True
            current = step.next_inside_delta

        self.records.append(
            PyBLPInnerSolveRecord(self.max_evaluations, final_residual, False)
        )
        return current, False

    def configuration(self) -> pyblp.Iteration:
        """Construct the public PyBLP iteration configuration."""

        return pyblp.Iteration(self.__call__)
