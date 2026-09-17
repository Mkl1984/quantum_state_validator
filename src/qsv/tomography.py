"""
Exact finite-shot counting model - replaces the Gaussian simplification.

Everywhere else in this project a tomographic reconstruction is simulated by
adding Gaussian noise to the amplitudes, with sigma = 1 / (2 sqrt(N)) (see
``qsv.features.sigma_from_shots``). That model was always declared as a
simplification. This module implements what a counting experiment actually
produces: integer counts, one per measurement outcome.

Model
-----
The detector is run for a calibrated exposure corresponding to N shots. The
number of clicks in outcome i is

    k_i ~ Poisson(N * |c_i|^2)

Reading: "k i follows a Poisson law of parameter N times the modulus squared
of c i".

The norm estimator is then the total click rate:

    ||psi_hat||^2 = (sum_i k_i) / N ,      sum_i k_i ~ Poisson(N * ||psi||^2)

Why Poisson rather than multinomial: a multinomial over the d outcomes is
what you obtain *after conditioning on the total count*, and conditioning on
the total is exactly what destroys the norm. Under k -> k / sum(k) the global
scale cancels identically, so the conditional outcome distribution of a state
of norm 0.85 and of the same state renormalised to 1 are the same
distribution. All the norm information lives in the total count, nowhere
else. This is the counting-statistics restatement of the scale-invariance
that drove the leakage analysis in notebook 07: a scale-invariant summary of
the data cannot, by construction, answer a question about the scale.

The practical consequence is that a norm check requires a calibrated exposure
(a known N), not just the measured outcome frequencies. An experiment that
only reports relative frequencies has already thrown the answer away.

Two properties separate this model from the Gaussian one:

* the estimator is unbiased - E[||psi_hat||^2] = ||psi||^2 exactly, with no
  2*d*sigma^2 correction. That correction is an artefact of squaring additive
  amplitude noise, not a fact about counting;
* its spread is Var = ||psi||^2 / N, which at ||psi||^2 = 1 gives
  sd = 1 / sqrt(N) - the same leading-order spread as the Gaussian model.

So the Gaussian simplification was right about the spread and wrong about the
bias. Notebook 15 quantifies what that costs operationally.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CountingResult",
    "NormEstimate",
    "sample_counts",
    "estimate_norm_squared",
    "conditional_probabilities",
    "validate_counts",
    "poisson_two_sided_p_value",
]

# Range of the exact Poisson sum, in standard deviations around the mean.
# Beyond 40 sd the neglected tail is far below float64 resolution.
SUM_RANGE_SD = 40.0


@dataclass
class NormEstimate:
    """Point estimate of ||psi||^2 from counting data, with its uncertainty."""

    value: float
    variance: float
    std_error: float
    total_counts: int
    n_shots: int


@dataclass
class CountingResult:
    """Decision taken on counting data, under the exact Poisson model."""

    valid: bool
    norm_squared: float
    statistic: float
    threshold: float
    p_value: float
    total_counts: int
    n_shots: int
    budget_ok: bool
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


def _as_complex(real: Sequence[float], imag: Sequence[float]) -> np.ndarray:
    re = np.asarray(real, dtype=float).ravel()
    im = np.asarray(imag, dtype=float).ravel()
    if re.shape != im.shape:
        raise ValueError(
            f"real and imag must have the same length, got {re.size} and {im.size}"
        )
    if re.size == 0:
        raise ValueError("an empty state has no norm to validate")
    return re + 1j * im


def sample_counts(
    real: Sequence[float],
    imag: Sequence[float],
    n_shots: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Draw one realisation of the counting experiment: k_i ~ Poisson(N |c_i|^2).

    Unlike the Gaussian model, the output is integer-valued and non-negative,
    which is what a detector actually returns.
    """
    if n_shots <= 0:
        raise ValueError(f"n_shots must be > 0, got: {n_shots}")
    state = _as_complex(real, imag)
    rng = np.random.default_rng(seed)
    return rng.poisson(n_shots * np.abs(state) ** 2)


def estimate_norm_squared(counts: Sequence[int], n_shots: int) -> NormEstimate:
    """
    Estimate ||psi||^2 from counts. Unbiased, no correction term.

    The variance is the plug-in estimate K / N^2: under Poisson the variance
    equals the mean, so the total count estimates its own variance.
    """
    if n_shots <= 0:
        raise ValueError(f"n_shots must be > 0, got: {n_shots}")
    k = np.asarray(counts, dtype=float).ravel()
    if k.size == 0:
        raise ValueError("counts is empty")
    if np.any(k < 0):
        raise ValueError("counts must be non-negative")

    total = float(k.sum())
    value = total / n_shots
    variance = total / (n_shots**2)
    return NormEstimate(
        value=value,
        variance=variance,
        std_error=float(np.sqrt(variance)),
        total_counts=int(round(total)),
        n_shots=int(n_shots),
    )


def conditional_probabilities(counts: Sequence[int]) -> np.ndarray:
    """
    Outcome frequencies k / sum(k) - the scale-invariant part of the data.

    Kept as a first-class function because it makes the non-identifiability
    explicit: this quantity is what survives when the exposure is unknown, and
    it is blind to the norm by construction.
    """
    k = np.asarray(counts, dtype=float).ravel()
    total = k.sum()
    if total <= 0:
        # No click at all: nothing observed, fall back to the uniform law
        # rather than dividing by zero.
        return np.full(k.size, 1.0 / k.size)
    return k / total


def poisson_two_sided_p_value(total_counts: int, expected: float) -> float:
    """
    Exact two-sided p-value for K ~ Poisson(expected), observed total_counts.

    Conventional doubling rule: p = min(1, 2 * min(P(K <= k), P(K >= k))).
    Computed by direct summation in log space - no Gaussian approximation,
    which matters precisely in the small-N regime where the approximation is
    weakest and the decision hardest.
    """
    if expected <= 0:
        raise ValueError(f"expected must be > 0, got: {expected}")
    k = int(total_counts)
    if k < 0:
        raise ValueError("total_counts must be >= 0")

    hi = int(np.ceil(expected + SUM_RANGE_SD * np.sqrt(expected))) + 10
    grid = np.arange(0, max(hi, k + 1) + 1)
    # log pmf = -lambda + n log lambda - log(n!)
    log_pmf = -expected + grid * np.log(expected) - _log_factorial(grid)
    pmf = np.exp(log_pmf)

    lower = float(pmf[: k + 1].sum())
    upper = float(pmf[k:].sum())
    return float(min(1.0, 2.0 * min(lower, upper)))


def _log_factorial(n: np.ndarray) -> np.ndarray:
    """log(n!) via a cumulative sum of log(i) - exact enough and dependency free."""
    n = np.asarray(n)
    max_n = int(n.max())
    table = np.zeros(max_n + 1)
    if max_n >= 1:
        table[1:] = np.cumsum(np.log(np.arange(1, max_n + 1)))
    return table[n]


def validate_counts(
    counts: Sequence[int],
    n_shots: int,
    margin: float = 0.05,
    alpha: float = 0.05,
) -> CountingResult:
    """
    Decide validity from counting data, under the exact model.

    Same decision geometry as ``qsv.validators.validate_state`` in noisy mode
    - two-sided test of ||psi||^2 = 1 with a tolerance of margin / 2 - but the
    statistic carries no bias correction, because the counting estimator has
    no bias to correct.

    ``p_value`` is the exact Poisson p-value of the observed total against the
    expectation N under H0. It is reported alongside the margin decision
    rather than replacing it: the margin encodes the physical class boundary
    (see the F2 guarantee), the p-value encodes what the data alone can say.

    ``budget_ok`` answers the sizing question of notebook 12: one standard
    deviation of the estimator must fit inside the decision tolerance,
    1 / sqrt(N) <= margin / 2, i.e. N >= 4 / margin^2. At margin = 0.05 that
    is N >= 1600.
    """
    if not 0 < margin < 1:
        raise ValueError(f"margin must be in (0, 1), got: {margin}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got: {alpha}")

    est = estimate_norm_squared(counts, n_shots)
    threshold = margin / 2
    statistic = abs(est.value - 1.0)
    valid = bool(statistic <= threshold)

    p_value = poisson_two_sided_p_value(est.total_counts, float(n_shots))
    budget_ok = bool(1.0 / np.sqrt(n_shots) <= threshold)

    verdict = "compatible with" if valid else "incompatible with"
    explanation = (
        f"Counting model: {est.total_counts} clicks for a calibrated exposure "
        f"of N = {n_shots}, so ||psi_hat||^2 = {est.value:.4f} "
        f"+/- {est.std_error:.4f} (1 sd). The deviation from 1 is "
        f"{statistic:.4f}, {verdict} the tolerance of {threshold:.4f}. "
        f"Exact Poisson two-sided p-value against ||psi||^2 = 1: {p_value:.3g}."
    )
    if not budget_ok:
        needed = int(np.ceil(4.0 / margin**2))
        explanation += (
            f" Budget warning: at N = {n_shots} one standard deviation "
            f"({est.std_error:.4f}) exceeds the tolerance, so this decision is "
            f"dominated by counting noise; N >= {needed} is required for the "
            f"tolerance to be resolvable at all."
        )
    if p_value < alpha and valid:
        explanation += (
            f" Note: the exact test rejects at alpha = {alpha} while the "
            f"margin rule accepts - the deviation is statistically resolved "
            f"but physically inside the tolerance band."
        )

    return CountingResult(
        valid=valid,
        norm_squared=float(est.value),
        statistic=float(statistic),
        threshold=float(threshold),
        p_value=float(p_value),
        total_counts=int(est.total_counts),
        n_shots=int(n_shots),
        budget_ok=budget_ok,
        explanation=explanation,
    )
