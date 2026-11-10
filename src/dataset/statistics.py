"""Statistical evaluation and confidence interval routines for BB84 QBER.

This module provides:
1. `compute_wilson_score_interval`: Statistically exact Wilson score confidence intervals
   for binomial proportions, ensuring physical [0, 1] bounds and proper coverage at small n
   and extreme proportions (k = 0, k = n).
2. `classify_feasibility`: Evaluates whether a candidate policy is statistically Feasible,
   Infeasible, or Indeterminate relative to a target protocol error budget.
3. `compute_empirical_frequencies`: Computes sample pass/abort frequencies and their
   associated binomial confidence intervals over repeated simulation trials.
"""

import math
from typing import NamedTuple, Sequence, Tuple
from src.dataset.schema import StatisticalFeasibility


class WilsonInterval(NamedTuple):
    """Container for Wilson score confidence interval outputs.

    Attributes:
        center: Center point of the Wilson interval in [0.0, 1.0].
        lcb: Lower confidence bound in [0.0, 1.0].
        ucb: Upper confidence bound in [0.0, 1.0].
        margin: Half-width / margin of the interval.
        confidence: Nominal confidence level (e.g. 0.95).
        k: Observed successes / errors.
        n: Total trials / bits.
    """

    center: float
    lcb: float
    ucb: float
    margin: float
    confidence: float
    k: int
    n: int


class EmpiricalOutcomeStats(NamedTuple):
    """Container for repeated-trial empirical protocol pass/abort frequencies.

    Attributes:
        n_trials: Total independent simulation runs evaluated (R).
        n_pass: Number of runs where sample QBER was at or below budget.
        n_abort: Number of runs where sample QBER exceeded budget.
        pass_frequency: Empirical pass rate (n_pass / n_trials).
        abort_frequency: Empirical abort rate (n_abort / n_trials).
        pass_lcb: Wilson 95% lower bound on pass frequency.
        pass_ucb: Wilson 95% upper bound on pass frequency.
        abort_lcb: Wilson 95% lower bound on abort frequency.
        abort_ucb: Wilson 95% upper bound on abort frequency.
    """

    n_trials: int
    n_pass: int
    n_abort: int
    pass_frequency: float
    abort_frequency: float
    pass_lcb: float
    pass_ucb: float
    abort_lcb: float
    abort_ucb: float


def get_standard_normal_quantile(confidence: float) -> float:
    """Return two-sided standard normal critical value z for a given confidence level.

    Args:
        confidence: Desired two-sided confidence level in (0.0, 1.0), e.g. 0.95 or 0.99.

    Returns:
        float: Two-sided critical value z_crit.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"Confidence must be strictly between 0 and 1, got {confidence}")

    # Standard precomputed quantiles for high precision
    if abs(confidence - 0.95) < 1e-6:
        return 1.959963984540054
    if abs(confidence - 0.99) < 1e-6:
        return 2.5758293035489004
    if abs(confidence - 0.90) < 1e-6:
        return 1.6448536269514722

    # Normal approximation via inverse error function: z = sqrt(2) * erfinv(confidence)
    # Using scipy.special if available, otherwise Abramowitz-Stegun approximation
    try:
        from scipy.special import erfinv

        return float(math.sqrt(2.0) * erfinv(confidence))
    except Exception:
        # Fallback accurate approximation
        alpha = 1.0 - confidence
        p = 1.0 - alpha / 2.0
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0)
        return float(z)


def compute_wilson_score_interval(
    k: int,
    n: int,
    confidence: float = 0.95,
) -> WilsonInterval:
    """Compute the exact two-sided Wilson score confidence interval for a binomial proportion.

    Formula:
        p_hat = k / n
        center = (p_hat + z^2 / (2n)) / (1 + z^2 / n) = (k + z^2 / 2) / (n + z^2)
        margin = (z / (1 + z^2 / n)) * sqrt(p_hat * (1 - p_hat) / n + z^2 / (4n^2))
        lcb = max(0.0, center - margin)
        ucb = min(1.0, center + margin)

    Args:
        k: Observed successes / errors (0 <= k <= n).
        n: Total trials / evaluated bits (n >= 1).
        confidence: Two-sided confidence level in (0.0, 1.0), default 0.95.

    Returns:
        WilsonInterval: NamedTuple containing center, lcb, ucb, margin, confidence, k, n.

    Raises:
        ValueError: If n < 1 or k < 0 or k > n.
    """
    if n < 1:
        raise ValueError(f"Total trials n must be >= 1, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"Observed count k must be between 0 and n ({n}), got {k}")

    z = get_standard_normal_quantile(confidence)
    z2 = z * z
    n_float = float(n)
    k_float = float(k)
    p_hat = k_float / n_float

    denom = n_float + z2
    center = (k_float + 0.5 * z2) / denom

    # Radicand: n * p_hat * (1 - p_hat) + z^2 / 4
    # Bounded >= 0 analytically since p_hat in [0, 1]
    variance_term = p_hat * (1.0 - p_hat) / n_float
    radicand = variance_term + (z2 / (4.0 * n_float * n_float))
    margin = (z / (1.0 + z2 / n_float)) * math.sqrt(max(0.0, radicand))

    lcb = max(0.0, center - margin)
    ucb = min(1.0, center + margin)

    # Clean exact zero and one boundaries
    if k == 0:
        lcb = 0.0
    if k == n:
        ucb = 1.0

    return WilsonInterval(
        center=float(center),
        lcb=float(lcb),
        ucb=float(ucb),
        margin=float(margin),
        confidence=float(confidence),
        k=int(k),
        n=int(n),
    )


def classify_feasibility(
    q_lcb: float,
    q_ucb: float,
    q_budget: float,
) -> StatisticalFeasibility:
    """Classify candidate policy feasibility using the tri-state statistical partition.

    Classification Rule:
        Feasible (Category A): Q_UCB <= Q_budget
        Infeasible (Category B): Q_LCB > Q_budget
        Indeterminate (Category C): Q_LCB <= Q_budget < Q_UCB

    Args:
        q_lcb: Wilson score lower confidence bound in [0.0, 1.0].
        q_ucb: Wilson score upper confidence bound in [0.0, 1.0].
        q_budget: Protocol security abort budget in [0.0, 1.0].

    Returns:
        StatisticalFeasibility: FEASIBLE, INFEASIBLE, or INDETERMINATE.

    Raises:
        ValueError: If q_lcb > q_ucb or if arguments are outside [0.0, 1.0].
    """
    if not (0.0 <= q_lcb <= 1.0):
        raise ValueError(f"q_lcb must be in [0.0, 1.0], got {q_lcb}")
    if not (0.0 <= q_ucb <= 1.0):
        raise ValueError(f"q_ucb must be in [0.0, 1.0], got {q_ucb}")
    if q_lcb > q_ucb:
        raise ValueError(f"q_lcb ({q_lcb}) cannot exceed q_ucb ({q_ucb})")
    if not (0.0 <= q_budget <= 1.0):
        raise ValueError(f"q_budget must be in [0.0, 1.0], got {q_budget}")

    if q_ucb <= q_budget:
        return StatisticalFeasibility.FEASIBLE
    elif q_lcb > q_budget:
        return StatisticalFeasibility.INFEASIBLE
    else:
        return StatisticalFeasibility.INDETERMINATE


def compute_empirical_frequencies(
    abort_outcomes: Sequence[bool],
    confidence: float = 0.95,
) -> EmpiricalOutcomeStats:
    """Compute empirical pass/abort frequencies and binomial confidence bounds over repeated runs.

    Args:
        abort_outcomes: Sequence of boolean outcomes where True = Abort, False = Pass.
        confidence: Confidence level for binomial intervals on empirical rates.

    Returns:
        EmpiricalOutcomeStats: Container with trial counts, rates, and Wilson intervals.

    Raises:
        ValueError: If abort_outcomes is empty.
    """
    n_trials = len(abort_outcomes)
    if n_trials == 0:
        raise ValueError("Cannot compute empirical frequencies on empty outcome sequence")

    n_abort = int(sum(bool(x) for x in abort_outcomes))
    n_pass = n_trials - n_abort

    pass_freq = float(n_pass / n_trials)
    abort_freq = float(n_abort / n_trials)

    pass_interval = compute_wilson_score_interval(n_pass, n_trials, confidence=confidence)
    abort_interval = compute_wilson_score_interval(n_abort, n_trials, confidence=confidence)

    return EmpiricalOutcomeStats(
        n_trials=n_trials,
        n_pass=n_pass,
        n_abort=n_abort,
        pass_frequency=pass_freq,
        abort_frequency=abort_freq,
        pass_lcb=pass_interval.lcb,
        pass_ucb=pass_interval.ucb,
        abort_lcb=abort_interval.lcb,
        abort_ucb=abort_interval.ucb,
    )

