"""
Diebold-Mariano test, HAC-corrected, per horizon day (master §8.9, §13.4).
Chosen over a naive paired t-test since per-row forecast errors are
autocorrelated (both across consecutive days and across horizon days
within one forecast).
"""

import numpy as np


def _newey_west_variance(d: np.ndarray, max_lag: int) -> float:
    """HAC (Newey-West) variance estimate of the mean of d."""
    n = len(d)
    d_centered = d - d.mean()
    gamma_0 = np.sum(d_centered ** 2) / n
    variance = gamma_0
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        gamma_lag = np.sum(d_centered[lag:] * d_centered[:-lag]) / n
        variance += 2 * weight * gamma_lag
    return variance / n  # variance of the MEAN, not of d itself


def diebold_mariano_test(loss_model_a: np.ndarray, loss_model_b: np.ndarray,
                          max_lag: int = 10) -> tuple:
    """
    Tests whether model_a's loss is significantly different from model_b's.
    loss_model_a/b: per-sample squared error for ONE horizon day (1D arrays,
    same length, same samples).
    max_lag: HAC truncation lag, approx horizon length per master §8.9.

    Returns (dm_statistic, p_value). Negative statistic + small p-value
    means model_a has significantly LOWER loss than model_b.
    """
    d = loss_model_a - loss_model_b
    mean_d = d.mean()
    var_d = _newey_west_variance(d, max_lag)

    if var_d <= 0:
        return 0.0, 1.0  # degenerate case, no distinguishable difference

    dm_stat = mean_d / np.sqrt(var_d)

    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return float(dm_stat), float(p_value)


def holm_correction(p_values: list) -> list:
    """
    Holm-Bonferroni step-down correction across multiple p-values
    (master §8.9, §13.4 -- required across the 10 per-horizon-day tests
    within a single tier comparison).
    Returns corrected p-values in the ORIGINAL order passed in.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    corrected = [0.0] * n
    max_so_far = 0.0
    for rank, (original_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        adjusted = max(adjusted, max_so_far)  # enforce monotonicity
        adjusted = min(adjusted, 1.0)
        max_so_far = adjusted
        corrected[original_idx] = adjusted

    return corrected