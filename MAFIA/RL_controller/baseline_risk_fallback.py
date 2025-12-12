"""
Baseline Risk Fallback Implementation (Market Baseline Strategy)

This module provides intelligent fallback strategies for computing baseline risk (σ_base)
when direct computation from RL action fails (NaN, Inf, or ≤ 0).

Strategy: Market Baseline Risk
- Uses equal-weight portfolio volatility on K assets
- Phản ánh rủi ro baseline của thị trường hiện tại
- Adaptive & không static config-dependent
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_baseline_risk_with_fallback(
    a_rl, cov_matrix, fallback_mode="market_baseline"
):
    """
    Compute baseline risk (σ_base) with intelligent fallback.

    Args:
        a_rl: (K,) RL action weights, should sum to 1
        cov_matrix: (K, K) covariance matrix of top-K assets
        fallback_mode: str, fallback strategy
                      - 'market_baseline': Equal-weight portfolio volatility (default)
                      - 'config': Use risk_market config value (deprecated)

    Returns:
        sigma_base: float, baseline risk (volatility)
        used_fallback: bool, True if fallback was used
        fallback_reason: str, description of why fallback was used (if applicable)
    """

    # 1. Try direct computation from a_rl
    try:
        a_rl_array = np.array(a_rl, dtype=float)
        if a_rl_array.ndim != 1:
            raise ValueError(f"a_rl must be 1D, got shape {a_rl_array.shape}")

        # σ_base = sqrt(a_rl @ Σ @ a_rl.T)
        variance = float(np.matmul(a_rl_array, np.matmul(cov_matrix, a_rl_array.T)))
        sigma_base = float(
            np.sqrt(max(variance, 0.0))
        )  # Clamp to avoid negative due to numerical errors

        # Check validity
        if sigma_base > 0 and np.isfinite(sigma_base):
            logger.debug(
                f"[BaselineRisk] Direct computation successful: σ_base={sigma_base:.6f}"
            )
            return sigma_base, False, None
        else:
            fallback_reason = (
                f"Invalid direct computation: σ_base={sigma_base} (not finite or ≤0)"
            )
            logger.warning(f"[BaselineRisk] {fallback_reason}")

    except Exception as e:
        fallback_reason = f"Direct computation failed: {str(e)}"
        logger.warning(f"[BaselineRisk] {fallback_reason}")

    # 2. Fallback: Market Baseline Risk (Equal-Weight Portfolio Volatility)
    if fallback_mode == "market_baseline":
        try:
            K = cov_matrix.shape[0]
            ones = np.ones(K)

            # σ_baseline = sqrt(1^T Σ 1 / K)
            # This is the average portfolio volatility if all assets have equal weight
            variance_equal_weight = float(
                np.matmul(ones, np.matmul(cov_matrix, ones.T))
            )
            sigma_base_market = float(
                np.sqrt(max(variance_equal_weight, 0.0))
            ) / np.sqrt(K)

            if sigma_base_market > 0 and np.isfinite(sigma_base_market):
                fallback_reason_detail = (
                    f"Used market baseline (equal-weight portfolio volatility)"
                )
                logger.info(
                    f"[BaselineRisk] Fallback activated: {fallback_reason_detail}"
                )
                logger.info(
                    f"[BaselineRisk] σ_base_market={sigma_base_market:.6f} (from 1^T Σ 1 / √K)"
                )
                return sigma_base_market, True, fallback_reason_detail
            else:
                raise ValueError(f"Market baseline invalid: σ={sigma_base_market}")

        except Exception as e:
            fallback_reason = f"Market baseline fallback failed: {str(e)}"
            logger.error(f"[BaselineRisk] {fallback_reason}")

    # 3. Final fallback: Use min-variance portfolio volatility as last resort
    try:
        logger.warning("[BaselineRisk] Attempting min-variance portfolio fallback...")
        sigma_min_var = estimate_min_variance_risk(cov_matrix)
        if (
            sigma_min_var is not None
            and sigma_min_var > 0
            and np.isfinite(sigma_min_var)
        ):
            logger.info(
                f"[BaselineRisk] Final fallback: min-variance σ={sigma_min_var:.6f}"
            )
            return sigma_min_var, True, "Min-variance portfolio fallback"
    except Exception as e:
        logger.warning(f"[BaselineRisk] Min-variance fallback also failed: {e}")

    # 4. Emergency fallback: Use small safe default
    sigma_emergency = 0.01  # 1% as absolute minimum
    logger.error(
        f"[BaselineRisk] All fallbacks exhausted, using emergency default: σ={sigma_emergency}"
    )
    return sigma_emergency, True, "Emergency default (0.01)"


def estimate_min_variance_risk(cov_matrix):
    """
    Approximate minimum achievable portfolio risk (volatility) under sum(x)=1, x≥0.

    Args:
        cov_matrix: (K, K) covariance matrix

    Returns:
        sigma_min_var: float, minimum variance portfolio volatility (or None if computation fails)
    """
    try:
        K = cov_matrix.shape[0]
        ones = np.ones(K)
        inv_cov = np.linalg.pinv(cov_matrix)
        weights = inv_cov @ ones

        if np.sum(weights) <= 1e-10:
            return None

        weights = weights / np.sum(weights)
        variance = float(np.matmul(weights, np.matmul(cov_matrix, weights.T)))
        return float(np.sqrt(max(variance, 0.0)))

    except Exception as e:
        logger.warning(f"[MinVariance] Computation failed: {e}")
        return None


def log_baseline_risk_diagnostics(
    a_rl, cov_matrix, sigma_base, used_fallback, fallback_reason
):
    """
    Log diagnostic information about baseline risk computation.

    Args:
        a_rl: (K,) RL action weights
        cov_matrix: (K, K) covariance matrix
        sigma_base: float, computed baseline risk
        used_fallback: bool, whether fallback was used
        fallback_reason: str, reason for fallback (if applicable)
    """
    K = len(a_rl)

    # Compute various statistics for diagnostics
    a_rl_array = np.array(a_rl, dtype=float)

    try:
        # Direct computation (for comparison)
        direct_var = float(np.matmul(a_rl_array, np.matmul(cov_matrix, a_rl_array.T)))
        direct_sigma = float(np.sqrt(max(direct_var, 0.0)))
    except:
        direct_sigma = None

    # Equal-weight comparison
    try:
        ones = np.ones(K)
        ew_var = float(np.matmul(ones, np.matmul(cov_matrix, ones.T)))
        ew_sigma = float(np.sqrt(ew_var)) / np.sqrt(K)
    except:
        ew_sigma = None

    # Min-variance comparison
    min_var_sigma = estimate_min_variance_risk(cov_matrix)

    # Log summary
    msg = (
        f"[BaselineRisk Diagnostics] K={K} | "
        f"σ_base={sigma_base:.6f} | "
        f"Fallback={'Yes' if used_fallback else 'No'}"
    )
    if used_fallback:
        msg += f" | Reason: {fallback_reason}"
    if direct_sigma is not None:
        msg += f" | Direct={direct_sigma:.6f}"
    if ew_sigma is not None:
        msg += f" | EqualWeight={ew_sigma:.6f}"
    if min_var_sigma is not None:
        msg += f" | MinVar={min_var_sigma:.6f}"

    logger.info(msg)


# ===== Integration with cbf_opt (Example) =====


def integrate_baseline_risk_fallback_example():
    """
    Example of how to integrate market baseline fallback into cbf_opt.

    In controllers.py, replace:
        sigma_base = None
        try:
            base_var = float(np.matmul(a_rl_opt, np.matmul(cov_matrix, a_rl_opt.T)))
            sigma_base = float(np.sqrt(max(base_var, 0.0)))
        except Exception:
            sigma_base = None

    With:
        from baseline_risk_fallback import compute_baseline_risk_with_fallback, log_baseline_risk_diagnostics

        sigma_base, used_fallback, fallback_reason = compute_baseline_risk_with_fallback(
            a_rl=a_rl_opt,
            cov_matrix=cov_matrix,
            fallback_mode='market_baseline'
        )

        log_baseline_risk_diagnostics(
            a_rl=a_rl_opt,
            cov_matrix=cov_matrix,
            sigma_base=sigma_base,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason
        )
    """
    pass


if __name__ == "__main__":
    """
    Simple test of baseline risk fallback.
    """
    import logging

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    # Test 1: Valid case
    print("=" * 60)
    print("Test 1: Valid RL action")
    print("=" * 60)
    K = 5
    cov_matrix_1 = np.array(
        [
            [0.04, 0.01, 0.00, 0.01, 0.00],
            [0.01, 0.05, 0.01, 0.00, 0.01],
            [0.00, 0.01, 0.03, 0.00, 0.00],
            [0.01, 0.00, 0.00, 0.06, 0.01],
            [0.00, 0.01, 0.00, 0.01, 0.04],
        ]
    )
    a_rl_1 = np.array([0.3, 0.2, 0.2, 0.15, 0.15])

    sigma_1, fallback_1, reason_1 = compute_baseline_risk_with_fallback(
        a_rl_1, cov_matrix_1
    )
    log_baseline_risk_diagnostics(a_rl_1, cov_matrix_1, sigma_1, fallback_1, reason_1)
    print(f"Result: σ={sigma_1:.6f}, Fallback={fallback_1}\n")

    # Test 2: NaN/Inf case (trigger fallback)
    print("=" * 60)
    print("Test 2: Invalid covariance (NaN) - should trigger fallback")
    print("=" * 60)
    cov_matrix_2 = cov_matrix_1.copy()
    cov_matrix_2[0, 0] = np.nan  # Inject NaN to trigger fallback
    a_rl_2 = np.array([0.3, 0.2, 0.2, 0.15, 0.15])

    sigma_2, fallback_2, reason_2 = compute_baseline_risk_with_fallback(
        a_rl_2, cov_matrix_2
    )
    log_baseline_risk_diagnostics(a_rl_2, cov_matrix_2, sigma_2, fallback_2, reason_2)
    print(f"Result: σ={sigma_2:.6f}, Fallback={fallback_2}\n")

    # Test 3: All-zero weights (edge case)
    print("=" * 60)
    print("Test 3: Zero weights")
    print("=" * 60)
    a_rl_3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    sigma_3, fallback_3, reason_3 = compute_baseline_risk_with_fallback(
        a_rl_3, cov_matrix_1
    )
    log_baseline_risk_diagnostics(a_rl_3, cov_matrix_1, sigma_3, fallback_3, reason_3)
    print(f"Result: σ={sigma_3:.6f}, Fallback={fallback_3}\n")
