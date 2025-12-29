"""
Portfolio Allocator Reward Function Implementation
Implements simplified 2-component reward: return + divergence penalty

According to spec: r_t = w_return · r_return - λ_js · D_JS(a_alloc || a_final)
"""

import numpy as np
from scipy.special import rel_entr

# LiveDisplay smart_print for terminal-safe logging
try:
    from utils.display_integration import smart_print
except ImportError:
    smart_print = print  # Fallback


def _normalize_prob(x):
    """Normalize vector to probability distribution (sum=1, all≥0)."""
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)  # Clamp negatives to 0
    s = np.sum(x)
    if s > 1e-8:
        return x / s
    else:
        return np.ones_like(x) / len(x)


def compute_log_return(portfolio_return):
    """
    Compute log return for the day.

    Args:
        portfolio_return (float): Daily portfolio return (e.g., 0.02 for +2%, -0.01 for -1%)

    Returns:
        float: log(1 + portfolio_return)
    """
    # Clamp to avoid log of negative values
    portfolio_return = np.asarray(portfolio_return, dtype=np.float64).item()
    if portfolio_return <= -1.0:
        # Avoid log(non-positive), return large negative value
        return -10.0
    return float(np.log(1.0 + portfolio_return))


def compute_jensen_shannon_divergence(a_alloc, a_final):
    """
    Compute Jensen-Shannon divergence between a_alloc and a_final.

    JS(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m), where m = 0.5 * (p + q)

    Args:
        a_alloc (array-like): Portfolio Allocator raw output (K,)
        a_final (array-like): Controller's final adjusted allocation (K,)

    Returns:
        float: JS divergence ∈ [0, 1]
    """
    try:
        a_alloc = np.asarray(a_alloc, dtype=np.float64).flatten()
        a_final = np.asarray(a_final, dtype=np.float64).flatten()

        # Normalize to probability distributions
        p = _normalize_prob(a_alloc)
        q = _normalize_prob(a_final)

        # Compute JS divergence
        m = 0.5 * (p + q)
        kl_pq = np.sum(rel_entr(p, m))  # KL(p||m)
        kl_qp = np.sum(rel_entr(q, m))  # KL(q||m)
        js_div = 0.5 * kl_pq + 0.5 * kl_qp

        # Clamp NaN/inf
        if np.isnan(js_div) or np.isinf(js_div):
            js_div = 0.0

        return float(js_div)
    except Exception as e:
        smart_print(
            f"[Warning] compute_jensen_shannon_divergence failed: {e}", flush=True
        )
        return 0.0


def compute_reward(
    portfolio_return,
    a_alloc,
    a_final,
    w_return=1.0,
    lambda_js=0.1,
    reward_scale=1.0,
    normalize_return=False,
    running_mean_return=None,
    running_std_return=None,
):
    """
    Compute Portfolio Allocator reward using simplified 2-component function.

    Formula: r_t = scale × (w_return · r_return - λ_js · D_JS(a_alloc || a_final))

    Args:
        portfolio_return (float): Daily portfolio return from environment
        a_alloc (array-like): RL agent's raw allocation output (K,)
        a_final (array-like): Controller's final adjusted allocation (K,)
        w_return (float): Weight for return component (default 1.0)
        lambda_js (float): Weight for JS divergence penalty (default 0.1)
        reward_scale (float): Scaling factor for reward (default 1.0, recommended 100.0)
            - Amplifies gradient signal for faster TD3 learning
            - Daily log returns ~0.001, scaling to ~0.1 improves convergence
        normalize_return (bool): Whether to normalize return component
        running_mean_return (float): Running mean for normalization
        running_std_return (float): Running std for normalization

    Returns:
        dict: {
            'reward_total': float (scaled),
            'reward_unscaled': float (before scaling, for debugging),
            'r_return': float,
            'r_divergence': float,
            'js_divergence': float,
            'log_return': float,
            'reward_scale': float
        }
    """
    # Compute components
    log_return = compute_log_return(portfolio_return)
    js_divergence = compute_jensen_shannon_divergence(a_alloc, a_final)

    # Scale components
    r_return = w_return * log_return
    r_divergence = -lambda_js * js_divergence

    # Optionally normalize return component
    if (
        normalize_return
        and running_mean_return is not None
        and running_std_return is not None
    ):
        if running_std_return > 1e-8:
            r_return = (r_return - running_mean_return) / (running_std_return + 1e-8)

    # Total reward (unscaled)
    r_unscaled = r_return + r_divergence

    # Apply scaling for stronger gradient signal
    r_total = reward_scale * r_unscaled

    # Clamp NaN/inf
    if np.isnan(r_total) or np.isinf(r_total):
        r_total = 0.0
        r_unscaled = 0.0

    return {
        "reward_total": float(r_total),
        "reward_unscaled": float(r_unscaled),
        "r_return": float(r_return),
        "r_divergence": float(r_divergence),
        "js_divergence": float(js_divergence),
        "log_return": float(log_return),
        "reward_scale": float(reward_scale),
    }


class RewardNormalizer:
    """
    Running normalizer for reward components using EMA.

    Maintains running mean and std for normalization.
    """

    def __init__(self, alpha=0.01):
        """
        Initialize normalizer.

        Args:
            alpha (float): EMA coefficient (default 0.01)
        """
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0  # variance
        self.count = 0

    def update(self, value):
        """
        Update running statistics with new value.

        Args:
            value (float): New observation
        """
        value = float(value)

        if self.count == 0:
            self.mean = value
            self.var = 0.0
            self.count = 1
        else:
            # EMA update for mean
            self.mean = (1 - self.alpha) * self.mean + self.alpha * value

            # EMA update for variance
            delta = value - self.mean
            self.var = (1 - self.alpha) * self.var + self.alpha * (delta**2)
            self.count += 1

    def normalize(self, value):
        """
        Normalize value using running statistics.

        Args:
            value (float): Value to normalize

        Returns:
            float: Normalized value
        """
        value = float(value)
        std = np.sqrt(max(self.var, 1e-8))
        return (value - self.mean) / (std + 1e-8)

    def get_stats(self):
        """Get current running statistics."""
        return {
            "mean": float(self.mean),
            "std": float(np.sqrt(max(self.var, 1e-8))),
            "count": self.count,
        }


def debug_log_reward_components(
    log_return,
    r_return,
    js_divergence,
    r_divergence,
    r_total,
    verbose=False,
):
    """
    Debug log for reward components.

    Args:
        log_return (float): Log return component
        r_return (float): Scaled return
        js_divergence (float): JS divergence value
        r_divergence (float): Scaled divergence penalty
        r_total (float): Total reward
        verbose (bool): Whether to print
    """
    if verbose:
        smart_print(
            f"[Reward] log_return={log_return:.6f} | "
            f"r_return={r_return:.6f} | "
            f"js_div={js_divergence:.6f} | "
            f"r_divergence={r_divergence:.6f} | "
            f"r_total={r_total:.6f}",
            flush=True,
        )
