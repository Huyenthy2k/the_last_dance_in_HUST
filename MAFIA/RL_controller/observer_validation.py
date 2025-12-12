"""
Observer Validation Module for Walk-Forward Training

Implements composite score computation for checkpoint selection:
Score = w_sharpe × SR + w_ic × IC + w_f1 × F1

Components:
1. Top-K Sharpe Ratio: Backtest of equal-weight Top-K portfolio
2. Information Coefficient: Spearman correlation of predictions vs returns
3. Direction F1-Macro: Classification accuracy for Bull/Bear/Side regimes
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from typing import Dict, List, Optional, Tuple
import warnings

# LiveDisplay smart_print for terminal-safe logging
try:
    from utils.display_integration import smart_print
except ImportError:
    smart_print = print  # Fallback


def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute annualized Sharpe Ratio from daily returns.

    Args:
        returns: Array of daily returns
        risk_free_rate: Daily risk-free rate (default 0)

    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess < 1e-8:
        return 0.0

    # Annualize: multiply by sqrt(252)
    daily_sharpe = mean_excess / std_excess
    annualized_sharpe = daily_sharpe * np.sqrt(252)

    return float(annualized_sharpe)


def backtest_topk_equalweight(
    daily_topk_indices: List[np.ndarray],
    daily_returns: np.ndarray,
    rebalance_interval: int = 14,
) -> np.ndarray:
    """
    Backtest equal-weight Top-K portfolio with periodic rebalancing.

    Args:
        daily_topk_indices: List of Top-K indices per day (length = num_days)
        daily_returns: (num_days, num_stocks) array of daily returns
        rebalance_interval: Days between rebalances

    Returns:
        Array of daily portfolio returns
    """
    num_days = len(daily_topk_indices)
    portfolio_returns = []
    current_topk = None

    for day in range(num_days):
        # Rebalance on first day or every rebalance_interval days
        if day == 0 or (day % rebalance_interval == 0):
            current_topk = daily_topk_indices[day]

        if current_topk is None or len(current_topk) == 0:
            portfolio_returns.append(0.0)
            continue

        # Equal-weight portfolio return
        valid_indices = current_topk[
            (current_topk >= 0) & (current_topk < daily_returns.shape[1])
        ]
        if len(valid_indices) == 0:
            portfolio_returns.append(0.0)
        else:
            day_return = np.mean(daily_returns[day, valid_indices])
            portfolio_returns.append(float(day_return))

    return np.array(portfolio_returns)


def compute_information_coefficient(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
) -> float:
    """
    Compute Information Coefficient (Spearman rank correlation).

    Args:
        predictions: Predicted scores/logits (N,)
        actual_returns: Actual future returns (N,)

    Returns:
        Spearman correlation coefficient
    """
    if len(predictions) < 3:
        return 0.0

    # Flatten if needed
    predictions = np.asarray(predictions).flatten()
    actual_returns = np.asarray(actual_returns).flatten()

    # Remove NaN/Inf
    valid_mask = ~(
        np.isnan(predictions)
        | np.isnan(actual_returns)
        | np.isinf(predictions)
        | np.isinf(actual_returns)
    )
    predictions = predictions[valid_mask]
    actual_returns = actual_returns[valid_mask]

    if len(predictions) < 3:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = spearmanr(predictions, actual_returns)

    if np.isnan(corr):
        return 0.0

    return float(corr)


def compute_direction_f1(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> float:
    """
    Compute F1-Macro for direction classification.

    Args:
        predictions: Predicted direction labels (0=Bear, 1=Side, 2=Bull)
        actuals: Actual direction labels

    Returns:
        F1-Macro score
    """
    predictions = np.asarray(predictions).flatten().astype(int)
    actuals = np.asarray(actuals).flatten().astype(int)

    if len(predictions) == 0 or len(actuals) == 0:
        return 0.0

    # Clip to valid range [0, 2]
    predictions = np.clip(predictions, 0, 2)
    actuals = np.clip(actuals, 0, 2)

    try:
        f1 = f1_score(actuals, predictions, average="macro", zero_division=0.0)
        return float(f1)
    except Exception:
        return 0.0


class CompositeScoreNormalizer:
    """
    Normalizer for composite score components using running min-max.

    Tracks min/max values across epochs for stable normalization.
    """

    def __init__(self):
        self.sharpe_min = None
        self.sharpe_max = None
        self.ic_min = None
        self.ic_max = None
        self.f1_min = None
        self.f1_max = None

    def update(self, sharpe: float, ic: float, f1: float):
        """Update running min/max with new values."""
        # Sharpe
        if self.sharpe_min is None:
            self.sharpe_min = sharpe
            self.sharpe_max = sharpe
        else:
            self.sharpe_min = min(self.sharpe_min, sharpe)
            self.sharpe_max = max(self.sharpe_max, sharpe)

        # IC
        if self.ic_min is None:
            self.ic_min = ic
            self.ic_max = ic
        else:
            self.ic_min = min(self.ic_min, ic)
            self.ic_max = max(self.ic_max, ic)

        # F1
        if self.f1_min is None:
            self.f1_min = f1
            self.f1_max = f1
        else:
            self.f1_min = min(self.f1_min, f1)
            self.f1_max = max(self.f1_max, f1)

    def normalize(
        self, value: float, vmin: Optional[float], vmax: Optional[float]
    ) -> float:
        """Min-max normalize a value to [0, 1]."""
        if vmin is None or vmax is None:
            return 0.5
        if vmax - vmin < 1e-8:
            return 0.5
        return (value - vmin) / (vmax - vmin)

    def get_normalized(
        self, sharpe: float, ic: float, f1: float
    ) -> Tuple[float, float, float]:
        """Get normalized values for all components."""
        norm_sharpe = self.normalize(sharpe, self.sharpe_min, self.sharpe_max)
        norm_ic = self.normalize(ic, self.ic_min, self.ic_max)
        norm_f1 = self.normalize(f1, self.f1_min, self.f1_max)
        return norm_sharpe, norm_ic, norm_f1


def compute_composite_score(
    sharpe: float,
    ic: float,
    f1: float,
    w_sharpe: float = 0.5,
    w_ic: float = 0.3,
    w_f1: float = 0.2,
    normalizer: Optional[CompositeScoreNormalizer] = None,
) -> Dict[str, float]:
    """
    Compute composite validation score for checkpoint selection.

    Score = w_sharpe × norm(SR) + w_ic × norm(IC) + w_f1 × norm(F1)

    Args:
        sharpe: Top-K Sharpe Ratio
        ic: Information Coefficient (Spearman correlation)
        f1: Direction F1-Macro score
        w_sharpe: Weight for Sharpe (default 0.5)
        w_ic: Weight for IC (default 0.3)
        w_f1: Weight for F1 (default 0.2)
        normalizer: Optional normalizer for min-max normalization

    Returns:
        Dict with composite_score and individual components
    """
    if normalizer is not None:
        normalizer.update(sharpe, ic, f1)
        norm_sharpe, norm_ic, norm_f1 = normalizer.get_normalized(sharpe, ic, f1)
    else:
        # Without normalizer, use raw values (not recommended)
        norm_sharpe = sharpe
        norm_ic = ic
        norm_f1 = f1

    composite = w_sharpe * norm_sharpe + w_ic * norm_ic + w_f1 * norm_f1

    return {
        "composite_score": float(composite),
        "sharpe": float(sharpe),
        "ic": float(ic),
        "f1_direction": float(f1),
        "norm_sharpe": float(norm_sharpe),
        "norm_ic": float(norm_ic),
        "norm_f1": float(norm_f1),
    }


def validate_observer_epoch(
    observer,
    valid_data: Dict,
    topk: int = 10,
    rebalance_interval: int = 14,
    config=None,
) -> Dict[str, float]:
    """
    Run full validation for one epoch and compute composite score.

    Args:
        observer: MAFIAObserver instance
        valid_data: Validation data dict with keys:
            - 'features': Input features for observer
            - 'returns': (num_days, num_stocks) daily returns
            - 'direction_labels': Ground truth direction labels
        topk: Number of top stocks to select
        rebalance_interval: Days between rebalances
        config: Config object for score weights

    Returns:
        Dict with validation metrics and composite score
    """
    # Get score weights from config
    if config is not None:
        w_sharpe = getattr(config, "walkforward_score_w_sharpe", 0.5)
        w_ic = getattr(config, "walkforward_score_w_ic", 0.3)
        w_f1 = getattr(config, "walkforward_score_w_f1", 0.2)
    else:
        w_sharpe, w_ic, w_f1 = 0.5, 0.3, 0.2

    # Run observer predictions on validation data
    daily_topk_indices = []
    daily_predictions = []
    daily_direction_preds = []

    num_days = len(valid_data.get("features", []))

    for day in range(num_days):
        # Get observer predictions for this day
        day_features = valid_data["features"][day]

        # Call observer.predict() or equivalent
        # This is a placeholder - actual implementation depends on observer interface
        try:
            topk_indices, risk_eta, direction_logits = observer.predict(
                day_features, mode="valid"
            )
            daily_topk_indices.append(np.array(topk_indices))
            daily_direction_preds.append(np.argmax(direction_logits))

            # Get market_logits for IC calculation
            if hasattr(observer, "get_market_logits"):
                market_logits = observer.get_market_logits(day_features)
                daily_predictions.append(market_logits)
        except Exception as e:
            smart_print(f"[Warning] Observer prediction failed at day {day}: {e}")
            daily_topk_indices.append(np.array([]))
            daily_direction_preds.append(1)  # Default to Side

    # 1. Compute Top-K Sharpe Ratio
    returns = valid_data.get("returns", np.array([]))
    if len(returns) > 0 and len(daily_topk_indices) > 0:
        portfolio_returns = backtest_topk_equalweight(
            daily_topk_indices, returns, rebalance_interval
        )
        sharpe = compute_sharpe_ratio(portfolio_returns)
    else:
        sharpe = 0.0

    # 2. Compute Information Coefficient
    if len(daily_predictions) > 0:
        # Aggregate predictions and returns for IC
        all_predictions = np.concatenate(daily_predictions)
        all_returns = returns.flatten() if len(returns) > 0 else np.array([])
        # Align lengths
        min_len = min(len(all_predictions), len(all_returns))
        if min_len > 0:
            ic = compute_information_coefficient(
                all_predictions[:min_len], all_returns[:min_len]
            )
        else:
            ic = 0.0
    else:
        ic = 0.0

    # 3. Compute Direction F1-Macro
    direction_labels = valid_data.get("direction_labels", np.array([]))
    if len(direction_labels) > 0 and len(daily_direction_preds) > 0:
        f1 = compute_direction_f1(np.array(daily_direction_preds), direction_labels)
    else:
        f1 = 0.0

    # Compute composite score
    result = compute_composite_score(
        sharpe=sharpe,
        ic=ic,
        f1=f1,
        w_sharpe=w_sharpe,
        w_ic=w_ic,
        w_f1=w_f1,
    )

    return result


class ObserverCheckpointTracker:
    """
    Tracks best observer checkpoint based on composite validation score.
    """

    def __init__(self, config=None):
        self.best_score = float("-inf")
        self.best_epoch = -1
        self.best_checkpoint = None
        self.history = []
        self.normalizer = CompositeScoreNormalizer()

        # Get weights from config
        if config is not None:
            self.w_sharpe = getattr(config, "walkforward_score_w_sharpe", 0.5)
            self.w_ic = getattr(config, "walkforward_score_w_ic", 0.3)
            self.w_f1 = getattr(config, "walkforward_score_w_f1", 0.2)
        else:
            self.w_sharpe, self.w_ic, self.w_f1 = 0.5, 0.3, 0.2

    def update(
        self,
        epoch: int,
        sharpe: float,
        ic: float,
        f1: float,
        checkpoint_path: Optional[str] = None,
    ) -> bool:
        """
        Update tracker with new epoch metrics.

        Args:
            epoch: Current epoch number
            sharpe: Top-K Sharpe Ratio
            ic: Information Coefficient
            f1: Direction F1-Macro
            checkpoint_path: Path to saved checkpoint

        Returns:
            True if this is a new best checkpoint
        """
        # Compute composite score with normalization
        result = compute_composite_score(
            sharpe=sharpe,
            ic=ic,
            f1=f1,
            w_sharpe=self.w_sharpe,
            w_ic=self.w_ic,
            w_f1=self.w_f1,
            normalizer=self.normalizer,
        )

        score = result["composite_score"]
        self.history.append(
            {
                "epoch": epoch,
                "score": score,
                "sharpe": sharpe,
                "ic": ic,
                "f1": f1,
                "checkpoint_path": checkpoint_path,
            }
        )

        is_best = score > self.best_score
        if is_best:
            self.best_score = score
            self.best_epoch = epoch
            self.best_checkpoint = checkpoint_path

        return is_best

    def get_best(self) -> Dict:
        """Get best checkpoint info."""
        return {
            "epoch": self.best_epoch,
            "score": self.best_score,
            "checkpoint_path": self.best_checkpoint,
        }

    def get_history(self) -> List[Dict]:
        """Get full validation history."""
        return self.history
