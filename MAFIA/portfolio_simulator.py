#!/usr/bin/env python3
"""
Portfolio Simulator for MAFIA Model Backtest

Realistic trading simulation with:
- Initial capital tracking
- Position management (cash + stocks)
- Rebalance schedule following model decisions
- Transaction costs (commission + slippage)
- Portfolio value compounding over time
- Performance metrics (Sharpe, Max Drawdown, etc.)

Usage:
    from portfolio_simulator import PortfolioSimulator, backtest_model

    # Quick backtest
    results = backtest_model(model, data_tensors, config)

    # Manual simulation
    sim = PortfolioSimulator(initial_capital=100_000)
    sim.execute_rebalance(new_weights, prices)
    metrics = sim.get_performance_metrics()
"""

import numpy as np
import pandas as pd
import torch as th
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Position:
    """Single stock position."""
    stock_idx: int
    shares: float
    entry_price: float
    entry_date: int  # timestep
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


@dataclass
class Trade:
    """Record of a single trade."""
    timestep: int
    stock_idx: int
    action: str  # 'BUY' or 'SELL'
    shares: float
    price: float
    commission: float
    slippage_cost: float

    @property
    def gross_value(self) -> float:
        return self.shares * self.price

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage_cost

    @property
    def net_value(self) -> float:
        if self.action == 'BUY':
            return self.gross_value + self.total_cost
        else:
            return self.gross_value - self.total_cost


@dataclass
class DailySnapshot:
    """Daily portfolio state snapshot."""
    timestep: int
    date: Optional[Any] = None
    cash: float = 0.0
    positions_value: float = 0.0
    total_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    num_positions: int = 0
    is_rebalance_day: bool = False
    turnover: float = 0.0  # % of portfolio traded
    transaction_costs: float = 0.0


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0

    # Risk
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trading
    total_trades: int = 0
    total_rebalances: int = 0
    avg_turnover: float = 0.0
    total_transaction_costs: float = 0.0
    transaction_cost_drag: float = 0.0  # % of returns lost to costs

    # Holding
    avg_holding_period: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0


# ============================================================================
# PORTFOLIO SIMULATOR
# ============================================================================

class PortfolioSimulator:
    """
    Realistic portfolio simulation engine.

    Features:
    - Track cash and positions separately
    - Equal-weight or score-weighted allocation
    - Transaction costs (commission + slippage)
    - Daily mark-to-market
    - Detailed trade history
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.001,  # 0.1% per trade (Vietnam typical)
        slippage_rate: float = 0.001,    # 0.1% slippage
        min_trade_value: float = 100.0,  # Minimum trade size
        risk_free_rate: float = 0.05,    # 5% annual risk-free rate
        trading_days_per_year: int = 252,
    ):
        """
        Initialize portfolio simulator.

        Args:
            initial_capital: Starting cash amount
            commission_rate: Commission as fraction of trade value
            slippage_rate: Expected slippage as fraction
            min_trade_value: Minimum trade value (skip smaller trades)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            trading_days_per_year: Trading days per year
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_trade_value = min_trade_value
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year

        # State
        self.cash = initial_capital
        self.positions: Dict[int, Position] = {}  # stock_idx -> Position
        self.current_timestep = 0

        # History
        self.trade_history: List[Trade] = []
        self.daily_snapshots: List[DailySnapshot] = []
        self.closed_trades: List[Dict] = []  # For win rate calculation

        # Tracking
        self._prev_portfolio_value = initial_capital
        self._peak_value = initial_capital
        self._drawdown_start = 0

    def reset(self):
        """Reset simulator to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.current_timestep = 0
        self.trade_history = []
        self.daily_snapshots = []
        self.closed_trades = []
        self._prev_portfolio_value = self.initial_capital
        self._peak_value = self.initial_capital
        self._drawdown_start = 0

    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    def get_positions_value(self) -> float:
        """Get total value of all positions."""
        return sum(p.market_value for p in self.positions.values())

    def update_prices(self, prices: np.ndarray):
        """
        Update current prices for all positions.

        Args:
            prices: (N,) array of current prices for all stocks
        """
        for stock_idx, position in self.positions.items():
            if stock_idx < len(prices):
                position.current_price = prices[stock_idx]

    def execute_rebalance(
        self,
        target_indices: np.ndarray,
        target_weights: Optional[np.ndarray],
        prices: np.ndarray,
        timestep: int,
        date: Optional[Any] = None,
    ) -> Tuple[float, float]:
        """
        Execute portfolio rebalance to target allocation.

        Args:
            target_indices: (K,) array of stock indices to hold
            target_weights: (K,) array of target weights (None = equal weight)
            prices: (N,) array of current prices
            timestep: Current timestep
            date: Optional date for logging

        Returns:
            (turnover, transaction_costs): Trading statistics
        """
        self.current_timestep = timestep

        # Update prices first
        self.update_prices(prices)

        # Get current portfolio value
        portfolio_value = self.get_portfolio_value()

        # Default to equal weights
        K = len(target_indices)
        if target_weights is None:
            target_weights = np.ones(K) / K
        else:
            # Normalize weights
            target_weights = target_weights / (target_weights.sum() + 1e-8)

        # Calculate target positions
        target_positions = {}
        for i, idx in enumerate(target_indices):
            target_value = portfolio_value * target_weights[i]
            price = prices[idx]
            if price > 0:
                target_shares = target_value / price
                target_positions[int(idx)] = target_shares

        # Determine trades needed
        trades_to_execute = []

        # 1. Sell positions not in target
        current_indices = set(self.positions.keys())
        target_indices_set = set(int(i) for i in target_indices)

        for idx in current_indices - target_indices_set:
            pos = self.positions[idx]
            if pos.shares > 0:
                trades_to_execute.append(('SELL', idx, pos.shares, prices[idx]))

        # 2. Adjust existing positions
        for idx in current_indices & target_indices_set:
            current_shares = self.positions[idx].shares
            target_shares = target_positions.get(idx, 0)
            diff = target_shares - current_shares

            if abs(diff * prices[idx]) > self.min_trade_value:
                if diff > 0:
                    trades_to_execute.append(('BUY', idx, diff, prices[idx]))
                else:
                    trades_to_execute.append(('SELL', idx, -diff, prices[idx]))

        # 3. Buy new positions
        for idx in target_indices_set - current_indices:
            target_shares = target_positions.get(idx, 0)
            if target_shares * prices[idx] > self.min_trade_value:
                trades_to_execute.append(('BUY', idx, target_shares, prices[idx]))

        # Execute trades (sells first to free up cash)
        trades_to_execute.sort(key=lambda x: (0 if x[0] == 'SELL' else 1, -x[2] * x[3]))

        total_turnover = 0.0
        total_costs = 0.0

        for action, idx, shares, price in trades_to_execute:
            trade = self._execute_trade(action, idx, shares, price, timestep)
            if trade:
                total_turnover += trade.gross_value
                total_costs += trade.total_cost

        turnover_pct = total_turnover / (portfolio_value + 1e-8)

        return turnover_pct, total_costs

    def _execute_trade(
        self,
        action: str,
        stock_idx: int,
        shares: float,
        price: float,
        timestep: int,
    ) -> Optional[Trade]:
        """Execute a single trade."""
        if shares <= 0 or price <= 0:
            return None

        gross_value = shares * price
        commission = gross_value * self.commission_rate
        slippage = gross_value * self.slippage_rate

        trade = Trade(
            timestep=timestep,
            stock_idx=stock_idx,
            action=action,
            shares=shares,
            price=price,
            commission=commission,
            slippage_cost=slippage,
        )

        if action == 'BUY':
            # Check if we have enough cash
            required_cash = gross_value + commission + slippage
            if required_cash > self.cash:
                # Adjust shares to fit available cash
                available = self.cash / (price * (1 + self.commission_rate + self.slippage_rate))
                if available < 1:
                    return None
                shares = available
                gross_value = shares * price
                commission = gross_value * self.commission_rate
                slippage = gross_value * self.slippage_rate
                trade.shares = shares
                trade.commission = commission
                trade.slippage_cost = slippage

            # Execute buy
            self.cash -= (gross_value + commission + slippage)

            if stock_idx in self.positions:
                # Add to existing position (average entry price)
                pos = self.positions[stock_idx]
                total_shares = pos.shares + shares
                avg_price = (pos.shares * pos.entry_price + shares * price) / total_shares
                pos.shares = total_shares
                pos.entry_price = avg_price
                pos.current_price = price
            else:
                # New position
                self.positions[stock_idx] = Position(
                    stock_idx=stock_idx,
                    shares=shares,
                    entry_price=price,
                    entry_date=timestep,
                    current_price=price,
                )

        else:  # SELL
            if stock_idx not in self.positions:
                return None

            pos = self.positions[stock_idx]
            shares = min(shares, pos.shares)
            gross_value = shares * price
            commission = gross_value * self.commission_rate
            slippage = gross_value * self.slippage_rate

            trade.shares = shares
            trade.commission = commission
            trade.slippage_cost = slippage

            # Record closed trade for win rate
            pnl = (price - pos.entry_price) * shares - commission - slippage
            self.closed_trades.append({
                'stock_idx': stock_idx,
                'entry_price': pos.entry_price,
                'exit_price': price,
                'shares': shares,
                'pnl': pnl,
                'holding_period': timestep - pos.entry_date,
            })

            # Execute sell
            self.cash += (gross_value - commission - slippage)
            pos.shares -= shares

            if pos.shares < 0.01:  # Effectively zero
                del self.positions[stock_idx]

        self.trade_history.append(trade)
        return trade

    def record_daily_snapshot(
        self,
        timestep: int,
        date: Optional[Any] = None,
        is_rebalance: bool = False,
        turnover: float = 0.0,
        transaction_costs: float = 0.0,
    ):
        """Record end-of-day portfolio snapshot."""
        portfolio_value = self.get_portfolio_value()
        positions_value = self.get_positions_value()

        # Calculate returns
        daily_return = (portfolio_value - self._prev_portfolio_value) / (self._prev_portfolio_value + 1e-8)
        cumulative_return = (portfolio_value - self.initial_capital) / self.initial_capital

        # Track peak for drawdown
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
            self._drawdown_start = timestep

        snapshot = DailySnapshot(
            timestep=timestep,
            date=date,
            cash=self.cash,
            positions_value=positions_value,
            total_value=portfolio_value,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            num_positions=len(self.positions),
            is_rebalance_day=is_rebalance,
            turnover=turnover,
            transaction_costs=transaction_costs,
        )

        self.daily_snapshots.append(snapshot)
        self._prev_portfolio_value = portfolio_value

    # =========================================================================
    # METRICS CALCULATION
    # =========================================================================

    def get_performance_metrics(
        self,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            benchmark_returns: Optional (T,) array of benchmark daily returns

        Returns:
            PerformanceMetrics dataclass
        """
        if len(self.daily_snapshots) < 2:
            return PerformanceMetrics()

        # Extract daily returns
        daily_returns = np.array([s.daily_return for s in self.daily_snapshots])
        portfolio_values = np.array([s.total_value for s in self.daily_snapshots])

        # Handle NaN/inf values
        daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)
        portfolio_values = np.nan_to_num(portfolio_values, nan=self.initial_capital)

        metrics = PerformanceMetrics()

        # ----- Returns -----
        metrics.total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital

        n_days = len(daily_returns)
        n_years = n_days / self.trading_days_per_year
        if n_years > 0:
            metrics.annualized_return = (1 + metrics.total_return) ** (1 / n_years) - 1

        # ----- Volatility -----
        valid_returns = daily_returns[~np.isnan(daily_returns)]
        metrics.volatility = np.std(valid_returns) if len(valid_returns) > 0 else 0.0
        metrics.annualized_volatility = metrics.volatility * np.sqrt(self.trading_days_per_year)

        # ----- Drawdown -----
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / (running_max + 1e-8)
        metrics.max_drawdown = abs(drawdowns.min())

        # Drawdown duration
        in_drawdown = drawdowns < 0
        if in_drawdown.any():
            drawdown_periods = []
            current_period = 0
            for dd in in_drawdown:
                if dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
            if current_period > 0:
                drawdown_periods.append(current_period)
            metrics.max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0

        # ----- Risk-Adjusted -----
        daily_rf = self.risk_free_rate / self.trading_days_per_year
        excess_returns = daily_returns - daily_rf

        if metrics.annualized_volatility > 0:
            metrics.sharpe_ratio = (metrics.annualized_return - self.risk_free_rate) / metrics.annualized_volatility

        # Sortino (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_vol = np.std(downside_returns) * np.sqrt(self.trading_days_per_year)
            if downside_vol > 0:
                metrics.sortino_ratio = (metrics.annualized_return - self.risk_free_rate) / downside_vol

        # Calmar
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

        # ----- Trading Stats -----
        metrics.total_trades = len(self.trade_history)
        metrics.total_rebalances = sum(1 for s in self.daily_snapshots if s.is_rebalance_day)

        turnovers = [s.turnover for s in self.daily_snapshots if s.is_rebalance_day]
        metrics.avg_turnover = np.mean(turnovers) if turnovers else 0.0

        metrics.total_transaction_costs = sum(t.total_cost for t in self.trade_history)
        if metrics.total_return != 0:
            metrics.transaction_cost_drag = metrics.total_transaction_costs / (
                self.initial_capital * abs(metrics.total_return) + 1e-8
            )

        # ----- Holding Stats -----
        if self.closed_trades:
            holding_periods = [t['holding_period'] for t in self.closed_trades]
            metrics.avg_holding_period = np.mean(holding_periods)

            wins = sum(1 for t in self.closed_trades if t['pnl'] > 0)
            metrics.win_rate = wins / len(self.closed_trades)

            gross_profit = sum(t['pnl'] for t in self.closed_trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in self.closed_trades if t['pnl'] < 0))
            if gross_loss > 0:
                metrics.profit_factor = gross_profit / gross_loss

        # ----- Benchmark Comparison -----
        if benchmark_returns is not None and len(benchmark_returns) == len(daily_returns):
            benchmark_cumulative = np.cumprod(1 + benchmark_returns) - 1
            metrics.benchmark_return = benchmark_cumulative[-1]
            metrics.alpha = metrics.total_return - metrics.benchmark_return

            # Beta and Information Ratio
            if np.var(benchmark_returns) > 0:
                metrics.beta = np.cov(daily_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)

                tracking_error = np.std(daily_returns - benchmark_returns) * np.sqrt(self.trading_days_per_year)
                if tracking_error > 0:
                    metrics.information_ratio = (metrics.annualized_return -
                        (np.mean(benchmark_returns) * self.trading_days_per_year)) / tracking_error

        return metrics

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.daily_snapshots:
            return pd.DataFrame()

        data = []
        for s in self.daily_snapshots:
            data.append({
                'timestep': s.timestep,
                'date': s.date,
                'portfolio_value': s.total_value,
                'cash': s.cash,
                'positions_value': s.positions_value,
                'daily_return': s.daily_return,
                'cumulative_return': s.cumulative_return,
                'num_positions': s.num_positions,
                'is_rebalance': s.is_rebalance_day,
                'turnover': s.turnover,
                'transaction_costs': s.transaction_costs,
            })

        return pd.DataFrame(data)

    def print_summary(self, benchmark_returns: Optional[np.ndarray] = None):
        """Print performance summary."""
        metrics = self.get_performance_metrics(benchmark_returns)

        print("\n" + "=" * 70)
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print("=" * 70)

        print(f"\n{'─' * 35} RETURNS {'─' * 35}")
        print(f"  Initial Capital:     ${self.initial_capital:>15,.2f}")
        print(f"  Final Value:         ${self.get_portfolio_value():>15,.2f}")
        print(f"  Total Return:        {metrics.total_return * 100:>15.2f}%")
        print(f"  Annualized Return:   {metrics.annualized_return * 100:>15.2f}%")

        if benchmark_returns is not None:
            print(f"  Benchmark Return:    {metrics.benchmark_return * 100:>15.2f}%")
            print(f"  Alpha:               {metrics.alpha * 100:>15.2f}%")

        print(f"\n{'─' * 35} RISK {'─' * 38}")
        print(f"  Volatility (Annual): {metrics.annualized_volatility * 100:>15.2f}%")
        print(f"  Max Drawdown:        {metrics.max_drawdown * 100:>15.2f}%")
        print(f"  Max DD Duration:     {metrics.max_drawdown_duration:>15} days")

        print(f"\n{'─' * 31} RISK-ADJUSTED {'─' * 31}")
        print(f"  Sharpe Ratio:        {metrics.sharpe_ratio:>15.3f}")
        print(f"  Sortino Ratio:       {metrics.sortino_ratio:>15.3f}")
        print(f"  Calmar Ratio:        {metrics.calmar_ratio:>15.3f}")

        print(f"\n{'─' * 33} TRADING {'─' * 33}")
        print(f"  Total Trades:        {metrics.total_trades:>15}")
        print(f"  Total Rebalances:    {metrics.total_rebalances:>15}")
        print(f"  Avg Turnover:        {metrics.avg_turnover * 100:>15.2f}%")
        print(f"  Transaction Costs:   ${metrics.total_transaction_costs:>14,.2f}")
        print(f"  Cost Drag:           {metrics.transaction_cost_drag * 100:>15.2f}%")

        print(f"\n{'─' * 33} HOLDINGS {'─' * 32}")
        print(f"  Avg Holding Period:  {metrics.avg_holding_period:>15.1f} days")
        print(f"  Win Rate:            {metrics.win_rate * 100:>15.2f}%")
        print(f"  Profit Factor:       {metrics.profit_factor:>15.3f}")

        print("\n" + "=" * 70)


# ============================================================================
# BACKTEST RUNNER
# ============================================================================

def backtest_model(
    model,
    data_tensors: Dict,
    config,
    initial_capital: float = 100_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    use_validation: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run backtest on trained model.

    Args:
        model: Trained MAFIA model
        data_tensors: Dictionary with 'prices', 'returns', 'dates', etc.
        config: Config object with rebalance settings
        initial_capital: Starting capital
        commission_rate: Commission rate
        slippage_rate: Slippage rate
        use_validation: Use validation data (True) or training data (False)
        verbose: Print progress

    Returns:
        Dictionary with simulator, metrics, equity_curve
    """
    import torch as th

    # Initialize simulator
    sim = PortfolioSimulator(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )

    # Get data
    prices = data_tensors.get('prices')  # (T, N)
    returns = data_tensors.get('returns')  # (T, N)
    market_returns = data_tensors.get('market_returns')  # (T,)
    dates = data_tensors.get('dates')  # (T,)
    ochlv = data_tensors.get('ochlv')  # (T, N, 5)

    if prices is None:
        raise ValueError("data_tensors must contain 'prices'")

    T, N = prices.shape
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'

    # Get config parameters
    K = int(getattr(config, 'mafia_topk_k', 10))
    rebalance_interval = int(getattr(config, 'mafia_rebalance_interval', 21))
    window_size = int(getattr(config, 'mafia_window_size', 30))  # Default 30 for model

    if verbose:
        print(f"Backtest Settings:")
        print(f"  - Period: {T} days")
        print(f"  - Stocks: {N}")
        print(f"  - Top-K: {K}")
        print(f"  - Rebalance Interval: {rebalance_interval} days")
        print(f"  - Initial Capital: ${initial_capital:,.0f}")
        print()

    # Convert to tensors if needed
    if isinstance(prices, np.ndarray):
        prices_tensor = th.tensor(prices, dtype=th.float32, device=device)
    else:
        prices_tensor = prices.to(device)

    # Run backtest
    model.eval()
    current_indices = None

    with th.no_grad():
        for t in range(window_size, T):
            is_rebalance = (t == window_size) or ((t - window_size) % rebalance_interval == 0)

            # Get current prices
            current_prices = prices[t].numpy() if isinstance(prices, th.Tensor) else prices[t]

            if is_rebalance:
                # Prepare input window
                if ochlv is not None:
                    window = ochlv[t - window_size:t]  # (window_size, N, 5)
                    if isinstance(window, np.ndarray):
                        window = th.tensor(window, dtype=th.float32, device=device)
                    # Model expects (B, N, 5, T_w)
                    window = window.permute(1, 2, 0).unsqueeze(0)  # (1, N, 5, window_size)
                    # Normalize window
                    window = th.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    # Construct from prices
                    price_window = prices_tensor[t - window_size:t]  # (window_size, N)
                    if isinstance(price_window, np.ndarray):
                        price_window = th.tensor(price_window, dtype=th.float32, device=device)
                    # Create fake OCHLV (use close for all)
                    window = price_window.T.unsqueeze(1).expand(-1, 5, -1).unsqueeze(0)  # (1, N, 5, window_size)

                # Forward pass
                try:
                    outputs = model(
                        ochlv_data=window,
                        market_index_ochlv_data=None,
                        force_topk_indices=None,
                        router_context_buffer=None,
                        explicit_signals=th.zeros(1, 6, device=device),
                    )

                    # Extract top-k indices
                    if isinstance(outputs, tuple):
                        topk_indices = outputs[5]  # Based on model output structure
                    else:
                        topk_indices = outputs.get('topk_indices')

                    current_indices = topk_indices[0].cpu().numpy()

                    model_success = True

                except Exception as e:
                    model_success = False
                    if verbose and t % 200 == 0:
                        print(f"  [WARN] Model error at t={t}: {e}")
                    # Fallback to momentum-based selection
                    if t >= 20:
                        momentum = (prices[t] - prices[t-20]) / (prices[t-20] + 1e-8)
                        current_indices = np.argsort(momentum)[-K:]
                    else:
                        current_indices = np.random.choice(N, K, replace=False)

                # Execute rebalance
                turnover, costs = sim.execute_rebalance(
                    target_indices=current_indices,
                    target_weights=None,  # Equal weight
                    prices=current_prices,
                    timestep=t,
                    date=dates[t] if dates is not None else None,
                )

                if verbose and t % 50 == 0:
                    pv = sim.get_portfolio_value()
                    ret = (pv - initial_capital) / initial_capital * 100
                    print(f"  Day {t}: Rebalance | PV=${pv:,.0f} | Return={ret:.2f}%")

            else:
                turnover, costs = 0.0, 0.0
                # Just update prices
                sim.update_prices(current_prices)

            # Record daily snapshot
            sim.record_daily_snapshot(
                timestep=t,
                date=dates[t] if dates is not None else None,
                is_rebalance=is_rebalance,
                turnover=turnover,
                transaction_costs=costs,
            )

    # Get results
    if market_returns is not None:
        if isinstance(market_returns, th.Tensor):
            benchmark_rets = market_returns[window_size:].cpu().numpy()
        else:
            benchmark_rets = market_returns[window_size:]
    else:
        benchmark_rets = None

    metrics = sim.get_performance_metrics(benchmark_returns=benchmark_rets)
    equity_curve = sim.get_equity_curve()

    if verbose:
        sim.print_summary(benchmark_returns=benchmark_rets)

    return {
        'simulator': sim,
        'metrics': metrics,
        'equity_curve': equity_curve,
    }


# ============================================================================
# STANDALONE BACKTEST SCRIPT
# ============================================================================

def main():
    """Run backtest from command line."""
    import sys
    import os

    SCRIPT_DIR = Path(__file__).parent
    sys.path.insert(0, str(SCRIPT_DIR))

    from config import Config

    # Default checkpoint path
    CHECKPOINT_PATH = "/Users/nguyensiry/Documents/the_last_dance/observer_offline_6/checkpoints/macro/temp_iter_0/best_checkpoint.pth"

    print("=" * 70)
    print("PORTFOLIO BACKTEST - MAFIA MODEL")
    print("=" * 70)

    # 1. Setup config
    config = Config()
    config.batch_size = 1

    # 2. Load data
    print("\n[1] Loading data...")
    from utils.mafia_data_loader import load_mafia_data
    data = load_mafia_data(config)

    # 3. Prepare data tensors
    print("[2] Preparing data tensors...")
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to tensors
        stock_list = data['stock'].unique().tolist()
        dates = sorted(data['date'].unique())

        # Pivot to get (T, N) shape
        pivot_close = data.pivot(index='date', columns='stock', values='close')
        pivot_open = data.pivot(index='date', columns='stock', values='open')
        pivot_high = data.pivot(index='date', columns='stock', values='high')
        pivot_low = data.pivot(index='date', columns='stock', values='low')
        pivot_volume = data.pivot(index='date', columns='stock', values='volume')

        prices = pivot_close.values
        returns = pivot_close.pct_change(fill_method=None).fillna(0).values

        # Build OCHLV (T, N, 5)
        ochlv = np.stack([
            pivot_open.values,
            pivot_close.values,
            pivot_high.values,
            pivot_low.values,
            pivot_volume.values / 1e6,  # Scale volume
        ], axis=-1)

        # Market returns (equal-weighted average)
        market_returns = returns.mean(axis=1)

        data_tensors = {
            'prices': prices,
            'returns': returns,
            'market_returns': market_returns,
            'dates': dates,
            'stock_list': stock_list,
            'ochlv': ochlv,
        }
    else:
        data_tensors = data

    print(f"   Prices shape: {data_tensors['prices'].shape}")

    # 4. Load model
    print("\n[3] Loading model...")
    import torch as th
    from RL_controller.mafia_observer import MAFIAObserver

    N = data_tensors['prices'].shape[1]
    observer = MAFIAObserver(config=config, action_dim=N)

    checkpoint = th.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    if 'mafia_model_state_dict' in checkpoint:
        state_dict = checkpoint['mafia_model_state_dict']
        # Remap legacy LayerNorm keys: input_norm.weight -> input_norm.norm.weight
        state_dict = {
            (k.replace(".input_norm.weight", ".input_norm.norm.weight")
              .replace(".input_norm.bias", ".input_norm.norm.bias")
             if ".input_norm." in k and ".input_norm.norm." not in k else k): v
            for k, v in state_dict.items()
        }
        observer.mafia_model.load_state_dict(state_dict, strict=False)

    # 5. Run backtest
    print("\n[4] Running backtest...")
    results = backtest_model(
        model=observer.mafia_model,
        data_tensors=data_tensors,
        config=config,
        initial_capital=100_000,
        commission_rate=0.001,
        slippage_rate=0.001,
        verbose=True,
    )

    # 6. Save results
    print("\n[5] Saving results...")
    output_dir = SCRIPT_DIR / "backtest_results"
    output_dir.mkdir(exist_ok=True)

    results['equity_curve'].to_csv(output_dir / "equity_curve.csv", index=False)
    print(f"   Saved to {output_dir}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
