# ！/usr/bin/python
# -*- coding: utf-8 -*-#

"""
MAFIA Lightweight Data Loader

This module provides efficient data loading for MAFIA without heavy legacy preprocessing.
MAFIA only needs raw OCHLV data and computes features on-the-fly.

Author: MASA
"""

import pandas as pd
import numpy as np
import copy

# LiveDisplay smart_print for terminal-safe logging
try:
    from utils.display_integration import smart_print
except ImportError:
    smart_print = print  # Fallback


class MAFIADataLoader:
    """
    Lightweight data loader for MAFIA.

    Skips heavy legacy preprocessing (volume_w1...w31, etc.) and only:
    - Splits data by dates (train/valid/test)
    - Prepares fine-grained windowed features (for extra_data)
    - Keeps raw OCHLV columns intact
    """

    def __init__(self, config):
        """
        Initialize MAFIA data loader.

        Args:
            config: Configuration object
        """
        self.config = config
        self.use_features = (
            config.use_features
        )  # ['close', 'open', 'high', 'low', 'volume']
        self.fine_window_size = config.fine_window_size  # 30
        self.freq = config.freq  # '1d'
        self.finefreq = config.finefreq  # '1d'

    def load_and_split_data(self, data: pd.DataFrame) -> dict:
        """
        Load and split data into train/valid/test WITHOUT heavy preprocessing.

        Args:
            data: Raw DataFrame with columns: date, stock, open, high, low, close, volume

        Returns:
            dict with keys: 'train', 'valid', 'test', 'extra_train', 'extra_valid', 'extra_test'
        """
        smart_print(
            "[MAFIA] Loading data with lightweight preprocessing (no legacy features)...",
            flush=True,
        )

        # Ensure date is datetime
        data["date"] = pd.to_datetime(data["date"])

        # Sort by date and stock
        data = data.sort_values(["date", "stock"], ascending=True, ignore_index=True)

        # Validate required columns
        required_cols = ["date", "stock", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Keep only necessary columns for MAFIA
        # MAFIA computes change features on-the-fly, no need for pre-computed columns
        base_data = data[required_cols].copy()

        # Remove rows with invalid price values (<=0 or NaN)
        price_cols = ["open", "high", "low", "close"]
        price_mask = np.ones(len(base_data), dtype=bool)
        for col in price_cols:
            price_mask &= base_data[col] > 0
        invalid_price_rows = len(base_data) - price_mask.sum()
        if invalid_price_rows > 0:
            smart_print(
                f"[MAFIA] Filtering out {invalid_price_rows} rows with non-positive price values",
                flush=True,
            )
        base_data = base_data[price_mask].copy()
        base_data.reset_index(drop=True, inplace=True)

        # Remove rows with negative volume (allow zero volume for inactive days)
        vol_mask = base_data["volume"] >= 0
        invalid_vol_rows = len(base_data) - vol_mask.sum()
        if invalid_vol_rows > 0:
            smart_print(
                f"[MAFIA] Filtering out {invalid_vol_rows} rows with negative volume",
                flush=True,
            )
        base_data = base_data[vol_mask].copy()
        base_data.reset_index(drop=True, inplace=True)

        # Drop stocks that have excessive zero-volume days
        zero_volume_counts = base_data[base_data["volume"] == 0].groupby("stock").size()
        zero_threshold = getattr(self.config, "max_zero_volume_days", 100)
        drop_stocks = zero_volume_counts[
            zero_volume_counts > zero_threshold
        ].index.tolist()
        if drop_stocks:
            smart_print(
                f"[MAFIA] Removing {len(drop_stocks)} stocks with more than {zero_threshold} zero-volume days",
                flush=True,
            )
            base_data = base_data[~base_data["stock"].isin(drop_stocks)].copy()
            base_data.reset_index(drop=True, inplace=True)

        # Pre-compute date masks for splits
        train_mask = (base_data["date"] >= self.config.train_date_start) & (
            base_data["date"] <= self.config.train_date_end
        )
        valid_mask_dates = None
        test_mask_dates = None
        if (self.config.valid_date_start is not None) and (
            self.config.valid_date_end is not None
        ):
            valid_mask_dates = (base_data["date"] >= self.config.valid_date_start) & (
                base_data["date"] <= self.config.valid_date_end
            )
        if (self.config.test_date_start is not None) and (
            self.config.test_date_end is not None
        ):
            test_mask_dates = (base_data["date"] >= self.config.test_date_start) & (
                base_data["date"] <= self.config.test_date_end
            )

        # Ensure we only keep stocks available in every split
        common_stocks = set(base_data.loc[train_mask, "stock"])
        if valid_mask_dates is not None:
            common_stocks &= set(base_data.loc[valid_mask_dates, "stock"])
        if test_mask_dates is not None:
            common_stocks &= set(base_data.loc[test_mask_dates, "stock"])
        if not common_stocks:
            raise ValueError(
                "No common stocks found across train/valid/test splits. Check data availability."
            )
        total_before = base_data["stock"].nunique()
        if len(common_stocks) < total_before:
            smart_print(
                f"[MAFIA] Using {len(common_stocks)} common stocks across splits (dropped {total_before - len(common_stocks)})",
                flush=True,
            )
        base_data = base_data[base_data["stock"].isin(common_stocks)].copy()
        base_data.reset_index(drop=True, inplace=True)

        # Recompute masks after filtering
        train_mask = (base_data["date"] >= self.config.train_date_start) & (
            base_data["date"] <= self.config.train_date_end
        )
        if valid_mask_dates is not None:
            valid_mask_dates = (base_data["date"] >= self.config.valid_date_start) & (
                base_data["date"] <= self.config.valid_date_end
            )
        if test_mask_dates is not None:
            test_mask_dates = (base_data["date"] >= self.config.test_date_start) & (
                base_data["date"] <= self.config.test_date_end
            )

        # Compute DAILYRETURNS for compatibility with environment (used in reward calc)
        # This is a lightweight computation compared to full preprocessing
        lookback = getattr(self.config, "dailyRetun_lookback", 5)
        base_data = self._add_daily_returns(base_data, lookback=lookback)

        # Compute MA (Moving Average) for compatibility with environment (used in ctl_state)
        ma_window = getattr(self.config, "otherRef_indicator_ma_window", 5)
        base_data = self._add_moving_average(base_data, window=ma_window)

        # Split by date ranges
        dataset_dict = {}

        # Train split
        train_data = base_data[
            (base_data["date"] >= self.config.train_date_start)
            & (base_data["date"] <= self.config.train_date_end)
        ].copy()
        train_data = train_data.sort_values(
            ["date", "stock"], ascending=True, ignore_index=True
        )
        dataset_dict["train"] = train_data
        smart_print(
            f"[MAFIA] Train data: {len(train_data)} rows, "
            f"dates: {train_data['date'].min()} to {train_data['date'].max()}",
            flush=True,
        )

        # Valid split
        if (self.config.valid_date_start is not None) and (
            self.config.valid_date_end is not None
        ):
            valid_data = base_data[
                (base_data["date"] >= self.config.valid_date_start)
                & (base_data["date"] <= self.config.valid_date_end)
            ].copy()
            valid_data = valid_data.sort_values(
                ["date", "stock"], ascending=True, ignore_index=True
            )
            dataset_dict["valid"] = valid_data
            smart_print(
                f"[MAFIA] Valid data: {len(valid_data)} rows, "
                f"dates: {valid_data['date'].min()} to {valid_data['date'].max()}",
                flush=True,
            )
        else:
            dataset_dict["valid"] = None

        # Test split
        if (self.config.test_date_start is not None) and (
            self.config.test_date_end is not None
        ):
            test_data = base_data[
                (base_data["date"] >= self.config.test_date_start)
                & (base_data["date"] <= self.config.test_date_end)
            ].copy()
            test_data = test_data.sort_values(
                ["date", "stock"], ascending=True, ignore_index=True
            )
            dataset_dict["test"] = test_data
            smart_print(
                f"[MAFIA] Test data: {len(test_data)} rows, "
                f"dates: {test_data['date'].min()} to {test_data['date'].max()}",
                flush=True,
            )
        else:
            dataset_dict["test"] = None

        # Process fine-grained features (for extra_data)
        # This is lightweight windowed data needed by MAFIAObserver
        smart_print(
            "[MAFIA] Processing fine-grained features for extra_data...", flush=True
        )
        dataset_dict["extra_train"] = self._process_fine_data(base_data, "train")
        if dataset_dict["valid"] is not None:
            dataset_dict["extra_valid"] = self._process_fine_data(base_data, "valid")
        else:
            dataset_dict["extra_valid"] = None
        if dataset_dict["test"] is not None:
            dataset_dict["extra_test"] = self._process_fine_data(base_data, "test")
        else:
            dataset_dict["extra_test"] = None

        smart_print("[MAFIA] Data loading complete (lightweight mode).", flush=True)
        return dataset_dict

    def _process_fine_data(self, data: pd.DataFrame, split: str) -> dict:
        """
        Process fine-grained windowed features for extra_data.

        This creates windowed features needed by MAFIAObserver for fine market/stock data.
        Much lighter than legacy preprocessing.

        Args:
            data: Full raw DataFrame
            split: 'train', 'valid', or 'test'

        Returns:
            dict with 'fine_market' and 'fine_stock' DataFrames
        """
        # Determine date range for this split
        if split == "train":
            start_date = self.config.train_date_start
            end_date = self.config.train_date_end
        elif split == "valid":
            start_date = self.config.valid_date_start
            end_date = self.config.valid_date_end
        elif split == "test":
            start_date = self.config.test_date_start
            end_date = self.config.test_date_end
        else:
            raise ValueError(f"Unknown split: {split}")

        if start_date is None or end_date is None:
            return {"fine_market": None, "fine_stock": None}

        # Extract data for this split + lookback window
        lookback_days = self.fine_window_size + 5  # Extra buffer for windowing
        lookback_start = start_date - pd.Timedelta(days=lookback_days)
        split_data = data[
            (data["date"] >= lookback_start) & (data["date"] <= end_date)
        ].copy()

        # Process fine market data (aggregated across all stocks)
        fine_market = self._create_fine_market_data(split_data, start_date, end_date)

        # Process fine stock data (per-stock windowed features)
        fine_stock = self._create_fine_stock_data(split_data, start_date, end_date)

        return {"fine_market": fine_market, "fine_stock": fine_stock}

    def _create_fine_market_data(
        self, data: pd.DataFrame, start_date, end_date
    ) -> pd.DataFrame:
        """
        Create fine-grained market data (windowed features).

        For each date, compute windowed changes of market aggregates.
        """
        # Group by date and compute market aggregates (e.g., mean close, total volume)
        market_agg = (
            data.groupby("date")
            .agg(
                {
                    "close": "mean",
                    "open": "mean",
                    "high": "mean",
                    "low": "mean",
                    "volume": "sum",
                }
            )
            .reset_index()
        )

        market_agg = market_agg.sort_values("date", ascending=True, ignore_index=True)

        # Create windowed features (batch insert to avoid fragmentation)
        windowed_cols = {}
        for feat in self.use_features:
            feat_vals = market_agg[feat].values
            for widx in range(1, self.fine_window_size + 1):
                col_name = f"mkt_{self.finefreq}_{feat}_w{widx}"
                if widx == 1:
                    # Current value (change = 0 for first day)
                    windowed = self._compute_pct_change(feat_vals)
                else:
                    # Shifted values
                    windowed = self._compute_pct_change(feat_vals)
                    windowed = np.roll(windowed, widx - 1)
                    windowed[: widx - 1] = 0  # Pad with zeros
                windowed_cols[col_name] = windowed

        # Batch insert all windowed columns at once
        market_agg = pd.concat(
            [market_agg, pd.DataFrame(windowed_cols, index=market_agg.index)], axis=1
        )

        # Add required columns for environment compatibility
        market_agg[f"mkt_{self.finefreq}_close"] = market_agg["close"]
        market_agg[f"mkt_{self.finefreq}_ma"] = (
            market_agg["close"].rolling(window=5, min_periods=1).mean()
        )

        # Filter to split dates only
        fine_market = market_agg[
            (market_agg["date"] >= start_date) & (market_agg["date"] <= end_date)
        ].copy()

        return fine_market

    def _create_fine_stock_data(
        self, data: pd.DataFrame, start_date, end_date
    ) -> pd.DataFrame:
        """
        Create fine-grained stock data (windowed features per stock).
        """
        stock_lst = data["stock"].unique()
        fine_stock_lst = []

        for stock in stock_lst:
            stock_data = data[data["stock"] == stock].sort_values(
                "date", ascending=True, ignore_index=True
            )

            # Create windowed features for this stock (batch insert to avoid fragmentation)
            windowed_cols = {}
            for feat in self.use_features:
                feat_vals = stock_data[feat].values
                for widx in range(1, self.fine_window_size + 1):
                    col_name = f"stock_{self.finefreq}_{feat}_w{widx}"
                    if widx == 1:
                        windowed = self._compute_pct_change(feat_vals)
                    else:
                        windowed = self._compute_pct_change(feat_vals)
                        windowed = np.roll(windowed, widx - 1)
                        windowed[: widx - 1] = 0
                    windowed_cols[col_name] = windowed

            # Batch insert all windowed columns at once
            stock_data = pd.concat(
                [stock_data, pd.DataFrame(windowed_cols, index=stock_data.index)],
                axis=1,
            )

            # Add required columns for environment compatibility
            stock_data[f"stock_{self.finefreq}_close"] = stock_data["close"]
            stock_data[f"stock_{self.finefreq}_ma"] = (
                stock_data["close"].rolling(window=5, min_periods=1).mean()
            )

            fine_stock_lst.append(stock_data)

        if len(fine_stock_lst) > 0:
            fine_stock = pd.concat(fine_stock_lst, ignore_index=True)
        else:
            # No data for this split
            fine_stock = pd.DataFrame()

        # Filter to split dates only
        fine_stock = fine_stock[
            (fine_stock["date"] >= start_date) & (fine_stock["date"] <= end_date)
        ].copy()

        return fine_stock

    def _compute_pct_change(self, values: np.ndarray) -> np.ndarray:
        """
        Compute percentage change: (val[t] - val[t-1]) / val[t-1]

        Args:
            values: Array of values

        Returns:
            Array of percentage changes (first element = 0)
        """
        if len(values) <= 1:
            return np.zeros_like(values)

        prev_vals = values[:-1]
        cur_vals = values[1:]
        change = np.divide(
            cur_vals,
            prev_vals,
            out=np.ones_like(cur_vals, dtype=float),
            where=prev_vals != 0,
        )
        change = change - 1.0
        change = np.concatenate([np.array([0.0]), change])
        return change

    def _add_daily_returns(
        self, data: pd.DataFrame, lookback: int = 20
    ) -> pd.DataFrame:
        """
        Add DAILYRETURNS column for compatibility with environment.

        This computes rolling returns which are used in reward calculation.

        Args:
            data: DataFrame with 'date', 'stock', 'close' columns
            lookback: Lookback window for daily returns

        Returns:
            DataFrame with added DAILYRETURNS column
        """
        stock_lst = data["stock"].unique()
        result_lst = []

        for stock in stock_lst:
            stock_data = (
                data[data["stock"] == stock].sort_values("date", ascending=True).copy()
            )

            # Compute daily returns
            close_prices = stock_data["close"].values
            if len(close_prices) <= 1:
                daily_returns = np.zeros(len(close_prices))
            else:
                pct_changes = self._compute_pct_change(close_prices)
                # Rolling window lookback
                daily_returns = (
                    pd.Series(pct_changes)
                    .rolling(window=lookback, min_periods=1)
                    .mean()
                    .values
                )

            stock_data[f"DAILYRETURNS-{lookback}"] = daily_returns
            result_lst.append(stock_data)

        result_data = pd.concat(result_lst, ignore_index=True)
        return result_data

    def _add_moving_average(self, data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Add Moving Average (MA) column for compatibility with environment.

        This computes simple moving average on close prices.

        Args:
            data: DataFrame with 'date', 'stock', 'close' columns
            window: Window size for moving average

        Returns:
            DataFrame with added MA column
        """
        stock_lst = data["stock"].unique()
        result_lst = []

        for stock in stock_lst:
            stock_data = (
                data[data["stock"] == stock].sort_values("date", ascending=True).copy()
            )

            # Compute moving average on close prices
            close_prices = stock_data["close"].values
            ma_values = (
                pd.Series(close_prices)
                .rolling(window=window, min_periods=1)
                .mean()
                .values
            )

            stock_data[f"MA-{window}"] = ma_values
            result_lst.append(stock_data)

        result_data = pd.concat(result_lst, ignore_index=True)
        return result_data


def load_mafia_data(config):
    """
    Load data using MAFIADataLoader.
    
    This is a helper function to allow direct import usage.
    """
    from utils.data_validator import get_stock_data_file

    file_path, error_msg = get_stock_data_file(config)
    if file_path is None:
        raise FileNotFoundError(
            f"Cannot load stock data file. {error_msg or 'No valid CSV found.'}"
        )

    smart_print(f"[MAFIA] Loading stock data from: {file_path}", flush=True)
    data = pd.read_csv(file_path)

    # Ensure required columns exist
    required_cols = ["date", "stock", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {file_path}: {', '.join(missing_cols)}"
        )

    # Normalize date column
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "stock"], ascending=True, ignore_index=True)
    return data
