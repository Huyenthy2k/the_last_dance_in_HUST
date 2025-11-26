#！/usr/bin/python
# -*- coding: utf-8 -*-#

"""
MAFIA (Multi-Agent Fusion and Intelligent Adaptation) Market Observer

This module implements the MAFIA Market Observer, which replaces the original
Market Observer in the MASA framework. MAFIA consists of:
- 1 Technical Agent: Analyzes raw prices + technical indicators
- 3 DC Agents: Analyze Directional Change events at different thresholds

Each agent has CSA, TA, and ST-Fusion modules. Outputs are aggregated to
generate market_vector and boundary_risk.
See: Đặc Tả Kỹ Thuật (Technical Specification) - MAFIA.md
"""
import os
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, Callable, Any, Optional

th.autograd.set_detect_anomaly(True)


class MAFIAObserver:
    """
    MAFIA Market Observer implementing the same interface as MarketObserver.
    
    MAFIA consists of:
    - 1 Technical Agent (analyzes raw prices + technical indicators)
    - 3 DC Agents (analyze Directional Change events at different thresholds)
    
    Each agent has:
    - Cross-Sectional Analysis (CSA) module
    - Temporal Analysis (TA) module
    - Spatial-Temporal Fusion (ST-Fusion) module
    
    Outputs:
    - market_vector: (batch, N) - Market trend vector for N assets
    - boundary_risk: (batch,) - Continuous risk boundary value
    """
    
    def __init__(self, config, action_dim):
        """
        Initialize MAFIA Observer.
        
        Args:
            config: Configuration object with MAFIA hyperparameters
            action_dim: Number of assets (N)
        """
        self.config = config
        self.action_dim = action_dim  # N (number of stocks)
        self.last_topk_indices = None
        self.last_gate_weights = None
        self.last_market_context = None
        
        # Validate MAFIA hyperparameters exist
        self._validate_config()
        
        # Setup device first
        if th.cuda.is_available():
            self.device = th.device('cuda:0')
        else:
            self.device = th.device('cpu')
        
        # Initialize MAFIA model
        from RL_controller.mafia_modules import MAFIAModel
        self.mafia_model = MAFIAModel(config=config, action_dim=action_dim)
        self.mafia_model = self.mafia_model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            self.mafia_model.parameters(),
            lr=config.mafia_learning_rate,
            weight_decay=config.mafia_weight_decay
        )
        # Learning-rate schedule: align with TD3-style decay
        schedule_mode = getattr(config, "mafia_lr_schedule", "linear")
        start_lr = config.mafia_learning_rate
        end_lr = start_lr * getattr(config, "mafia_lr_end_factor", 0.2)
        frac = getattr(config, "mafia_lr_end_fraction", 0.5)

        if schedule_mode == "linear":
            decay_epochs = max(1, int(config.num_epochs * frac))

            def _lr_lambda(epoch):
                if epoch >= decay_epochs:
                    return end_lr / start_lr
                # Linear decay from start_lr to end_lr over decay_epochs (one-shot over full run)
                return 1.0 - (1.0 - end_lr / start_lr) * (epoch / float(decay_epochs))

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lr_lambda)
        elif schedule_mode == "linear_per_epoch":
            # Repeat linear decay every epoch: use a cycle on scheduler steps
            cycle_steps = max(
                1,
                int(getattr(config, "observer_mini_epoch_steps", 1)),
            )
            decay_steps = max(1, int(cycle_steps * frac))

            def _lr_lambda(step_idx):
                step_in_cycle = step_idx % cycle_steps
                if step_in_cycle >= decay_steps:
                    return end_lr / start_lr
                return 1.0 + (end_lr / start_lr - 1.0) * (step_in_cycle / float(decay_steps))

            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lr_lambda)
        else:
            decay_steps = max(1, config.num_epochs // 3)  # Ensure at least 1
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=decay_steps, gamma=0.1
            )
        
        # Training buffers (for Policy Gradient)
        self.market_vector_lst = []  # Store market_vector for each step (with gradient)
        self.boundary_risk_lst = []   # Store boundary_risk for each step
        self.rate_of_price_change_lst = []  # Store rate_of_price_change for reward calculation
        self.mkt_direction_lst = []   # Store market direction labels
        self.sigma_log_p_lst = []  # Store log-probabilities for market direction classification
        self.last_sigma_log_p: Optional[th.Tensor] = None
        self.last_sigma_pred: Optional[th.Tensor] = None
        
    def _validate_config(self):
        """Validate that config has all required MAFIA hyperparameters."""
        required_params = [
            'mafia_T_w', 'mafia_DC_thresholds', 'mafia_D', 'mafia_D_h',
            'mafia_encoder_layers', 'mafia_encoder_heads', 'mafia_M_tech', 'mafia_M_dc',
            'mafia_learning_rate', 'mafia_weight_decay'
        ]
        
        for param in required_params:
            if not hasattr(self.config, param):
                raise ValueError(f"Config missing required MAFIA parameter: {param}")
    
    def predict(self, finemkt_feat=None, finestock_feat=None, raw_ochlv_data=None, **kwargs):
        """
        Predict market_vector and boundary_risk.
        
        Supports two input modes for compatibility:
        1. Legacy mode: finemkt_feat, finestock_feat (from MarketObserver interface)
        2. Direct mode: raw_ochlv_data (for MAFIA native format)
        
        Args:
            finemkt_feat: (batch, features, window_size) - Optional, for compatibility
            finestock_feat: (batch, features, num_of_stocks, window_size) - Optional, for compatibility
            raw_ochlv_data: (N, M, T_w) - Optional, direct MAFIA input
            **kwargs: Must include 'mode' ('train', 'valid', 'test')
        
        Returns:
            tuple: (market_vector, boundary_risk, market_scores_full, gate_weights, market_context, fused_stock_embedding, sigma_val, sigma_log_p)
                - market_vector: (batch, N) numpy array - Top-K weights, zero elsewhere
                - boundary_risk: (batch,) numpy array (continuous risk value)
                - market_scores_full: (batch, N) numpy array - Full market scores (softmax on all N assets, no Top-K mask)
                - gate_weights: (batch, 4) numpy array - Dense MoE gate weights for 4 stock experts
                - market_context: (batch, D) numpy array - Market condition embedding from gating encoder
                - fused_stock_embedding: (batch, N, D) numpy array - Per-stock embedding fused via gate weights
                - sigma_val: (batch,) numpy array - Direction class prediction (0/1/2)
                - sigma_log_p: (batch, 3) numpy array - Log-probabilities for direction classes
        """
        mode = kwargs.get('mode', 'train')
        
        # Validate inputs
        self._validate_input_shapes(finemkt_feat, finestock_feat, **kwargs)
        
        # Determine input source and prepare OCHLV tensor
        if raw_ochlv_data is not None:
            # Direct MAFIA input
            ochlv_tensor = self._prepare_ochlv_tensor(raw_ochlv_data)
        elif finestock_feat is not None:
            # Legacy MarketObserver format - extract from rawdata if available
            if 'rawdata_window' in kwargs:
                ochlv_tensor = self._prepare_ochlv_tensor(kwargs['rawdata_window'])
            else:
                # For now, raise error - will implement conversion in later step
                raise NotImplementedError(
                    "Conversion from fine features to raw OCHLV not yet implemented. "
                    "Please provide raw_ochlv_data or rawdata_window in kwargs."
                )
        else:
            raise ValueError("Either raw_ochlv_data or finestock_feat must be provided")
        
        # Forward pass through MAFIA model
        # ochlv_tensor shape: (batch, N, M, T_w) where M=5 (OCHLV)
        # Need to ensure correct format: (batch, N, 5, T_w)
        if ochlv_tensor.shape[2] != 5:
            # If shape is (batch, N, T_w, 5), transpose to (batch, N, 5, T_w)
            if ochlv_tensor.shape[3] == 5:
                ochlv_tensor = ochlv_tensor.permute(0, 1, 3, 2)
        
        # Extract market-index OCHLV data if available
        market_index_ochlv_tensor = None
        if 'market_index_ochlv_data' in kwargs and kwargs['market_index_ochlv_data'] is not None:
            # Direct market-index data provided
            market_index_ochlv_tensor = self._prepare_market_index_ochlv_tensor(kwargs['market_index_ochlv_data'])
        elif 'env' in kwargs and kwargs['env'] is not None:
            # Try to extract from environment
            env = kwargs['env']
            if hasattr(env, '_extract_market_index_ochlv_window') and hasattr(env, 'curData'):
                try:
                    cur_date = env.curData['date'].iloc[0] if hasattr(env.curData, 'iloc') else env.curData['date'][0]
                    market_index_ochlv_np = env._extract_market_index_ochlv_window(cur_date, window_size=self.config.mafia_T_w)
                    market_index_ochlv_tensor = self._prepare_market_index_ochlv_tensor(market_index_ochlv_np)
                except Exception as e:
                    # If extraction fails, skip market-index agent
                    print(f"Warning: Could not extract market-index OCHLV data: {e}")
                    market_index_ochlv_tensor = None
        
        if mode == 'train':
            self.mafia_model.train()
            (
                market_vector,
                boundary_risk,
                topk_indices,
                market_scores_full,
                gate_weights,
                market_context,
                stock_embedding,
                sigma_logits,
            ) = self.mafia_model(
                ochlv_tensor, market_index_ochlv_data=market_index_ochlv_tensor
            )
        else:
            self.mafia_model.eval()
            with th.no_grad():
                (
                    market_vector,
                    boundary_risk,
                    topk_indices,
                    market_scores_full,
                    gate_weights,
                    market_context,
                    stock_embedding,
                    sigma_logits,
                ) = self.mafia_model(
                    ochlv_tensor, market_index_ochlv_data=market_index_ochlv_tensor
                )
        
        # Market direction prediction
        sigma_log_p = F.log_softmax(sigma_logits, dim=-1)  # (batch, 3)
        sigma_prob = sigma_log_p.exp()
        sigma_pred = th.argmax(sigma_prob, dim=-1)  # (batch,)
        self.last_sigma_log_p = sigma_log_p if mode == 'train' else sigma_log_p.detach()
        self.last_sigma_pred = sigma_pred.detach()

        if mode == 'train':
            # Store for training (keep gradient for Policy Gradient)
            self.market_vector_lst.append(market_vector)  # Don't detach - need gradient
            self.boundary_risk_lst.append(boundary_risk.detach())
        
        # Store gate_weights for analysis/visualization (optional)
        self.last_gate_weights = gate_weights.detach()
        
        # Convert to numpy
        market_vector_np = market_vector.detach().cpu().numpy()
        boundary_risk_np = boundary_risk.detach().cpu().numpy()
        topk_indices_np = topk_indices.detach().cpu().numpy()
        market_scores_full_np = market_scores_full.detach().cpu().numpy()
        market_context_np = market_context.detach().cpu().numpy()
        sigma_val_np = sigma_pred.detach().cpu().numpy()
        sigma_log_p_np = sigma_log_p.detach().cpu().numpy()
        stock_embedding_np = None
        if stock_embedding is not None:
            stock_embedding_np = stock_embedding.detach().cpu().numpy()
        self.last_topk_indices = topk_indices_np
        
        # Clean NaN/Inf from outputs
        market_vector_np = np.nan_to_num(market_vector_np, nan=0.0, posinf=0.0, neginf=0.0)
        boundary_risk_np = np.nan_to_num(boundary_risk_np, nan=self.config.risk_market, posinf=self.config.risk_market, neginf=self.config.risk_market)
        market_scores_full_np = np.nan_to_num(market_scores_full_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure correct shapes
        if market_vector_np.ndim == 1:
            market_vector_np = market_vector_np.reshape(1, -1)
        if boundary_risk_np.ndim == 0:
            boundary_risk_np = boundary_risk_np.reshape(1)
        if market_scores_full_np.ndim == 1:
            market_scores_full_np = market_scores_full_np.reshape(1, -1)
        
        # Handle size mismatch: pad or truncate market_vector to match action_dim
        actual_N = market_vector_np.shape[1]
        if actual_N != self.action_dim:
            if actual_N < self.action_dim:
                # Pad with zeros
                padding = np.zeros((market_vector_np.shape[0], self.action_dim - actual_N))
                market_vector_np = np.concatenate([market_vector_np, padding], axis=1)
            else:
                # Truncate
                market_vector_np = market_vector_np[:, :self.action_dim]
        
        # Handle size mismatch for market_scores_full: pad or truncate to match action_dim
        actual_N_full = market_scores_full_np.shape[1]
        if actual_N_full != self.action_dim:
            if actual_N_full < self.action_dim:
                # Pad with uniform distribution (1/N) to maintain probability distribution
                padding = np.ones((market_scores_full_np.shape[0], self.action_dim - actual_N_full)) / self.action_dim
                market_scores_full_np = np.concatenate([market_scores_full_np, padding], axis=1)
                # Renormalize to ensure sum = 1
                market_scores_full_np = market_scores_full_np / (np.sum(market_scores_full_np, axis=1, keepdims=True) + 1e-8)
            else:
                # Truncate
                market_scores_full_np = market_scores_full_np[:, :self.action_dim]
                # Renormalize to ensure sum = 1
                market_scores_full_np = market_scores_full_np / (np.sum(market_scores_full_np, axis=1, keepdims=True) + 1e-8)
        
        # Final validation: if market_vector is all zeros or contains NaN/Inf, use uniform distribution
        if np.any(np.isnan(market_vector_np)) or np.any(np.isinf(market_vector_np)) or np.sum(np.abs(market_vector_np)) < 1e-8:
            market_vector_np = np.ones((market_vector_np.shape[0], self.action_dim)) / self.action_dim
        
        # Final validation for market_scores_full: if all zeros or contains NaN/Inf, use uniform distribution
        if np.any(np.isnan(market_scores_full_np)) or np.any(np.isinf(market_scores_full_np)) or np.sum(np.abs(market_scores_full_np)) < 1e-8:
            market_scores_full_np = np.ones((market_scores_full_np.shape[0], self.action_dim)) / self.action_dim
        
        # Extract gate_weights
        if hasattr(self, 'last_gate_weights') and self.last_gate_weights is not None:
            gate_weights_np = self.last_gate_weights.detach().cpu().numpy()
        else:
            # Fallback: uniform weights for 4 experts
            gate_weights_np = np.ones((market_vector_np.shape[0], 4)) / 4.0

        # Market context cache
        if hasattr(self, 'last_market_context') and self.last_market_context is not None:
            last_context_np = self.last_market_context.detach().cpu().numpy()
        else:
            last_context_np = np.zeros((market_vector_np.shape[0], self.config.mafia_D))

        if market_context_np.ndim == 1:
            market_context_np = market_context_np.reshape(1, -1)
        self.last_market_context = market_context.detach()

        # Fallback if NaN/Inf
        market_context_np = np.nan_to_num(market_context_np, nan=0.0, posinf=0.0, neginf=0.0)
        if market_context_np.shape[1] != self.config.mafia_D:
            market_context_np = last_context_np

        # Stock embedding cache
        fused_stock_embedding_np = None
        if stock_embedding_np is not None:
            if stock_embedding_np.ndim == 3:
                fused_stock_embedding_np = stock_embedding_np
            elif stock_embedding_np.ndim == 2:
                fused_stock_embedding_np = stock_embedding_np.reshape(stock_embedding_np.shape[0], -1, self.config.mafia_D)
            if fused_stock_embedding_np is not None:
                fused_stock_embedding_np = np.nan_to_num(fused_stock_embedding_np, nan=0.0, posinf=0.0, neginf=0.0)
                if fused_stock_embedding_np.shape[1] != self.action_dim:
                    # pad/truncate to action_dim
                    if fused_stock_embedding_np.shape[1] < self.action_dim:
                        pad = np.zeros((fused_stock_embedding_np.shape[0], self.action_dim - fused_stock_embedding_np.shape[1], self.config.mafia_D))
                        fused_stock_embedding_np = np.concatenate([fused_stock_embedding_np, pad], axis=1)
                    else:
                        fused_stock_embedding_np = fused_stock_embedding_np[:, :self.action_dim, :]
        if fused_stock_embedding_np is None:
            fused_stock_embedding_np = np.zeros((market_vector_np.shape[0], self.action_dim, self.config.mafia_D), dtype=np.float32)

        # Market direction outputs
        if sigma_val_np.ndim == 0:
            sigma_val_np = sigma_val_np.reshape(1)
        sigma_val_np = np.nan_to_num(sigma_val_np, nan=1.0, posinf=1.0, neginf=1.0).astype(np.int64)
        sigma_log_p_np = np.nan_to_num(sigma_log_p_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Validate output shapes
        self._validate_output_shapes(market_vector_np, boundary_risk_np)

        return (
            market_vector_np,
            boundary_risk_np,
            market_scores_full_np,
            gate_weights_np,
            market_context_np,
            fused_stock_embedding_np,
            sigma_val_np,
            sigma_log_p_np,
        )
    
    def _prepare_ochlv_tensor(self, raw_ochlv_data):
        """
        Prepare raw OCHLV data as tensor.
        
        Args:
            raw_ochlv_data: (N, M, T_w) numpy array
        
        Returns:
            torch.Tensor: (1, N, M, T_w) tensor on device
        """
        ochlv_tensor = th.from_numpy(raw_ochlv_data).to(th.float32)
        ochlv_tensor = ochlv_tensor.unsqueeze(0)  # Add batch dim: (1, N, M, T_w)
        ochlv_tensor = ochlv_tensor.to(self.device)
        return ochlv_tensor
    
    def _prepare_market_index_ochlv_tensor(self, market_index_ochlv_data):
        """
        Prepare market-index OCHLV data as tensor.
        
        Args:
            market_index_ochlv_data: (1, 5, T_w) numpy array
        
        Returns:
            torch.Tensor: (1, 1, 5, T_w) tensor on device
        """
        mkt_ochlv_tensor = th.from_numpy(market_index_ochlv_data).to(th.float32)
        mkt_ochlv_tensor = mkt_ochlv_tensor.unsqueeze(0)  # Add batch dim: (1, 1, 5, T_w)
        mkt_ochlv_tensor = mkt_ochlv_tensor.to(self.device)
        return mkt_ochlv_tensor
    
    def _validate_input_shapes(self, finemkt_feat, finestock_feat, **kwargs):
        """Validate input shapes match expected format."""
        if finestock_feat is not None:
            assert finestock_feat.ndim == 4, f"finestock_feat must be 4D, got {finestock_feat.ndim}D"
            batch, features, num_stocks, window = finestock_feat.shape
            assert num_stocks == self.action_dim, f"num_stocks mismatch: {num_stocks} != {self.action_dim}"
        
        if finemkt_feat is not None:
            assert finemkt_feat.ndim == 3, f"finemkt_feat must be 3D, got {finemkt_feat.ndim}D"
        
        assert 'mode' in kwargs, "kwargs must include 'mode'"
        assert kwargs['mode'] in ['train', 'valid', 'test'], f"Invalid mode: {kwargs['mode']}"
    
    def _validate_output_shapes(self, market_vector, boundary_risk):
        """Validate output shapes match MarketObserver format."""
        assert market_vector.shape[1] == self.action_dim, \
            f"market_vector shape mismatch: {market_vector.shape[1]} != {self.action_dim}"
        assert market_vector.ndim == 2, f"market_vector must be 2D, got {market_vector.ndim}D"
        assert boundary_risk.ndim == 1, f"boundary_risk must be 1D, got {boundary_risk.ndim}D"
        assert market_vector.shape[0] == boundary_risk.shape[0], \
            "Batch sizes must match"
    
    def train(self, **label_kwargs):
        """
        Train MAFIA using Policy Gradient method.
        Compatible with MarketObserver.train() interface.
        
        Called at end of episode.
        
        Args:
            **label_kwargs: May contain:
                - mode: 'train', 'valid', 'test'
                - ori_profit: Original profit rates
                - adj_profit: Adjusted profit rates
                - ori_risk: Original risk values
                - adj_risk: Adjusted risk values
        """
        if len(self.market_vector_lst) == 0 or len(self.rate_of_price_change_lst) == 0:
            # No data collected, skip training
            return
        
        self.mafia_model.train()
        mode = label_kwargs.get('mode', 'train')
        
        # Compute rewards from market_vectors with gradient
        # This allows gradients to flow through the computation graph
        reward_list = []
        for market_vector, rate_of_price_change in zip(self.market_vector_lst, self.rate_of_price_change_lst):
            # Compute reward: log(1 + sum((rate - 1) * market_vector))
            # This measures how well market_vector predicted the actual returns
            portfolio_return = th.sum((rate_of_price_change - 1.0) * market_vector, dim=-1)  # (batch,)
            step_reward = th.log(portfolio_return + 1.0)  # (batch,)
            reward_list.append(step_reward)
        
        # Concatenate all rewards
        reward_tensor = th.cat(reward_list, dim=0)  # (total_steps,)
        mkt_direction_tensor = th.cat(self.mkt_direction_lst, dim=0).long()  # (total_steps,)
        
        # Policy Gradient Loss for market_vector
        # Loss = -mean(reward) (maximize expected reward)
        loss_market_vector = -th.mean(reward_tensor)
        
        # Total loss
        total_loss = self.config.hidden_vec_loss_weight * loss_market_vector

        # Optional market direction classification loss
        direction_loss = None
        if len(self.sigma_log_p_lst) > 0 and len(self.mkt_direction_lst) > 0:
            sigma_log_p_tensor = th.cat(self.sigma_log_p_lst, dim=0)  # (total_steps, 3)
            min_len = min(sigma_log_p_tensor.shape[0], mkt_direction_tensor.shape[0])
            if min_len > 0:
                sigma_log_p_tensor = sigma_log_p_tensor[:min_len]
                direction_labels = mkt_direction_tensor[:min_len]
                direction_loss = F.nll_loss(sigma_log_p_tensor, direction_labels)
                weight = getattr(self.config, "market_direction_loss_weight", 1.0)
                if weight > 0:
                    total_loss = total_loss + weight * direction_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (optional but recommended)
        th.nn.utils.clip_grad_norm_(self.mafia_model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.lr_scheduler.step()

        # Persist latest losses on config for downstream logging/plots
        if hasattr(self, "config"):
            try:
                self.config.last_mafia_loss = float(total_loss.detach().cpu().item())
            except Exception:
                self.config.last_mafia_loss = None
            if direction_loss is not None:
                try:
                    self.config.last_mafia_direction_loss = float(direction_loss.detach().cpu().item())
                except Exception:
                    self.config.last_mafia_direction_loss = None
            else:
                self.config.last_mafia_direction_loss = None
        
        # Logging
        if th.cuda.is_available():
            th.cuda.synchronize()
        disp_str = '[MAFIA] Mode: {} | Loss(MAFIA): {:.6f}'.format(
            mode, 
            total_loss.detach().cpu().item()
        )
        if direction_loss is not None:
            disp_str += " | dir_loss={:.6f}".format(direction_loss.detach().cpu().item())
        td3_actor = getattr(self.config, 'last_td3_actor_loss', None)
        td3_critic = getattr(self.config, 'last_td3_critic_loss', None)
        td3_reward = getattr(self.config, 'last_td3_mean_reward', None)
        td3_updates = getattr(self.config, 'last_td3_updates', None)
        if td3_actor is not None and td3_critic is not None:
            disp_str += " | [TD3] updates={} actor_loss={:.6f} critic_loss={:.6f}".format(
                td3_updates if td3_updates is not None else 'N/A',
                td3_actor,
                td3_critic
            )
            if td3_reward is not None:
                disp_str += " reward={:.6f}".format(td3_reward)
        print(disp_str)
        
        # Reset buffers
        self.reset()
        
        if th.cuda.is_available():
            th.cuda.empty_cache()
    
    def reset(self):
        """Reset training buffers at start of new episode."""
        self.market_vector_lst = []
        self.boundary_risk_lst = []
        self.rate_of_price_change_lst = []
        self.mkt_direction_lst = []
        self.sigma_log_p_lst = []
        self.last_sigma_log_p = None
        self.last_sigma_pred = None
    
    def update_hidden_vec_reward(self, mode, rate_of_price_change, mkt_direction):
        """
        Collect rewards for Policy Gradient training.
        Compatible with MarketObserver.update_hidden_vec_reward() interface.
        
        Args:
            mode: 'train', 'valid', 'test'
            rate_of_price_change: (batch, N+1) - includes cash as first element
            mkt_direction: (batch,) - market direction label {0, 1, 2}
        """
        if mode != 'train':
            return
        
        # Get last market_vector (from predict() call)
        if len(self.market_vector_lst) == 0:
            return
        
        last_market_vector = self.market_vector_lst[-1]  # (batch, N)
        rate_of_price_change = th.from_numpy(rate_of_price_change).to(th.float32).to(self.device)
        
        # Remove cash (first element) to match market_vector shape
        rate_of_price_change_stocks = rate_of_price_change[:, 1:]  # (batch, N_actual)
        
        # Handle size mismatch: rate_of_price_change_stocks may have different size than market_vector
        # This can happen if curData has different number of stocks than expected
        expected_size = last_market_vector.shape[1]  # Expected number of stocks
        actual_size = rate_of_price_change_stocks.shape[1]  # Actual number of stocks
        
        if actual_size != expected_size:
            # Pad or truncate to match expected size
            if actual_size < expected_size:
                # Pad with 1.0 (no change) for missing stocks
                padding_size = expected_size - actual_size
                padding = th.ones(rate_of_price_change_stocks.shape[0], padding_size, 
                                device=rate_of_price_change_stocks.device, 
                                dtype=rate_of_price_change_stocks.dtype)
                rate_of_price_change_stocks = th.cat([rate_of_price_change_stocks, padding], dim=1)
            else:
                # Truncate to expected size (take first N stocks)
                rate_of_price_change_stocks = rate_of_price_change_stocks[:, :expected_size]
        
        # Store rate_of_price_change for reward calculation in train()
        # We'll compute reward from market_vector with gradient in train() method
        self.rate_of_price_change_lst.append(rate_of_price_change_stocks)
        mkt_direction_tensor = th.from_numpy(mkt_direction).to(self.device)
        self.mkt_direction_lst.append(mkt_direction_tensor)

        # Store sigma log-probabilities for direction classification (keep gradient)
        if self.last_sigma_log_p is not None:
            self.sigma_log_p_lst.append(self.last_sigma_log_p)
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int):
        """
        Save MAFIA observer checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'mafia_model_state_dict': self.mafia_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': {
                'mafia_T_w': self.config.mafia_T_w,
                'mafia_DC_thresholds': self.config.mafia_DC_thresholds,
                'mafia_D': self.config.mafia_D,
                'mafia_D_h': self.config.mafia_D_h,
                'mafia_encoder_layers': self.config.mafia_encoder_layers,
                'mafia_encoder_heads': self.config.mafia_encoder_heads,
                'mafia_M_tech': self.config.mafia_M_tech,
                'mafia_M_dc': self.config.mafia_M_dc,
                'mafia_learning_rate': self.config.mafia_learning_rate,
                'mafia_weight_decay': self.config.mafia_weight_decay,
            },
            'action_dim': self.action_dim,
        }
        th.save(checkpoint, checkpoint_path)
        print(f"MAFIA Observer checkpoint saved to {checkpoint_path} (epoch {epoch})", flush=True)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load MAFIA observer checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            epoch: Epoch number from checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = th.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.mafia_model.load_state_dict(checkpoint['mafia_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        try:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        except KeyError as e:
            # Backward compatibility: scheduler type changed (e.g., StepLR -> LambdaLR)
            print(f"[MAFIA] Warning: LR scheduler state missing key {e}; using freshly-initialized scheduler state.", flush=True)
        
        epoch = checkpoint.get('epoch', 0)
        print(f"MAFIA Observer checkpoint loaded from {checkpoint_path} (epoch {epoch})", flush=True)
        
        return epoch
