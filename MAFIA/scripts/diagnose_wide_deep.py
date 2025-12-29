#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wide/Deep Path Diagnostic Tool

This script monitors the health of Wide Path (explicit signals) and Deep Path
(learned embeddings) during MAFIA Observer training.

Key Diagnostics:
1. Gradient Magnitude Ratio: grad(Wide) / grad(Deep) - Detects dominance
2. Activation Statistics: Mean/Std of each path - Detects dead neurons
3. Contribution Analysis: Output attribution to each path
4. Signal Health: NaN/Inf detection, scale mismatches

Usage:
    python3 agents/MAFIA/scripts/diagnose_wide_deep.py --checkpoint <path>
"""

import argparse
import os
import sys
import torch as th
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.getcwd(), "agents", "MAFIA"))

from config import Config
from RL_controller.mafia_modules import MAFIAModel
from RL_controller.mafia_observer import MAFIAObserver
from RL_controller.observer_offline_trainer import ObserverOfflineBatchTrainer
from scripts.train_observer_offline import load_mafia_data


def analyze_gradients(model, loss, verbose=True):
    """Analyze gradient flow through Wide and Deep paths."""
    
    # Backward pass to compute gradients
    loss.backward(retain_graph=True)
    
    results = {}
    
    # === DirectionHead ===
    dir_head = model.signal_generator.direction_head
    
    # Deep Path gradients
    deep_grad_expand = dir_head.expand_proj.weight.grad
    deep_grad_compress = dir_head.compress_proj.weight.grad
    
    if deep_grad_expand is not None and deep_grad_compress is not None:
        deep_grad_norm = (deep_grad_expand.norm().item() + deep_grad_compress.norm().item()) / 2
    else:
        deep_grad_norm = 0.0
    
    # Wide Path gradients (classifier weights for explicit signals start at D)
    classifier_grad = dir_head.classifier.weight.grad
    if classifier_grad is not None:
        D = dir_head.D
        wide_grad_norm = classifier_grad[:, D:].norm().item()
        deep_classifier_grad = classifier_grad[:, :D].norm().item()
    else:
        wide_grad_norm = 0.0
        deep_classifier_grad = 0.0
    
    # Gradient ratio
    total_deep = deep_grad_norm + deep_classifier_grad
    ratio = wide_grad_norm / (total_deep + 1e-8)
    
    results["direction"] = {
        "deep_grad_norm": total_deep,
        "wide_grad_norm": wide_grad_norm,
        "wide_to_deep_ratio": ratio,
        "status": "BALANCED" if 0.1 <= ratio <= 10 else ("WIDE_DOMINANT" if ratio > 10 else "DEEP_DOMINANT")
    }
    
    # === RiskHead ===
    risk_head = model.signal_generator.risk_head
    
    risk_deep_expand = risk_head.expand_proj.weight.grad
    risk_deep_compress = risk_head.compress_proj.weight.grad
    
    if risk_deep_expand is not None and risk_deep_compress is not None:
        risk_deep_grad = (risk_deep_expand.norm().item() + risk_deep_compress.norm().item()) / 2
    else:
        risk_deep_grad = 0.0
    
    risk_fusion_grad = risk_head.fusion_net.weight.grad
    if risk_fusion_grad is not None:
        D = risk_head.D
        risk_wide_grad = risk_fusion_grad[:, D:].norm().item()
        risk_deep_fusion = risk_fusion_grad[:, :D].norm().item()
    else:
        risk_wide_grad = 0.0
        risk_deep_fusion = 0.0
    
    risk_total_deep = risk_deep_grad + risk_deep_fusion
    risk_ratio = risk_wide_grad / (risk_total_deep + 1e-8)
    
    results["risk"] = {
        "deep_grad_norm": risk_total_deep,
        "wide_grad_norm": risk_wide_grad,
        "wide_to_deep_ratio": risk_ratio,
        "status": "BALANCED" if 0.1 <= risk_ratio <= 10 else ("WIDE_DOMINANT" if risk_ratio > 10 else "DEEP_DOMINANT")
    }
    
    if verbose:
        print("\n" + "="*60)
        print("GRADIENT ANALYSIS")
        print("="*60)
        print("\n[DirectionHead]")
        print(f"  Deep Path Grad Norm:  {results['direction']['deep_grad_norm']:.6f}")
        print(f"  Wide Path Grad Norm:  {results['direction']['wide_grad_norm']:.6f}")
        print(f"  Wide/Deep Ratio:      {results['direction']['wide_to_deep_ratio']:.4f}")
        print(f"  Status:               {results['direction']['status']}")
        
        print("\n[RiskHead]")
        print(f"  Deep Path Grad Norm:  {results['risk']['deep_grad_norm']:.6f}")
        print(f"  Wide Path Grad Norm:  {results['risk']['wide_grad_norm']:.6f}")
        print(f"  Wide/Deep Ratio:      {results['risk']['wide_to_deep_ratio']:.4f}")
        print(f"  Status:               {results['risk']['status']}")
    
    return results


def analyze_activations(model, c_mkt, delta_c, explicit_signals, verbose=True):
    """Analyze activation statistics for Wide and Deep paths."""
    
    results = {}
    
    # === DirectionHead ===
    dir_head = model.signal_generator.direction_head
    
    # Deep Path activations (matching forward pass)
    x_latent = th.cat([c_mkt, delta_c], dim=-1)
    h_expand = th.nn.functional.gelu(dir_head.expand_proj(x_latent))
    h_deep = dir_head.compress_proj(h_expand)
    h_deep = dir_head.deep_norm(h_deep)  # [FIX] Include LayerNorm
    
    deep_mean = h_deep.mean().item()
    deep_std = h_deep.std().item()
    deep_dead = (h_deep.abs() < 1e-6).float().mean().item() * 100  # % dead neurons
    
    # Wide Path activations
    signals_clamped = explicit_signals.clamp(-3.0, 3.0)
    signals_norm = dir_head.explicit_norm(signals_clamped)
    signals_interact = dir_head.wide_interaction(signals_norm)
    signals_post = dir_head.post_interaction_norm(signals_interact)
    
    wide_mean = signals_post.mean().item()
    wide_std = signals_post.std().item()
    wide_dead = (signals_post.abs() < 1e-6).float().mean().item() * 100
    
    # Scale mismatch detection
    scale_ratio = deep_std / (wide_std + 1e-8)
    
    results["direction"] = {
        "deep_mean": deep_mean,
        "deep_std": deep_std,
        "deep_dead_pct": deep_dead,
        "wide_mean": wide_mean,
        "wide_std": wide_std,
        "wide_dead_pct": wide_dead,
        "scale_ratio": scale_ratio,
        "scale_status": "BALANCED" if 0.1 <= scale_ratio <= 10 else "MISMATCH"
    }
    
    # === RiskHead ===
    risk_head = model.signal_generator.risk_head
    
    # Similar analysis for RiskHead
    x_latent_risk = th.cat([c_mkt, delta_c], dim=-1)
    h_expand_risk = th.nn.functional.gelu(risk_head.expand_proj(x_latent_risk))
    h_deep_risk = risk_head.compress_proj(h_expand_risk)
    h_deep_risk = risk_head.deep_norm(h_deep_risk)  # [FIX] Include LayerNorm
    
    risk_deep_mean = h_deep_risk.mean().item()
    risk_deep_std = h_deep_risk.std().item()
    risk_deep_dead = (h_deep_risk.abs() < 1e-6).float().mean().item() * 100
    
    signals_risk = risk_head.explicit_norm(signals_clamped)
    signals_risk_interact = risk_head.wide_interaction(signals_risk)
    signals_risk_post = risk_head.post_interaction_norm(signals_risk_interact)
    
    risk_wide_mean = signals_risk_post.mean().item()
    risk_wide_std = signals_risk_post.std().item()
    risk_wide_dead = (signals_risk_post.abs() < 1e-6).float().mean().item() * 100
    
    risk_scale_ratio = risk_deep_std / (risk_wide_std + 1e-8)
    
    results["risk"] = {
        "deep_mean": risk_deep_mean,
        "deep_std": risk_deep_std,
        "deep_dead_pct": risk_deep_dead,
        "wide_mean": risk_wide_mean,
        "wide_std": risk_wide_std,
        "wide_dead_pct": risk_wide_dead,
        "scale_ratio": risk_scale_ratio,
        "scale_status": "BALANCED" if 0.1 <= risk_scale_ratio <= 10 else "MISMATCH"
    }
    
    if verbose:
        print("\n" + "="*60)
        print("ACTIVATION ANALYSIS")
        print("="*60)
        print("\n[DirectionHead]")
        print(f"  Deep Path: Mean={deep_mean:.4f}, Std={deep_std:.4f}, Dead={deep_dead:.1f}%")
        print(f"  Wide Path: Mean={wide_mean:.4f}, Std={wide_std:.4f}, Dead={wide_dead:.1f}%")
        print(f"  Scale Ratio (Deep/Wide): {scale_ratio:.4f} [{results['direction']['scale_status']}]")
        
        print("\n[RiskHead]")
        print(f"  Deep Path: Mean={risk_deep_mean:.4f}, Std={risk_deep_std:.4f}, Dead={risk_deep_dead:.1f}%")
        print(f"  Wide Path: Mean={risk_wide_mean:.4f}, Std={risk_wide_std:.4f}, Dead={risk_wide_dead:.1f}%")
        print(f"  Scale Ratio (Deep/Wide): {risk_scale_ratio:.4f} [{results['risk']['scale_status']}]")
    
    return results


def analyze_signal_health(explicit_signals, verbose=True):
    """Check for NaN, Inf, and abnormal values in explicit signals."""
    
    results = {}
    
    nan_count = th.isnan(explicit_signals).sum().item()
    inf_count = th.isinf(explicit_signals).sum().item()
    total = explicit_signals.numel()
    
    # Per-signal statistics
    signal_names = ["Vol_Std20", "DC_Event", "Breadth_Gap", "Div_Signal", "Signed_VPI", "Drawdown60"]
    
    signal_stats = []
    for i in range(min(explicit_signals.shape[-1], len(signal_names))):
        sig = explicit_signals[..., i]
        stat = {
            "name": signal_names[i] if i < len(signal_names) else f"Signal_{i}",
            "mean": sig.mean().item(),
            "std": sig.std().item(),
            "min": sig.min().item(),
            "max": sig.max().item(),
            "nan_pct": (th.isnan(sig).sum().item() / sig.numel()) * 100,
            "status": "OK" if not th.isnan(sig).any() and 0.01 < sig.std().item() < 10 else "CHECK"
        }
        signal_stats.append(stat)
    
    results = {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "total_elements": total,
        "health_status": "HEALTHY" if nan_count == 0 and inf_count == 0 else "UNHEALTHY",
        "signals": signal_stats
    }
    
    if verbose:
        print("\n" + "="*60)
        print("EXPLICIT SIGNAL HEALTH CHECK")
        print("="*60)
        print(f"\nOverall: NaN={nan_count}, Inf={inf_count}, Status={results['health_status']}")
        print("\nPer-Signal Statistics:")
        print(f"{'Signal':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Status':>8}")
        print("-" * 65)
        for s in signal_stats:
            print(f"{s['name']:<15} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f} {s['status']:>8}")
    
    return results


def run_diagnosis(checkpoint_path=None):
    """Run full Wide/Deep path diagnosis."""
    
    print("="*60)
    print("WIDE/DEEP PATH DIAGNOSTIC TOOL")
    print("="*60)
    
    # 1. Setup
    config = Config(create_dirs=False)
    config.device = th.device("cpu")
    
    # 2. Load Data
    print("\n[1/4] Loading data...")
    stock_data = load_mafia_data(config)
    stock_list = sorted(stock_data["stock"].unique().tolist())
    
    # 3. Create Model
    print("[2/4] Creating model...")
    action_dim = len(stock_list)
    observer = MAFIAObserver(config, action_dim=action_dim)
    model = observer.mafia_model
    model.to(config.device)  # Force CPU
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[3/4] Loading checkpoint: {checkpoint_path}")
        observer.load_checkpoint(checkpoint_path, load_optimizer=False)
    else:
        print("[3/4] Using randomly initialized model (no checkpoint)")
    
    model.eval()
    
    # 4. Generate sample data for analysis
    print("[4/4] Generating sample data...")
    B = 16
    D = config.mafia_D
    
    # Mock inputs
    c_mkt = th.randn(B, D) * 0.1
    delta_c = th.randn(B, D) * 0.1
    explicit_signals = th.randn(B, 6) * 0.5
    
    # === Run Diagnostics ===
    
    # A. Signal Health
    _ = analyze_signal_health(explicit_signals)
    
    # B. Activation Analysis
    with th.no_grad():
        _ = analyze_activations(model, c_mkt, delta_c, explicit_signals)
    
    # C. Gradient Analysis (requires forward + backward)
    model.train()
    model.zero_grad()
    
    # Forward pass through Direction and Risk heads
    dir_logits = model.signal_generator.direction_head(c_mkt, delta_c, explicit_signals)
    risk_eta = model.signal_generator.risk_head(c_mkt, delta_c, explicit_signals)
    
    # Fake targets
    dir_target = th.randint(0, 3, (B,))
    risk_target = th.ones(B) * 1.0
    
    # Losses
    dir_loss = th.nn.functional.cross_entropy(dir_logits, dir_target)
    risk_loss = th.nn.functional.mse_loss(risk_eta.squeeze(), risk_target)
    total_loss = dir_loss + risk_loss
    
    _ = analyze_gradients(model, total_loss)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("\nRecommendations:")
    print("- If DEEP_DOMINANT: Wide Path signals may not be contributing. Check signal preprocessing.")
    print("- If WIDE_DOMINANT: Deep Path may be underfitting. Check latent embedding quality.")
    print("- If MISMATCH: Scale signals before fusion (LayerNorm should handle this).")
    print("- If high Dead %: Check for vanishing gradients or initialization issues.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wide/Deep Path Diagnostic Tool")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    
    run_diagnosis(args.checkpoint)
