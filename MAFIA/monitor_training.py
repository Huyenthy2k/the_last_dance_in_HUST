#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple training monitor - prints clear progress updates
No external dependencies needed (no tqdm)
"""

import os
import sys
import datetime
import time


class SimpleProgressMonitor:
    """Simple progress monitor with clear output."""
    
    def __init__(self, total_epochs, steps_per_epoch):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.start_time = time.time()
        self.current_epoch = 0
        self.current_step = 0
        
    def update_epoch(self, epoch, portfolio_value=None, actor_loss=None, critic_loss=None):
        """Update epoch progress."""
        self.current_epoch = epoch
        elapsed = time.time() - self.start_time
        progress_pct = (epoch / self.total_epochs) * 100
        
        print("\n" + "="*80)
        print(f"📊 EPOCH {epoch}/{self.total_epochs} ({progress_pct:.1f}%)")
        print("="*80)
        print(f"⏱️  Elapsed Time: {self._format_time(elapsed)}")
        
        if portfolio_value:
            print(f"💰 Portfolio Value: ${portfolio_value:,.2f}")
        
        if actor_loss is not None:
            print(f"📉 Actor Loss: {actor_loss:.6f}")
        
        if critic_loss is not None:
            print(f"📉 Critic Loss: {critic_loss:.6f}")
        
        # Estimate remaining time
        if epoch > 0:
            avg_epoch_time = elapsed / epoch
            remaining_epochs = self.total_epochs - epoch
            estimated_remaining = avg_epoch_time * remaining_epochs
            print(f"⏳ Estimated Remaining: {self._format_time(estimated_remaining)}")
        
        print("="*80)
    
    def update_step(self, step, reward=None, info=None):
        """Update step progress."""
        self.current_step = step
        step_in_epoch = step % self.steps_per_epoch
        progress_in_epoch = (step_in_epoch / self.steps_per_epoch) * 100
        
        # Print every 10% of epoch
        if step_in_epoch % (self.steps_per_epoch // 10) == 0:
            print(f"  Step {step_in_epoch}/{self.steps_per_epoch} ({progress_in_epoch:.0f}%)", end="")
            if reward is not None:
                print(f" | Reward: {reward:.4f}", end="")
            print()
            sys.stdout.flush()
    
    def _format_time(self, seconds):
        """Format seconds to readable string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def print_summary(self):
        """Print training summary."""
        elapsed = time.time() - self.start_time
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED")
        print("="*80)
        print(f"Total Time: {self._format_time(elapsed)}")
        print(f"Total Epochs: {self.current_epoch}/{self.total_epochs}")
        print(f"Total Steps: {self.current_step}")
        print("="*80 + "\n")


def print_training_header(config, total_timesteps):
    """Print training configuration header."""
    print("\n" + "="*80)
    print("🚀 MAFIA TRAINING - REAL-TIME MONITOR".center(80))
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Algorithm: {config.benchmark_algo}")
    print(f"   Market: {config.market_name}")
    print(f"   Stocks: Top-{config.topK}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Total Timesteps: {total_timesteps:,}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Mode: {config.mode}")
    print(f"\n📁 Results Directory:")
    print(f"   {config.res_dir}")
    print("="*80 + "\n")
    print("⏰ Started at:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80 + "\n")


def print_metrics_table(metrics_history):
    """Print a table of metrics."""
    if not metrics_history:
        return
    
    print("\n" + "="*80)
    print("📈 TRAINING METRICS HISTORY".center(80))
    print("="*80)
    print(f"{'Epoch':<8} {'Portfolio Value':<20} {'Actor Loss':<15} {'Critic Loss':<15}")
    print("-"*80)
    
    for metric in metrics_history[-10:]:  # Last 10 epochs
        epoch = metric.get('epoch', '-')
        pv = metric.get('portfolio_value', 0)
        actor = metric.get('actor_loss', 0)
        critic = metric.get('critic_loss', 0)
        
        print(f"{epoch:<8} ${pv:>18,.2f} {actor:>14.6f} {critic:>14.6f}")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    # Example usage
    print("\n" + "="*80)
    print("Training Monitor - Usage Examples".center(80))
    print("="*80)
    print("""
This module provides simple progress monitoring for MAFIA training.

Usage in your training script:
    
    from monitor_training import SimpleProgressMonitor, print_training_header
    
    # At start of training
    monitor = SimpleProgressMonitor(
        total_epochs=config.num_epochs,
        steps_per_epoch=env_train.totalTradeDay
    )
    print_training_header(config, total_timesteps)
    
    # During training (in callback)
    monitor.update_epoch(
        epoch=current_epoch,
        portfolio_value=env.cur_capital,
        actor_loss=mean_actor_loss,
        critic_loss=mean_critic_loss
    )
    
    # At end of training
    monitor.print_summary()

Or simply run the enhanced test script:
    
    python test_training_progress.py

This will show:
    ✅ Real-time progress bars (with tqdm)
    ✅ Epoch-by-epoch metrics
    ✅ Loss values
    ✅ Portfolio value changes
    ✅ Time estimates
    """)
    print("="*80 + "\n")

