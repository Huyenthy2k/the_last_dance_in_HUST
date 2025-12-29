"""
Simple Terminal Logger for Training Progress
=============================================
Clean, minimal logging without fancy formatting.
Dashboard updates are separate and unaffected.
"""

import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any


class SimpleLogger:
    """Simple logger that prints clean progress to terminal."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.last_step = 0

        # Phase tracking
        self.current_phase = "INIT"
        self.current_epoch = 0
        self.total_epochs = 1

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _elapsed(self) -> str:
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)
        if hours > 0:
            return f"{hours}h{mins:02d}m"
        elif mins > 0:
            return f"{mins}m{secs:02d}s"
        else:
            return f"{secs}s"

    def _write(self, msg: str, newline: bool = True):
        """Write to stdout and optionally to log file."""
        if newline:
            print(msg, flush=True)
        else:
            sys.stdout.write(f"\r{msg}")
            sys.stdout.flush()

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(msg + "\n")

    # =========================================================================
    # Phase Announcements
    # =========================================================================

    def phase_start(self, phase: str, details: str = ""):
        """Announce start of a training phase."""
        self.current_phase = phase
        separator = "=" * 60
        self._write(f"\n{separator}")
        self._write(f"[{self._timestamp()}] PHASE: {phase}")
        if details:
            self._write(f"  {details}")
        self._write(separator)

    def epoch_start(self, epoch: int, total_epochs: int, date_range: str = ""):
        """Announce start of an epoch."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        self._write(
            f"\n[{self._timestamp()}] Epoch {epoch}/{total_epochs} started {date_range}"
        )

    def epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Announce end of an epoch with metrics."""
        sharpe = metrics.get("sharpeRatio", 0)
        mdd = metrics.get("mdd", 0)
        annual_ret = metrics.get("annualReturn_pct", 0)
        final_cap = metrics.get(
            "final_capital", metrics.get("netProfit", 0) + 1_000_000
        )

        self._write(
            f"[{self._timestamp()}] Epoch {epoch} done | "
            f"Sharpe: {sharpe:.3f} | MDD: {mdd:.1f}% | "
            f"Annual: {annual_ret * 100:.1f}% | Capital: ${final_cap:,.0f}"
        )

    # =========================================================================
    # Progress Updates
    # =========================================================================

    def step_progress(
        self,
        step: int,
        total_steps: int,
        epoch: int,
        total_epochs: int,
        date: str = "",
        extra: str = "",
    ):
        """Update step progress - prints new line each time for clean terminal."""
        # Calculate speed
        now = time.time()
        delta_time = now - self.last_step_time
        delta_steps = step - self.last_step

        if delta_time > 0 and delta_steps > 0:
            speed = delta_steps / delta_time
        else:
            speed = 0

        self.last_step_time = now
        self.last_step = step

        pct = (step / total_steps * 100) if total_steps > 0 else 0

        msg = (
            f"[{self._elapsed()}] Ep {epoch}/{total_epochs} | "
            f"Day {step}/{total_steps} ({pct:.0f}%)"
        )
        if date:
            msg += f" | {date}"
        if extra:
            msg += f" | {extra}"
        if speed > 0:
            msg += f" | {speed:.1f}/s"

        # Use newline=True to avoid terminal conflicts
        self._write(msg, newline=True)

    def observer_update(
        self,
        step: int,
        buffer_size: int,
        warmup_target: int,
        loss_dir: float = 0,
        loss_eta: float = 0,
        loss_pg: float = 0,
    ):
        """Log Observer training progress."""
        warmup_pct = (buffer_size / warmup_target * 100) if warmup_target > 0 else 0
        is_warmup = buffer_size < warmup_target

        if is_warmup:
            msg = (
                f"[{self._elapsed()}] OBSERVER WARMUP | "
                f"Buffer: {buffer_size}/{warmup_target} ({warmup_pct:.0f}%)"
            )
        else:
            total_loss = loss_dir + loss_eta + loss_pg
            msg = (
                f"[{self._elapsed()}] OBSERVER | "
                f"Loss: {total_loss:.4f} (dir={loss_dir:.4f}, eta={loss_eta:.4f}, pg={loss_pg:.4f})"
            )
        self._write(msg, newline=True)

    # =========================================================================
    # Validation / Test
    # =========================================================================

    def validation_start(self, epoch: int):
        """Announce validation phase."""
        self._write(f"\n[{self._timestamp()}] Validation for Epoch {epoch}...")

    def validation_end(
        self, epoch: int, metrics: Dict[str, Any], is_best: bool = False
    ):
        """Report validation results."""
        sharpe = metrics.get("sharpeRatio", 0)
        mdd = metrics.get("mdd", 0)
        best_marker = " *** BEST ***" if is_best else ""
        self._write(
            f"[{self._timestamp()}] Validation done | Sharpe: {sharpe:.3f} | MDD: {mdd:.1f}%{best_marker}"
        )

    def test_start(self, epoch: int):
        """Announce test phase."""
        self._write(f"\n[{self._timestamp()}] Testing for Epoch {epoch}...")

    def test_end(self, epoch: int, metrics: Dict[str, Any]):
        """Report test results."""
        sharpe = metrics.get("sharpeRatio", 0)
        mdd = metrics.get("mdd", 0)
        annual_ret = metrics.get("annualReturn_pct", 0)
        self._write(
            f"[{self._timestamp()}] Test done | Sharpe: {sharpe:.3f} | "
            f"MDD: {mdd:.1f}% | Annual: {annual_ret * 100:.1f}%"
        )

    # =========================================================================
    # Walk-Forward
    # =========================================================================

    def walkforward_iteration_start(
        self,
        iteration: int,
        total: int,
        train_range: str,
        valid_range: str,
        infer_year: int,
    ):
        """Announce walk-forward iteration."""
        self._write(f"\n{'=' * 60}")
        self._write(f"WALK-FORWARD ITERATION {iteration}/{total}")
        self._write(f"  Train:  {train_range}")
        self._write(f"  Valid:  {valid_range}")
        self._write(f"  Infer:  {infer_year}")
        self._write("=" * 60)

    def walkforward_iteration_end(self, iteration: int, metrics: Dict[str, Any]):
        """Report walk-forward iteration results."""
        sharpe = metrics.get("sharpeRatio", 0)
        mdd = metrics.get("mdd", 0)
        self._write(
            f"[{self._timestamp()}] WF Iter {iteration} done | "
            f"Sharpe: {sharpe:.3f} | MDD: {mdd:.1f}%"
        )

    # =========================================================================
    # Errors / Warnings
    # =========================================================================

    def error(self, msg: str):
        """Log an error."""
        self._write(f"\n[ERROR] {msg}")

    def warning(self, msg: str):
        """Log a warning."""
        self._write(f"[WARN] {msg}")

    def info(self, msg: str):
        """Log info message."""
        self._write(f"[INFO] {msg}")

    def debug(self, msg: str):
        """Log debug message (only to file if log_file set)."""
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"[DEBUG] {msg}\n")


# Global logger instance
_logger: Optional[SimpleLogger] = None


def get_simple_logger() -> SimpleLogger:
    """Get or create the global simple logger."""
    global _logger
    if _logger is None:
        _logger = SimpleLogger()
    return _logger


def init_simple_logger(log_file: Optional[str] = None) -> SimpleLogger:
    """Initialize global simple logger with optional log file."""
    global _logger
    _logger = SimpleLogger(log_file=log_file)
    return _logger
