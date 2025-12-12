# -*- coding: utf-8 -*-
"""
Training Progress Logger with Colored Output

Provides enhanced terminal logging for MAFIA training pipeline:
- Colored phase indicators (Pretrain, Train, Valid, Test)
- Real-time progress updates on single line
- Phase transition highlights
- Clear epoch/step tracking
"""

import os
import sys
import time
from enum import Enum
from typing import Optional, Dict, Any


def _is_display_suppressed() -> bool:
    """Check if display output should be suppressed (simple_logger mode)."""
    # In simple logging mode, always allow printing even if LiveDisplay is off
    if os.environ.get("MAFIA_SIMPLE_LOGGING", "").lower() in ("1", "true", "yes"):
        return False
    return os.environ.get("MAFIA_NO_LIVE_DISPLAY", "").lower() in ("1", "true", "yes")


# Import LiveDisplay check for stdout suppression
try:
    from utils.display_integration import get_display, smart_print

    LIVE_DISPLAY_AVAILABLE = True
except ImportError:
    LIVE_DISPLAY_AVAILABLE = False
    smart_print = print  # Fallback

    def get_display():
        return None


def _is_live_display_active() -> bool:
    """Check if LiveDisplay is currently active and rendering."""
    if not LIVE_DISPLAY_AVAILABLE:
        return False
    display = get_display()
    return display is not None and display.enabled


class Colors:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class TrainingPhase(Enum):
    """Training phases with associated colors and icons.

    2-Phase Training Architecture:
    - Phase 1 (OBSERVER_PRETRAIN): Observer Walk-Forward Training (TD3 frozen)
    - Phase 2 (RL_TRAIN): TD3 Training (Observer frozen as Static Expert)
    """

    INIT = ("KHỞI TẠO", Colors.BRIGHT_WHITE, "⚙️")

    # Legacy (backward compat) - maps to OBSERVER_PRETRAIN
    PRETRAIN = ("PRETRAIN", Colors.BRIGHT_MAGENTA, "🔮")

    # 2-Phase Training: Phase 1 - Observer Walk-Forward Training
    OBSERVER_PRETRAIN = ("OBSERVER TRAIN", Colors.BRIGHT_MAGENTA, "🔮")

    WARMUP = ("WARMUP", Colors.BRIGHT_YELLOW, "🔥")

    # Legacy (backward compat) - maps to RL_TRAIN
    TRAIN = ("TRAIN", Colors.BRIGHT_GREEN, "🎯")

    # 2-Phase Training: Phase 2 - TD3 Training with frozen Observer
    RL_TRAIN = ("TD3 TRAIN", Colors.BRIGHT_GREEN, "🎯")

    VALID = ("VALID", Colors.BRIGHT_CYAN, "📊")
    TEST = ("TEST", Colors.BRIGHT_BLUE, "🧪")
    CHECKPOINT = ("CHECKPOINT", Colors.BRIGHT_YELLOW, "💾")
    COMPLETE = ("HOÀN THÀNH", Colors.BRIGHT_GREEN, "✅")
    ERROR = ("LỖI", Colors.BRIGHT_RED, "❌")


class TrainingLogger:
    """
    Enhanced training logger with colored output and real-time updates.

    Features:
    - Phase-specific coloring
    - Real-time single-line progress updates
    - Phase transition banners
    - Metric tracking and display
    """

    def __init__(self, enable_colors: bool = True, log_file: Optional[str] = None):
        self.enable_colors = enable_colors
        self.log_file = log_file
        self.current_phase = TrainingPhase.INIT
        self.start_time = time.time()
        self.phase_start_time = time.time()
        self._last_line_length = 0
        self._last_progress_time = 0
        self._progress_update_interval = 0.1  # Update at most 10 times per second

        # Tracking metrics
        self.total_epochs = 0
        self.total_steps = 0
        self.current_epoch = 0
        self.current_step = 0
        self.steps_per_second = 0.0

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.enable_colors:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _clear_line(self):
        """Clear the current line."""
        # Always allow terminal logging for debugging
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()

    def _write_to_file(self, message: str):
        """Write message to log file if configured."""
        if self.log_file:
            # Strip ANSI codes for file output
            import re

            clean_message = re.sub(r"\033\[[0-9;]*m", "", message)
            with open(self.log_file, "a") as f:
                f.write(clean_message + "\n")

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def _format_eta(self, remaining_steps: int, steps_per_sec: float) -> str:
        """Format ETA based on remaining steps and speed."""
        if steps_per_sec <= 0 or remaining_steps <= 0:
            return "--:--"
        eta_seconds = remaining_steps / steps_per_sec
        return self._format_time(eta_seconds)

    def print_banner(
        self, title: str, phase: Optional[TrainingPhase] = None, width: int = 100
    ):
        """Print a prominent banner for phase transitions."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return

        self._clear_line()

        if phase is None:
            phase = self.current_phase

        phase_name, color, icon = phase.value

        # Top border
        border = "═" * width
        smart_print(f"\n{self._colorize(border, color)}")

        # Title line
        title_with_icon = f"{icon} {title}"
        padding = (width - len(title_with_icon) - 4) // 2
        title_line = f"║{' ' * padding}{title_with_icon}{' ' * (width - padding - len(title_with_icon) - 2)}║"
        smart_print(self._colorize(title_line, color + Colors.BOLD))

        # Bottom border
        smart_print(f"{self._colorize(border, color)}")
        sys.stdout.flush()

    def print_phase_transition(
        self, from_phase: TrainingPhase, to_phase: TrainingPhase, details: str = ""
    ):
        """Print a highlighted phase transition message."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            self.current_phase = to_phase
            self.phase_start_time = time.time()
            return

        self._clear_line()

        from_name, from_color, from_icon = from_phase.value
        to_name, to_color, to_icon = to_phase.value

        elapsed = time.time() - self.phase_start_time

        # Transition message
        arrow = self._colorize(" ══► ", Colors.BRIGHT_WHITE + Colors.BOLD)
        from_str = self._colorize(f"[{from_icon} {from_name}]", from_color)
        to_str = self._colorize(f"[{to_icon} {to_name}]", to_color + Colors.BOLD)

        message = f"\n{'─' * 100}"
        message += f"\n🔄 CHUYỂN GIAI ĐOẠN: {from_str}{arrow}{to_str}"
        if details:
            message += f"\n   {self._colorize('Chi tiết:', Colors.DIM)} {details}"
        message += f"\n   {self._colorize('Thời gian giai đoạn trước:', Colors.DIM)} {self._format_time(elapsed)}"
        message += f"\n{'─' * 100}\n"

        smart_print(message, flush=True)
        self._write_to_file(message)

        self.current_phase = to_phase
        self.phase_start_time = time.time()

    def set_phase(self, phase: TrainingPhase):
        """Set current phase without printing transition."""
        self.current_phase = phase
        self.phase_start_time = time.time()

    def print_epoch_start(
        self, epoch: int, total_epochs: int, phase: TrainingPhase = TrainingPhase.TRAIN
    ):
        """Print epoch start banner."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            self.current_epoch = epoch
            self.total_epochs = total_epochs
            return

        self._clear_line()
        self.current_epoch = epoch
        self.total_epochs = total_epochs

        phase_name, color, icon = phase.value

        progress_pct = (epoch / total_epochs) * 100 if total_epochs > 0 else 0
        elapsed = time.time() - self.start_time

        # Create progress bar
        bar_width = 30
        filled = int(bar_width * epoch / total_epochs) if total_epochs > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        banner = f"""
{self._colorize("═" * 100, color)}
{self._colorize(f"{icon} EPOCH {epoch}/{total_epochs}", color + Colors.BOLD)} [{self._colorize(bar, Colors.BRIGHT_GREEN)}] {progress_pct:.1f}%
{self._colorize("─" * 100, Colors.DIM)}
  📍 Giai đoạn: {self._colorize(phase_name, color)}
  ⏱️  Thời gian đã chạy: {self._format_time(elapsed)}
  🔢 Tổng bước (Global Steps): {self.current_step:,}
{self._colorize("═" * 100, color)}
"""
        smart_print(banner, flush=True)
        self._write_to_file(banner)

    def print_epoch_complete(self, epoch: int, metrics: Dict[str, Any]):
        """Print epoch completion summary."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return

        self._clear_line()

        phase_name, color, icon = self.current_phase.value
        epoch_time = time.time() - self.phase_start_time

        # Format metrics
        metrics_str = ""
        if metrics:
            metric_items = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    if abs(value) < 0.01:
                        metric_items.append(f"{key}: {value:.6f}")
                    else:
                        metric_items.append(f"{key}: {value:.4f}")
                else:
                    metric_items.append(f"{key}: {value}")
            metrics_str = " | ".join(metric_items)

        summary = f"""
{self._colorize("┌" + "─" * 98 + "┐", color)}
{self._colorize("│", color)} ✅ EPOCH {epoch} HOÀN THÀNH {self._colorize("│", color)}
{self._colorize("├" + "─" * 98 + "┤", color)}
{self._colorize("│", color)}   ⏱️  Thời gian epoch: {self._format_time(epoch_time):>10} {self._colorize("│", color)}
{self._colorize("│", color)}   📊 Kết quả: {metrics_str[:75]:75} {self._colorize("│", color)}
{self._colorize("└" + "─" * 98 + "┘", color)}
"""
        smart_print(summary, flush=True)
        self._write_to_file(summary)

    def update_progress(
        self,
        step: int,
        total_steps: int,
        epoch: int,
        total_epochs: int,
        day_in_epoch: int,
        total_days: int,
        speed: float,
        buffer_size: Optional[int] = None,
        buffer_target: Optional[int] = None,
        portfolio_value: Optional[float] = None,
        portfolio_return: Optional[float] = None,
        force: bool = False,
    ):
        """
        Update progress on a single line (real-time).

        This method updates the progress display in-place without creating new lines.
        Rate-limited to avoid excessive terminal updates.
        """
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            # Still update internal tracking
            self.current_step = step
            self.current_epoch = epoch
            self.steps_per_second = speed
            return

        current_time = time.time()

        # Rate limiting
        if (
            not force
            and (current_time - self._last_progress_time)
            < self._progress_update_interval
        ):
            return
        self._last_progress_time = current_time

        self.current_step = step
        self.current_epoch = epoch
        self.steps_per_second = speed

        phase_name, color, icon = self.current_phase.value

        # Calculate ETA
        remaining_steps = total_steps - step
        eta = self._format_eta(remaining_steps, speed)

        # Progress percentage
        progress_pct = (step / total_steps * 100) if total_steps > 0 else 0

        # Build progress line
        parts = []

        # Phase indicator
        phase_indicator = self._colorize(f"[{icon} {phase_name}]", color + Colors.BOLD)
        parts.append(phase_indicator)

        # Step progress
        step_str = f"Bước {step:,}/{total_steps:,}"
        parts.append(self._colorize(step_str, Colors.WHITE))

        # Epoch/Day info
        epoch_str = f"Epoch {epoch}/{total_epochs} Ngày {day_in_epoch}/{total_days}"
        parts.append(self._colorize(epoch_str, Colors.BRIGHT_CYAN))

        # Speed
        speed_str = f"{speed:.1f} steps/s"
        parts.append(self._colorize(speed_str, Colors.BRIGHT_YELLOW))

        # ETA
        eta_str = f"ETA: {eta}"
        parts.append(self._colorize(eta_str, Colors.BRIGHT_MAGENTA))

        # Buffer info (for warmup)
        if buffer_size is not None and buffer_target is not None:
            buffer_pct = (buffer_size / buffer_target * 100) if buffer_target > 0 else 0
            buffer_str = f"Buffer {buffer_size}/{buffer_target} ({buffer_pct:.0f}%)"
            parts.append(self._colorize(buffer_str, Colors.BRIGHT_YELLOW))

        # Portfolio value
        if portfolio_value is not None:
            pv_str = f"${portfolio_value:,.0f}"
            if portfolio_return is not None:
                sign = "+" if portfolio_return >= 0 else ""
                return_color = (
                    Colors.BRIGHT_GREEN if portfolio_return >= 0 else Colors.BRIGHT_RED
                )
                pv_str += self._colorize(
                    f" ({sign}{portfolio_return:.2f}%)", return_color
                )
            parts.append(pv_str)

        # Combine parts
        line = " │ ".join(parts)

        # Always allow terminal logging for debugging
        # Clear line and write
        sys.stdout.write(f"\r\033[2K{line}")
        sys.stdout.flush()

        self._last_line_length = len(line)

    def print_warmup_status(self, buffer_size: int, buffer_target: int, speed: float):
        """Print warmup status on a single line."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return

        phase_name, color, icon = TrainingPhase.WARMUP.value

        progress_pct = (buffer_size / buffer_target * 100) if buffer_target > 0 else 0
        bar_width = 20
        filled = (
            int(bar_width * buffer_size / buffer_target) if buffer_target > 0 else 0
        )
        bar = "█" * filled + "░" * (bar_width - filled)

        line = f"{self._colorize(f'[{icon} {phase_name}]', color + Colors.BOLD)} "
        line += f"Thu thập kinh nghiệm [{self._colorize(bar, Colors.BRIGHT_YELLOW)}] "
        line += f"{buffer_size:,}/{buffer_target:,} ({progress_pct:.1f}%) "
        line += f"│ {speed:.1f} steps/s"

        # Always allow terminal logging for debugging
        sys.stdout.write(f"\r\033[2K{line}")
        sys.stdout.flush()

    def print_pretrain_progress(
        self,
        step: int,
        total_steps: int,
        mini_epoch: int,
        total_mini_epochs: int,
        config=None,
    ):
        """Print pretrain progress on a single line with descriptive info including Observer losses."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return

        phase_name, color, icon = TrainingPhase.PRETRAIN.value

        progress_pct = (step / total_steps * 100) if total_steps > 0 else 0

        # Calculate ETA
        elapsed = time.time() - self.phase_start_time
        steps_per_sec = step / elapsed if elapsed > 0 else 0
        remaining_steps = total_steps - step
        eta = self._format_eta(remaining_steps, steps_per_sec)

        line = f"{self._colorize(f'[{icon} {phase_name}]', color + Colors.BOLD)} "
        line += f"Observer-only warmup │ "
        line += f"Mini-epoch {mini_epoch}/{total_mini_epochs} │ "
        line += f"Bước {step:,}/{total_steps:,} ({progress_pct:.1f}%) │ "
        line += f"Tốc độ: {steps_per_sec:.1f} steps/s │ "
        line += f"ETA: {eta}"

        # Add Observer loss info if available
        if config is not None:
            total_loss = getattr(config, "last_mafia_loss", None)
            eta_loss = getattr(config, "last_mafia_eta_loss", None)
            dir_loss = getattr(config, "last_mafia_direction_loss", None)

            if total_loss is not None:
                line += f" │ 📉 Loss: {total_loss:.4f}"
                # Add component breakdown if available
                components = []
                if eta_loss is not None:
                    components.append(f"η={eta_loss:.4f}")
                if dir_loss is not None:
                    components.append(f"dir={dir_loss:.4f}")
                if components:
                    line += f" ({', '.join(components)})"

        # Always allow terminal logging for debugging
        sys.stdout.write(f"\r\033[2K{line}")
        sys.stdout.flush()

    def print_validation_start(self, epoch: int):
        """Print validation phase start."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return

        self._clear_line()
        phase_name, color, icon = TrainingPhase.VALID.value

        message = f"\n{self._colorize('─' * 80, Colors.BRIGHT_CYAN)}"
        message += f"\n{self._colorize(f'{icon} BẮT ĐẦU VALIDATION - Epoch {epoch}', color + Colors.BOLD)}"
        message += f"\n{self._colorize('  Đánh giá mô hình với deterministic actions (không có noise)', Colors.DIM)}"
        message += f"\n{self._colorize('  Gradient updates: TẮT', Colors.DIM)}"
        message += f"\n{self._colorize('─' * 80, Colors.BRIGHT_CYAN)}\n"

        smart_print(message, flush=True)
        self._write_to_file(message)

    def print_validation_complete(
        self, epoch: int, metrics: Dict[str, Any], is_best: bool = False
    ):
        """Print validation completion with metrics."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return

        self._clear_line()
        phase_name, color, icon = TrainingPhase.VALID.value

        # Format key metrics
        sharpe = metrics.get("sharpeRatio", "N/A")
        mdd = metrics.get("mdd", "N/A")
        annual_return = metrics.get("annualReturn_pct", "N/A")

        message = f"\n{self._colorize('┌' + '─' * 78 + '┐', color)}"
        message += f"\n{self._colorize('│', color)} {icon} VALIDATION HOÀN THÀNH - Epoch {epoch}"

        if is_best:
            message += (
                f" {self._colorize('★ BEST ★', Colors.BRIGHT_YELLOW + Colors.BOLD)}"
            )

        message += f"\n{self._colorize('├' + '─' * 78 + '┤', color)}"

        # Metrics
        if isinstance(sharpe, (int, float)):
            sharpe_color = Colors.BRIGHT_GREEN if sharpe > 0 else Colors.BRIGHT_RED
            message += f"\n{self._colorize('│', color)}   Sharpe Ratio: {self._colorize(f'{sharpe:.4f}', sharpe_color)}"
        if isinstance(mdd, (int, float)):
            message += f"\n{self._colorize('│', color)}   Max Drawdown: {self._colorize(f'{mdd:.2%}', Colors.BRIGHT_RED)}"
        if isinstance(annual_return, (int, float)):
            ar_color = Colors.BRIGHT_GREEN if annual_return > 0 else Colors.BRIGHT_RED
            message += f"\n{self._colorize('│', color)}   Annual Return: {self._colorize(f'{annual_return:.2f}%', ar_color)}"

        message += f"\n{self._colorize('└' + '─' * 78 + '┘', color)}\n"

        smart_print(message, flush=True)
        self._write_to_file(message)

    def print_test_start(self):
        """Print test phase start."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return

        self._clear_line()
        phase_name, color, icon = TrainingPhase.TEST.value

        message = f"\n{self._colorize('═' * 80, color)}"
        message += f"\n{self._colorize(f'{icon} BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP TEST', color + Colors.BOLD)}"
        message += f"\n{self._colorize('  Chế độ: Inference only (không cập nhật weights)', Colors.DIM)}"
        message += f"\n{self._colorize('═' * 80, color)}\n"

        smart_print(message, flush=True)
        self._write_to_file(message)

    def print_test_complete(self, metrics: Dict[str, Any]):
        """Print test completion with final metrics."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return

        self._clear_line()
        phase_name, color, icon = TrainingPhase.TEST.value

        message = f"\n{self._colorize('╔' + '═' * 78 + '╗', color)}"
        message += (
            f"\n{self._colorize('║', color)} {icon} KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST"
        )
        message += f"\n{self._colorize('╠' + '═' * 78 + '╣', color)}"

        # Key metrics with formatting
        for key, value in metrics.items():
            if isinstance(value, float):
                if "pct" in key.lower() or "return" in key.lower():
                    val_color = Colors.BRIGHT_GREEN if value > 0 else Colors.BRIGHT_RED
                    val_str = f"{value:.2f}%"
                elif "sharpe" in key.lower():
                    val_color = Colors.BRIGHT_GREEN if value > 0 else Colors.BRIGHT_RED
                    val_str = f"{value:.4f}"
                elif "mdd" in key.lower():
                    val_color = Colors.BRIGHT_RED
                    val_str = f"{value:.2%}"
                elif "capital" in key.lower():
                    val_color = Colors.BRIGHT_CYAN
                    val_str = f"${value:,.2f}"
                else:
                    val_color = Colors.WHITE
                    val_str = f"{value:.4f}"
                message += f"\n{self._colorize('║', color)}   {key}: {self._colorize(val_str, val_color)}"
            else:
                message += f"\n{self._colorize('║', color)}   {key}: {value}"

        message += f"\n{self._colorize('╚' + '═' * 78 + '╝', color)}\n"

        smart_print(message, flush=True)
        self._write_to_file(message)

    def print_checkpoint_saved(
        self, checkpoint_name: str, checkpoint_type: str, epoch: int
    ):
        """Print checkpoint save notification."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return

        self._clear_line()
        phase_name, color, icon = TrainingPhase.CHECKPOINT.value

        type_color = (
            Colors.BRIGHT_GREEN
            if "best" in checkpoint_type.lower()
            else Colors.BRIGHT_YELLOW
        )

        message = (
            f"\n{self._colorize(f'{icon} CHECKPOINT ĐÃ LƯU', color + Colors.BOLD)}: "
        )
        message += f"{self._colorize(checkpoint_name, type_color)} "
        message += f"(Epoch {epoch}, Type: {checkpoint_type})\n"

        smart_print(message, flush=True)
        self._write_to_file(message)

    def print_training_complete(self, total_time: float, final_metrics: Dict[str, Any]):
        """Print training completion summary."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return

        self._clear_line()
        phase_name, color, icon = TrainingPhase.COMPLETE.value

        message = f"""
{self._colorize("╔" + "═" * 98 + "╗", Colors.BRIGHT_GREEN)}
{self._colorize("║", Colors.BRIGHT_GREEN)}{"TRAINING HOÀN THÀNH":^98}{self._colorize("║", Colors.BRIGHT_GREEN)}
{self._colorize("╠" + "═" * 98 + "╣", Colors.BRIGHT_GREEN)}
{self._colorize("║", Colors.BRIGHT_GREEN)}   ⏱️  Tổng thời gian: {self._format_time(total_time):>15}
{self._colorize("║", Colors.BRIGHT_GREEN)}   🔢 Tổng số bước: {self.current_step:>18,}
{self._colorize("║", Colors.BRIGHT_GREEN)}   📊 Số epochs: {self.current_epoch:>21}
{self._colorize("╠" + "═" * 98 + "╣", Colors.BRIGHT_GREEN)}
{self._colorize("║", Colors.BRIGHT_GREEN)} 📈 KẾT QUẢ CUỐI CÙNG:
"""
        for key, value in final_metrics.items():
            if isinstance(value, float):
                message += f"{self._colorize('║', Colors.BRIGHT_GREEN)}      {key}: {value:.4f}\n"
            else:
                message += (
                    f"{self._colorize('║', Colors.BRIGHT_GREEN)}      {key}: {value}\n"
                )

        message += f"{self._colorize('╚' + '═' * 98 + '╝', Colors.BRIGHT_GREEN)}\n"

        smart_print(message, flush=True)
        self._write_to_file(message)

    def print_error(self, message: str, exception: Optional[Exception] = None):
        """Print error message."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return

        self._clear_line()
        phase_name, color, icon = TrainingPhase.ERROR.value

        error_msg = f"\n{self._colorize(f'{icon} LỖI: {message}', Colors.BRIGHT_RED + Colors.BOLD)}"
        if exception:
            error_msg += (
                f"\n{self._colorize(f'   Chi tiết: {str(exception)}', Colors.RED)}"
            )
        error_msg += "\n"

        smart_print(error_msg, flush=True)
        self._write_to_file(error_msg)

    def newline(self):
        """Print a newline to break from single-line updates."""
        # Skip if display is suppressed (simple_logger mode)
        if _is_display_suppressed():
            return
        smart_print("", flush=True)


# Global logger instance
_logger: Optional[TrainingLogger] = None


def get_logger() -> TrainingLogger:
    """Get or create the global training logger."""
    global _logger
    if _logger is None:
        _logger = TrainingLogger()
    return _logger


def reset_logger():
    """Reset the global logger."""
    global _logger
    _logger = None
