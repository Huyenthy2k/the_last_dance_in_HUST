"""
Live Display Manager for MAFIA Training
========================================
Real-time terminal display with FIXED layout (no scrolling spam).
Only values update - structure stays constant.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
import threading

# Terminal control for disabling keyboard echo
try:
    import termios
    import tty

    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False


# ANSI escape codes
class ANSI:
    """ANSI escape codes for terminal control."""

    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    MOVE_UP = "\033[{}A"
    MOVE_DOWN = "\033[{}B"  # Move cursor down N lines
    MOVE_TO_START = "\033[H"  # Move cursor to top-left (1,1)
    MOVE_TO_ROW = "\033[{};1H"  # Move cursor to row N, column 1
    CLEAR_LINE = "\033[2K"
    CLEAR_SCREEN = "\033[2J"
    # Alternate screen buffer (like vim, less, etc.)
    ENTER_ALT_SCREEN = "\033[?1049h"
    EXIT_ALT_SCREEN = "\033[?1049l"

    # Colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"


class _PickleSafeLogRedirect:
    """Pickle-safe stdout/stderr redirect that logs to file and suppresses terminal output.

    This class is pickle-safe because it only stores the log file path (a string),
    not an open file handle. The file is opened temporarily for each write operation.
    """

    def __init__(self, log_file_path: Optional[str], stream_name: str = "stdout"):
        self._log_file_path = log_file_path  # Just the path, not the file handle
        self._stream_name = stream_name
        self._closed = False

    def write(self, text):
        """Write to log file only (suppress terminal output).

        IMPORTANT: We only filter truly empty strings, NOT whitespace.
        ANSI escape codes (like cursor movement) contain only 'whitespace' chars
        and would be incorrectly filtered by .strip() check.
        """
        if self._closed:
            return
        if not text:  # Only filter empty strings, keep ANSI codes
            return
        if self._log_file_path:
            try:
                # Open, write, close - pickle safe
                with open(self._log_file_path, "a", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        pass  # No-op, we flush on each write

    def close(self):
        self._closed = True

    def fileno(self):
        # Raise error - we don't have a persistent file descriptor
        raise OSError("_PickleSafeLogRedirect does not support fileno()")

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"

    @property
    def errors(self):
        return "replace"

    def __getstate__(self):
        """Return pickle-safe state."""
        return {
            "_log_file_path": self._log_file_path,
            "_stream_name": self._stream_name,
            "_closed": self._closed,
        }

    def __setstate__(self, state):
        """Restore from pickled state."""
        self._log_file_path = state.get("_log_file_path")
        self._stream_name = state.get("_stream_name", "stdout")
        self._closed = state.get("_closed", False)


@dataclass
class DisplayState:
    """Current state of all display metrics."""

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING INFO
    # ═══════════════════════════════════════════════════════════════════════════
    epoch: int = 0
    total_epochs: int = 50
    phase: str = "RL_TRAIN"  # OBSERVER_PRETRAIN, RL_TRAIN, WARMUP, VALID, TEST
    stage: str = "INIT"  # Current detailed stage (e.g., "Buffer Fill", "Training")
    window: int = 0
    seed: int = 2025

    # Step info
    step: int = 0
    total_steps: int = 252
    epoch_day: int = 0  # Day within current epoch (1-based)
    epoch_total_days: int = 0  # Total days in current epoch
    date: str = "2024-01-01"

    # ═══════════════════════════════════════════════════════════════════════════
    # 2-PHASE TRAINING MODE - No COMBINED mode!
    # Phase 1: Observer training (TD3 frozen, uniform weights)
    # Phase 2: TD3 training (Observer frozen, Static Expert)
    # ═══════════════════════════════════════════════════════════════════════════
    training_mode: str = "RL_ONLY"  # "OBSERVER_ONLY" | "RL_ONLY" (NO COMBINED!)
    observer_frozen: bool = True  # True when Phase 2 (RL training)
    td3_frozen: bool = False  # True when Phase 1 (Observer training)
    observer_checkpoint_source: str = ""  # Path to loaded checkpoint (when frozen)

    # Walk-Forward Iteration Info (Phase 1)
    walkforward_iteration: int = 0  # Current iteration (0-indexed)
    walkforward_total_iterations: int = 5
    walkforward_train_years: str = ""  # e.g., "2015→2017" (legacy, year only)
    walkforward_valid_year: int = 0  # e.g., 2017 (legacy, year only)
    walkforward_infer_year: int = 0  # e.g., 2018 (legacy, year only)
    # Full date ranges for Expanding Window display
    walkforward_train_range: str = ""  # e.g., "01/2015 → 06/2017"
    walkforward_valid_range: str = ""  # e.g., "07/2017 → 12/2017"
    walkforward_infer_range: str = ""  # e.g., "01/2018 → 12/2018"
    walkforward_is_finetune: bool = False  # True if finetuning from previous checkpoint
    walkforward_best_score: float = 0.0  # Best composite score so far
    walkforward_best_epoch: int = 0  # Epoch with best score
    walkforward_current_score: float = 0.0  # Current validation score

    # ═══════════════════════════════════════════════════════════════════════════
    # MARKET & REGIME
    # ═══════════════════════════════════════════════════════════════════════════
    direction: int = 1  # 0=Bear, 1=Flat, 2=Bull
    trend_z: float = 0.0
    eta_risk: float = 1.0
    volatility: float = 0.01
    regime_shift_count: int = 0
    last_regime_date: str = ""
    last_regime_from: str = ""  # Previous regime (BEAR/FLAT/BULL)
    last_regime_to: str = ""  # New regime
    last_regime_reason: str = ""

    # ═══════════════════════════════════════════════════════════════════════════
    # PORTFOLIO & SELECTION
    # ═══════════════════════════════════════════════════════════════════════════
    is_rebalance: bool = False
    days_since_rebal: int = 0
    rebal_interval: int = 10
    top_k: int = 10
    capital: float = 1_000_000
    daily_return: float = 0.0
    cumul_return: float = 0.0

    # Selection stats
    n_kept: int = 10
    n_added: int = 0
    n_removed: int = 0
    rebalance_count: int = 0

    # Selection details (for event display)
    last_selection_date: str = ""
    last_selection_trigger: str = ""
    last_kept_tickers: str = ""  # e.g. "AAPL,MSFT,GOOGL..."
    last_added_tickers: str = ""  # e.g. "+NVDA,+AMD"
    last_removed_tickers: str = ""  # e.g. "-TSLA,-META"

    # ═══════════════════════════════════════════════════════════════════════════
    # RETURNS & PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════════
    sharpe: float = 0.0
    mdd: float = 0.0
    win_rate: float = 0.5
    epoch_return: float = 0.0
    net_profit: float = 0.0  # Net profit in currency
    vol_max: float = 0.0  # Maximum volatility
    annual_return_pct: float = 0.0  # Annualized return percentage

    # ═══════════════════════════════════════════════════════════════════════════
    # OBSERVER (MAFIA) - 3 Tasks
    # ═══════════════════════════════════════════════════════════════════════════
    # Task 1: Direction Prediction
    obs_dir_pred: str = "FLAT"  # Predicted direction
    obs_dir_conf: float = 0.0  # Confidence [0,1]
    obs_loss_dir: float = 0.0  # Direction loss (CE)
    obs_samples_dir: int = 0  # Training samples

    # Task 2: Risk (Eta) Prediction
    obs_eta_pred: float = 1.0  # Predicted risk multiplier
    obs_loss_eta: float = 0.0  # Eta loss (MSE)
    obs_samples_risk: int = 0  # Training samples

    # Task 3: Stock Selection (Policy Gradient)
    # PG Reward: R_t = mean_return - α_turnover×turnover - α_change×symdiff
    obs_loss_pg: float = 0.0  # PG loss
    obs_r_sel_return: float = 0.0  # mean_return component
    obs_r_sel_turnover: float = 0.0  # α_turnover × turnover penalty
    obs_r_sel_change: float = 0.0  # α_change × symdiff penalty
    obs_r_sel_total: float = 0.0  # Total shaped return (R_t)
    obs_r_sel_div: float = 0.0  # Legacy: diversity (kept for compat)
    obs_samples_sel: int = 0  # Training samples
    rebalance_ratio: float = 10.0  # Rebalance trigger threshold %

    # Observer totals
    obs_loss_total: float = 0.0  # L_total = L_dir + L_eta + L_pg

    # Observer Warmup Phase
    obs_warmup_samples: int = 0  # Current buffer size
    obs_warmup_target: int = 300  # Target warmup samples (default 300)
    obs_warmup_complete: bool = False  # True when warmup is done

    # ═══════════════════════════════════════════════════════════════════════════
    # TD3 AGENT (Portfolio Allocator)
    # ═══════════════════════════════════════════════════════════════════════════
    td3_actor_loss: float = 0.0
    td3_critic_loss: float = 0.0
    td3_buffer_size: int = 0
    td3_buffer_max: int = 156000
    td3_learning_starts: int = 1000  # Buffer threshold before training starts
    td3_updates: int = 0
    td3_mean_q: float = 0.0
    td3_lr: float = 0.0001
    td3_noise_sigma: float = 0.15

    # TD3 Reward Components
    # r_total = (r_return + r_js) * reward_scale
    td3_r_return: float = 0.0  # w_return * log(1 + portfolio_return)
    td3_r_js: float = 0.0  # -lambda_js * JS_divergence
    td3_r_unscaled: float = 0.0  # r_return + r_js
    td3_reward: float = 0.0  # Final scaled reward
    td3_reward_sum: float = 0.0  # Sum of rewards in episode
    td3_reward_mean: float = 0.0  # Mean reward
    td3_w_return: float = 1.0  # Weight for return
    td3_lambda_js: float = 0.1  # Weight for JS penalty
    td3_reward_scale: float = 100.0  # Scaling factor
    td3_js_divergence: float = 0.0  # Raw JS divergence

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTROLLER CBF (Control Barrier Function)
    # ═══════════════════════════════════════════════════════════════════════════
    cbf_enabled: bool = True
    cbf_alpha: float = 0.1  # CBF relaxation parameter
    cbf_interventions: int = 0  # Number of safety interventions
    cbf_last_intervention: str = ""  # Last intervention reason
    cbf_safety_margin: float = 0.0  # Current safety margin
    cbf_constraint_active: bool = False  # Is constraint currently active

    # CBF detailed parameters
    cbf_sigma_base: float = 0.15  # Base noise sigma
    cbf_sigma_current: float = 0.15  # Current adjusted sigma
    cbf_min_variance: float = 0.01  # Minimum variance threshold
    cbf_max_position: float = 0.25  # Max position size constraint
    cbf_adjust_direction: str = ""  # "↑"=increase, "↓"=decrease, ""=no change
    cbf_adjust_magnitude: float = 0.0  # How much adjustment was applied

    # CBF Action adjustment "from → to" values
    cbf_action_before: str = ""  # e.g., "[0.10, 0.15, 0.20, ...]"
    cbf_action_after: str = ""  # e.g., "[0.08, 0.12, 0.18, ...]"
    cbf_action_norm_before: float = 0.0  # L2 norm before
    cbf_action_norm_after: float = 0.0  # L2 norm after

    # Training step counters
    total_train_steps: int = 0  # Total steps across all epochs
    current_epoch_steps: int = 0  # Steps in current epoch

    # ═══════════════════════════════════════════════════════════════════════════
    # TIMING
    # ═══════════════════════════════════════════════════════════════════════════
    speed: float = 0.0
    elapsed_seconds: float = 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT LOG
    # ═══════════════════════════════════════════════════════════════════════════
    event_log: List[str] = field(default_factory=list)
    max_events: int = 20  # Show last 20 events


class LiveDisplay:
    """
    Real-time terminal display with FIXED layout.
    Only values update - structure never changes.
    """

    # FIXED number of display lines - NEVER changes
    FIXED_LINES = 64  # Header(3) + Flow(3) + Progress(5) + Market(7) + Observer(9) + TD3(12) + CBF(9) + Returns(7) + Events(7) + Padding(2)
    FIXED_WIDTH = 110  # Wider for more detail

    def __init__(self, enabled: bool = True, log_file: Optional[str] = None):
        self._tty = None
        self._tty_available = False
        self._original_stdout = None
        self._original_stderr = None
        self._original_term_settings = None  # Store original terminal settings
        # Store the REAL stdout at init time, BEFORE any redirection
        # This is critical for fallback writing if TTY fails
        self._fallback_stdout = (
            sys.__stdout__
        )  # Use sys.__stdout__ which is never redirected

        if enabled:
            try:
                self._tty = open("/dev/tty", "w")
                self._tty_available = True
            except (OSError, IOError):
                self._tty_available = False

        self.enabled = enabled and self._tty_available
        self.log_file = log_file
        self.state = DisplayState()
        self._lock = threading.Lock()
        self._last_render_time = 0
        self._min_render_interval = 0.1
        self._initialized = False

    def _write(self, text: str):
        """Write to the display output (always to TTY, never to redirected stdout).

        CRITICAL: This method must ONLY write to:
        1. /dev/tty (preferred) - direct terminal access
        2. sys.__stdout__ (fallback) - the ORIGINAL stdout, never redirected

        NEVER use sys.stdout here as it may be redirected to a log file!
        """
        if self._tty_available and self._tty:
            try:
                self._tty.write(text)
                self._tty.flush()
                return
            except (OSError, IOError, ValueError):
                # TTY closed or unavailable, try fallback
                pass

        # Fallback: Use the original stdout (sys.__stdout__), NOT sys.stdout
        # sys.stdout may be redirected to _PickleSafeLogRedirect
        if self._fallback_stdout and hasattr(self._fallback_stdout, "write"):
            try:
                self._fallback_stdout.write(text)
                self._fallback_stdout.flush()
            except (OSError, IOError, ValueError):
                # Even fallback failed, silently ignore
                pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("_tty", None)
        state.pop("_fallback_stdout", None)  # Can't pickle stdout
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._tty = None
        self._fallback_stdout = sys.__stdout__  # Restore fallback
        try:
            self._tty = open("/dev/tty", "w")
            self._tty_available = True
        except (OSError, IOError):
            self._tty_available = False

    def initialize(self):
        """Initialize the display using alternate screen buffer."""
        if not self.enabled:
            return

        with self._lock:
            if self._initialized:
                return

            # Disable keyboard echo to prevent arrow key escape codes showing
            if TERMIOS_AVAILABLE:
                try:
                    stdin_fd = sys.stdin.fileno()
                    self._original_term_settings = termios.tcgetattr(stdin_fd)
                    # Set terminal to cbreak mode (disable echo, but allow signals)
                    new_settings = termios.tcgetattr(stdin_fd)
                    new_settings[3] = new_settings[3] & ~termios.ECHO  # Disable echo
                    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, new_settings)
                except (termios.error, OSError, ValueError):
                    # Failed to set terminal mode, continue anyway
                    self._original_term_settings = None

            # Enter alternate screen buffer (preserves terminal scrollback)
            self._write(ANSI.ENTER_ALT_SCREEN)
            self._write(ANSI.CLEAR_SCREEN)
            self._write(ANSI.HIDE_CURSOR)
            self._write(ANSI.MOVE_TO_START)
            self._initialized = True

    def cleanup(self):
        """Cleanup display and restore terminal state.

        Exit alternate screen buffer to restore original terminal content.
        """
        if not self.enabled:
            return

        with self._lock:
            # Show cursor before exiting alternate screen
            self._write(ANSI.SHOW_CURSOR)
            # Exit alternate screen buffer (restores original terminal content)
            self._write(ANSI.EXIT_ALT_SCREEN)
            self._initialized = False

            # Restore original terminal settings (re-enable echo)
            if TERMIOS_AVAILABLE and self._original_term_settings is not None:
                try:
                    stdin_fd = sys.stdin.fileno()
                    termios.tcsetattr(
                        stdin_fd, termios.TCSADRAIN, self._original_term_settings
                    )
                except (termios.error, OSError, ValueError):
                    pass
                self._original_term_settings = None

            # Close TTY handle
            if self._tty:
                try:
                    self._tty.close()
                except:
                    pass
                self._tty = None

    def update(self, **kwargs):
        """Update display state."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)

    def add_event(self, event_type: str, message: str):
        """Add an event to the event log."""
        with self._lock:
            timestamp = time.strftime("%H:%M:%S")
            # Color code by event type
            if event_type == "REBALANCE":
                color = ANSI.CYAN
                icon = "🔄"
            elif event_type == "REGIME":
                color = ANSI.YELLOW
                icon = "⚡"
            elif event_type == "EPOCH":
                color = ANSI.GREEN
                icon = "📊"
            else:
                color = ANSI.WHITE
                icon = "📝"

            entry = f"{color}{icon} [{timestamp}] {message}{ANSI.RESET}"
            self.state.event_log.append(entry)
            # Keep only last N events
            if len(self.state.event_log) > self.state.max_events:
                self.state.event_log = self.state.event_log[-self.state.max_events :]

    def capture_stdout(self):
        """Capture stdout/stderr to prevent breaking the display.

        Uses a pickle-safe approach: stores log file path instead of file handle.
        Display output goes directly to /dev/tty, bypassing stdout.
        """
        if not self.enabled:
            return

        # Store for later restoration
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # Create pickle-safe redirect that opens file on each write
        sys.stdout = _PickleSafeLogRedirect(self.log_file, "stdout")
        sys.stderr = _PickleSafeLogRedirect(self.log_file, "stderr")

    def release_stdout(self):
        """Release captured stdout/stderr."""
        if self._original_stdout:
            sys.stdout = self._original_stdout
        if self._original_stderr:
            sys.stderr = self._original_stderr

    def render(self, force: bool = False):
        """Render the FIXED layout display.

        CRITICAL: This method ensures the display stays FIXED by:
        1. Always writing to /dev/tty directly (bypassing stdout)
        2. Using ANSI escape codes to overwrite previous content
        3. Moving cursor back to start position after rendering
        """
        if not self.enabled:
            return

        now = time.time()
        if not force and (now - self._last_render_time) < self._min_render_interval:
            return
        self._last_render_time = now

        with self._lock:
            if not self._initialized:
                self.initialize()

            lines = self._build_fixed_display()

            # Ensure we have EXACTLY FIXED_LINES lines
            while len(lines) < self.FIXED_LINES:
                lines.append("")
            lines = lines[: self.FIXED_LINES]

            # Move cursor to top-left of alternate screen (absolute positioning)
            self._write(ANSI.MOVE_TO_START)

            # Write exactly FIXED_LINES lines
            for line in lines:
                self._write(ANSI.CLEAR_LINE + line + "\n")

            # No need to move cursor back - alternate screen is isolated

    def _build_fixed_display(self) -> List[str]:
        """Build FIXED layout - exactly FIXED_LINES lines with comprehensive training info."""
        s = self.state
        W = self.FIXED_WIDTH

        # Helper functions for FIXED width formatting
        def hline(char="─"):
            return char * (W - 2)

        def dhline():
            return "═" * (W - 2)

        def box(content: str) -> str:
            """Create a fixed-width box line."""
            visible = self._strip_ansi(content)
            padding = W - 4 - len(visible)
            return f"│ {content}{' ' * max(0, padding)} │"

        def dbox(content: str) -> str:
            """Create a double-border box line."""
            visible = self._strip_ansi(content)
            padding = W - 4 - len(visible)
            return f"║ {content}{' ' * max(0, padding)} ║"

        def center(text: str, width: int) -> str:
            """Center text with ANSI support."""
            visible = self._strip_ansi(text)
            pad = (width - len(visible)) // 2
            return " " * pad + text + " " * (width - pad - len(visible))

        def fmt_f(val: float, width: int = 8, precision: int = 4) -> str:
            """Format float to fixed width."""
            txt = f"{val:+.{precision}f}" if val != 0 else f"{val:.{precision}f}"
            return txt[:width].rjust(width)

        def fmt_pct(val: float, width: int = 7) -> str:
            """Format percentage to fixed width."""
            txt = f"{val:+.2f}%" if val != 0 else f"{val:.2f}%"
            return txt[:width].rjust(width)

        def fmt_int(val: int, width: int = 6) -> str:
            """Format int to fixed width."""
            return f"{val:,}"[:width].rjust(width)

        def color_val(val: float, fmt_str: str) -> str:
            """Color value based on sign."""
            color = ANSI.GREEN if val > 0 else (ANSI.RED if val < 0 else ANSI.WHITE)
            return f"{color}{fmt_str}{ANSI.RESET}"

        def progress_bar(current: int, total: int, width: int = 20) -> str:
            """Create a progress bar."""
            pct = min(100, current / max(1, total) * 100)
            filled = int(width * pct / 100)
            return f"[{'█' * filled}{'░' * (width - filled)}] {pct:5.1f}%"

        def mini_bar(current: float, max_val: float, width: int = 10) -> str:
            """Create a mini progress bar for metrics."""
            pct = min(1.0, current / max(0.001, max_val))
            filled = int(width * pct)
            return f"{'▓' * filled}{'░' * (width - filled)}"

        # Direction emoji/text
        dir_map = {
            0: ("🐻", "BEAR", ANSI.RED),
            1: ("➡️", "FLAT", ANSI.YELLOW),
            2: ("🐂", "BULL", ANSI.GREEN),
        }
        dir_emoji, dir_text, dir_color = dir_map.get(
            s.direction, ("➡️", "FLAT", ANSI.YELLOW)
        )

        # Observer direction prediction
        obs_dir_map = {"BEAR": ANSI.RED, "FLAT": ANSI.YELLOW, "BULL": ANSI.GREEN}
        obs_dir_color = obs_dir_map.get(s.obs_dir_pred, ANSI.WHITE)

        # Phase indicators with status (2-Phase Training - NO COMBINED!)
        phase_icons = {
            # Phase 1: Observer Walk-Forward Training
            "INIT": ("⚙️", ANSI.WHITE, "Initialization"),
            "OBSERVER_PRETRAIN": ("🔮", ANSI.MAGENTA, "Observer Pre-training"),
            # Phase 2: TD3 Training with Frozen Observer
            "WARMUP": ("🔥", ANSI.YELLOW, "Buffer Fill"),
            "RL_TRAIN": ("🎯", ANSI.GREEN, "TD3 Training"),
            # Shared phases
            "VALID": ("📊", ANSI.CYAN, "Validation"),
            "TEST": ("🧪", ANSI.BLUE, "Testing"),
            "COMPLETE": ("✅", ANSI.GREEN, "Complete"),
        }
        phase_icon, phase_color, phase_desc = phase_icons.get(
            s.phase, ("▶️", ANSI.WHITE, s.phase)
        )

        # 2-Phase Training: Determine status based on training_mode (NO COMBINED!)
        if s.training_mode == "OBSERVER_ONLY":
            # Phase 1: Observer training, TD3 frozen
            obs_status = f"{ANSI.GREEN}● TRAINING{ANSI.RESET}"
            td3_status = f"{ANSI.CYAN}❄️ FROZEN{ANSI.RESET}"
            phase_title = "🔮 PHASE 1: OBSERVER WALK-FORWARD TRAINING"
            show_td3_losses = False
            show_observer_losses = True
        else:  # RL_ONLY
            # Phase 2: Observer frozen (Static Expert), TD3 training
            obs_status = f"{ANSI.CYAN}❄️ FROZEN (Static Expert){ANSI.RESET}"
            td3_status = f"{ANSI.GREEN}● TRAINING{ANSI.RESET}"
            phase_title = "🎯 PHASE 2: TD3 TRAINING"
            show_td3_losses = True
            show_observer_losses = False

        # Action text
        action_text = "🔄 REBALANCE" if s.is_rebalance else "💤 HOLD"
        action_color = ANSI.CYAN if s.is_rebalance else ANSI.DIM

        # Timing
        elapsed = self._format_time(s.elapsed_seconds)
        eta_seconds = (s.total_steps - s.step) / max(0.1, s.speed)
        eta = self._format_time(eta_seconds) if s.speed > 0 else "---"

        # Buffer status - use actual learning_starts threshold
        buf_pct = s.td3_buffer_size / max(1, s.td3_buffer_max) * 100
        buf_ready = s.td3_buffer_size >= s.td3_learning_starts

        # Training status determination based on 2-phase mode
        td3_active = (
            s.training_mode == "RL_ONLY"
            and s.phase in ["RL_TRAIN", "WARMUP"]
            and buf_ready
        )
        obs_active = (
            s.training_mode == "OBSERVER_ONLY" and s.phase == "OBSERVER_PRETRAIN"
        )

        # Build EXACTLY FIXED_LINES lines
        lines = []

        # ═══════════════════════════════════════════════════════════════════════════════════════════
        # HEADER SECTION (Double border) - Use phase_title based on training_mode
        # ═══════════════════════════════════════════════════════════════════════════════════════════
        lines.append(f"╔{dhline()}╗")
        lines.append(
            dbox(
                f"{ANSI.BOLD}{phase_icon} {phase_title}{ANSI.RESET}"
                f"                              "
                f"Seed:{s.seed} │ Window:{s.window}"
            )
        )
        lines.append(f"╠{dhline()}╣")

        # ═══════════════════════════════════════════════════════════════════════════════════════════
        # TRAINING FLOW SECTION - Different flow for each phase
        # ═══════════════════════════════════════════════════════════════════════════════════════════
        if s.training_mode == "OBSERVER_ONLY":
            # Phase 1 flow: INIT → OBS-TRAIN → VALID → INFER
            init_style = f"{ANSI.BOLD}{ANSI.WHITE}" if s.phase == "INIT" else ANSI.DIM
            obs_train_style = (
                f"{ANSI.BOLD}{ANSI.MAGENTA}"
                if s.phase == "OBSERVER_PRETRAIN"
                else ANSI.DIM
            )
            valid_style = f"{ANSI.BOLD}{ANSI.CYAN}" if s.phase == "VALID" else ANSI.DIM
            infer_style = (
                f"{ANSI.BOLD}{ANSI.GREEN}" if s.phase == "COMPLETE" else ANSI.DIM
            )

            lines.append(
                dbox(
                    f"  {ANSI.BOLD}FLOW:{ANSI.RESET} "
                    f"{init_style}[INIT]{ANSI.RESET} → "
                    f"{obs_train_style}[OBS-TRAIN]{ANSI.RESET} → "
                    f"{valid_style}[VALID]{ANSI.RESET} → "
                    f"{infer_style}[INFER]{ANSI.RESET}"
                    f"      "
                    f"Iter: {s.walkforward_iteration}/{s.walkforward_total_iterations} │ "
                    f"Epoch: {ANSI.BOLD}{s.epoch:2d}/{s.total_epochs:2d}{ANSI.RESET}"
                )
            )
            # Show walk-forward info
            wf_info = (
                f"Train: {s.walkforward_train_years}"
                if s.walkforward_train_years
                else "Train: ---"
            )
            lines.append(
                dbox(
                    f"  {ANSI.BOLD}STATUS:{ANSI.RESET} {phase_color}{phase_desc}{ANSI.RESET} │ "
                    f"{wf_info} │ Valid: {s.walkforward_valid_year or '---'}"
                )
            )
        else:
            # Phase 2 flow: WARMUP → RL-TRAIN → VALID → TEST
            warmup_style = (
                f"{ANSI.BOLD}{ANSI.YELLOW}" if s.phase == "WARMUP" else ANSI.DIM
            )
            rl_train_style = (
                f"{ANSI.BOLD}{ANSI.GREEN}" if s.phase == "RL_TRAIN" else ANSI.DIM
            )
            valid_style = f"{ANSI.BOLD}{ANSI.CYAN}" if s.phase == "VALID" else ANSI.DIM
            test_style = f"{ANSI.BOLD}{ANSI.BLUE}" if s.phase == "TEST" else ANSI.DIM

            lines.append(
                dbox(
                    f"  {ANSI.BOLD}FLOW:{ANSI.RESET} "
                    f"{warmup_style}[WARMUP]{ANSI.RESET} → "
                    f"{rl_train_style}[RL-TRAIN]{ANSI.RESET} → "
                    f"{valid_style}[VALID]{ANSI.RESET} → "
                    f"{test_style}[TEST]{ANSI.RESET}"
                    f"      "
                    f"Iter: {s.walkforward_iteration}/{s.walkforward_total_iterations} │ "
                    f"Epoch: {ANSI.BOLD}{s.epoch:2d}/{s.total_epochs:2d}{ANSI.RESET}"
                )
            )
            # Show observer checkpoint source AND walk-forward info
            obs_src = (
                s.observer_checkpoint_source[-25:]
                if len(s.observer_checkpoint_source) > 25
                else s.observer_checkpoint_source
            )
            # Use train range if available
            train_info = s.walkforward_train_range or s.walkforward_train_years or "---"
            finetune = " [FT]" if s.walkforward_is_finetune else ""
            
            lines.append(
                dbox(
                    f"  {ANSI.BOLD}STATUS:{ANSI.RESET} {phase_color}{phase_desc}{ANSI.RESET} │ "
                    f"Train: {train_info}{finetune} │ Obs: ..{obs_src or 'N/A'}❄️"
                )
            )
        lines.append(f"╠{dhline()}╣")

        # ═══════════════════════════════════════════════════════════════════════════════════════════
        # PROGRESS SECTION
        # ═══════════════════════════════════════════════════════════════════════════════════════════
        pct = min(100, s.step / max(1, s.total_steps) * 100)
        filled = int(30 * pct / 100)
        bar = "█" * filled + "░" * (30 - filled)

        # Per-epoch progress (use epoch_day/epoch_total_days when available)
        epoch_day = s.epoch_day if s.epoch_day else s.step
        epoch_total_days = s.epoch_total_days if s.epoch_total_days else s.total_steps
        epoch_pct = min(100, epoch_day / max(1, epoch_total_days) * 100)
        epoch_bar = "█" * int(30 * epoch_pct / 100) + "░" * (
            30 - int(30 * epoch_pct / 100)
        )

        lines.append(dbox(f"  {ANSI.BOLD}📅 PROGRESS{ANSI.RESET}"))
        lines.append(
            dbox(
                f"  Epoch Day: {ANSI.BOLD}{epoch_day:4d}{ANSI.RESET}/{epoch_total_days:4d}  │  "
                f"Date: {ANSI.BOLD}{s.date:10s}{ANSI.RESET}  │  "
                f"[{epoch_bar}] {epoch_pct:5.1f}%"
            )
        )
        # Total training steps info
        total_expected = s.total_epochs * (
            s.epoch_total_days if s.epoch_total_days else s.total_steps
        )
        total_pct = min(100, s.total_train_steps / max(1, total_expected) * 100)
        lines.append(
            dbox(
                f"  Total Steps: {ANSI.BOLD}{s.total_train_steps:7d}{ANSI.RESET}/{total_expected:7d} ({total_pct:5.1f}%)  │  "
                f"Speed: {s.speed:5.1f} st/s  │  Elapsed: {elapsed:>7s}  │  ETA: {eta:>7s}"
            )
        )
        lines.append(f"╚{dhline()}╝")

        # ═══════════════════════════════════════════════════════════════════════════════════════════
        # MARKET & PORTFOLIO SECTION (Combined using box() for consistent width)
        # ═══════════════════════════════════════════════════════════════════════════════════════════
        lines.append(f"┌{hline()}┐")
        lines.append(
            box(
                f"{ANSI.BOLD}📊 MARKET{ANSI.RESET}                                              │ {ANSI.BOLD}💼 PORTFOLIO{ANSI.RESET}"
            )
        )
        lines.append(f"├{hline()}┤")
        lines.append(
            box(
                f"  Dir: {dir_color}{dir_emoji}{dir_text:4s}{ANSI.RESET}  Z:{fmt_f(s.trend_z, 6, 2)}  "
                f"η:{fmt_f(s.eta_risk, 5, 2)}  Vol:{fmt_pct(s.volatility * 100, 6)}  │  "
                f"{action_color}{action_text:12s}{ANSI.RESET}  Rebals:{s.rebalance_count:3d}"
            )
        )
        # Regime shift info - format based on trigger type
        if s.last_regime_reason:
            # Determine trigger type from reason string
            if s.last_regime_reason.startswith("direction_reversal"):
                # Direction change: Show from→to (only for actual direction changes)
                regime_info = f"{s.last_regime_from}→{s.last_regime_to}"
            elif s.last_regime_reason.startswith("vol_shock"):
                # Vol shock: Show current direction with indicator
                regime_info = f"⚡VOL[{s.last_regime_to}]"
            elif s.last_regime_reason.startswith("dc_trigger"):
                # DC reversal: Show current direction with indicator
                regime_info = f"⚡DC[{s.last_regime_to}]"
            else:
                # Unknown/legacy: Show from→to
                regime_info = f"{s.last_regime_from}→{s.last_regime_to}"
        else:
            regime_info = "(none)"
        lines.append(
            box(
                f"  Regime Shifts: {s.regime_shift_count:3d}  Last: {regime_info:12s}            │  "
                f"Next Rebal: {s.days_since_rebal:2d}/{s.rebal_interval:2d} days"
            )
        )
        # Portfolio selection details - clearer format
        add_color = ANSI.GREEN if s.n_added > 0 else ANSI.DIM
        rem_color = ANSI.RED if s.n_removed > 0 else ANSI.DIM
        lines.append(
            box(
                f"  Top-K: {s.top_k:2d} stocks                                      │  "
                f"{add_color}Added:{s.n_added:2d}{ANSI.RESET} {rem_color}Removed:{s.n_removed:2d}{ANSI.RESET} Kept:{s.n_kept:2d}"
            )
        )
        lines.append(f"└{hline()}┘")

        # ═══════════════════════════════════════════════════════════════════════════════════════════
        # OBSERVER (MAFIA) SECTION - FIXED 9 LINES regardless of training_mode
        # ═══════════════════════════════════════════════════════════════════════════════════════════
        # obs_status is already set above based on training_mode
        obs_title = (
            "🧠 OBSERVER (MAFIA)"
            if s.training_mode == "OBSERVER_ONLY"
            else "🧠 OBSERVER (Static Expert)"
        )
        lines.append(f"┌{hline()}┐")  # Line 1
        lines.append(
            box(
                f"{ANSI.BOLD}{obs_title}{ANSI.RESET}                                                    {obs_status}"
            )
        )  # Line 2
        lines.append(f"├{hline()}┤")  # Line 3

        if s.training_mode == "RL_ONLY":
            # Phase 2: Observer is FROZEN - show only predictions, no losses (pad to 5 content lines)
            lines.append(
                box(
                    f"  {ANSI.BOLD}Source:{ANSI.RESET}       {s.observer_checkpoint_source[-50:] if s.observer_checkpoint_source else 'N/A'}"
                )
            )  # Line 4
            conf_bar = mini_bar(s.obs_dir_conf, 1.0, 10)
            lines.append(
                box(
                    f"  {ANSI.BOLD}Predictions:{ANSI.RESET}  "
                    f"Direction: {obs_dir_color}{s.obs_dir_pred:4s}{ANSI.RESET} ({s.obs_dir_conf * 100:4.1f}%)  │  "
                    f"η (Risk): {fmt_f(s.obs_eta_pred, 5, 2)}  │  "
                    f"Top-K refreshed: {s.rebalance_count} times"
                )
            )  # Line 5
            lines.append(
                box(
                    f"  {ANSI.DIM}Note: Parameters FROZEN - acting as Static Expert, no gradient updates{ANSI.RESET}"
                )
            )  # Line 6
            lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 7 (padding)
            lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 8 (padding)
        else:
            # Phase 1: Observer is TRAINING - show full details (5 content lines)
            # Check if still in warmup phase
            if not s.obs_warmup_complete and s.obs_warmup_samples < s.obs_warmup_target:
                # ━━━━━ WARMUP PHASE: Collecting samples before training ━━━━━
                warmup_pct = (
                    (s.obs_warmup_samples / s.obs_warmup_target) * 100
                    if s.obs_warmup_target > 0
                    else 0
                )
                warmup_bar = mini_bar(s.obs_warmup_samples, s.obs_warmup_target, 30)
                remaining = s.obs_warmup_target - s.obs_warmup_samples

                lines.append(
                    box(
                        f"  {ANSI.BOLD}{ANSI.YELLOW}⏳ WARMUP PHASE{ANSI.RESET} - Collecting samples before training starts"
                    )
                )  # Line 4
                lines.append(
                    box(
                        f"  {ANSI.BOLD}Buffer:{ANSI.RESET}       {warmup_bar}  {s.obs_warmup_samples:4d}/{s.obs_warmup_target} ({warmup_pct:5.1f}%)"
                    )
                )  # Line 5
                lines.append(
                    box(
                        f"  {ANSI.BOLD}Remaining:{ANSI.RESET}    {remaining:4d} samples needed to start training"
                    )
                )  # Line 6
                lines.append(
                    box(
                        f"  {ANSI.DIM}Note: Observer will start gradient updates once warmup is complete{ANSI.RESET}"
                    )
                )  # Line 7
                lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 8
            else:
                # ━━━━━ TRAINING PHASE: Normal training display ━━━━━
                conf_bar = mini_bar(s.obs_dir_conf, 1.0, 10)
                lines.append(
                    box(
                        f"  {ANSI.BOLD}Predictions:{ANSI.RESET}  "
                        f"Direction: {obs_dir_color}{s.obs_dir_pred:4s}{ANSI.RESET} (conf: {conf_bar} {s.obs_dir_conf * 100:4.1f}%)  │  "
                        f"η (Risk): {fmt_f(s.obs_eta_pred, 5, 2)}"
                    )
                )  # Line 4

                # Observer losses (only show when training)
                lines.append(
                    box(
                        f"  {ANSI.BOLD}Losses:{ANSI.RESET}       "
                        f"Total: {fmt_f(s.obs_loss_total, 8, 4)}  │  "
                        f"Dir: {fmt_f(s.obs_loss_dir, 8, 4)}  │  "
                        f"Eta: {fmt_f(s.obs_loss_eta, 8, 4)}  │  "
                        f"PG: {fmt_f(s.obs_loss_pg, 8, 4)}"
                    )
                )  # Line 5

                # Observer samples
                total_samples = (
                    s.obs_samples_dir + s.obs_samples_risk + s.obs_samples_sel
                )
                lines.append(
                    box(
                        f"  {ANSI.BOLD}Samples:{ANSI.RESET}      "
                        f"Dir: {s.obs_samples_dir:6d}  │  "
                        f"Risk: {s.obs_samples_risk:6d}  │  "
                        f"Sel: {s.obs_samples_sel:6d}  │  "
                        f"Total: {total_samples:8d}"
                    )
                )  # Line 6

                # PG Reward breakdown: R_t = mean_return - α_turnover×turnover - α_change×symdiff
                pg_ret_color = ANSI.GREEN if s.obs_r_sel_return > 0 else ANSI.RED
                pg_turn_color = ANSI.RED if s.obs_r_sel_turnover > 0 else ANSI.DIM
                pg_chg_color = ANSI.RED if s.obs_r_sel_change > 0 else ANSI.DIM
                pg_total_color = ANSI.GREEN if s.obs_r_sel_total > 0 else ANSI.RED
                lines.append(
                    box(
                        f"  {ANSI.BOLD}PG Reward:{ANSI.RESET}    "
                        f"R_t: {pg_total_color}{s.obs_r_sel_total:+8.4f}{ANSI.RESET}  │  "
                        f"mean_ret: {pg_ret_color}{s.obs_r_sel_return:+7.4f}{ANSI.RESET}  │  "
                        f"turnover: {pg_turn_color}{-s.obs_r_sel_turnover:+6.4f}{ANSI.RESET}  │  "
                        f"change: {pg_chg_color}{-s.obs_r_sel_change:+6.4f}{ANSI.RESET}"
                    )
                )  # Line 7

                # Walk-forward best score
                lines.append(
                    box(
                        f"  {ANSI.BOLD}Best Score:{ANSI.RESET}   "
                        f"{s.walkforward_best_score:.4f} (Epoch {s.walkforward_best_epoch})  │  "
                        f"Current: {s.walkforward_current_score:.4f}"
                    )
                )  # Line 8
        lines.append(f"└{hline()}┘")  # Line 9

        # ═══════════════════════════════════════════════════════════════════════════════════════════
        # TD3 AGENT SECTION - FIXED 12 LINES regardless of training_mode
        # ═══════════════════════════════════════════════════════════════════════════════════════════
        # td3_status is already set above based on training_mode
        if s.training_mode == "RL_ONLY":
            # Phase 2: Update status based on buffer fill
            if td3_active:
                td3_status = f"{ANSI.GREEN}● TRAINING{ANSI.RESET}"
            elif s.phase == "WARMUP":
                td3_status = f"{ANSI.YELLOW}◐ BUFFER FILL{ANSI.RESET}"
            else:
                td3_status = f"{ANSI.DIM}○ IDLE{ANSI.RESET}"

        lines.append(f"┌{hline()}┐")  # Line 1
        lines.append(
            box(
                f"{ANSI.BOLD}🤖 TD3 AGENT{ANSI.RESET}                                                                  {td3_status}"
            )
        )  # Line 2
        lines.append(f"├{hline()}┤")  # Line 3

        if s.training_mode == "OBSERVER_ONLY":
            # Phase 1: TD3 is FROZEN - show minimal info (pad to 8 content lines)
            lines.append(
                box(
                    f"  {ANSI.DIM}Status: NOT TRAINING (Phase 1 - Observer only){ANSI.RESET}"
                )
            )  # Line 4
            lines.append(
                box(
                    f"  {ANSI.DIM}Actions: Uniform weights [1/K, 1/K, ..., 1/K] for portfolio simulation{ANSI.RESET}"
                )
            )  # Line 5
            lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 6 (padding)
            lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 7 (padding)
            lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 8 (padding)
            lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 9 (padding)
            lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 10 (padding)
            lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 11 (padding)
        else:
            # Phase 2: TD3 is TRAINING - show full details (8 content lines)
            # TD3 Buffer with progress bar
            buf_bar = progress_bar(s.td3_buffer_size, s.td3_buffer_max, 30)
            buf_color = ANSI.GREEN if buf_ready else ANSI.YELLOW
            lines.append(
                box(
                    f"  {ANSI.BOLD}Buffer:{ANSI.RESET}       "
                    f"{buf_color}{s.td3_buffer_size:7d}{ANSI.RESET}/{s.td3_buffer_max:7d}  "
                    f"{buf_bar}"
                )
            )  # Line 4

            # TD3 Losses
            lines.append(
                box(
                    f"  {ANSI.BOLD}Losses:{ANSI.RESET}       "
                    f"Actor: {fmt_f(s.td3_actor_loss, 10, 6)}       │  "
                    f"Critic: {fmt_f(s.td3_critic_loss, 10, 6)}"
                )
            )  # Line 5

            # TD3 Reward & Q-value
            rew_fmt = color_val(s.td3_reward, fmt_f(s.td3_reward, 10, 4))
            lines.append(
                box(
                    f"  {ANSI.BOLD}Reward:{ANSI.RESET}       {rew_fmt}                   │  "
                    f"Mean Q: {fmt_f(s.td3_mean_q, 10, 4)}"
                )
            )  # Line 6

            # TD3 Training params
            lines.append(
                box(
                    f"  {ANSI.BOLD}Training:{ANSI.RESET}     "
                    f"Updates: {s.td3_updates:8d}         │  "
                    f"LR: {s.td3_lr:.6f}  │  "
                    f"Noise σ: {s.td3_noise_sigma:.4f}"
                )
            )  # Line 7

            # TD3 Reward Breakdown header
            lines.append(f"├{hline()}┤")  # Line 8
            lines.append(
                box(
                    f"  {ANSI.BOLD}📊 REWARD BREAKDOWN (TD3 Portfolio Allocator){ANSI.RESET}"
                )
            )  # Line 9

            # r_total = (r_return + r_js) * reward_scale
            r_ret_color = ANSI.GREEN if s.td3_r_return > 0 else ANSI.RED
            r_js_color = ANSI.RED if s.td3_r_js < 0 else ANSI.GREEN
            lines.append(
                box(
                    f"  R_return: {r_ret_color}{s.td3_r_return:+8.4f}{ANSI.RESET} (w={s.td3_w_return:.2f})  │  "
                    f"R_js: {r_js_color}{s.td3_r_js:+8.4f}{ANSI.RESET} (λ={s.td3_lambda_js:.2f})  │  "
                    f"JS_div: {s.td3_js_divergence:.4f}"
                )
            )  # Line 10

            # Show unscaled and scaled total
            unscaled_color = ANSI.GREEN if s.td3_r_unscaled > 0 else ANSI.RED
            lines.append(
                box(
                    f"  Unscaled: {unscaled_color}{s.td3_r_unscaled:+8.4f}{ANSI.RESET}  →  "
                    f"Scaled (×{s.td3_reward_scale:.0f}): {rew_fmt}"
                )
            )  # Line 11
        lines.append(f"└{hline()}┘")  # Line 12

        # ═══════════════════════════════════════════════════════════════════════════════════════════
        # CONTROLLER CBF SECTION - FIXED 9 LINES regardless of training_mode
        # ═══════════════════════════════════════════════════════════════════════════════════════════
        lines.append(f"┌{hline()}┐")  # Line 1

        if s.training_mode == "OBSERVER_ONLY":
            # Phase 1: CBF not active (TD3 frozen, no actions to constrain) - pad to 7 content lines
            cbf_status = f"{ANSI.DIM}○ N/A (TD3 Frozen){ANSI.RESET}"
            lines.append(
                box(
                    f"{ANSI.BOLD}🛡️ CONTROLLER CBF{ANSI.RESET}                                                         {cbf_status}"
                )
            )  # Line 2
            lines.append(f"├{hline()}┤")  # Line 3
            lines.append(
                box(
                    f"  {ANSI.DIM}Status: NOT ACTIVE (Phase 1 - Observer only, TD3 uses uniform weights){ANSI.RESET}"
                )
            )  # Line 4
            lines.append(
                box(
                    f"  {ANSI.DIM}CBF will be active in Phase 2 when TD3 is training{ANSI.RESET}"
                )
            )  # Line 5
            lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 6 (padding)
            lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 7 (padding)
            lines.append(box(f"  {ANSI.DIM}---{ANSI.RESET}"))  # Line 8 (padding)
        else:
            # Phase 2: CBF is active - show full details (7 content lines)
            cbf_status = (
                f"{ANSI.GREEN}● ENABLED{ANSI.RESET}"
                if s.cbf_enabled
                else f"{ANSI.DIM}○ DISABLED{ANSI.RESET}"
            )
            cbf_active_status = (
                f"{ANSI.YELLOW}⚠ ACTIVE{ANSI.RESET}"
                if s.cbf_constraint_active
                else f"{ANSI.GREEN}✓ SAFE{ANSI.RESET}"
            )
            lines.append(
                box(
                    f"{ANSI.BOLD}🛡️ CONTROLLER CBF (Control Barrier Function){ANSI.RESET}                            {cbf_status}"
                )
            )  # Line 2
            lines.append(f"├{hline()}┤")  # Line 3
            # CBF Parameters
            lines.append(
                box(
                    f"  α: {s.cbf_alpha:.4f}  │  "
                    f"Safety Margin: {s.cbf_safety_margin:+8.4f}  │  "
                    f"Max Position: {s.cbf_max_position:.2f}  │  "
                    f"Status: {cbf_active_status}"
                )
            )  # Line 4
            # Sigma and Variance
            sigma_change = s.cbf_sigma_current - s.cbf_sigma_base
            sigma_dir = (
                "↑" if sigma_change > 0.001 else ("↓" if sigma_change < -0.001 else "=")
            )
            sigma_color = (
                ANSI.RED
                if sigma_change > 0.01
                else (ANSI.GREEN if sigma_change < -0.01 else ANSI.WHITE)
            )
            lines.append(
                box(
                    f"  σ_base: {s.cbf_sigma_base:.4f}  →  σ_current: {sigma_color}{s.cbf_sigma_current:.4f}{ANSI.RESET} ({sigma_dir})  │  "
                    f"Min Variance: {s.cbf_min_variance:.4f}"
                )
            )  # Line 5
            # Adjustment info with from→to values
            lines.append(
                box(
                    f"  Interventions: {s.cbf_interventions:5d}  │  "
                    f"Last: {s.cbf_last_intervention[:40] if s.cbf_last_intervention else '(none)':40s}"
                )
            )  # Line 6
            # Action adjustment: show norm change and direction (always show 2 lines)
            norm_before = s.cbf_action_norm_before
            norm_after = s.cbf_action_norm_after
            norm_change = norm_after - norm_before
            norm_dir = (
                "↑" if norm_change > 0.001 else ("↓" if norm_change < -0.001 else "=")
            )
            norm_color = (
                ANSI.RED
                if norm_change > 0.01
                else (ANSI.GREEN if norm_change < -0.01 else ANSI.WHITE)
            )
            lines.append(
                box(
                    f"  Action Norm: {norm_before:.4f} → {norm_color}{norm_after:.4f}{ANSI.RESET} ({norm_dir}{abs(norm_change):+.4f})  │  "
                    f"Δmag: {s.cbf_adjust_magnitude:+.4f}"
                )
            )  # Line 7
            # Show action values or placeholder
            if s.cbf_action_before and s.cbf_action_after:
                lines.append(
                    box(
                        f"  Before: {s.cbf_action_before[:40]:40s}  →  After: {s.cbf_action_after[:35]:35s}"
                    )
                )  # Line 8
            else:
                lines.append(
                    box(f"  {ANSI.DIM}(no action adjustment this step){ANSI.RESET}")
                )  # Line 8
        lines.append(f"└{hline()}┘")  # Line 9

        # ═══════════════════════════════════════════════════════════════════════════════════════════
        # RETURNS SECTION
        # ═══════════════════════════════════════════════════════════════════════════════════════════
        lines.append(f"┌{hline()}┐")
        lines.append(box(f"{ANSI.BOLD}💰 PERFORMANCE METRICS{ANSI.RESET}"))
        lines.append(f"├{hline()}┤")

        # Returns
        daily_fmt = color_val(s.daily_return, fmt_pct(s.daily_return, 8))
        cumul_fmt = color_val(s.cumul_return, fmt_pct(s.cumul_return, 8))
        epoch_fmt = color_val(s.epoch_return, fmt_pct(s.epoch_return, 8))

        lines.append(
            box(
                f"  Daily Return: {daily_fmt}  │  "
                f"Epoch Return: {epoch_fmt}  │  "
                f"Cumulative: {cumul_fmt}"
            )
        )

        # Risk metrics
        sharpe_color = (
            ANSI.GREEN if s.sharpe > 1 else (ANSI.YELLOW if s.sharpe > 0 else ANSI.RED)
        )
        # MDD is reported as positive percentage (e.g., 15.3% means lost 15.3% from peak)
        # Higher MDD = worse risk, so RED for high values
        mdd_color = (
            ANSI.RED if s.mdd > 10 else (ANSI.YELLOW if s.mdd > 5 else ANSI.GREEN)
        )
        win_color = (
            ANSI.GREEN
            if s.win_rate > 0.55
            else (ANSI.YELLOW if s.win_rate > 0.45 else ANSI.RED)
        )

        lines.append(
            box(
                f"  Sharpe: {sharpe_color}{s.sharpe:+6.3f}{ANSI.RESET}       │  "
                f"Max DD: {mdd_color}-{s.mdd:5.2f}%{ANSI.RESET}          │  "
                f"Win Rate: {win_color}{s.win_rate * 100:5.1f}%{ANSI.RESET}"
            )
        )

        # Capital and Net Profit
        cap_str = f"${s.capital:,.0f}"
        profit_color = ANSI.GREEN if s.net_profit > 0 else ANSI.RED
        profit_str = f"${s.net_profit:+,.0f}"
        lines.append(
            box(
                f"  Capital: {ANSI.BOLD}{cap_str:>12s}{ANSI.RESET}  │  "
                f"Net Profit: {profit_color}{profit_str:>12s}{ANSI.RESET}"
            )
        )

        # Annual Return and Volatility
        annual_color = ANSI.GREEN if s.annual_return_pct > 0 else ANSI.RED
        vol_color = (
            ANSI.RED
            if s.vol_max > 30
            else (ANSI.YELLOW if s.vol_max > 20 else ANSI.GREEN)
        )
        lines.append(
            box(
                f"  Annual Return: {annual_color}{s.annual_return_pct:+7.2f}%{ANSI.RESET}  │  "
                f"Max Volatility: {vol_color}{s.vol_max:6.2f}%{ANSI.RESET}"
            )
        )
        lines.append(f"└{hline()}┘")

        # ═══════════════════════════════════════════════════════════════════════════════════════════
        # EVENT LOG SECTION (Recent events: rebalances, regime shifts, etc.)
        # ═══════════════════════════════════════════════════════════════════════════════════════════
        lines.append(f"┌{hline()}┐")
        lines.append(
            box(f"{ANSI.BOLD}📋 EVENT LOG{ANSI.RESET} (Last {s.max_events} events)")
        )
        lines.append(f"├{hline()}┤")

        # Show events or placeholder
        if s.event_log:
            for event in s.event_log[-s.max_events :]:
                # Truncate event to fit in box
                visible = self._strip_ansi(event)
                if len(visible) > W - 6:
                    # Need to truncate
                    event = event[: W - 9] + "..."
                lines.append(box(f"  {event}"))
            # Pad remaining event slots
            for _ in range(s.max_events - len(s.event_log)):
                lines.append(box(f"  {ANSI.DIM}(waiting for events...){ANSI.RESET}"))
        else:
            for _ in range(s.max_events):
                lines.append(box(f"  {ANSI.DIM}(waiting for events...){ANSI.RESET}"))

        lines.append(f"└{hline()}┘")

        # Padding to reach FIXED_LINES
        while len(lines) < self.FIXED_LINES:
            lines.append("")

        return lines[: self.FIXED_LINES]

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from text."""
        import re

        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:5.0f}s"
        elif seconds < 3600:
            m, s = divmod(seconds, 60)
            return f"{int(m):2d}m {int(s):02d}s"
        else:
            h, rem = divmod(seconds, 3600)
            m, _ = divmod(rem, 60)
            return f"{int(h):2d}h {int(m):02d}m"

    def log(self, message: str, level: str = "INFO"):
        """Log a message to file only."""
        if self.log_file:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a") as f:
                f.write(f"[{timestamp}] [{level}] {message}\n")


# Global instance
_display: Optional[LiveDisplay] = None


def get_display() -> LiveDisplay:
    """Get or create the global display instance."""
    global _display
    if _display is None:
        _display = LiveDisplay(enabled=False)
    return _display


def init_display(enabled: bool = True, log_file: Optional[str] = None) -> LiveDisplay:
    """Initialize the global display instance.

    If a display already exists and is enabled, reuse it (just reset state).
    This prevents multiple displays from being created during walk-forward iterations.
    """
    global _display

    # Cleanup existing display before creating new one
    if _display is not None and _display.enabled:
        _display.cleanup()
        _display.release_stdout()

    _display = LiveDisplay(enabled=enabled, log_file=log_file)
    return _display
