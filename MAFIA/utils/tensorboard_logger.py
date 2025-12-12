#!/usr/bin/env python3
"""
TensorBoard Logger for MAFIA Observer Training

Provides a unified interface for logging training metrics, charts, and diagnostics
to TensorBoard for real-time visualization and post-training analysis.

Features:
- Scalar logging (losses, metrics, rewards)
- Image logging (training charts)
- Histogram logging (weights, gradients, actions)
- Text logging (configuration, events)
- Graceful fallback if TensorBoard unavailable
"""
import os
from typing import Dict, Optional, Any, Union
import numpy as np
import torch as th
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class TensorBoardLogger:
    """
    Centralized TensorBoard logger for MAFIA Observer training.

    Handles all TensorBoard logging operations with automatic directory management,
    graceful fallback, and memory-efficient image encoding.
    """

    def __init__(
        self,
        log_dir: str,
        enabled: bool = True,
        log_images: bool = True,
        log_histograms: bool = True,
        histogram_freq: int = 10,
        comment: str = "",
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Base directory for TensorBoard logs
            enabled: Enable/disable logging (master switch)
            log_images: Enable image logging
            log_histograms: Enable histogram logging
            histogram_freq: Log histograms every N batches
            comment: Optional comment for run identification
        """
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        self.log_images = log_images
        self.log_histograms = log_histograms
        self.histogram_freq = histogram_freq
        self.comment = comment

        self.writer: Optional[SummaryWriter] = None
        # Store absolute path to avoid issues with working directory changes
        self.log_dir = str(Path(log_dir).resolve())
        self.global_step = 0
        self._writer_invalid = False  # Track if writer needs recreation

        if self.enabled:
            self._init_writer()
        else:
            if not TENSORBOARD_AVAILABLE:
                print("[WARN] TensorBoard not available. Install with: pip install tensorboard")
            else:
                print("[INFO] TensorBoard logging disabled")

    def _init_writer(self):
        """Initialize or reinitialize the SummaryWriter."""
        try:
            # Close existing writer if any
            if self.writer is not None:
                try:
                    self.writer.close()
                except Exception:
                    pass

            # Create log directory (use absolute path)
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)

            # Initialize SummaryWriter with absolute path
            self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.comment)
            self._writer_invalid = False
            print(f"[TensorBoard] Logging to: {self.log_dir}")
            print(f"[TensorBoard] Start server: tensorboard --logdir {Path(self.log_dir).parent}")
        except Exception as e:
            print(f"[WARN] Failed to initialize TensorBoard: {e}")
            self.enabled = False
            self._writer_invalid = True

    def _ensure_writer_valid(self) -> bool:
        """Ensure writer is valid, recreate if needed. Returns True if valid."""
        if not self.enabled or self._writer_invalid:
            if self._writer_invalid and self.enabled:
                # Try to recreate the writer once
                self._init_writer()
            return self.enabled and self.writer is not None and not self._writer_invalid
        return self.writer is not None
    
    def log_scalar(
        self,
        tag: str,
        value: Union[float, int, th.Tensor],
        step: Optional[int] = None,
        phase: str = "train",
    ):
        """
        Log a scalar value.

        Args:
            tag: Metric name (e.g., "loss/pg", "metrics/sharpe")
            value: Scalar value to log
            step: Global step (uses internal counter if None)
            phase: Training phase (train/valid/test) - prepended to tag
        """
        if not self._ensure_writer_valid():
            return

        try:
            # Convert tensor to scalar
            if isinstance(value, th.Tensor):
                value = value.detach().cpu().item()

            # Prepend phase to tag
            full_tag = f"{phase}/{tag}"

            # Use internal step counter if not provided
            if step is None:
                step = self.global_step

            self.writer.add_scalar(full_tag, value, step)
        except OSError as e:
            print(f"[WARN] Failed to log scalar {tag}: {e}")
            self._writer_invalid = True
            self._init_writer()
        except Exception as e:
            print(f"[WARN] Failed to log scalar {tag}: {e}")
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, int, th.Tensor]],
        step: Optional[int] = None,
        phase: str = "train",
    ):
        """
        Log multiple scalars under a common tag.

        Args:
            main_tag: Main category (e.g., "losses", "metrics")
            tag_scalar_dict: Dictionary of {sub_tag: value}
            step: Global step
            phase: Training phase
        """
        if not self._ensure_writer_valid():
            return

        try:
            if step is None:
                step = self.global_step

            # Convert tensors to scalars
            scalar_dict = {}
            for k, v in tag_scalar_dict.items():
                if isinstance(v, th.Tensor):
                    v = v.detach().cpu().item()
                scalar_dict[k] = v

            full_tag = f"{phase}/{main_tag}"
            self.writer.add_scalars(full_tag, scalar_dict, step)
        except OSError as e:
            print(f"[WARN] Failed to log scalars {main_tag}: {e}")
            self._writer_invalid = True
            self._init_writer()
        except Exception as e:
            print(f"[WARN] Failed to log scalars {main_tag}: {e}")
    
    def log_image(
        self,
        tag: str,
        image_path: str,
        step: Optional[int] = None,
        phase: str = "train",
    ):
        """
        Log an image from file path.

        Args:
            tag: Image name (e.g., "charts/dashboard")
            image_path: Path to image file
            step: Global step
            phase: Training phase
        """
        if not self.log_images or not self._ensure_writer_valid():
            return

        try:
            if not os.path.exists(image_path):
                print(f"[WARN] Image not found: {image_path}")
                return

            # Read image using PIL
            from PIL import Image
            import torchvision.transforms as transforms

            img = Image.open(image_path)
            # Convert to tensor (C, H, W) format
            img_tensor = transforms.ToTensor()(img)

            if step is None:
                step = self.global_step

            full_tag = f"{phase}/{tag}"
            self.writer.add_image(full_tag, img_tensor, step)
        except OSError as e:
            print(f"[WARN] Failed to log image {tag}: {e}")
            self._writer_invalid = True
            self._init_writer()
        except Exception as e:
            print(f"[WARN] Failed to log image {tag}: {e}")
    
    def log_histogram(
        self,
        tag: str,
        values: Union[th.Tensor, np.ndarray],
        step: Optional[int] = None,
        phase: str = "train",
    ):
        """
        Log a histogram of values.

        Args:
            tag: Histogram name (e.g., "weights/layer1", "gradients/encoder")
            values: Tensor or array of values
            step: Global step
            phase: Training phase
        """
        if not self.log_histograms or not self._ensure_writer_valid():
            return

        try:
            # Convert to numpy if tensor
            if isinstance(values, th.Tensor):
                values = values.detach().cpu().numpy()

            if step is None:
                step = self.global_step

            full_tag = f"{phase}/{tag}"
            self.writer.add_histogram(full_tag, values, step)
        except OSError as e:
            # Handle file not found errors (e.g., event file deleted)
            print(f"[WARN] Failed to log histogram {tag}: {e}")
            self._writer_invalid = True
            # Try to recreate writer for next call
            self._init_writer()
        except Exception as e:
            print(f"[WARN] Failed to log histogram {tag}: {e}")
    
    def log_model_weights(
        self,
        model: th.nn.Module,
        step: Optional[int] = None,
        phase: str = "train",
    ):
        """
        Log histograms of all model weights.
        
        Args:
            model: PyTorch model
            step: Global step
            phase: Training phase
        """
        if not self.enabled or not self.log_histograms or self.writer is None:
            return
        
        try:
            if step is None:
                step = self.global_step
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.log_histogram(f"weights/{name}", param.data, step, phase)
        except Exception as e:
            print(f"[WARN] Failed to log model weights: {e}")
    
    def log_model_gradients(
        self,
        model: th.nn.Module,
        step: Optional[int] = None,
        phase: str = "train",
    ):
        """
        Log histograms of all model gradients.
        
        Args:
            model: PyTorch model
            step: Global step
            phase: Training phase
        """
        if not self.enabled or not self.log_histograms or self.writer is None:
            return
        
        try:
            if step is None:
                step = self.global_step
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.log_histogram(f"gradients/{name}", param.grad.data, step, phase)
        except Exception as e:
            print(f"[WARN] Failed to log model gradients: {e}")
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
    ):
        """
        Log text data.

        Args:
            tag: Text identifier
            text: Text content
            step: Global step
        """
        if not self._ensure_writer_valid():
            return

        try:
            if step is None:
                step = self.global_step

            self.writer.add_text(tag, text, step)
        except OSError as e:
            print(f"[WARN] Failed to log text {tag}: {e}")
            self._writer_invalid = True
            self._init_writer()
        except Exception as e:
            print(f"[WARN] Failed to log text {tag}: {e}")
    
    def log_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, float],
    ):
        """
        Log hyperparameters and final metrics.

        Args:
            hparam_dict: Dictionary of hyperparameters
            metric_dict: Dictionary of final metrics
        """
        if not self._ensure_writer_valid():
            return

        try:
            # Filter out non-serializable values
            clean_hparams = {}
            for k, v in hparam_dict.items():
                if isinstance(v, (int, float, str, bool)):
                    clean_hparams[k] = v
                elif isinstance(v, (list, tuple)) and len(v) > 0:
                    clean_hparams[k] = str(v)

            self.writer.add_hparams(clean_hparams, metric_dict)
        except OSError as e:
            print(f"[WARN] Failed to log hparams: {e}")
            self._writer_invalid = True
            self._init_writer()
        except Exception as e:
            print(f"[WARN] Failed to log hparams: {e}")
    
    def increment_step(self):
        """Increment global step counter."""
        self.global_step += 1
    
    def set_step(self, step: int):
        """Set global step counter."""
        self.global_step = step
    
    def flush(self):
        """Flush pending logs to disk."""
        if self.enabled and self.writer is not None:
            try:
                self.writer.flush()
            except Exception as e:
                print(f"[WARN] Failed to flush TensorBoard logs: {e}")
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer is not None:
            try:
                self.writer.close()
                print(f"[TensorBoard] Logs saved to: {self.log_dir}")
            except Exception as e:
                print(f"[WARN] Failed to close TensorBoard writer: {e}")
            finally:
                self.writer = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def create_tensorboard_logger(
    base_dir: str,
    run_name: str,
    config: Optional[Any] = None,
    enabled: bool = True,
) -> TensorBoardLogger:
    """
    Factory function to create a TensorBoard logger with standard settings.
    
    Args:
        base_dir: Base directory for all TensorBoard logs
        run_name: Unique name for this run (e.g., "win0_year2017")
        config: Optional config object with TensorBoard settings
        enabled: Master enable switch
    
    Returns:
        TensorBoardLogger instance
    """
    # Extract settings from config if provided
    log_images = getattr(config, "tensorboard_log_images", True) if config else True
    log_histograms = getattr(config, "tensorboard_log_histograms", True) if config else True
    histogram_freq = getattr(config, "tensorboard_histogram_freq", 10) if config else 10
    
    # Construct log directory
    log_dir = os.path.join(base_dir, run_name)
    
    return TensorBoardLogger(
        log_dir=log_dir,
        enabled=enabled,
        log_images=log_images,
        log_histograms=log_histograms,
        histogram_freq=histogram_freq,
        comment=run_name,
    )
