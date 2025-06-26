"""
Shared utilities and common functions for the AI Assistant project.

This module contains utility functions that are used across different phases
of the project, including logging, configuration management, model utilities,
and training helpers.
"""

import os
import json
import logging
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_memory_usage(model: nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Estimate model memory usage.
    
    Args:
        model: PyTorch model
        device: Device to check memory on
        
    Returns:
        Dictionary with memory usage information
    """
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    
    memory_info = {
        'parameters_mb': param_memory / (1024 ** 2),
        'buffers_mb': buffer_memory / (1024 ** 2),
        'total_mb': (param_memory + buffer_memory) / (1024 ** 2)
    }
    
    if device.type == 'cuda':
        memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated(device) / (1024 ** 2)
        memory_info['gpu_cached_mb'] = torch.cuda.memory_reserved(device) / (1024 ** 2)
    
    return memory_info


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warmup.
    """
    
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, 
                 min_lr_ratio: float = 0.1, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: linear increase
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor) 
                   for base_lr in self.base_lrs]


class TrainingMetrics:
    """
    Track training metrics and provide utilities for logging and visualization.
    """
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics for a given step."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((step, value))
            self.metrics[key] = value
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        return self.metrics.get(metric_name)
    
    def get_history(self, metric_name: str) -> List[tuple]:
        """Get the history for a metric."""
        return self.history.get(metric_name, [])
    
    def save(self, filepath: str):
        """Save metrics to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, filepath: str):
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        
        # Update current metrics with latest values
        for key, values in self.history.items():
            if values:
                self.metrics[key] = values[-1][1]


class ModelCheckpoint:
    """
    Handle model checkpointing and loading.
    """
    
    def __init__(self, checkpoint_dir: str, save_total_limit: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
    
    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
             scheduler: Optional[_LRScheduler], step: int, 
             metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            step: Current training step
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'step': step,
            'metrics': metrics
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def load(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[_LRScheduler] = None, 
             checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            checkpoint_path: Specific checkpoint path, or None for latest
            
        Returns:
            Dictionary with loaded information
        """
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob("checkpoint-*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = max(checkpoints, key=lambda x: int(x.stem.split('-')[1]))
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {}),
            'checkpoint_path': str(checkpoint_path)
        }
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain save_total_limit."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*.pt"),
            key=lambda x: int(x.stem.split('-')[1])
        )
        
        if len(checkpoints) > self.save_total_limit:
            for checkpoint in checkpoints[:-self.save_total_limit]:
                checkpoint.unlink()


class TimerContext:
    """
    Context manager for timing operations.
    """
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.operation_name} completed in {duration:.2f} seconds")
    
    @property
    def duration(self) -> float:
        """Get the duration of the timed operation."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def estimate_training_time(steps_per_second: float, total_steps: int, 
                          completed_steps: int = 0) -> str:
    """
    Estimate remaining training time.
    
    Args:
        steps_per_second: Training speed
        total_steps: Total training steps
        completed_steps: Already completed steps
        
    Returns:
        Formatted time estimate
    """
    remaining_steps = total_steps - completed_steps
    remaining_seconds = remaining_steps / steps_per_second
    return format_time(remaining_seconds)


def print_model_summary(model: nn.Module, input_shape: tuple = None):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape for parameter calculation
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    total_params = count_parameters(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")  # Assuming float32
    
    if input_shape:
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        model.eval()
        with torch.no_grad():
            try:
                output = model(dummy_input)
                if isinstance(output, dict):
                    print(f"Output shapes:")
                    for key, tensor in output.items():
                        print(f"  {key}: {tensor.shape}")
                else:
                    print(f"Output shape: {output.shape}")
            except Exception as e:
                print(f"Could not compute output shape: {e}")
    
    print("=" * 60)


# Default configuration for the project
DEFAULT_CONFIG = {
    "project_name": "AI Assistant From Scratch",
    "version": "1.0.0",
    "random_seed": 42,
    "device": "auto",
    "mixed_precision": True,
    "compile_model": False
}


if __name__ == "__main__":
    # Test utilities
    print("Testing shared utilities...")
    
    # Test logging
    logger = setup_logging("INFO")
    logger.info("Logging system initialized")
    
    # Test device detection
    device = get_device()
    print(f"Using device: {device}")
    
    # Test timer
    with TimerContext("Test operation"):
        time.sleep(1)
    
    # Test metrics
    metrics = TrainingMetrics()
    metrics.update({"loss": 1.5, "accuracy": 0.8}, step=100)
    print(f"Latest loss: {metrics.get_latest('loss')}")
    
    print("✅ Shared utilities test completed!")
