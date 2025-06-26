"""
Environment Setup and Configuration

This module handles the setup of the training environment, including
GPU configuration, CUDA setup, distributed training preparation,
and system optimization for large-scale model training.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import psutil
import platform
import subprocess
import json
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemChecker:
    """
    Check system requirements and capabilities for AI training.
    """
    
    @staticmethod
    def check_cuda_availability() -> Dict[str, any]:
        """Check CUDA availability and GPU information."""
        cuda_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_details': []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                cuda_info['gpu_details'].append({
                    'device_id': i,
                    'name': gpu_props.name,
                    'memory_total': gpu_props.total_memory // (1024**3),  # GB
                    'memory_available': torch.cuda.get_device_properties(i).total_memory // (1024**3),
                    'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                })
        
        return cuda_info
    
    @staticmethod
    def check_system_specs() -> Dict[str, any]:
        """Check system specifications."""
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.machine(),
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_physical_cores': psutil.cpu_count(logical=False),
            'memory_total_gb': psutil.virtual_memory().total // (1024**3),
            'memory_available_gb': psutil.virtual_memory().available // (1024**3),
            'disk_space_gb': psutil.disk_usage('/').free // (1024**3) if platform.system() != 'Windows' 
                           else psutil.disk_usage('C:').free // (1024**3)
        }
    
    @staticmethod
    def check_python_environment() -> Dict[str, any]:
        """Check Python environment and installed packages."""
        try:
            import torch
            import transformers
            import datasets
            import numpy as np
            import pandas as pd
            
            env_info = {
                'python_version': platform.python_version(),
                'torch_version': torch.__version__,
                'transformers_version': transformers.__version__,
                'datasets_version': datasets.__version__,
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__,
                'environment_ready': True
            }
        except ImportError as e:
            env_info = {
                'python_version': platform.python_version(),
                'environment_ready': False,
                'missing_packages': str(e)
            }
        
        return env_info


class TrainingEnvironment:
    """
    Manage training environment configuration and optimization.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/training_config.json"
        self.system_info = self._gather_system_info()
        self.config = self._load_or_create_config()
    
    def _gather_system_info(self) -> Dict[str, any]:
        """Gather comprehensive system information."""
        checker = SystemChecker()
        return {
            'cuda_info': checker.check_cuda_availability(),
            'system_specs': checker.check_system_specs(),
            'python_env': checker.check_python_environment()
        }
    
    def _load_or_create_config(self) -> Dict[str, any]:
        """Load existing config or create default based on system capabilities."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, any]:
        """Create default configuration based on system capabilities."""
        cuda_info = self.system_info['cuda_info']
        system_specs = self.system_info['system_specs']
        
        # Determine optimal settings based on available hardware
        if cuda_info['cuda_available'] and cuda_info['gpu_count'] > 0:
            # Calculate optimal batch size based on GPU memory
            gpu_memory_gb = cuda_info['gpu_details'][0]['memory_total']
            if gpu_memory_gb >= 24:  # High-end GPU
                batch_size = 32
                gradient_accumulation_steps = 1
            elif gpu_memory_gb >= 12:  # Mid-range GPU
                batch_size = 16
                gradient_accumulation_steps = 2
            else:  # Lower-end GPU
                batch_size = 8
                gradient_accumulation_steps = 4
        else:
            # CPU-only configuration
            batch_size = 4
            gradient_accumulation_steps = 8
        
        config = {
            'hardware': {
                'use_cuda': cuda_info['cuda_available'],
                'gpu_count': cuda_info['gpu_count'],
                'use_mixed_precision': cuda_info['cuda_available'],
                'gradient_checkpointing': True
            },
            'training': {
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'max_grad_norm': 1.0,
                'learning_rate': 5e-4,
                'warmup_steps': 1000,
                'max_steps': 100000,
                'save_steps': 5000,
                'eval_steps': 1000,
                'logging_steps': 100
            },
            'model': {
                'd_model': 512,
                'num_heads': 8,
                'num_layers': 6,
                'vocab_size': 50000,
                'max_seq_length': 1024
            },
            'data': {
                'dataset_path': 'data/processed',
                'tokenizer_path': 'tokenizer/',
                'num_workers': min(8, system_specs['cpu_count']),
                'pin_memory': cuda_info['cuda_available']
            },
            'optimization': {
                'optimizer': 'AdamW',
                'weight_decay': 0.1,
                'beta1': 0.9,
                'beta2': 0.95,
                'epsilon': 1e-8,
                'scheduler': 'cosine'
            },
            'distributed': {
                'backend': 'nccl' if cuda_info['cuda_available'] else 'gloo',
                'world_size': cuda_info['gpu_count'],
                'find_unused_parameters': False
            }
        }
        
        # Save the default configuration
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def optimize_pytorch_settings(self):
        """Optimize PyTorch settings for training performance."""
        # Enable optimized attention if available
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Set optimal number of threads
        if self.config['hardware']['use_cuda']:
            torch.set_num_threads(min(8, psutil.cpu_count(logical=True)))
        else:
            torch.set_num_threads(psutil.cpu_count(logical=True))
        
        # Enable cuDNN benchmark mode for consistent input sizes
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        logger.info("PyTorch settings optimized for training")
    
    def setup_distributed_training(self, rank: int, world_size: int):
        """Setup distributed training environment."""
        if world_size > 1:
            # Set up process group
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            dist.init_process_group(
                backend=self.config['distributed']['backend'],
                rank=rank,
                world_size=world_size
            )
            
            # Set device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(rank)
            
            logger.info(f"Distributed training setup complete. Rank: {rank}, World size: {world_size}")
    
    def print_environment_summary(self):
        """Print a comprehensive environment summary."""
        print("=" * 60)
        print("AI ASSISTANT TRAINING ENVIRONMENT SUMMARY")
        print("=" * 60)
        
        # System Information
        system = self.system_info['system_specs']
        print(f"\n🖥️  SYSTEM INFORMATION:")
        print(f"   OS: {system['os']} {system['os_version']}")
        print(f"   Architecture: {system['architecture']}")
        print(f"   CPU Cores: {system['cpu_physical_cores']} physical, {system['cpu_count']} logical")
        print(f"   Memory: {system['memory_available_gb']:.1f}GB available / {system['memory_total_gb']:.1f}GB total")
        print(f"   Disk Space: {system['disk_space_gb']:.1f}GB available")
        
        # CUDA Information
        cuda = self.system_info['cuda_info']
        print(f"\n🚀 CUDA INFORMATION:")
        print(f"   CUDA Available: {cuda['cuda_available']}")
        if cuda['cuda_available']:
            print(f"   CUDA Version: {cuda['cuda_version']}")
            print(f"   cuDNN Version: {cuda['cudnn_version']}")
            print(f"   GPU Count: {cuda['gpu_count']}")
            for gpu in cuda['gpu_details']:
                print(f"   GPU {gpu['device_id']}: {gpu['name']} ({gpu['memory_total']}GB)")
        
        # Python Environment
        python_env = self.system_info['python_env']
        print(f"\n🐍 PYTHON ENVIRONMENT:")
        print(f"   Python Version: {python_env['python_version']}")
        print(f"   Environment Ready: {python_env['environment_ready']}")
        if python_env['environment_ready']:
            print(f"   PyTorch: {python_env['torch_version']}")
            print(f"   Transformers: {python_env['transformers_version']}")
            print(f"   Datasets: {python_env['datasets_version']}")
        
        # Training Configuration
        print(f"\n⚙️  TRAINING CONFIGURATION:")
        training = self.config['training']
        print(f"   Batch Size: {training['batch_size']}")
        print(f"   Gradient Accumulation: {training['gradient_accumulation_steps']}")
        print(f"   Learning Rate: {training['learning_rate']}")
        print(f"   Mixed Precision: {self.config['hardware']['use_mixed_precision']}")
        print(f"   Gradient Checkpointing: {self.config['hardware']['gradient_checkpointing']}")
        
        # Recommendations
        self._print_recommendations()
        
        print("=" * 60)
    
    def _print_recommendations(self):
        """Print recommendations based on system capabilities."""
        print(f"\n💡 RECOMMENDATIONS:")
        
        cuda_info = self.system_info['cuda_info']
        system_specs = self.system_info['system_specs']
        
        if not cuda_info['cuda_available']:
            print("   ⚠️  No CUDA GPU detected. Training will be slow on CPU.")
            print("   💰 Consider using cloud GPU instances (Lambda Labs, RunPod, etc.)")
        
        if cuda_info['cuda_available'] and cuda_info['gpu_details']:
            gpu_memory = cuda_info['gpu_details'][0]['memory_total']
            if gpu_memory < 8:
                print("   ⚠️  Limited GPU memory. Consider gradient checkpointing and smaller batch sizes.")
            elif gpu_memory >= 24:
                print("   ✅ Excellent GPU memory! You can train larger models efficiently.")
        
        if system_specs['memory_total_gb'] < 16:
            print("   ⚠️  Limited system RAM. Consider reducing data loader workers.")
        
        if system_specs['disk_space_gb'] < 100:
            print("   ⚠️  Limited disk space. Large datasets may require external storage.")


def setup_environment():
    """
    Main function to set up the training environment.
    Call this function to initialize and configure everything needed for training.
    """
    print("Setting up AI Assistant training environment...")
    
    # Initialize environment
    env = TrainingEnvironment()
    
    # Optimize PyTorch settings
    env.optimize_pytorch_settings()
    
    # Print comprehensive summary
    env.print_environment_summary()
    
    return env


if __name__ == "__main__":
    # Setup the environment when running this script directly
    environment = setup_environment()
    
    # Additional checks
    print("\n🔍 RUNNING ADDITIONAL CHECKS...")
    
    # Test basic tensor operations
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            test_tensor = torch.randn(1000, 1000, device=device)
            result = torch.matmul(test_tensor, test_tensor)
            print("   ✅ GPU tensor operations working correctly")
        else:
            test_tensor = torch.randn(1000, 1000)
            result = torch.matmul(test_tensor, test_tensor)
            print("   ✅ CPU tensor operations working correctly")
    except Exception as e:
        print(f"   ❌ Tensor operations failed: {e}")
    
    # Test mixed precision if available
    if torch.cuda.is_available():
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            print("   ✅ Mixed precision training available")
        except ImportError:
            print("   ⚠️  Mixed precision not available")
    
    print("\n✨ Environment setup complete! Ready to proceed to Phase 3.")
    print("💡 Next: Run data collection and preprocessing scripts.")
