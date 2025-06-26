#!/usr/bin/env python3
"""
Training Monitor for AI Assistant

This script monitors the training progress and provides real-time updates.
"""

import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def monitor_training(output_dir="outputs", refresh_interval=10):
    """Monitor training progress by reading logs and metrics."""
    
    print("🔍 AI Assistant Training Monitor")
    print("=" * 50)
    
    metrics_file = Path(output_dir) / "training_metrics.json"
    log_dir = Path("logs")
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
            
            print("🚀 AI Assistant Training Monitor")
            print("=" * 50)
            print(f"⏰ Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Check if metrics file exists
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    print("📊 Training Metrics:")
                    print("-" * 30)
                    
                    if metrics:
                        latest_step = max(metrics.keys(), key=int)
                        latest_metrics = metrics[latest_step]
                        
                        print(f"🔢 Global Step: {latest_step}")
                        print(f"📉 Train Loss: {latest_metrics.get('train_loss', 'N/A'):.4f}")
                        print(f"📈 Eval Loss: {latest_metrics.get('eval_loss', 'N/A'):.4f}")
                        print(f"🎯 Perplexity: {latest_metrics.get('eval_perplexity', 'N/A'):.2f}")
                        print(f"📚 Learning Rate: {latest_metrics.get('learning_rate', 'N/A'):.2e}")
                        print(f"🏆 Epoch: {latest_metrics.get('epoch', 'N/A')}")
                    else:
                        print("📊 No metrics available yet...")
                        
                except Exception as e:
                    print(f"❌ Error reading metrics: {e}")
            else:
                print("📊 Training Metrics:")
                print("-" * 30)
                print("⏳ Metrics file not found. Training may still be initializing...")
            
            print()
            
            # Check training status
            print("💾 Checkpoints:")
            print("-" * 30)
            
            checkpoint_dir = Path("checkpoints")
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pt"))
                if checkpoints:
                    print(f"📁 Found {len(checkpoints)} checkpoint(s)")
                    latest_checkpoint = max(checkpoints, key=os.path.getctime)
                    print(f"🔄 Latest: {latest_checkpoint.name}")
                    print(f"📅 Modified: {time.ctime(os.path.getctime(latest_checkpoint))}")
                else:
                    print("📁 No checkpoints found yet")
            else:
                print("📁 Checkpoint directory not found")
            
            print()
            print("🔧 System Info:")
            print("-" * 30)
            
            # Check GPU/CPU usage (simplified)
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
                    print(f"💾 GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB used")
                else:
                    print("🖥️  Device: CPU (Training will be slower)")
            except:
                print("🖥️  Device info not available")
            
            print()
            print(f"⏱️  Refreshing in {refresh_interval} seconds... (Ctrl+C to exit)")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Training monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

def plot_training_curves(output_dir="outputs"):
    """Create training curve plots."""
    print("📈 Creating training plots...")
    
    metrics_file = Path(output_dir) / "training_metrics.json"
    
    if not metrics_file.exists():
        print("❌ No metrics file found. Training may not have started yet.")
        return
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if not metrics:
            print("❌ No metrics data available.")
            return
        
        # Extract data
        steps = [int(step) for step in metrics.keys()]
        train_losses = [metrics[str(step)].get('train_loss', 0) for step in steps]
        eval_losses = [metrics[str(step)].get('eval_loss', 0) for step in steps if 'eval_loss' in metrics[str(step)]]
        eval_steps = [step for step in steps if 'eval_loss' in metrics[str(step)]]
        
        # Create plots
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(steps, train_losses, label='Train Loss', color='blue', alpha=0.7)
        if eval_losses:
            plt.plot(eval_steps, eval_losses, label='Eval Loss', color='red', alpha=0.7, marker='o')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(1, 2, 2)
        learning_rates = [metrics[str(step)].get('learning_rate', 0) for step in steps]
        plt.plot(steps, learning_rates, color='green', alpha=0.7)
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(output_dir) / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Training curves saved to {plot_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_training_curves()
    else:
        monitor_training()

if __name__ == "__main__":
    main()
