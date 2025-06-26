#!/usr/bin/env python3
"""
AI Assistant From Scratch - Main Startup Script

This script helps you get started with building your AI assistant from scratch.
It provides guidance, checks your environment, and helps you navigate through
the different phases of the project.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import subprocess

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent / "shared"))

try:
    from utils import setup_logging, get_device, print_model_summary
    from phase2_environment.setup_environment import setup_environment
except ImportError:
    print("⚠️  Some modules not found. Make sure to install requirements first:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logger = setup_logging("INFO")

def print_welcome():
    """Print welcome message and project overview."""
    print("=" * 70)
    print("🧠 WELCOME TO: AI ASSISTANT FROM SCRATCH")
    print("=" * 70)
    print()
    print("🎯 Project Goal: Build a complete AI assistant without using pre-trained models")
    print("📚 Learning Focus: Deep understanding of transformers, attention, and language models")
    print("🔧 Technology Stack: PyTorch, NumPy, Custom implementations")
    print()
    print("📅 Development Phases:")
    phases = [
        ("Phase 1", "Learn the Fundamentals", "✅ Ready"),
        ("Phase 2", "Set Up Training Environment", "✅ Ready"),
        ("Phase 3", "Data Collection & Preprocessing", "✅ Ready"),
        ("Phase 4", "Implement Transformer Model", "✅ Ready"),
        ("Phase 5", "Train Small Language Model", "🔄 Next"),
        ("Phase 6", "Scale Up Training", "⏳ Future"),
        ("Phase 7", "Build Assistant Application", "⏳ Future"),
        ("Phase 8", "Add Advanced Features", "⏳ Future")
    ]
    
    for phase, description, status in phases:
        print(f"   {phase}: {description:<30} {status}")
    
    print()
    print("🚀 Let's build something amazing together!")
    print("=" * 70)
    print()

def check_requirements() -> bool:
    """Check if all requirements are installed."""
    print("🔍 Checking Requirements...")
    
    required_packages = [
        "torch", "numpy", "matplotlib", "seaborn", "tqdm", 
        "transformers", "datasets", "tokenizers"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📦 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All requirements satisfied!")
    return True

def show_phase_menu() -> str:
    """Show phase selection menu."""
    print("\n📋 SELECT A PHASE TO START:")
    print()
    
    phases = {
        "1": ("Phase 1: Fundamentals", "Learn PyTorch, attention, transformers"),
        "2": ("Phase 2: Environment", "Set up training environment and GPU"),
        "3": ("Phase 3: Data Pipeline", "Collect and preprocess training data"),
        "4": ("Phase 4: Model Implementation", "Build transformer from scratch"),
        "5": ("Phase 5: Training", "Train your first small language model"),
        "env": ("Check Environment", "Verify system capabilities"),
        "demo": ("Quick Demo", "See a mini transformer in action"),
        "help": ("Help & Documentation", "Get help and view documentation"),
        "quit": ("Exit", "Exit the application")
    }
    
    for key, (title, description) in phases.items():
        print(f"   [{key}] {title}")
        print(f"       {description}")
        print()
    
    while True:
        choice = input("Enter your choice: ").strip().lower()
        if choice in phases:
            return choice
        print("❌ Invalid choice. Please try again.")

def run_phase1():
    """Start Phase 1: Fundamentals."""
    print("\n🧠 PHASE 1: FUNDAMENTALS")
    print("=" * 50)
    print()
    print("📚 Learning Materials Available:")
    print("   • PyTorch Basics Notebook")
    print("   • Attention Mechanism Implementation")
    print("   • Transformer Components")
    print()
    
    notebooks = [
        "phase1_fundamentals/01_pytorch_basics.ipynb",
        "phase1_fundamentals/attention_mechanism.py",
        "phase1_fundamentals/pytorch_basics.py"
    ]
    
    print("🎯 Recommended Learning Path:")
    for i, notebook in enumerate(notebooks, 1):
        if os.path.exists(notebook):
            print(f"   {i}. {notebook} ✅")
        else:
            print(f"   {i}. {notebook} ❌ (Missing)")
    
    print()
    print("💡 To start:")
    print("   1. Open the Jupyter notebook: phase1_fundamentals/01_pytorch_basics.ipynb")
    print("   2. Work through the examples step by step")
    print("   3. Run the Python scripts to test your understanding")
    print()
    
    if input("Would you like to open the notebook now? (y/n): ").lower() == 'y':
        try:
            notebook_path = "phase1_fundamentals/01_pytorch_basics.ipynb"
            if os.path.exists(notebook_path):
                print(f"📝 Opening {notebook_path}...")
                # This will open in VS Code's notebook interface
                os.system(f"code {notebook_path}")
            else:
                print("❌ Notebook not found. Please check the file path.")
        except Exception as e:
            print(f"❌ Error opening notebook: {e}")

def run_phase2():
    """Start Phase 2: Environment Setup."""
    print("\n⚙️  PHASE 2: ENVIRONMENT SETUP")
    print("=" * 50)
    print()
    
    try:
        # Run environment setup
        env = setup_environment()
        print("\n✅ Environment setup completed!")
        
        # Show system summary
        env.print_environment_summary()
        
    except Exception as e:
        print(f"❌ Error setting up environment: {e}")
        print("💡 Try running: python phase2_environment/setup_environment.py")

def run_phase3():
    """Start Phase 3: Data Collection."""
    print("\n📊 PHASE 3: DATA COLLECTION & PREPROCESSING")
    print("=" * 50)
    print()
    print("🔄 This phase involves:")
    print("   • Downloading training datasets")
    print("   • Text cleaning and preprocessing")
    print("   • Building custom tokenizers")
    print("   • Creating efficient data loaders")
    print()
    print("💡 To start:")
    print("   python phase3_data/data_pipeline.py")
    print()
    
    if input("Would you like to run the data pipeline now? (y/n): ").lower() == 'y':
        try:
            import subprocess
            subprocess.run([sys.executable, "phase3_data/data_pipeline.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running data pipeline: {e}")
        except FileNotFoundError:
            print("❌ Data pipeline script not found.")

def run_phase4():
    """Start Phase 4: Model Implementation."""
    print("\n🏗️  PHASE 4: TRANSFORMER MODEL")
    print("=" * 50)
    print()
    print("🔧 This phase covers:")
    print("   • Complete transformer architecture")
    print("   • Multi-head attention implementation")
    print("   • Positional encoding and embeddings")
    print("   • Causal language modeling")
    print()
    print("💡 To start:")
    print("   python phase4_model/transformer_model.py")
    print()
    
    if input("Would you like to test the transformer model? (y/n): ").lower() == 'y':
        try:
            subprocess.run([sys.executable, "phase4_model/transformer_model.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running transformer model: {e}")
        except FileNotFoundError:
            print("❌ Transformer model script not found.")

def run_quick_demo():
    """Run a quick demo of the mini transformer."""
    print("\n🎪 QUICK DEMO: MINI TRANSFORMER")
    print("=" * 50)
    print()
    
    try:
        # Import required modules
        import torch
        import torch.nn as nn
        from phase4_model.transformer_model import create_gpt_model
        
        print("🤖 Creating a mini GPT model...")
        
        # Create a small model for demo
        config = {
            'vocab_size': 1000,
            'd_model': 256,
            'num_heads': 4,
            'num_layers': 2,
            'd_ff': 512,
            'max_seq_length': 128,
            'dropout': 0.1
        }
        
        model = create_gpt_model(config)
        print(f"✅ Model created with {model.count_parameters():,} parameters")
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        
        print(f"🔄 Running forward pass...")
        print(f"   Input shape: {input_ids.shape}")
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs['logits']
        
        print(f"   Output shape: {logits.shape}")
        print(f"✅ Demo completed successfully!")
        
        # Show model architecture
        print(f"\n📊 Model Summary:")
        print(f"   Vocabulary Size: {config['vocab_size']:,}")
        print(f"   Model Dimension: {config['d_model']}")
        print(f"   Attention Heads: {config['num_heads']}")
        print(f"   Layers: {config['num_layers']}")
        print(f"   Parameters: {model.count_parameters():,}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure all dependencies are installed and Phase 4 files exist.")

def show_help():
    """Show help and documentation."""
    print("\n📖 HELP & DOCUMENTATION")
    print("=" * 50)
    print()
    print("📚 Key Resources:")
    print("   • README.md - Project overview and setup instructions")
    print("   • phase*/README.md - Phase-specific documentation")
    print("   • configs/ - Configuration files and examples")
    print("   • shared/utils.py - Utility functions and helpers")
    print()
    print("🔗 Important Links:")
    print("   • PyTorch Documentation: https://pytorch.org/docs/")
    print("   • Attention Paper: 'Attention is All You Need' (Vaswani et al.)")
    print("   • GPT Papers: OpenAI's GPT-1, GPT-2, GPT-3 papers")
    print()
    print("💡 Getting Help:")
    print("   • Check error messages carefully")
    print("   • Review the phase README files")
    print("   • Ensure all requirements are installed")
    print("   • Start with smaller examples before scaling up")
    print()
    print("🐛 Common Issues:")
    print("   • CUDA out of memory: Reduce batch size or model size")
    print("   • Import errors: Run 'pip install -r requirements.txt'")
    print("   • Slow training: Check if GPU is being used")

def main():
    """Main application loop."""
    print_welcome()
    
    # Check requirements
    if not check_requirements():
        return
    
    while True:
        choice = show_phase_menu()
        
        if choice == "1":
            run_phase1()
        elif choice == "2":
            run_phase2()
        elif choice == "3":
            run_phase3()
        elif choice == "4":
            run_phase4()
        elif choice == "5":
            print("\n🚀 PHASE 5: TRAINING")
            print("Phase 5 implementation coming soon!")
            print("For now, you can experiment with the models from Phase 4.")
        elif choice == "env":
            run_phase2()  # Environment check is part of Phase 2
        elif choice == "demo":
            run_quick_demo()
        elif choice == "help":
            show_help()
        elif choice == "quit":
            print("\n👋 Thanks for using AI Assistant From Scratch!")
            print("Happy coding and building! 🚀")
            break
        
        print("\n" + "="*50)
        input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for building AI from scratch!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please check the error message and try again.")
