#!/usr/bin/env python3
"""
Quick setup test for AI Assistant project
Tests if all dependencies and modules are working correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("🔧 Testing Python environment setup...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import tqdm
        print(f"✅ tqdm: {tqdm.__version__}")
    except ImportError as e:
        print(f"❌ tqdm import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test if project structure is correct."""
    print("\n📁 Testing project structure...")
    
    required_dirs = [
        "phase1_fundamentals",
        "phase2_environment", 
        "phase3_data",
        "phase4_model",
        "phase5_training",
        "phase6_scaling",
        "phase7_application",
        "phase8_features",
        "shared",
        "configs"
    ]
    
    project_root = Path(__file__).parent
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ (missing)")
            return False
    
    return True

def test_custom_modules():
    """Test if our custom modules can be imported."""
    print("\n🧩 Testing custom modules...")
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        from shared.utils import setup_logging, get_device
        print("✅ shared.utils imported successfully")
        
        # Test basic functionality
        device = get_device()
        print(f"   Detected device: {device}")
        
        logger = setup_logging("INFO")
        logger.info("✅ Logging system working")
        
    except ImportError as e:
        print(f"❌ shared.utils import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ shared.utils functionality failed: {e}")
        return False
    
    try:
        from phase4_model.transformer_model import create_gpt_model
        print("✅ phase4_model.transformer_model imported successfully")
        
        # Test model creation
        model_config = {
            'vocab_size': 1000,
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'd_ff': 512,
            'max_seq_length': 256,
            'dropout': 0.1
        }
        
        model = create_gpt_model(model_config)
        print(f"   ✅ Test model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    except ImportError as e:
        print(f"❌ transformer_model import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ model creation failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 AI Assistant Project Setup Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_imports()
    all_tests_passed &= test_project_structure()
    all_tests_passed &= test_custom_modules()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Your AI Assistant project is ready!")
        print("\n🔥 Next steps:")
        print("   1. Run: python phase5_training/train_model.py")
        print("   2. Or explore: python start_ai_assistant.py")
        print("   3. Open notebooks: jupyter lab")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
