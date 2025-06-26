"""
Complete Training and Deployment Pipeline for Basic AI Assistant

This script handles the complete pipeline:
1. Generate training data
2. Train the basic AI model
3. Test the model
4. Launch the web interface
"""

import subprocess
import sys
import time
from pathlib import Path
import os

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    packages = [
        'torch',
        'numpy',
        'tqdm',
        'flask',
        'flask-cors'
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Warning: Could not install {package}")

def generate_training_data():
    """Generate basic training data."""
    print("\n🎯 Generating training data...")
    
    try:
        subprocess.run([sys.executable, 'basic_data_generator.py'], 
                      cwd=Path(__file__).parent, check=True)
        print("✅ Training data generated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate training data: {e}")
        return False

def train_model():
    """Train the basic AI model."""
    print("\n🧠 Training the AI model...")
    print("This may take a few minutes...")
    
    try:
        subprocess.run([sys.executable, 'train_basic_model.py'], 
                      cwd=Path(__file__).parent, check=True)
        print("✅ Model training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to train model: {e}")
        return False

def test_model():
    """Test the trained model."""
    print("\n🧪 Testing the AI model...")
    
    try:
        # Run a quick test
        from basic_ai_interface import BasicAIAssistant
        
        model_path = Path(__file__).parent.parent / "outputs" / "basic_ai_model.pth"
        assistant = BasicAIAssistant(str(model_path) if model_path.exists() else None)
        
        # Test cases
        test_cases = [
            "Hello",
            "What is 5 + 3?",
            "How are you?"
        ]
        
        print("Test results:")
        for test in test_cases:
            response = assistant.generate_response(test)
            print(f"  👤 Human: {test}")
            print(f"  🤖 Assistant: {response}")
            print()
        
        return True
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def launch_web_interface():
    """Launch the web interface."""
    print("\n🌐 Launching web interface...")
    print("The web interface will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, 'web_api.py'], 
                      cwd=Path(__file__).parent, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down web interface...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch web interface: {e}")

def main():
    """Main pipeline function."""
    print("🚀 Basic AI Assistant Training & Deployment Pipeline")
    print("=" * 60)
    
    # Change to the correct directory
    os.chdir(Path(__file__).parent)
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Generate training data
    if not generate_training_data():
        print("❌ Pipeline failed at data generation step")
        return
    
    # Step 3: Train model
    print("\n" + "="*50)
    print("🎓 TRAINING PHASE")
    print("="*50)
    
    user_choice = input("\n📝 Do you want to train the model? (y/n): ").lower().strip()
    
    if user_choice in ['y', 'yes']:
        if not train_model():
            print("❌ Pipeline failed at training step")
            print("💡 You can still use the rule-based assistant")
    else:
        print("⏭️  Skipping training - will use rule-based responses")
    
    # Step 4: Test model
    print("\n" + "="*50)
    print("🧪 TESTING PHASE")
    print("="*50)
    
    test_model()
    
    # Step 5: Launch web interface
    print("\n" + "="*50)
    print("🌐 WEB INTERFACE")
    print("="*50)
    
    user_choice = input("\n🌐 Do you want to launch the web interface? (y/n): ").lower().strip()
    
    if user_choice in ['y', 'yes']:
        launch_web_interface()
    else:
        print("✅ Pipeline completed! You can run the web interface later with:")
        print(f"   python {Path(__file__).parent / 'web_api.py'}")
    
    print("\n🎉 All done! Your AI assistant is ready to use!")

if __name__ == "__main__":
    main()
