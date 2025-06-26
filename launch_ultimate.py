#!/usr/bin/env python3
"""
Ultimate AI Assistant Launcher
Premium interface launcher with multiple options
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print the startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        ✨ ULTIMATE AI ASSISTANT LAUNCHER ✨                  ║
    ║                                                               ║
    ║        🎨 Pixel-Perfect UI Design                            ║
    ║        🚀 Premium User Experience                            ║
    ║        💎 Claude + ChatGPT + Gemini Inspired                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['gradio', 'torch', 'transformers']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ All dependencies installed!")
    
    return True

def launch_interface(interface_type="ultimate"):
    """Launch the selected interface"""
    base_path = Path(__file__).parent
    
    interfaces = {
        "ultimate": base_path / "ultimate_ai_assistant.py",
        "premium": base_path / "premium_ai_assistant.py",
        "standard": base_path / "ai_assistant_app.py",
        "ultra": base_path / "ultra_premium_assistant.py"
    }
    
    if interface_type not in interfaces:
        print(f"❌ Unknown interface type: {interface_type}")
        return False
    
    interface_file = interfaces[interface_type]
    
    if not interface_file.exists():
        print(f"❌ Interface file not found: {interface_file}")
        return False
    
    print(f"🚀 Launching {interface_type.title()} AI Assistant...")
    print(f"📁 Using file: {interface_file}")
    
    try:
        # Change to the correct directory
        os.chdir(interface_file.parent)
        
        # Launch the interface
        subprocess.run([sys.executable, str(interface_file)], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch interface: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return True
    
    return True

def main():
    """Main launcher function"""
    print_banner()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed!")
        return
    
    # Show interface options
    print("\n🎨 Available Interfaces:")
    print("1. ✨ Ultimate AI Assistant (Recommended)")
    print("2. 💎 Premium AI Assistant")
    print("3. 🔥 Ultra Premium Assistant")
    print("4. 📱 Standard AI Assistant")
    print("5. 🚪 Exit")
    
    try:
        choice = input("\n👉 Select interface (1-5): ").strip()
        
        interface_map = {
            "1": "ultimate",
            "2": "premium", 
            "3": "ultra",
            "4": "standard",
            "5": "exit"
        }
        
        if choice == "5":
            print("👋 Goodbye!")
            return
        
        if choice in interface_map:
            interface_type = interface_map[choice]
            print(f"\n🎯 Selected: {interface_type.title()} Interface")
            
            # Add delay for effect
            print("🔄 Initializing...")
            time.sleep(1)
            
            # Launch the interface
            launch_interface(interface_type)
        else:
            print("❌ Invalid selection. Please choose 1-5.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
