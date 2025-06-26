"""
Quick Test: AI Assistant with Date/Time Support

This script tests the enhanced AI assistant that can tell you the current date and time.
"""

import sys
from pathlib import Path

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

from quick_start_ai import SimpleAIAssistant

def quick_test():
    """Quick test of the date/time functionality."""
    print("🚀 Testing AI Assistant with Date/Time Support")
    print("=" * 50)
    
    assistant = SimpleAIAssistant()
    
    # Test date/time specific queries
    date_time_tests = [
        "What day is today?",
        "What time is it?",
        "What's today's date?",
        "What year is it?",
        "What month is it?",
        "What day of the week is it?",
        "What's the date and time?",
    ]
    
    print("📅 Testing Date/Time Queries:")
    print("-" * 30)
    
    for test in date_time_tests:
        response = assistant.generate_response(test)
        print(f"👤 Human: {test}")
        print(f"🤖 Assistant: {response}")
        print()
    
    print("🧮 Testing Math and Other Queries:")
    print("-" * 35)
    
    other_tests = [
        "Hello",
        "What is 10 + 5?",
        "How are you?",
        "What can you do?"
    ]
    
    for test in other_tests:
        response = assistant.generate_response(test)
        print(f"👤 Human: {test}")
        print(f"🤖 Assistant: {response}")
        print()

def main():
    """Main function."""
    quick_test()
    
    print("🎯 Interactive Chat with Date/Time Support")
    print("=" * 45)
    
    choice = input("\nWould you like to start an interactive chat? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        assistant = SimpleAIAssistant()
        assistant.chat()
    else:
        print("✅ Test completed! The AI can now tell you the current date and time!")
        print("\n💡 Try these commands in the web interface:")
        print("   - 'What day is today?'")
        print("   - 'What time is it?'") 
        print("   - 'What's today's date?'")
        print("   - 'What year is it?'")

if __name__ == "__main__":
    main()
