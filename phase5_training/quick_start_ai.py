"""
Quick Start: Basic AI Assistant Training

This script provides a quick way to train and test a basic AI assistant
without complex dependencies. Perfect for getting started!
"""

import json
import random
import re
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def create_basic_training_data():
    """Create basic training data for the AI assistant."""
    print("📝 Creating basic training data...")
    
    # Basic conversation data
    conversations = [
        # Greetings
        {"input": "Hello", "output": "Hello! How can I help you today?"},
        {"input": "Hi", "output": "Hi there! What can I do for you?"},
        {"input": "Hey", "output": "Hey! How are you doing?"},
        {"input": "Good morning", "output": "Good morning! Hope you're having a great day!"},
        
        # Basic questions
        {"input": "How are you?", "output": "I'm doing well, thank you for asking! How are you?"},
        {"input": "What's your name?", "output": "I'm Glitch Bot, created to help you with various tasks!"},
        {"input": "What can you do?", "output": "I can help with basic math, answer questions, and have conversations!"},
        
        # Politeness
        {"input": "Thank you", "output": "You're very welcome! Happy to help!"},
        {"input": "Thanks", "output": "You're welcome! Is there anything else I can help with?"},
        {"input": "Goodbye", "output": "Goodbye! Have a wonderful day!"},
        {"input": "Bye", "output": "Bye! Take care!"},
        
        # Math problems
        {"input": "What is 2 + 3?", "output": "2 + 3 = 5"},
        {"input": "What is 5 - 2?", "output": "5 - 2 = 3"},
        {"input": "What is 4 * 6?", "output": "4 × 6 = 24"},
        {"input": "What is 10 / 2?", "output": "10 ÷ 2 = 5"},
        {"input": "Calculate 7 + 8", "output": "7 + 8 = 15"},
        {"input": "What's 9 * 3?", "output": "9 × 3 = 27"},            # General knowledge
            {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
            {"input": "How many days in a week?", "output": "There are 7 days in a week."},
            {"input": "What color is the sky?", "output": "The sky is typically blue during the day."},
            {"input": "What is water made of?", "output": "Water is made of hydrogen and oxygen (H2O)."},
            
            # Date and time queries (these will be handled dynamically)
            {"input": "What day is today?", "output": "I can tell you the current date!"},
            {"input": "What's today's date?", "output": "I can get the current date for you!"},
            {"input": "What time is it?", "output": "I can tell you the current time!"},
            {"input": "What day of the week is it?", "output": "I can tell you the current day!"},
            {"input": "What month is it?", "output": "I can tell you the current month!"},
            {"input": "What year is it?", "output": "I can tell you the current year!"},
        ]
    
    # Generate more math problems
    math_problems = []
    for _ in range(100):
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        
        operations = [
            ('+', a + b, 'plus'),
            ('-', a - b, 'minus'),
            ('*', a * b, 'times'),
        ]
        
        op, result, word_op = random.choice(operations)
        
        # Different question formats
        formats = [
            f"What is {a} {op} {b}?",
            f"Calculate {a} {op} {b}",
            f"What's {a} {op} {b}?",
            f"What is {a} {word_op} {b}?",
        ]
        
        question = random.choice(formats)
        answer = f"{a} {op} {b} = {result}"
        
        math_problems.append({"input": question, "output": answer})
    
    # Combine all data
    all_data = conversations * 20 + math_problems  # Repeat conversations for more training
    random.shuffle(all_data)
    
    return all_data

class SimpleAIAssistant:
    """A simple rule-based AI assistant for basic tasks."""
    
    def __init__(self):
        self.responses = {}
        self.load_responses()
    
    def load_responses(self):
        """Load pre-defined responses."""
        training_data = create_basic_training_data()
        
        # Store exact matches
        for item in training_data:
            self.responses[item['input'].lower()] = item['output']
    
    def solve_math(self, prompt: str) -> str:
        """Solve basic math problems."""
        # Remove common words
        clean_prompt = re.sub(r'\b(what|is|equals?|calculate|solve)\b', '', prompt, flags=re.IGNORECASE)
        
        # Look for basic math operations
        patterns = [
            (r'(\d+)\s*\+\s*(\d+)', lambda a, b: f"{a} + {b} = {a + b}"),
            (r'(\d+)\s*\-\s*(\d+)', lambda a, b: f"{a} - {b} = {a - b}"),
            (r'(\d+)\s*\*\s*(\d+)', lambda a, b: f"{a} × {b} = {a * b}"),
            (r'(\d+)\s*x\s*(\d+)', lambda a, b: f"{a} × {b} = {a * b}"),
            (r'(\d+)\s*/\s*(\d+)', lambda a, b: f"{a} ÷ {b} = {a / b:.1f}" if b != 0 else "Cannot divide by zero!"),
            (r'(\d+)\s*times\s*(\d+)', lambda a, b: f"{a} × {b} = {a * b}"),
            (r'(\d+)\s*plus\s*(\d+)', lambda a, b: f"{a} + {b} = {a + b}"),
            (r'(\d+)\s*minus\s*(\d+)', lambda a, b: f"{a} - {b} = {a - b}"),
        ]
        
        for pattern, calculator in patterns:
            match = re.search(pattern, clean_prompt, re.IGNORECASE)
            if match:
                try:
                    a = int(match.group(1))
                    b = int(match.group(2))
                    return calculator(a, b)
                except (ValueError, ZeroDivisionError):
                    continue
        
        return None
    
    def get_current_date_time(self, query_type: str) -> str:
        """Get current date/time information."""
        now = datetime.now()
        
        if query_type == 'date':
            return now.strftime("Today is %A, %B %d, %Y")
        elif query_type == 'time':
            return now.strftime("The current time is %I:%M %p")
        elif query_type == 'day':
            return f"Today is {now.strftime('%A')}"
        elif query_type == 'month':
            return f"We're currently in {now.strftime('%B')}"
        elif query_type == 'year':
            return f"The current year is {now.year}"
        elif query_type == 'datetime':
            return now.strftime("It's currently %A, %B %d, %Y at %I:%M %p")
        
        return "I can tell you the current date and time!"
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response to the user's input."""
        prompt_lower = prompt.lower().strip()
        
        # Date and time queries
        if any(phrase in prompt_lower for phrase in ['what day today', 'what\'s today', 'today\'s date', 'what date today']):
            return self.get_current_date_time('date')
        
        if any(phrase in prompt_lower for phrase in ['what time', 'current time', 'time now']):
            return self.get_current_date_time('time')
        
        if any(phrase in prompt_lower for phrase in ['what day of week', 'day of week', 'what day it']):
            return self.get_current_date_time('day')
        
        if any(phrase in prompt_lower for phrase in ['what month', 'current month']):
            return self.get_current_date_time('month')
        
        if any(phrase in prompt_lower for phrase in ['what year', 'current year']):
            return self.get_current_date_time('year')
        
        if any(phrase in prompt_lower for phrase in ['date and time', 'time and date']):
            return self.get_current_date_time('datetime')
        
        # Check for exact matches first
        if prompt_lower in self.responses:
            return self.responses[prompt_lower]
        
        # Try math solving
        math_result = self.solve_math(prompt)
        if math_result:
            return math_result
        
        # Pattern matching for common phrases
        if any(greeting in prompt_lower for greeting in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?"
        
        if 'how are you' in prompt_lower:
            return "I'm doing well, thank you for asking! How are you?"
        
        if any(name in prompt_lower for name in ['name', 'who are you']):
            return "I'm Glitch Bot, created to help you with basic questions and math!"
        
        if 'thank' in prompt_lower:
            return "You're welcome! Happy to help!"
        
        if any(bye in prompt_lower for bye in ['goodbye', 'bye', 'see you']):
            return "Goodbye! Have a great day!"
        
        if 'help' in prompt_lower:
            return "I can help with basic math problems, tell you the current date and time, and answer simple questions. Try asking me 'What day is today?' or 'What is 5 + 3?'"
        
        if 'what can you do' in prompt_lower:
            return "I can help with basic math, tell you the current date and time, and answer simple questions. Try asking me 'What time is it?' or a math problem!"
        
        # Default response
        return "I'm still learning! I can help with basic math problems like '5 + 3', tell you the current date/time, and answer simple questions. What would you like to know?"
    
    def chat(self):
        """Start an interactive chat session."""
        print("🤖 Glitch Bot")
        print("I can help with basic math and simple questions!")
        print("Type 'quit' to exit")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = self.generate_response(user_input)
                print(f"🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

def test_assistant():
    """Test the AI assistant with various inputs."""
    assistant = SimpleAIAssistant()
    
    test_cases = [
        "Hello",
        "What is 5 + 3?",
        "How are you?",
        "What is 12 * 4?",
        "Calculate 20 - 7",
        "What's your name?",
        "Thank you",
        "What is 15 divided by 3?",
        "What day is today?",
        "What time is it?",
        "What year is it?",
        "Help me",
        "Goodbye"
    ]
    
    print("🧪 Testing Glitch Bot:")
    print("=" * 40)
    
    for test in test_cases:
        response = assistant.generate_response(test)
        print(f"👤 Human: {test}")
        print(f"🤖 Assistant: {response}")
        print()

def main():
    """Main function."""
    print("🚀 Quick Start: Glitch Bot Training")
    print("=" * 50)
    
    print("\n📊 Creating training data...")
    training_data = create_basic_training_data()
    print(f"✅ Created {len(training_data)} training examples")
    
    print("\n🧪 Testing the assistant...")
    test_assistant()
    
    print("\n" + "="*50)
    print("🎯 Interactive Chat")
    print("="*50)
    
    # Ask user if they want to chat
    choice = input("\nWould you like to start an interactive chat? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        assistant = SimpleAIAssistant()
        assistant.chat()
    else:
        print("✅ All done! You can run this script again to chat with the assistant.")

if __name__ == "__main__":
    main()
