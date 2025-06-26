"""
Basic Training Data Generator for AI Assistant

This module generates training data specifically for basic questions and math operations
to train the AI assistant for fundamental conversational and computational tasks.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any


class BasicTrainingDataGenerator:
    """Generate basic training data for fundamental AI assistant capabilities."""
    
    def __init__(self):
        self.math_operations = ['+', '-', '*', '/']
        self.conversation_templates = self._load_conversation_templates()
        self.math_templates = self._load_math_templates()
        self.general_qa = self._load_general_qa()
    
    def _load_conversation_templates(self) -> List[Dict[str, str]]:
        """Load basic conversation templates."""
        return [
            # Greetings
            {"input": "Hello", "output": "Hello! How can I help you today?"},
            {"input": "Hi", "output": "Hi there! What can I do for you?"},
            {"input": "Hey", "output": "Hey! How are you doing?"},
            {"input": "Good morning", "output": "Good morning! Hope you're having a great day!"},
            {"input": "Good afternoon", "output": "Good afternoon! How can I assist you?"},
            {"input": "Good evening", "output": "Good evening! What can I help you with?"},
            
            # Basic questions
            {"input": "How are you?", "output": "I'm doing well, thank you for asking! How are you?"},
            {"input": "What's your name?", "output": "I'm an AI assistant created to help you with various tasks. You can call me AI Assistant!"},
            {"input": "What can you do?", "output": "I can help with basic math, answer questions, have conversations, and assist with various tasks. What would you like help with?"},
            {"input": "Who made you?", "output": "I was created as an AI assistant to help people with their questions and tasks."},
            
            # Politeness
            {"input": "Thank you", "output": "You're very welcome! Happy to help!"},
            {"input": "Thanks", "output": "You're welcome! Is there anything else I can help with?"},
            {"input": "Please help me", "output": "Of course! I'd be happy to help. What do you need assistance with?"},
            {"input": "Goodbye", "output": "Goodbye! Have a wonderful day!"},
            {"input": "Bye", "output": "Bye! Take care!"},
            
            # Help requests
            {"input": "I need help", "output": "I'm here to help! What do you need assistance with?"},
            {"input": "Can you help me?", "output": "Absolutely! I'd be happy to help. What can I do for you?"},
            {"input": "I have a question", "output": "Great! I'm here to answer questions. What would you like to know?"},
        ]
    
    def _load_math_templates(self) -> List[Dict[str, Any]]:
        """Load math problem templates."""
        templates = []
        
        # Basic arithmetic templates
        arithmetic_templates = [
            "What is {a} {op} {b}?",
            "Calculate {a} {op} {b}",
            "What's {a} {op} {b}?",
            "Can you solve {a} {op} {b}?",
            "{a} {op} {b} = ?",
            "What does {a} {op} {b} equal?",
        ]
        
        return [{"templates": arithmetic_templates, "type": "arithmetic"}]
    
    def _load_general_qa(self) -> List[Dict[str, str]]:
        """Load general knowledge Q&A pairs."""
        return [
            # Basic facts
            {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
            {"input": "What is 2+2?", "output": "2 + 2 = 4"},
            {"input": "What color is the sky?", "output": "The sky is typically blue during the day."},
            {"input": "How many days in a week?", "output": "There are 7 days in a week."},
            {"input": "What is water made of?", "output": "Water is made of hydrogen and oxygen (H2O)."},
            {"input": "What is the largest planet?", "output": "Jupiter is the largest planet in our solar system."},
            {"input": "What is 10 x 10?", "output": "10 × 10 = 100"},
            
            # Basic science
            {"input": "What is gravity?", "output": "Gravity is the force that pulls objects toward each other, especially toward the Earth."},
            {"input": "What makes plants green?", "output": "Chlorophyll makes plants green. It helps them make food from sunlight."},
            {"input": "How many continents are there?", "output": "There are 7 continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia."},
            
            # Time and dates
            {"input": "How many hours in a day?", "output": "There are 24 hours in a day."},
            {"input": "How many minutes in an hour?", "output": "There are 60 minutes in an hour."},
            {"input": "How many seconds in a minute?", "output": "There are 60 seconds in a minute."},
            {"input": "How many months in a year?", "output": "There are 12 months in a year."},
            
            # Basic conversational AI responses
            {"input": "Are you human?", "output": "No, I'm an AI assistant designed to help answer questions and have conversations."},
            {"input": "Do you have feelings?", "output": "I don't have feelings like humans do, but I'm designed to be helpful and friendly!"},
            {"input": "What's the weather like?", "output": "I don't have access to current weather data, but you can check a weather app or website for current conditions in your area."},
            
            # Date and time queries
            {"input": "What day is today?", "output": "I can tell you the current date from your device when you ask me!"},
            {"input": "What's today's date?", "output": "I can get the current date from your device!"},
            {"input": "What time is it?", "output": "I can tell you the current time from your device!"},
            {"input": "What's the current time?", "output": "I can check the current time on your device for you!"},
            {"input": "What day of the week is it?", "output": "I can tell you what day of the week it is from your device!"},
            {"input": "What month is it?", "output": "I can tell you the current month!"},
            {"input": "What year is it?", "output": "I can tell you the current year!"},
            {"input": "What's the date and time?", "output": "I can get both the current date and time from your device!"},
        ]
    
    def generate_math_problems(self, num_problems: int = 1000) -> List[Dict[str, str]]:
        """Generate math problems and their solutions."""
        problems = []
        
        for _ in range(num_problems):
            # Random numbers for problems
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            
            # Choose operation
            op = random.choice(self.math_operations)
            op_word = ""  # Initialize op_word
            
            # Calculate answer
            if op == '+':
                answer = a + b
                op_word = "plus"
            elif op == '-':
                answer = a - b
                op_word = "minus"
            elif op == '*':
                answer = a * b
                op_word = "times"
            elif op == '/':
                # Ensure clean division
                answer = a / b
                if answer != int(answer):
                    answer = round(answer, 2)
                op_word = "divided by"
            
            # Choose random template
            template = random.choice(self.math_templates[0]["templates"])
            question = template.format(a=a, b=b, op=op)
            
            # Create response
            if op == '/':
                response = f"{a} {op} {b} = {answer}"
            else:
                response = f"{a} {op} {b} = {int(answer)}"
            
            problems.append({
                "input": question,
                "output": response
            })
            
            # Also add word problem version
            if random.random() < 0.3:  # 30% chance
                word_question = f"What is {a} {op_word} {b}?"
                problems.append({
                    "input": word_question,
                    "output": response
                })
        
        return problems
    
    def generate_training_data(self, num_math_problems: int = 2000) -> List[Dict[str, str]]:
        """Generate complete training dataset."""
        training_data = []
        
        # Add conversation templates (repeat them multiple times for training)
        for _ in range(10):  # Repeat 10 times
            training_data.extend(self.conversation_templates)
        
        # Add general Q&A (repeat multiple times)
        for _ in range(15):  # Repeat 15 times
            training_data.extend(self.general_qa)
        
        # Add math problems
        training_data.extend(self.generate_math_problems(num_math_problems))
        
        # Shuffle the data
        random.shuffle(training_data)
        
        return training_data
    
    def save_training_data(self, output_path: str, num_math_problems: int = 2000):
        """Generate and save training data to JSONL format."""
        training_data = self.generate_training_data(num_math_problems)
        
        # Create output directory
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in training_data:
                # Format for training (input -> output)
                training_item = {
                    "text": f"Human: {item['input']}\nAssistant: {item['output']}\n"
                }
                f.write(json.dumps(training_item) + '\n')
        
        print(f"Saved {len(training_data)} training examples to {output_path}")
        return len(training_data)


def main():
    """Generate basic training data for the AI assistant."""
    generator = BasicTrainingDataGenerator()
    
    # Generate and save training data
    base_dir = Path(__file__).parent.parent / "data" / "processed"
    
    # Generate main training set
    num_examples = generator.save_training_data(
        base_dir / "basic_training.jsonl", 
        num_math_problems=3000
    )
    
    # Generate smaller validation set
    validation_data = generator.generate_training_data(num_math_problems=500)
    
    with open(base_dir / "basic_validation.jsonl", 'w', encoding='utf-8') as f:
        for item in validation_data:
            training_item = {
                "text": f"Human: {item['input']}\nAssistant: {item['output']}\n"
            }
            f.write(json.dumps(training_item) + '\n')
    
    print(f"Generated basic training dataset with {num_examples} examples")
    print(f"Generated validation dataset with {len(validation_data)} examples")
    print("\nDataset includes:")
    print("- Basic conversations and greetings")
    print("- Math problems (addition, subtraction, multiplication, division)")
    print("- General knowledge Q&A")
    print("- Politeness and help requests")


if __name__ == "__main__":
    main()
