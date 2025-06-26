"""
Basic AI Assistant Interface

This script provides a simple interface to interact with the trained basic AI model.
It can handle basic conversations and math problems.
"""

import torch
from pathlib import Path
from typing import Optional
import re

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from phase4_model.transformer_model import create_gpt_model


class BasicAIAssistant:
    """Simple AI assistant for basic conversations and math."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            # Use rule-based responses if no model is available
            self.model = None
            self.tokenizer = None
            print("No trained model found. Using rule-based responses.")
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.tokenizer = checkpoint['tokenizer']
        
        # Recreate model
        self.model = create_gpt_model(self.config)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def generate_response(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate a response using the trained model."""
        if not self.model:
            return self.rule_based_response(prompt)
        
        # Format prompt
        formatted_prompt = f"Human: {prompt}\nAssistant:"
        
        # Encode prompt
        input_ids = self.tokenizer.encode(formatted_prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                if len(generated) >= self.config['max_seq_length']:
                    break
                
                # Get model output
                outputs = self.model(input_tensor)
                
                # Get next token probabilities
                next_token_logits = outputs[0, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1).item()
                
                # Check for natural stopping
                if next_token == 0:  # Padding token
                    break
                
                generated_char = self.tokenizer.id_to_char.get(next_token, '')
                if generated_char == '\n' and 'Assistant:' in self.tokenizer.decode(generated[-20:]):
                    break
                
                generated.append(next_token)
                
                # Update input tensor
                input_tensor = torch.tensor([generated], dtype=torch.long).to(self.device)
                
                # Truncate if too long
                if input_tensor.size(1) > self.config['max_seq_length']:
                    input_tensor = input_tensor[:, -self.config['max_seq_length']:]
        
        # Decode and extract response
        full_text = self.tokenizer.decode(generated)
        
        # Extract just the assistant's response
        if "Assistant:" in full_text:
            response = full_text.split("Assistant:")[-1].strip()
            # Clean up the response
            response = response.split("Human:")[0].strip()  # Remove any follow-up human text
            return response if response else "I'm not sure how to respond to that."
        
        return "I'm not sure how to respond to that."
    
    def rule_based_response(self, prompt: str) -> str:
        """Provide rule-based responses for basic questions and math."""
        prompt = prompt.lower().strip()
        
        # Greetings
        if any(greeting in prompt for greeting in ['hello', 'hi', 'hey']):
            return "Hello! How can I help you today?"
        
        # How are you
        if 'how are you' in prompt:
            return "I'm doing well, thank you for asking! How are you?"
        
        # Math operations
        math_result = self.solve_math(prompt)
        if math_result:
            return math_result
        
        # Basic questions
        if 'what is your name' in prompt or 'who are you' in prompt:
            return "I'm an AI assistant created to help you with basic questions and math!"
        
        if 'what can you do' in prompt:
            return "I can help with basic math, answer simple questions, and have conversations. Try asking me a math problem!"
        
        # Thank you
        if any(thanks in prompt for thanks in ['thank you', 'thanks']):
            return "You're welcome! Happy to help!"
        
        # Goodbye
        if any(bye in prompt for bye in ['goodbye', 'bye', 'see you']):
            return "Goodbye! Have a great day!"
        
        # Help
        if 'help' in prompt:
            return "I'm here to help! You can ask me math questions like '5 + 3' or basic questions about various topics."
        
        # Default response
        return "I'm still learning! I can help with basic math problems and simple questions. Try asking me something like 'What is 5 + 3?' or 'How are you?'"
    
    def solve_math(self, prompt: str) -> Optional[str]:
        """Solve basic math problems."""
        # Remove common words
        clean_prompt = re.sub(r'\b(what|is|equals?|calculate|solve)\b', '', prompt, flags=re.IGNORECASE)
        
        # Look for basic math operations
        patterns = [
            r'(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)',  # Addition
            r'(\d+\.?\d*)\s*\-\s*(\d+\.?\d*)',  # Subtraction
            r'(\d+\.?\d*)\s*\*\s*(\d+\.?\d*)',  # Multiplication
            r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)',   # Multiplication (x)
            r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)',   # Division
            r'(\d+\.?\d*)\s*times\s*(\d+\.?\d*)',  # Word multiplication
            r'(\d+\.?\d*)\s*plus\s*(\d+\.?\d*)',   # Word addition
            r'(\d+\.?\d*)\s*minus\s*(\d+\.?\d*)',  # Word subtraction
            r'(\d+\.?\d*)\s*divided\s*by\s*(\d+\.?\d*)',  # Word division
        ]
        
        operations = ['+', '-', '*', 'x', '/', 'times', 'plus', 'minus', 'divided by']
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, clean_prompt, re.IGNORECASE)
            if match:
                try:
                    a = float(match.group(1))
                    b = float(match.group(2))
                    op = '?'  # Initialize op variable
                    
                    if i in [0, 5]:  # Addition
                        result = a + b
                        op = '+'
                    elif i in [1, 7]:  # Subtraction
                        result = a - b
                        op = '-'
                    elif i in [2, 3, 6]:  # Multiplication
                        result = a * b
                        op = '×'
                    elif i in [4, 8]:  # Division
                        if b == 0:
                            return "Cannot divide by zero!"
                        result = a / b
                        op = '÷'
                    
                    # Format result
                    if result == int(result):
                        result = int(result)
                    else:
                        result = round(result, 2)
                    
                    # Format the numbers for display
                    a_display = int(a) if a == int(a) else a
                    b_display = int(b) if b == int(b) else b
                    
                    return f"{a_display} {op} {b_display} = {result}"
                    
                except (ValueError, ZeroDivisionError):
                    continue
        
        return None
    
    def chat(self):
        """Interactive chat interface."""
        print("🤖 Basic AI Assistant")
        print("Type 'quit' to exit, 'help' for help")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Generate response
                response = self.generate_response(user_input)
                print(f"🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the assistant."""
    # Look for trained model
    model_path = Path(__file__).parent.parent / "outputs" / "basic_ai_model.pth"
    
    # Initialize assistant
    assistant = BasicAIAssistant(str(model_path) if model_path.exists() else None)
    
    # Test some examples
    print("🧪 Testing the AI Assistant:")
    test_cases = [
        "Hello",
        "What is 5 + 3?",
        "How are you?",
        "What is 12 * 4?",
        "What is 20 divided by 4?",
        "Thank you",
    ]
    
    for test in test_cases:
        response = assistant.generate_response(test)
        print(f"👤 Human: {test}")
        print(f"🤖 Assistant: {response}")
        print()
    
    # Start interactive chat
    print("\n" + "="*50)
    assistant.chat()


if __name__ == "__main__":
    main()
