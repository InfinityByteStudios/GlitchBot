"""
AI Assistant Application Interface

This module implements the user-facing application for the AI assistant,
including conversation management, prompt handling, and various interface options.
This is Phase 7: Build Your AI Assistant Application.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn.functional as F
    import gradio as gr
    import streamlit as st
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Import our custom modules
    from shared.utils import setup_logging, get_device
    from phase4_model.transformer_model import create_gpt_model
    from phase3_data.data_pipeline import SimpleTokenizer
    
except ImportError as e:
    print(f"Some dependencies not available: {e}")
    print("For Phase 7, install: pip install gradio streamlit")

# Configure logging
logger = setup_logging("INFO")


class AIAssistant:
    """Main AI Assistant class handling conversation and generation."""
    
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.max_history_length = 10
        
        # Load model and tokenizer
        if model_path and tokenizer_path:
            self.load_model(model_path, tokenizer_path)
        else:
            logger.warning("No model/tokenizer path provided. Using dummy responses.")
    
    def load_model(self, model_path: str, tokenizer_path: str):
        """Load the trained model and tokenizer."""
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Load model configuration
            config_path = Path(model_path) / "model_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                
                # Create model
                self.model = create_gpt_model(model_config)
                
                # Load checkpoint
                checkpoint_path = Path(model_path) / "best_model.pt"
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Model loaded successfully")
                else:
                    logger.warning("No checkpoint found, using random weights")
            
            # Load tokenizer
            self.tokenizer = SimpleTokenizer()
            if Path(tokenizer_path).exists():
                self.tokenizer.load(tokenizer_path)
                logger.info("Tokenizer loaded successfully")
            else:
                logger.warning("Tokenizer not found")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_response(self, user_input: str, max_length: int = 100,
                         temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate response to user input."""
        
        if not self.model or not self.tokenizer:
            # Dummy responses for testing
            dummy_responses = [
                "I'm still learning! This is a demo response from your AI assistant built from scratch.",
                "Hello! I'm your custom AI assistant. I'm currently in development mode.",
                "That's an interesting question! I'm still training to provide better responses.",
                "I'm your AI assistant, built entirely from scratch using PyTorch. How can I help you today?",
                "As an AI assistant developed from the ground up, I'm continuously improving my responses."
            ]
            import random
            return random.choice(dummy_responses)
        
        try:
            # Prepare conversation context
            context = self._prepare_context(user_input)
            
            # Tokenize input
            input_ids = torch.tensor([self.tokenizer.encode(context)]).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.vocab.get('<pad>', 0)
                )
            
            # Decode response
            response_ids = generated[0][len(input_ids[0]):]
            response = self.tokenizer.decode(response_ids.tolist())
            
            # Clean up response
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def _prepare_context(self, user_input: str) -> str:
        """Prepare conversation context for generation."""
        # Add current input to history
        self.conversation_history.append(f"User: {user_input}")
        
        # Limit history length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        # Create context string
        context = "\n".join(self.conversation_history) + "\nAssistant:"
        
        return context
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response."""
        # Remove special tokens
        response = response.replace('<pad>', '').replace('<unk>', '')
        
        # Split on newlines and take first meaningful part
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('User:') and not line.startswith('Assistant:'):
                cleaned_lines.append(line)
            elif line.startswith('User:'):
                break  # Stop if we see another user input
        
        if cleaned_lines:
            response = ' '.join(cleaned_lines)
        else:
            response = "I'm sorry, I couldn't generate a proper response."
        
        # Limit response length
        if len(response) > 500:
            response = response[:500] + "..."
        
        return response.strip()
    
    def add_to_history(self, response: str):
        """Add assistant response to conversation history."""
        self.conversation_history.append(f"Assistant: {response}")
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> List[str]:
        """Get conversation history."""
        return self.conversation_history.copy()


def create_gradio_interface(assistant: AIAssistant) -> gr.Interface:
    """Create Gradio web interface for the AI assistant."""
    
    def chat_function(message, history):
        """Handle chat interaction."""
        if not message.strip():
            return history, ""
        
        # Generate response
        response = assistant.generate_response(message)
        assistant.add_to_history(response)
        
        # Update history
        history.append([message, response])
        
        return history, ""
    
    def clear_chat():
        """Clear chat history."""
        assistant.clear_history()
        return []
    
    # Create interface
    with gr.Blocks(title="🧠 AI Assistant From Scratch", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🧠 AI Assistant From Scratch")
        gr.Markdown("### Your custom AI assistant built entirely from scratch using PyTorch!")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_label=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Type your message here...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="Controls response creativity"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p",
                    info="Controls response diversity"
                )
                
                max_length = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=100,
                    step=10,
                    label="Max Length",
                    info="Maximum response length"
                )
                
                gr.Markdown("### 📊 Model Info")
                if assistant.model:
                    gr.Markdown(f"**Status:** ✅ Model Loaded")
                    gr.Markdown(f"**Parameters:** {assistant.model.count_parameters():,}")
                else:
                    gr.Markdown(f"**Status:** ⚠️ Demo Mode")
                    gr.Markdown(f"**Parameters:** N/A")
        
        # Event handlers
        def enhanced_chat_function(message, history, temp, top_p_val, max_len):
            if not message.strip():
                return history, ""
            
            response = assistant.generate_response(
                message, 
                max_length=max_len,
                temperature=temp,
                top_p=top_p_val
            )
            assistant.add_to_history(response)
            history.append([message, response])
            return history, ""
        
        send_btn.click(
            enhanced_chat_function,
            inputs=[msg, chatbot, temperature, top_p, max_length],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            enhanced_chat_function,
            inputs=[msg, chatbot, temperature, top_p, max_length],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot]
        )
    
    return interface


def create_streamlit_app():
    """Create Streamlit app for the AI assistant."""
    st.set_page_config(
        page_title="🧠 AI Assistant From Scratch",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 AI Assistant From Scratch")
    st.markdown("### Your custom AI assistant built entirely from scratch using PyTorch!")
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = AIAssistant()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
        max_length = st.slider("Max Length", 20, 200, 100, 10)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.assistant.clear_history()
            st.rerun()
        
        st.header("📊 Model Info")
        if st.session_state.assistant.model:
            st.success("✅ Model Loaded")
            st.info(f"Parameters: {st.session_state.assistant.model.count_parameters():,}")
        else:
            st.warning("⚠️ Demo Mode")
            st.info("Parameters: N/A")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.assistant.generate_response(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p
                )
            st.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.assistant.add_to_history(response)


def create_cli_interface(assistant: AIAssistant):
    """Create command-line interface for the AI assistant."""
    print("=" * 60)
    print("🧠 AI ASSISTANT FROM SCRATCH - CLI Mode")
    print("=" * 60)
    print()
    print("Welcome to your custom AI assistant!")
    print("Type 'quit' to exit, 'clear' to clear history, 'help' for commands.")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! Thanks for using your AI assistant! 👋")
                break
            
            elif user_input.lower() in ['clear', 'reset']:
                assistant.clear_history()
                print("🗑️ Conversation history cleared.")
                continue
            
            elif user_input.lower() in ['help', 'h']:
                print("\n📖 Available commands:")
                print("  quit/exit/q  - Exit the assistant")
                print("  clear/reset  - Clear conversation history")
                print("  help/h       - Show this help message")
                print("  status       - Show assistant status")
                print()
                continue
            
            elif user_input.lower() == 'status':
                print(f"\n📊 Assistant Status:")
                print(f"  Model loaded: {'✅ Yes' if assistant.model else '❌ No (Demo mode)'}")
                print(f"  Tokenizer loaded: {'✅ Yes' if assistant.tokenizer else '❌ No'}")
                print(f"  Conversation length: {len(assistant.conversation_history)} messages")
                if assistant.model:
                    print(f"  Model parameters: {assistant.model.count_parameters():,}")
                print()
                continue
            
            elif not user_input:
                continue
            
            # Generate response
            print("🤖 Assistant: ", end="", flush=True)
            response = assistant.generate_response(user_input)
            print(response)
            assistant.add_to_history(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for using your AI assistant! 👋")
            break
        except EOFError:
            print("\n\nGoodbye! Thanks for using your AI assistant! 👋")
            break


def main():
    """Main function to launch the AI assistant application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Assistant From Scratch - Application Interface")
    parser.add_argument("--interface", choices=["cli", "gradio", "streamlit"], 
                       default="cli", help="Interface type")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer")
    parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
    
    args = parser.parse_args()
    
    # Create assistant
    assistant = AIAssistant(args.model_path, args.tokenizer_path)
    
    # Launch appropriate interface
    if args.interface == "cli":
        create_cli_interface(assistant)
    
    elif args.interface == "gradio":
        try:
            interface = create_gradio_interface(assistant)
            interface.launch(server_port=args.port, share=False)
        except ImportError:
            print("❌ Gradio not available. Install with: pip install gradio")
    
    elif args.interface == "streamlit":
        try:
            print(f"🚀 Starting Streamlit app...")
            print(f"🌐 Open your browser to: http://localhost:8501")
            # Note: Streamlit runs via command line, this just shows the function
            create_streamlit_app()
        except ImportError:
            print("❌ Streamlit not available. Install with: pip install streamlit")


if __name__ == "__main__":
    main()
