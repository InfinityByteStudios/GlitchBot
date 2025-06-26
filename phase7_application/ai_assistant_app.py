"""
AI Assistant Application Interface

This module implements the user-facing application for the AI assistant,
including conversation management, prompt handling, and various interface options.
This is Phase 7: Build Your AI Assistant Application.
"""

import sys
import json
import random
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import gradio as gr
    import streamlit as st
    
    # Import our custom modules
    from shared.utils import setup_logging, get_device
    from phase4_model.transformer_model import create_gpt_model
    from phase3_data.data_pipeline import SimpleTokenizer
    
except ImportError as e:
    print("Some dependencies not available: %s", e)
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
            logger.info("Loading model from %s", model_path)
            
            # Load model configuration
            config_path = Path(model_path) / "model_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
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
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to load model: %s", e)
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
            
        except (RuntimeError, ValueError) as e:
            logger.error("Generation failed: %s", e)
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
    """Create modern Gradio web interface inspired by ChatGPT, Claude, and Gemini."""
    
    # Custom CSS for modern, clean design
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .user-message {
        background: #007bff;
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 10px rgba(0,123,255,0.3);
    }
    .assistant-message {
        background: #f8f9fa;
        color: #333;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 80%;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .input-container {
        background: white;
        border-radius: 25px;
        padding: 5px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        border: 2px solid #e9ecef;
    }
    .sidebar {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 5px 20px rgba(102,126,234,0.3);
    }
    .status-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .btn-primary {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        border: none;
        border-radius: 20px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,123,255,0.4);
    }
    """
    
    def enhanced_chat_function(message, history, temp, top_p_val, max_len):
        """Enhanced chat function with typing simulation."""
        if not message.strip():
            return history, ""
        
        # Generate response with parameters
        response = assistant.generate_response(
            message, 
            max_length=max_len,
            temperature=temp,
            top_p=top_p_val
        )
        assistant.add_to_history(response)
        
        # Add to history with proper formatting
        history.append([message, response])
        return history, ""
    
    def clear_chat():
        """Clear chat history with confirmation."""
        assistant.clear_history()
        return []
    
    def get_example_prompts():
        """Get example prompts for quick start."""
        return [
            "Tell me about yourself and your capabilities",
            "Explain quantum computing in simple terms",
            "Write a creative story about AI and humans",
            "Help me plan a productive day",
            "What's the future of artificial intelligence?"
        ]
    
    # Create the interface
    with gr.Blocks(
        title="🚀 AI Assistant - Built from Scratch",
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css=custom_css
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🚀 AI Assistant from Scratch</h1>
            <p>Experience the power of a custom-built AI assistant using PyTorch</p>
            <p>Inspired by ChatGPT, Claude, and Gemini - Built with ❤️ from the ground up</p>
        </div>
        """)
        
        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=500,
                    show_label=False,
                    avatar_images=("🧑", "🤖"),
                    bubble_full_width=False,
                    show_copy_button=True,
                    elem_classes=["chat-container"]
                )
                
                # Input area
                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="Ask me anything... I'm here to help! 💡",
                        scale=5,
                        lines=1,
                        max_lines=4,
                        elem_classes=["input-container"]
                    )
                    send_btn = gr.Button(
                        "Send 🚀", 
                        variant="primary", 
                        scale=1,
                        elem_classes=["btn-primary"]
                    )
                
                # Quick actions
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)
                    examples_btn = gr.Button("💡 Example Prompts", variant="secondary", scale=1)
            
            # Sidebar
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                gr.Markdown("## ⚙️ Settings")
                
                # Generation parameters
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="🔥 Temperature",
                    info="Higher = more creative"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="🎯 Top-p",
                    info="Higher = more diverse"
                )
                
                max_length = gr.Slider(
                    minimum=20,
                    maximum=300,
                    value=150,
                    step=10,
                    label="📏 Max Length",
                    info="Response length limit"
                )
                
                gr.Markdown("---")
                gr.Markdown("## 📊 Assistant Status")
                
                # Status display
                with gr.Column(elem_classes=["status-card"]):
                    if assistant.model:
                        gr.Markdown("**Status:** ✅ Model Loaded")
                        gr.Markdown(f"**Parameters:** {assistant.model.count_parameters():,}")
                        gr.Markdown("**Mode:** Full AI Model")
                    else:
                        gr.Markdown("**Status:** ⚠️ Demo Mode")
                        gr.Markdown("**Parameters:** N/A")
                        gr.Markdown("**Mode:** Simulated Responses")
                
                gr.Markdown("---")
                gr.Markdown("## 🎨 Features")
                gr.Markdown("""
                • **Smart Conversations** 🧠
                • **Context Awareness** 🔍
                • **Adjustable Creativity** 🎭
                • **Built from Scratch** ⚡
                • **PyTorch Powered** 🔥
                """)
        
        # Example prompts area
        with gr.Row(visible=False) as examples_row:
            with gr.Column():
                gr.Markdown("### 💡 Try these example prompts:")
                example_buttons = []
                for prompt in get_example_prompts():
                    btn = gr.Button(prompt, variant="secondary", size="sm")
                    example_buttons.append(btn)
        
        # Event handlers
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
        
        clear_btn.click(clear_chat, outputs=[chatbot])
        
        # Toggle examples visibility
        def toggle_examples():
            return gr.update(visible=True)
        
        examples_btn.click(toggle_examples, outputs=[examples_row])
        
        # Example button handlers
        for i, btn in enumerate(example_buttons):
            prompt = get_example_prompts()[i]
            btn.click(lambda p=prompt: p, outputs=[msg])
    
    return interface


def create_streamlit_app():
    """Create modern Streamlit app inspired by Claude and ChatGPT."""
    st.set_page_config(
        page_title="🚀 AI Assistant from Scratch",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for modern design
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: 20%;
    }
    .assistant-message {
        background: #f8f9fa;
        color: #333;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #28a745;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(102,126,234,0.3);
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>🚀 AI Assistant from Scratch</h1>
        <p>Experience the power of a custom-built AI assistant using PyTorch</p>
        <p>Inspired by ChatGPT, Claude, and Gemini • Built with ❤️ from the ground up</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = AIAssistant()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar with enhanced design
    with st.sidebar:
        st.markdown("## ⚙️ Assistant Settings")
        
        # Generation parameters with better descriptions
        temperature = st.slider(
            "🔥 Temperature", 
            0.1, 2.0, 0.8, 0.1,
            help="Higher values make responses more creative and unpredictable"
        )
        top_p = st.slider(
            "🎯 Top-p", 
            0.1, 1.0, 0.9, 0.05,
            help="Controls response diversity by limiting token choices"
        )
        max_length = st.slider(
            "📏 Max Length", 
            20, 300, 150, 10,
            help="Maximum number of tokens in the response"
        )
        
        st.markdown("---")
        
        # Enhanced control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.assistant.clear_history()
                st.rerun()
        
        with col2:
            if st.button("🔄 New Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.assistant.clear_history()
                st.success("Started a new conversation!")
                st.rerun()
        
        st.markdown("---")
        
        # Model status with beautiful cards
        st.markdown("## 📊 Assistant Status")
        if st.session_state.assistant.model:
            st.markdown("""
            <div class="status-success">
                <strong>✅ Model Status:</strong> Loaded<br>
                <strong>🔢 Parameters:</strong> {:,}<br>
                <strong>🚀 Mode:</strong> Full AI Model
            </div>
            """.format(st.session_state.assistant.model.count_parameters()), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-warning">
                <strong>⚠️ Model Status:</strong> Demo Mode<br>
                <strong>🔢 Parameters:</strong> N/A<br>
                <strong>🎭 Mode:</strong> Simulated Responses
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Features showcase
        st.markdown("## 🎨 Features")
        st.markdown("""
        <div class="feature-card">
            <h4>🧠 Smart Conversations</h4>
            <p>Context-aware responses with memory</p>
        </div>
        <div class="feature-card">
            <h4>⚡ Built from Scratch</h4>
            <p>Custom PyTorch transformer model</p>
        </div>
        <div class="feature-card">
            <h4>🎭 Adjustable Personality</h4>
            <p>Fine-tune creativity and style</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example prompts
        st.markdown("## 💡 Try These Prompts")
        example_prompts = [
            "Tell me about yourself",
            "Explain quantum computing simply",
            "Write a creative story",
            "Help me plan my day",
            "What's the future of AI?"
        ]
        
        for prompt in example_prompts:
            if st.button(f"💭 {prompt}", key=f"example_{prompt}", use_container_width=True):
                st.session_state.example_prompt = prompt
    
    # Main chat interface
    st.markdown("## 💬 Conversation")
    
    # Display chat history with modern styling
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
    
    # Handle example prompt
    if 'example_prompt' in st.session_state:
        prompt = st.session_state.example_prompt
        del st.session_state.example_prompt
    else:
        prompt = st.chat_input("Ask me anything... I'm here to help! 💡", key="chat_input")
    
    # Process chat input
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.assistant.generate_response(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p
                )
            
            # Stream-like effect for response
            placeholder = st.empty()
            full_response = ""
            
            # Simulate typing effect
            import time
            for chunk in response.split():
                full_response += chunk + " "
                placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
            
            placeholder.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.assistant.add_to_history(response)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🚀 Built with PyTorch • Powered by Custom Transformer Architecture</p>
        <p>Made with ❤️ for learning and experimentation</p>
    </div>
    """, unsafe_allow_html=True)


def create_cli_interface(assistant: AIAssistant):
    """Create enhanced command-line interface for the AI assistant."""
    print("=" * 70)
    print("🚀 AI ASSISTANT FROM SCRATCH - CLI Mode")
    print("=" * 70)
    print()
    print("🎉 Welcome to your custom AI assistant!")
    print("📝 Type 'quit' to exit, 'clear' to clear history, 'help' for commands.")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for using your AI assistant!")
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
                print("\n📊 Assistant Status:")
                print("  Model loaded: %s", '✅ Yes' if assistant.model else '❌ No (Demo mode)')
                print("  Tokenizer loaded: %s", '✅ Yes' if assistant.tokenizer else '❌ No')
                print("  Conversation length: %d messages", len(assistant.conversation_history))
                if assistant.model:
                    print("  Model parameters: %s", f"{assistant.model.count_parameters():,}")
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
            print("\n\n👋 Goodbye! Thanks for using your AI assistant!")
            break
        except EOFError:
            print("\n\n👋 Goodbye! Thanks for using your AI assistant!")
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
            print("🚀 Starting Gradio app...")
            print("🌐 Open your browser to: http://localhost:%d", args.port)
            interface.launch(server_port=args.port, share=False)
        except ImportError:
            print("❌ Gradio not available. Install with: pip install gradio")
    
    elif args.interface == "streamlit":
        try:
            print("🚀 Starting Streamlit app...")
            print("🌐 Open your browser to: http://localhost:8501")
            # Note: Streamlit runs via command line, this just shows the function
            create_streamlit_app()
        except ImportError:
            print("❌ Streamlit not available. Install with: pip install streamlit")


if __name__ == "__main__":
    main()
