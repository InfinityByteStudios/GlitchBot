"""
Premium AI Assistant Interface - Inspired by Claude, ChatGPT, and Gemini

This creates a beautiful, modern UI that combines the best visual elements
from all three major AI assistants while maintaining our unique branding.
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
    
    # Import our custom modules
    from shared.utils import setup_logging, get_device
    from phase4_model.transformer_model import create_gpt_model
    from phase3_data.data_pipeline import SimpleTokenizer
    
except ImportError as e:
    print("Dependencies not available: %s", e)
    print("Install with: pip install -r requirements.txt")

# Configure logging
logger = setup_logging("INFO")


class PremiumAIAssistant:
    """Premium AI Assistant with enhanced UI capabilities."""
    
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.max_history_length = 20
        
        # Enhanced demo responses with personality
        self.demo_responses = [
            "✨ Hello! I'm your AI assistant, built from scratch with PyTorch. I'm here to help you explore the fascinating world of artificial intelligence!",
            "🚀 That's a wonderful question! As a custom-built AI, I'm constantly learning and evolving. What specific aspect interests you most?",
            "🧠 I love discussing complex topics! My neural networks are designed to provide thoughtful, engaging responses across many domains.",
            "💡 Interesting perspective! Let me think about that... I'm powered by transformer architecture and trained to be helpful, harmless, and honest.",
            "🌟 I appreciate your curiosity! As an AI built from the ground up, I aim to combine creativity with accuracy in every response.",
            "🎯 That's exactly the kind of question that makes AI development exciting! I'm designed to be both informative and conversational.",
            "⚡ Great point! My training focuses on being genuinely helpful while maintaining the wonder and excitement of AI exploration.",
            "🔮 I find that topic fascinating too! My architecture allows me to engage with complex ideas while keeping things accessible.",
        ]
        
        # Load model if paths provided
        if model_path and tokenizer_path:
            self.load_model(model_path, tokenizer_path)
        else:
            logger.info("Running in demo mode with enhanced responses")
    
    def load_model(self, model_path: str, tokenizer_path: str):
        """Load trained model and tokenizer."""
        try:
            logger.info("Loading model from %s", model_path)
            
            # Load model configuration
            config_path = Path(model_path) / "model_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                
                self.model = create_gpt_model(model_config)
                
                # Load checkpoint if available
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
    
    def generate_response(self, user_input: str, max_length: int = 150,
                         temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate response with enhanced personality."""
        
        if not self.model or not self.tokenizer:
            return random.choice(self.demo_responses)
        
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
            
            # Decode and clean response
            response_ids = generated[0][len(input_ids[0]):]
            response = self.tokenizer.decode(response_ids.tolist())
            response = self._clean_response(response)
            
            return response
            
        except (RuntimeError, ValueError) as e:
            logger.error("Generation failed: %s", e)
            return "I apologize, but I encountered an error while generating a response. Please try again!"
    
    def _prepare_context(self, user_input: str) -> str:
        """Prepare conversation context."""
        self.conversation_history.append(f"Human: {user_input}")
        
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        context = "\n".join(self.conversation_history) + "\nAssistant:"
        return context
    
    def _clean_response(self, response: str) -> str:
        """Clean and format response."""
        response = response.replace('<pad>', '').replace('<unk>', '')
        
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Human:') and not line.startswith('Assistant:'):
                cleaned_lines.append(line)
            elif line.startswith('Human:'):
                break
        
        if cleaned_lines:
            response = ' '.join(cleaned_lines)
        else:
            response = "I'm here to help! Could you please rephrase your question?"
        
        if len(response) > 800:
            response = response[:800] + "..."
        
        return response.strip()
    
    def add_to_history(self, response: str):
        """Add response to conversation history."""
        self.conversation_history.append(f"Assistant: {response}")
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")


def create_premium_interface(assistant: PremiumAIAssistant) -> gr.Interface:
    """Create a premium interface inspired by Claude, ChatGPT, and Gemini."""
    
    # Premium CSS inspired by Claude, ChatGPT, and Gemini
    premium_css = """
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global CSS Variables - Inspired by all three UIs */
    :root {
        /* Claude Colors */
        --claude-orange: #D97757;
        --claude-orange-light: #E8B4A6;
        --claude-bg: #F7F7F5;
        --claude-text: #2F2F2F;
        
        /* ChatGPT Colors */
        --chatgpt-dark: #212121;
        --chatgpt-sidebar: #171717;
        --chatgpt-green: #10A37F;
        --chatgpt-text: #ECECEC;
        --chatgpt-input: #40414F;
        
        /* Gemini Colors */
        --gemini-blue: #4285F4;
        --gemini-purple: #9C27B0;
        --gemini-pink: #EA4335;
        --gemini-yellow: #FBBC04;
        --gemini-dark: #1F1F1F;
        
        /* Custom Gradients */
        --gradient-gemini: linear-gradient(90deg, #4285F4 0%, #9C27B0 25%, #EA4335 50%, #FBBC04 75%, #34A853 100%);
        --gradient-claude: linear-gradient(135deg, #D97757 0%, #E8B4A6 100%);
        --gradient-chatgpt: linear-gradient(135deg, #10A37F 0%, #1A7F64 100%);
        
        /* Shadows */
        --shadow-soft: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        --shadow-medium: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);
        --shadow-large: 0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23);
        
        /* Animation */
        --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    body, .gradio-container {
        background: var(--chatgpt-dark) !important;
        color: var(--chatgpt-text) !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .gradio-container {
        max-width: 100vw !important;
        margin: 0 !important;
        background: var(--chatgpt-dark) !important;
    }
    
    /* Header - Gemini Inspired with Gradient Text */
    .premium-header {
        background: var(--chatgpt-dark);
        padding: 4rem 2rem;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 300%;
        height: 1px;
        background: var(--gradient-gemini);
        animation: shimmer 3s infinite linear;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .premium-header h1 {
        background: var(--gradient-gemini);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin: 0 0 1rem 0 !important;
        animation: gradient-shift 4s ease-in-out infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .premium-header p {
        color: rgba(255,255,255,0.7) !important;
        font-size: 1.1rem !important;
        margin: 0.5rem 0 !important;
        font-weight: 400 !important;
    }
    
    /* Main Layout - ChatGPT Inspired */
    .main-container {
        display: flex;
        height: calc(100vh - 200px);
        background: var(--chatgpt-dark);
    }
    
    /* Sidebar - ChatGPT Style */
    .premium-sidebar {
        width: 280px;
        background: var(--chatgpt-sidebar) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
        padding: 1.5rem !important;
        overflow-y: auto;
        position: fixed;
        left: 0;
        top: 200px;
        height: calc(100vh - 200px);
        z-index: 10;
    }
    
    .sidebar-section {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        transition: var(--transition-smooth);
    }
    
    .sidebar-section:hover {
        background: rgba(255,255,255,0.05) !important;
        border-color: var(--claude-orange);
    }
    
    .sidebar-section h3 {
        color: white !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin: 0 0 1rem 0 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Chat Area - Center like ChatGPT */
    .chat-area {
        margin-left: 280px;
        flex: 1;
        display: flex;
        flex-direction: column;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        padding: 0 2rem;
    }
    
    /* Chat Container */
    .chat-container {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Messages - Claude Inspired Styling */
    .message {
        margin: 2rem 0;
        padding: 0;
    }
    
    .user-message {
        background: var(--chatgpt-input) !important;
        color: white !important;
        padding: 1.2rem 1.5rem !important;
        border-radius: 18px !important;
        margin-left: 25% !important;
        margin-right: 0 !important;
        border: 1px solid rgba(255,255,255,0.1);
        position: relative;
        box-shadow: var(--shadow-soft);
    }
    
    .assistant-message {
        background: rgba(255,255,255,0.02) !important;
        color: var(--chatgpt-text) !important;
        padding: 1.5rem !important;
        border-radius: 18px !important;
        margin-right: 25% !important;
        margin-left: 0 !important;
        border-left: 3px solid var(--claude-orange);
        position: relative;
        box-shadow: var(--shadow-soft);
        line-height: 1.6;
    }
    
    /* Input Area - Claude Inspired */
    .input-container {
        background: var(--chatgpt-input) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 25px !important;
        padding: 8px !important;
        margin: 2rem 0 !important;
        transition: var(--transition-smooth);
    }
    
    .input-container:focus-within {
        border-color: var(--claude-orange) !important;
        box-shadow: 0 0 0 3px rgba(217, 119, 87, 0.1) !important;
    }
    
    .input-field {
        background: transparent !important;
        border: none !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 1rem 1.5rem !important;
        border-radius: 20px !important;
    }
    
    .input-field::placeholder {
        color: rgba(255,255,255,0.5) !important;
    }
    
    /* Buttons */
    .premium-button {
        background: var(--gradient-claude) !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.8rem 1.5rem !important;
        color: white !important;
        font-weight: 500 !important;
        transition: var(--transition-smooth) !important;
        cursor: pointer !important;
    }
    
    .premium-button:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-medium) !important;
        background: var(--claude-orange) !important;
    }
    
    /* Sliders - Custom Styling */
    .premium-slider {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .premium-slider input[type="range"] {
        background: rgba(255,255,255,0.1) !important;
        height: 6px !important;
        border-radius: 3px !important;
    }
    
    .premium-slider input[type="range"]::-webkit-slider-thumb {
        background: var(--claude-orange) !important;
        border: none !important;
        border-radius: 50% !important;
        width: 18px !important;
        height: 18px !important;
        cursor: pointer !important;
    }
    
    /* Status Cards */
    .status-card {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        transition: var(--transition-smooth);
    }
    
    .status-card:hover {
        background: rgba(255,255,255,0.05) !important;
        border-color: var(--claude-orange) !important;
    }
    
    .status-card.success {
        border-left: 3px solid var(--chatgpt-green) !important;
    }
    
    .status-card.warning {
        border-left: 3px solid var(--claude-orange) !important;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .feature-card {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px !important;
        padding: 1rem !important;
        text-align: center;
        transition: var(--transition-smooth);
        cursor: pointer;
    }
    
    .feature-card:hover {
        background: rgba(255,255,255,0.05) !important;
        transform: translateY(-2px);
        border-color: var(--claude-orange);
    }
    
    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .feature-card strong {
        color: white !important;
        font-size: 0.85rem;
        display: block;
        margin-bottom: 0.25rem;
    }
    
    .feature-card div {
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.75rem;
    }
    
    /* Example Buttons */
    .example-button {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        margin: 0.5rem !important;
        color: rgba(255,255,255,0.9) !important;
        text-align: left !important;
        transition: var(--transition-smooth) !important;
        cursor: pointer !important;
    }
    
    .example-button:hover {
        background: rgba(255,255,255,0.05) !important;
        border-color: var(--claude-orange) !important;
        transform: translateX(3px) !important;
    }
    
    /* Welcome Message - Gemini Style */
    .welcome-message {
        text-align: center;
        padding: 4rem 2rem;
        color: rgba(255,255,255,0.6);
    }
    
    .welcome-message h2 {
        background: var(--gradient-gemini);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        animation: gradient-shift 4s ease-in-out infinite;
    }
    
    /* Loading Animation */
    .loading {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--claude-orange);
    }
    
    .loading::after {
        content: '';
        width: 12px;
        height: 12px;
        border: 2px solid rgba(217, 119, 87, 0.3);
        border-top-color: var(--claude-orange);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--claude-orange);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--claude-orange-light);
    }
    
    /* Responsive Design */
    @media (max-width: 1024px) {
        .premium-sidebar {
            transform: translateX(-100%);
            transition: transform 0.3s ease;
        }
        
        .premium-sidebar.open {
            transform: translateX(0);
        }
        
        .chat-area {
            margin-left: 0;
            padding: 0 1rem;
        }
        
        .user-message, .assistant-message {
            margin-left: 1rem !important;
            margin-right: 1rem !important;
        }
    }
    
    @media (max-width: 768px) {
        .premium-header h1 {
            font-size: 2.5rem !important;
        }
        
        .premium-header {
            padding: 2rem 1rem;
        }
        
        .feature-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    /* Dark theme text colors */
    .gradio-container label,
    .gradio-container p,
    .gradio-container span,
    .gradio-container div {
        color: rgba(255,255,255,0.9) !important;
    }
    
    /* Override Gradio defaults */
    .gradio-container .gr-button {
        background: var(--gradient-claude) !important;
        border: none !important;
        color: white !important;
    }
    
    .gradio-container .gr-textbox {
        background: var(--chatgpt-input) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    """
    
    def enhanced_chat_function(message, history, temp, top_p_val, max_len):
        """Enhanced chat with personality and better formatting."""
        if not message.strip():
            return history, ""
        
        # Add typing indicator
        typing_response = "🤔 Thinking..."
        history.append([message, typing_response])
        
        # Generate actual response
        response = assistant.generate_response(
            message, 
            max_length=max_len,
            temperature=temp,
            top_p=top_p_val
        )
        assistant.add_to_history(response)
        
        # Replace typing indicator with actual response
        history[-1] = [message, response]
        return history, ""
    
    def clear_chat():
        """Clear chat with animation."""
        assistant.clear_history()
        return []
    
    def get_example_prompts():
        """Premium example prompts."""
        return [
            "🌟 Tell me about yourself and your unique capabilities",
            "🚀 Explain how you were built from scratch using PyTorch",
            "🧠 What makes transformer architecture so powerful?",
            "✨ Write a creative story about AI and human collaboration",
            "🎯 Help me understand the future of artificial intelligence",
            "💡 What's the most fascinating thing about language models?",
            "🔮 How do you generate such human-like responses?",
            "⚡ What's your favorite thing about being an AI assistant?"
        ]
    
    # Create the premium interface
    with gr.Blocks(
        title="🚀 AI Assistant - Premium Edition",
        theme=gr.themes.Base(
            primary_hue="orange",
            secondary_hue="blue", 
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
        ),
        css=premium_css
    ) as interface:
        
        # Premium Header - Gemini Style Gradient Text
        gr.HTML("""
        <div class="premium-header">
            <h1>Hello, Hassan</h1>
            <p>Welcome to your AI Assistant, built from scratch</p>
            <p>Combining Claude's thoughtfulness • ChatGPT's reliability • Gemini's innovation</p>
        </div>
        """)
        
        # Main Layout Container
        with gr.Row():
            # ChatGPT-Style Sidebar
            with gr.Column(scale=1, elem_classes=["premium-sidebar"]):
                gr.HTML('<h2 style="color: white; text-align: center; margin-bottom: 2rem;">🚀 Assistant</h2>')
                
                # New Chat Button (ChatGPT Style)
                new_chat_btn = gr.Button("+ New chat", elem_classes=["premium-button"], size="lg")
                
                # Settings Section
                with gr.Group(elem_classes=["sidebar-section"]):
                    gr.HTML("<h3>🎨 Response Style</h3>")
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="🔥 Creativity",
                        info="Higher = more creative",
                        elem_classes=["premium-slider"]
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="🎯 Focus",
                        info="Higher = more diverse",
                        elem_classes=["premium-slider"]
                    )
                    
                    max_length = gr.Slider(
                        minimum=50,
                        maximum=400,
                        value=200,
                        step=25,
                        label="📏 Length",
                        info="Response length",
                        elem_classes=["premium-slider"]
                    )
                
                # Status Section
                with gr.Group(elem_classes=["sidebar-section"]):
                    gr.HTML("<h3>📊 Status</h3>")
                    
                    if assistant.model:
                        gr.HTML("""
                        <div class="status-card success">
                            <div style="font-weight: 600; color: #10A37F;">✅ Model Ready</div>
                            <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                                {:,} parameters
                            </div>
                        </div>
                        """.format(assistant.model.count_parameters()))
                    else:
                        gr.HTML("""
                        <div class="status-card warning">
                            <div style="font-weight: 600; color: #D97757;">⚡ Demo Mode</div>
                            <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                                Enhanced responses
                            </div>
                        </div>
                        """)
                
                # Features
                with gr.Group(elem_classes=["sidebar-section"]):
                    gr.HTML("<h3>🌟 Features</h3>")
                    gr.HTML("""
                    <div class="feature-grid">
                        <div class="feature-card">
                            <span class="feature-icon">🧠</span>
                            <strong>Smart</strong>
                            <div>Context aware</div>
                        </div>
                        <div class="feature-card">
                            <span class="feature-icon">✨</span>
                            <strong>Creative</strong>
                            <div>Adjustable style</div>
                        </div>
                        <div class="feature-card">
                            <span class="feature-icon">🚀</span>
                            <strong>Custom</strong>
                            <div>From scratch</div>
                        </div>
                        <div class="feature-card">
                            <span class="feature-icon">🎯</span>
                            <strong>Thoughtful</strong>
                            <div>Carefully designed</div>
                        </div>
                    </div>
                    """)
            
            # Main Chat Area - ChatGPT Style
            with gr.Column(scale=3, elem_classes=["chat-area"]):
                # Welcome Message when no conversation
                welcome_area = gr.HTML("""
                <div class="welcome-message">
                    <h2>Ready when you are.</h2>
                    <p>Ask me anything, and I'll provide thoughtful, helpful responses.</p>
                </div>
                """, visible=True)
                
                # Chat Interface
                chatbot = gr.Chatbot(
                    label="",
                    height=500,
                    show_label=False,
                    avatar_images=None,
                    bubble_full_width=False,
                    show_copy_button=True,
                    elem_classes=["chat-container"],
                    visible=False
                )
                
                # Input Area - Claude Style
                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="Ask anything...",
                        scale=5,
                        lines=1,
                        max_lines=4,
                        elem_classes=["input-container", "input-field"]
                    )
                    send_btn = gr.Button(
                        "↗", 
                        variant="primary", 
                        scale=1,
                        elem_classes=["premium-button"]
                    )
                
                # Action Buttons Row
                with gr.Row(visible=False) as action_buttons:
                    clear_btn = gr.Button("🗑️ Clear", elem_classes=["premium-button"])
                    examples_btn = gr.Button("💡 Examples", elem_classes=["premium-button"])
        
        # Example Prompts Section (Initially Hidden)
        with gr.Row(visible=False) as examples_row:
            with gr.Column():
                gr.HTML("<h3 style='text-align: center; color: rgba(255,255,255,0.8); margin: 2rem 0;'>💡 Try these prompts</h3>")
                with gr.Row():
                    example_buttons = []
                    prompts = get_example_prompts()
                    for i in range(0, len(prompts), 2):
                        with gr.Column():
                            for j in range(2):
                                if i + j < len(prompts):
                                    btn = gr.Button(
                                        prompts[i + j], 
                                        elem_classes=["example-button"],
                                        size="lg"
                                    )
                                    example_buttons.append((btn, prompts[i + j]))
        
        # Enhanced Event Handlers
        def first_message_handler(message, history, temp, top_p_val, max_len):
            """Handle the first message to show chat interface."""
            if not message.strip():
                return history, "", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
            
            # Generate response
            response = assistant.generate_response(
                message, 
                max_length=max_len,
                temperature=temp,
                top_p=top_p_val
            )
            assistant.add_to_history(response)
            
            # Add to history
            history.append([message, response])
            
            # Show chat interface, hide welcome
            return (history, "", 
                   gr.update(visible=False),  # welcome_area
                   gr.update(visible=True),   # chatbot
                   gr.update(visible=True))   # action_buttons
        
        def continue_chat(message, history, temp, top_p_val, max_len):
            """Handle subsequent messages."""
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
        
        def clear_conversation():
            """Clear chat and return to welcome screen."""
            assistant.clear_history()
            return ([], 
                   gr.update(visible=True),   # welcome_area
                   gr.update(visible=False),  # chatbot
                   gr.update(visible=False))  # action_buttons
        
        # Event Bindings
        send_btn.click(
            first_message_handler,
            inputs=[msg, chatbot, temperature, top_p, max_length],
            outputs=[chatbot, msg, welcome_area, chatbot, action_buttons]
        )
        
        msg.submit(
            first_message_handler,
            inputs=[msg, chatbot, temperature, top_p, max_length],
            outputs=[chatbot, msg, welcome_area, chatbot, action_buttons]
        )
        
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot, welcome_area, chatbot, action_buttons]
        )
        
        new_chat_btn.click(
            clear_conversation,
            outputs=[chatbot, welcome_area, chatbot, action_buttons]
        )
        
        # Toggle examples
        examples_btn.click(
            lambda: gr.update(visible=True),
            outputs=[examples_row]
        )
        
        # Example button handlers
        for btn, prompt in example_buttons:
            btn.click(lambda p=prompt: p, outputs=[msg])
    
    return interface


def main():
    """Launch the premium AI assistant interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Premium AI Assistant Interface")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer")
    parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    # Create premium assistant
    assistant = PremiumAIAssistant(args.model_path, args.tokenizer_path)
    
    # Launch premium interface
    interface = create_premium_interface(assistant)
    
    print("🚀 Launching Premium AI Assistant...")
    print("🌟 Inspired by Claude, ChatGPT, and Gemini")
    print(f"🌐 Open your browser to: http://localhost:{args.port}")
    
    interface.launch(
        server_port=args.port, 
        share=args.share,
        server_name="0.0.0.0" if args.share else "127.0.0.1",
        favicon_path=None,
        show_error=True
    )


if __name__ == "__main__":
    main()
