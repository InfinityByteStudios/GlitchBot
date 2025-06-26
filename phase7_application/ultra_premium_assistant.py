"""
Ultra-Premium AI Assistant Interface
Pixel-perfect recreation inspired by Claude, ChatGPT, and Gemini
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

# Configure logging
logger = setup_logging("INFO")


class UltraPremiumAIAssistant:
    """Ultra-premium AI Assistant with pixel-perfect UI."""
    
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.max_history_length = 20
        
        # Premium responses with personality
        self.premium_responses = [
            "Hello! I'm your AI assistant, thoughtfully designed and built from scratch. How can I help you today?",
            "That's a fascinating question! Let me think about that carefully and provide you with a thoughtful response.",
            "I appreciate your curiosity! As an AI built from the ground up, I love engaging with complex and interesting topics.",
            "Excellent point! My architecture is designed to provide nuanced, helpful responses while maintaining genuine conversation.",
            "I find that topic particularly interesting! Let me share some insights based on my training and design.",
            "That's exactly the kind of question I was built to handle! I'm excited to explore this with you.",
            "I love how you're thinking about this! My custom neural networks are optimized for exactly these kinds of discussions.",
            "What a wonderful way to put it! I'm designed to be both informative and genuinely engaging in our conversations."
        ]
        
        # Load model if available
        if model_path and tokenizer_path:
            self.load_model(model_path, tokenizer_path)
        else:
            logger.info("Running in ultra-premium demo mode")
    
    def load_model(self, model_path: str, tokenizer_path: str):
        """Load the trained model and tokenizer."""
        try:
            logger.info("Loading model from %s", model_path)
            
            config_path = Path(model_path) / "model_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                
                self.model = create_gpt_model(model_config)
                
                checkpoint_path = Path(model_path) / "best_model.pt"
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Model loaded successfully")
                else:
                    logger.warning("No checkpoint found, using random weights")
            
            self.tokenizer = SimpleTokenizer()
            if Path(tokenizer_path).exists():
                self.tokenizer.load(tokenizer_path)
                logger.info("Tokenizer loaded successfully")
            else:
                logger.warning("Tokenizer not found")
            
            if self.model:
                self.model.to(self.device)
                self.model.eval()
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to load model: %s", e)
            self.model = None
            self.tokenizer = None
    
    def generate_response(self, user_input: str, max_length: int = 200,
                         temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate thoughtful response."""
        
        if not self.model or not self.tokenizer:
            return random.choice(self.premium_responses)
        
        try:
            context = self._prepare_context(user_input)
            input_ids = torch.tensor([self.tokenizer.encode(context)]).to(self.device)
            
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.vocab.get('<pad>', 0)
                )
            
            response_ids = generated[0][len(input_ids[0]):]
            response = self.tokenizer.decode(response_ids.tolist())
            response = self._clean_response(response)
            
            return response
            
        except (RuntimeError, ValueError) as e:
            logger.error("Generation failed: %s", e)
            return "I apologize, but I encountered an error. Please try again!"
    
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
        
        if len(response) > 1000:
            response = response[:1000] + "..."
        
        return response.strip()
    
    def add_to_history(self, response: str):
        """Add response to conversation history."""
        self.conversation_history.append(f"Assistant: {response}")
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")


def create_ultra_premium_interface(assistant: UltraPremiumAIAssistant) -> gr.Interface:
    """Create ultra-premium interface matching Claude, ChatGPT, and Gemini exactly."""
    
    # Ultra-premium CSS matching the exact designs
    ultra_premium_css = """
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* CSS Variables matching exact color schemes */
    :root {
        /* Claude Colors */
        --claude-orange: #D2691E;
        --claude-orange-hover: #CD853F;
        --claude-bg: #FAF9F6;
        --claude-text: #2C2C2C;
        --claude-border: #E8E5E0;
        
        /* ChatGPT Colors */
        --chatgpt-dark: #0D1117;
        --chatgpt-sidebar: #161B22;
        --chatgpt-border: #21262D;
        --chatgpt-text: #F0F6FC;
        --chatgpt-text-muted: #8B949E;
        --chatgpt-input: #21262D;
        --chatgpt-button: #238636;
        
        /* Gemini Colors */
        --gemini-blue: #1A73E8;
        --gemini-purple: #9334E6;
        --gemini-red: #EA4335;
        --gemini-yellow: #FBBC04;
        --gemini-green: #34A853;
        --gemini-dark: #1F1F1F;
        --gemini-text: #E8EAED;
        
        /* Exact gradients */
        --gemini-gradient: linear-gradient(45deg, #1A73E8 0%, #9334E6 25%, #EA4335 50%, #FBBC04 75%, #34A853 100%);
        --claude-gradient: linear-gradient(135deg, #D2691E 0%, #CD853F 100%);
        
        /* Shadows */
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Global reset and base styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body, html, .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        background: var(--chatgpt-dark) !important;
        color: var(--chatgpt-text) !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 100vh !important;
        overflow-x: hidden;
    }
    
    .gradio-container {
        max-width: 100vw !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Header - Gemini style with exact gradient */
    .ultra-header {
        background: var(--chatgpt-dark);
        padding: 2rem 0;
        text-align: center;
        border-bottom: 1px solid var(--chatgpt-border);
        position: relative;
    }
    
    .ultra-header h1 {
        background: var(--gemini-gradient);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        animation: gradient-flow 3s ease-in-out infinite;
    }
    
    @keyframes gradient-flow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .ultra-header p {
        color: var(--chatgpt-text-muted) !important;
        font-size: 0.95rem !important;
        margin: 0.5rem 0 0 0 !important;
        font-weight: 400 !important;
    }
    
    /* Main layout - ChatGPT exact structure */
    .main-layout {
        display: flex;
        height: calc(100vh - 120px);
        background: var(--chatgpt-dark);
    }
    
    /* Sidebar - Exact ChatGPT design */
    .ultra-sidebar {
        width: 260px !important;
        background: var(--chatgpt-sidebar) !important;
        border-right: 1px solid var(--chatgpt-border);
        padding: 1rem !important;
        display: flex;
        flex-direction: column;
        overflow-y: auto;
        position: fixed;
        height: calc(100vh - 120px);
        z-index: 10;
    }
    
    /* New Chat Button - ChatGPT style */
    .new-chat-btn {
        background: transparent !important;
        border: 1px solid var(--chatgpt-border) !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        color: var(--chatgpt-text) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        margin-bottom: 1rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        text-align: left !important;
    }
    
    .new-chat-btn:hover {
        background: var(--chatgpt-input) !important;
        border-color: var(--chatgpt-text-muted) !important;
    }
    
    /* Sidebar sections */
    .sidebar-group {
        background: transparent !important;
        border: 1px solid var(--chatgpt-border) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .sidebar-group h3 {
        color: var(--chatgpt-text) !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        margin: 0 0 0.75rem 0 !important;
        text-transform: none !important;
    }
    
    /* Chat area - ChatGPT centered layout */
    .ultra-chat-area {
        margin-left: 260px;
        flex: 1;
        display: flex;
        flex-direction: column;
        max-width: 768px;
        margin-left: calc(260px + (100vw - 260px - 768px) / 2);
        margin-right: auto;
        padding: 0 1rem;
    }
    
    /* Welcome screen - ChatGPT style */
    .ultra-welcome {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
        text-align: center;
        padding: 2rem;
    }
    
    .ultra-welcome h2 {
        color: var(--chatgpt-text) !important;
        font-size: 2rem !important;
        font-weight: 400 !important;
        margin-bottom: 2rem !important;
    }
    
    /* Input area - Claude style rounded input */
    .ultra-input-container {
        position: sticky;
        bottom: 0;
        background: var(--chatgpt-dark);
        padding: 1rem 0;
        border-top: 1px solid var(--chatgpt-border);
    }
    
    .ultra-input-wrapper {
        position: relative;
        background: var(--chatgpt-input) !important;
        border: 1px solid var(--chatgpt-border) !important;
        border-radius: 24px !important;
        padding: 12px 48px 12px 16px !important;
        transition: all 0.2s ease;
    }
    
    .ultra-input-wrapper:focus-within {
        border-color: var(--claude-orange) !important;
        box-shadow: 0 0 0 2px rgba(210, 105, 30, 0.1) !important;
    }
    
    .ultra-input {
        background: transparent !important;
        border: none !important;
        color: var(--chatgpt-text) !important;
        font-size: 1rem !important;
        font-family: inherit !important;
        outline: none !important;
        width: 100% !important;
        resize: none !important;
        min-height: 24px !important;
        max-height: 200px !important;
    }
    
    .ultra-input::placeholder {
        color: var(--chatgpt-text-muted) !important;
    }
    
    /* Send button - Claude orange */
    .ultra-send-btn {
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        background: var(--claude-orange) !important;
        border: none !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: white !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    
    .ultra-send-btn:hover {
        background: var(--claude-orange-hover) !important;
        transform: translateY(-50%) scale(1.05) !important;
    }
    
    /* Chat messages */
    .ultra-chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 1rem 0;
    }
    
    .ultra-message {
        margin: 1.5rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .ultra-message.user {
        flex-direction: row-reverse;
    }
    
    .ultra-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    
    .ultra-avatar.user {
        background: var(--claude-orange);
        color: white;
    }
    
    .ultra-avatar.assistant {
        background: var(--chatgpt-input);
        color: var(--chatgpt-text);
        border: 1px solid var(--chatgpt-border);
    }
    
    .ultra-message-content {
        background: var(--chatgpt-input);
        border: 1px solid var(--chatgpt-border);
        border-radius: 16px;
        padding: 0.75rem 1rem;
        max-width: 70%;
        color: var(--chatgpt-text);
        line-height: 1.5;
        font-size: 0.95rem;
    }
    
    .ultra-message.user .ultra-message-content {
        background: var(--claude-orange);
        color: white;
        border-color: var(--claude-orange);
    }
    
    /* Sliders - Custom styling */
    .ultra-slider {
        margin: 0.75rem 0 !important;
    }
    
    .ultra-slider input[type="range"] {
        width: 100% !important;
        height: 4px !important;
        background: var(--chatgpt-input) !important;
        border-radius: 2px !important;
        outline: none !important;
        border: none !important;
    }
    
    .ultra-slider input[type="range"]::-webkit-slider-thumb {
        appearance: none !important;
        width: 16px !important;
        height: 16px !important;
        border-radius: 50% !important;
        background: var(--claude-orange) !important;
        cursor: pointer !important;
        border: none !important;
    }
    
    .ultra-slider input[type="range"]::-moz-range-thumb {
        width: 16px !important;
        height: 16px !important;
        border-radius: 50% !important;
        background: var(--claude-orange) !important;
        cursor: pointer !important;
        border: none !important;
    }
    
    /* Status cards */
    .ultra-status-card {
        background: var(--chatgpt-input) !important;
        border: 1px solid var(--chatgpt-border) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.75rem 0 !important;
    }
    
    .ultra-status-card.success {
        border-left: 3px solid var(--gemini-green) !important;
    }
    
    .ultra-status-card.warning {
        border-left: 3px solid var(--claude-orange) !important;
    }
    
    /* Feature grid */
    .ultra-feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .ultra-feature-card {
        background: var(--chatgpt-input) !important;
        border: 1px solid var(--chatgpt-border) !important;
        border-radius: 6px !important;
        padding: 0.75rem !important;
        text-align: center;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .ultra-feature-card:hover {
        border-color: var(--claude-orange) !important;
        background: rgba(210, 105, 30, 0.05) !important;
    }
    
    .ultra-feature-icon {
        font-size: 1.25rem;
        margin-bottom: 0.25rem;
        display: block;
    }
    
    .ultra-feature-title {
        color: var(--chatgpt-text) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.25rem;
    }
    
    .ultra-feature-desc {
        color: var(--chatgpt-text-muted) !important;
        font-size: 0.7rem !important;
    }
    
    /* Example buttons */
    .ultra-example-btn {
        background: var(--chatgpt-input) !important;
        border: 1px solid var(--chatgpt-border) !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.5rem 0 !important;
        color: var(--chatgpt-text) !important;
        text-align: left !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-size: 0.875rem !important;
        width: 100% !important;
    }
    
    .ultra-example-btn:hover {
        border-color: var(--claude-orange) !important;
        background: rgba(210, 105, 30, 0.05) !important;
    }
    
    /* Loading animation */
    .ultra-loading {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--claude-orange);
        font-size: 0.875rem;
    }
    
    .ultra-loading::after {
        content: '';
        width: 12px;
        height: 12px;
        border: 2px solid rgba(210, 105, 30, 0.3);
        border-top-color: var(--claude-orange);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--chatgpt-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--chatgpt-border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--chatgpt-text-muted);
    }
    
    /* Responsive design */
    @media (max-width: 1024px) {
        .ultra-sidebar {
            transform: translateX(-100%);
            transition: transform 0.3s ease;
        }
        
        .ultra-sidebar.open {
            transform: translateX(0);
        }
        
        .ultra-chat-area {
            margin-left: 0;
            width: 100%;
        }
    }
    
    @media (max-width: 768px) {
        .ultra-header h1 {
            font-size: 2rem !important;
        }
        
        .ultra-feature-grid {
            grid-template-columns: 1fr;
        }
        
        .ultra-message-content {
            max-width: 85%;
        }
    }
    
    /* Override gradio defaults */
    .gradio-container .gr-button {
        background: var(--claude-orange) !important;
        border: none !important;
        color: white !important;
        border-radius: 6px !important;
    }
    
    .gradio-container .gr-textbox {
        background: var(--chatgpt-input) !important;
        border: 1px solid var(--chatgpt-border) !important;
        color: var(--chatgpt-text) !important;
        border-radius: 8px !important;
    }
    
    .gradio-container label {
        color: var(--chatgpt-text) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    .gradio-container .gr-form {
        background: transparent !important;
        border: none !important;
    }
    """
    
    def get_ultra_example_prompts():
        """Ultra-premium example prompts."""
        return [
            "🤔 Tell me about your unique capabilities",
            "🚀 How were you built from scratch?",
            "🧠 Explain transformer architecture simply",
            "✨ Write a creative story about AI",
            "💡 What's fascinating about language models?",
            "🎯 Help me understand your training process",
            "🔮 What makes you different from other AIs?",
            "⚡ Share insights about neural networks"
        ]
    
    def ultra_chat_function(message, history, temp, top_p_val, max_len):
        """Ultra-premium chat function with enhanced responses."""
        if not message.strip():
            return history, ""
        
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
        return history, ""
    
    def ultra_clear_chat():
        """Clear chat with ultra-premium animation."""
        assistant.clear_history()
        return []
    
    # Create the ultra-premium interface
    with gr.Blocks(
        title="AI Assistant - Ultra Premium",
        theme=gr.themes.Base(
            primary_hue="orange",
            secondary_hue="blue",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
        ),
        css=ultra_premium_css
    ) as interface:
        
        # Ultra Header - Exact Gemini style
        gr.HTML("""
        <div class="ultra-header">
            <h1>Hello, Hassan</h1>
            <p>Welcome to your AI assistant, built from scratch with thoughtful design</p>
        </div>
        """)
        
        # Main Layout
        with gr.Row(elem_classes=["main-layout"]):
            # Ultra Sidebar - Exact ChatGPT layout
            with gr.Column(scale=0, min_width=260, elem_classes=["ultra-sidebar"]):
                # New Chat Button
                new_chat_btn = gr.Button("+ New chat", elem_classes=["new-chat-btn"])
                
                # Settings Group
                with gr.Group(elem_classes=["sidebar-group"]):
                    gr.HTML("<h3>🎨 Response Settings</h3>")
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="Creativity",
                        elem_classes=["ultra-slider"]
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Focus",
                        elem_classes=["ultra-slider"]
                    )
                    
                    max_length = gr.Slider(
                        minimum=50,
                        maximum=400,
                        value=200,
                        step=25,
                        label="Length",
                        elem_classes=["ultra-slider"]
                    )
                
                # Status Group
                with gr.Group(elem_classes=["sidebar-group"]):
                    gr.HTML("<h3>📊 Status</h3>")
                    
                    if assistant.model:
                        gr.HTML("""
                        <div class="ultra-status-card success">
                            <div style="font-weight: 600; color: #34A853;">✅ Model Loaded</div>
                            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--chatgpt-text-muted);">
                                {:,} parameters ready
                            </div>
                        </div>
                        """.format(assistant.model.count_parameters()))
                    else:
                        gr.HTML("""
                        <div class="ultra-status-card warning">
                            <div style="font-weight: 600; color: #D2691E;">⚡ Demo Mode</div>
                            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--chatgpt-text-muted);">
                                Premium responses active
                            </div>
                        </div>
                        """)
                
                # Features Group
                with gr.Group(elem_classes=["sidebar-group"]):
                    gr.HTML("<h3>🌟 Features</h3>")
                    gr.HTML("""
                    <div class="ultra-feature-grid">
                        <div class="ultra-feature-card">
                            <span class="ultra-feature-icon">🧠</span>
                            <div class="ultra-feature-title">Smart</div>
                            <div class="ultra-feature-desc">Context aware</div>
                        </div>
                        <div class="ultra-feature-card">
                            <span class="ultra-feature-icon">✨</span>
                            <div class="ultra-feature-title">Creative</div>
                            <div class="ultra-feature-desc">Adjustable style</div>
                        </div>
                        <div class="ultra-feature-card">
                            <span class="ultra-feature-icon">🚀</span>
                            <div class="ultra-feature-title">Custom</div>
                            <div class="ultra-feature-desc">From scratch</div>
                        </div>
                        <div class="ultra-feature-card">
                            <span class="ultra-feature-icon">🎯</span>
                            <div class="ultra-feature-title">Thoughtful</div>
                            <div class="ultra-feature-desc">Carefully designed</div>
                        </div>
                    </div>
                    """)
            
            # Ultra Chat Area - ChatGPT centered layout
            with gr.Column(scale=1, elem_classes=["ultra-chat-area"]):
                # Welcome Message
                welcome_area = gr.HTML("""
                <div class="ultra-welcome">
                    <h2>Ready when you are.</h2>
                </div>
                """, visible=True)
                
                # Chat Container
                chatbot = gr.Chatbot(
                    label="",
                    height=400,
                    show_label=False,
                    avatar_images=None,
                    bubble_full_width=False,
                    show_copy_button=True,
                    elem_classes=["ultra-chat-container"],
                    visible=False
                )
                
                # Ultra Input Area
                with gr.Row(elem_classes=["ultra-input-container"]):
                    with gr.Column():
                        msg = gr.Textbox(
                            label="",
                            placeholder="Ask anything...",
                            lines=1,
                            max_lines=4,
                            elem_classes=["ultra-input-wrapper", "ultra-input"]
                        )
                        with gr.Row():
                            send_btn = gr.Button("↗", elem_classes=["ultra-send-btn"])
        
        # Example Prompts (Initially Hidden)
        with gr.Row(visible=False) as examples_row:
            with gr.Column():
                gr.HTML("<h3 style='text-align: center; color: var(--chatgpt-text-muted); margin: 2rem 0;'>💡 Try these prompts</h3>")
                example_buttons = []
                prompts = get_ultra_example_prompts()
                for prompt in prompts:
                    btn = gr.Button(prompt, elem_classes=["ultra-example-btn"])
                    example_buttons.append((btn, prompt))
        
        # Event Handlers
        def first_message_handler(message, history, temp, top_p_val, max_len):
            if not message.strip():
                return history, "", gr.update(), gr.update()
            
            response = assistant.generate_response(message, max_len, temp, top_p_val)
            assistant.add_to_history(response)
            history.append([message, response])
            
            return (history, "", 
                   gr.update(visible=False),  # welcome
                   gr.update(visible=True))   # chatbot
        
        def clear_conversation():
            assistant.clear_history()
            return ([], 
                   gr.update(visible=True),   # welcome
                   gr.update(visible=False))  # chatbot
        
        # Bind events
        send_btn.click(
            first_message_handler,
            inputs=[msg, chatbot, temperature, top_p, max_length],
            outputs=[chatbot, msg, welcome_area, chatbot]
        )
        
        msg.submit(
            first_message_handler,
            inputs=[msg, chatbot, temperature, top_p, max_length],
            outputs=[chatbot, msg, welcome_area, chatbot]
        )
        
        new_chat_btn.click(
            clear_conversation,
            outputs=[chatbot, welcome_area, chatbot]
        )
        
        # Example button handlers
        for btn, prompt in example_buttons:
            btn.click(lambda p=prompt: p, outputs=[msg])
    
    return interface


def main():
    """Launch the ultra-premium AI assistant interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Premium AI Assistant")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer")
    parser.add_argument("--port", type=int, default=7861, help="Port for web interface")
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    # Create ultra-premium assistant
    assistant = UltraPremiumAIAssistant(args.model_path, args.tokenizer_path)
    
    # Launch ultra-premium interface
    interface = create_ultra_premium_interface(assistant)
    
    print("🚀 Launching Ultra-Premium AI Assistant...")
    print("🎨 Pixel-perfect design inspired by Claude, ChatGPT & Gemini")
    print(f"🌐 Open your browser to: http://localhost:{args.port}")
    
    interface.launch(
        server_port=args.port, 
        share=args.share,
        server_name="0.0.0.0" if args.share else "127.0.0.1"
    )


if __name__ == "__main__":
    main()
