#!/usr/bin/env python3
"""
Pure HTML AI Assistant - Completely Custom Interface
This creates a standalone HTML interface that exactly matches the screenshots
"""

import http.server
import socketserver
import webbrowser
import json
import logging
from datetime import datetime
import threading
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAssistantHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for the AI assistant"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html_content().encode('utf-8'))
        elif self.path == '/api/chat':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"message": "Please use POST for chat messages"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                message = data.get('message', '')
                
                # Generate response
                response_text = self.generate_response(message)
                
                response = {
                    "success": True,
                    "message": response_text,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                logger.error("Error processing chat message: %s", str(e))
                response = {
                    "success": False,
                    "error": "Failed to process message"
                }
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def generate_response(self, message):
        """Generate AI response"""
        responses = [
            f"I understand you're asking about '{message}'. Based on my analysis, here's what I can tell you...",
            f"That's a great question about '{message}'. Let me provide you with a comprehensive response...",
            f"Regarding '{message}', I'd be happy to help you explore this topic in detail...",
            f"Thank you for your question about '{message}'. Here's my thoughtful response..."
        ]
        import random
        return random.choice(responses)
    
    def get_html_content(self):
        """Get the complete HTML content"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Assistant from Scratch</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            height: 100vh;
            overflow: hidden;
        }
        
        .app-container {
            display: flex;
            height: 100vh;
            width: 100vw;
        }
        
        /* Sidebar */
        .sidebar {
            width: 260px;
            background: #1a1a1a;
            border-right: 1px solid #2a2a2a;
            display: flex;
            flex-direction: column;
            padding: 20px 16px;
        }
        
        .sidebar-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            padding: 0 4px;
        }
        
        .sidebar-title {
            font-size: 18px;
            font-weight: 600;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .upgrade-btn {
            background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .upgrade-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
        }
        
        .new-chat-btn {
            width: 100%;
            background: transparent;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            padding: 12px 16px;
            color: #ffffff;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .new-chat-btn:hover {
            background: #2a2a2a;
            border-color: #4a4a4a;
        }
        
        .chat-list {
            flex: 1;
            overflow-y: auto;
        }
        
        .chat-item {
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 13px;
            color: #b0b0b0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .chat-item:hover {
            background: #2a2a2a;
            color: #ffffff;
        }
        
        .chat-item.active {
            background: #3a3a3a;
            color: #ffffff;
        }
        
        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0a0a0a;
        }
        
        /* Header */
        .main-header {
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-bottom: 1px solid #2a2a2a;
            background: rgba(26, 26, 26, 0.8);
            backdrop-filter: blur(10px);
        }
        
        .model-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #2a2a2a;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            padding: 8px 16px;
            color: #ffffff;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .model-selector:hover {
            background: #3a3a3a;
            border-color: #4a4a4a;
        }
        
        /* Chat Area */
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .welcome-screen {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .welcome-title {
            font-size: 48px;
            font-weight: 300;
            color: #ffffff;
            margin-bottom: 16px;
            background: linear-gradient(135deg, #4285f4 0%, #9c27b0 50%, #e91e63 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .welcome-subtitle {
            font-size: 20px;
            color: #888888;
            margin-bottom: 40px;
            font-weight: 400;
        }
        
        .quick-actions {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 40px;
        }
        
        .quick-action {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 12px 20px;
            color: #ffffff;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .quick-action:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(66, 133, 244, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(66, 133, 244, 0.2);
        }
        
        /* Messages */
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: none;
        }
        
        .message {
            margin-bottom: 24px;
            display: flex;
            gap: 12px;
            animation: fadeIn 0.3s ease-out;
        }
        
        .message.user {
            flex-direction: row-reverse;
        }
        
        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            flex-shrink: 0;
        }
        
        .message.user .message-avatar {
            background: linear-gradient(135deg, #4285f4 0%, #9c27b0 100%);
            color: white;
        }
        
        .message.assistant .message-avatar {
            background: #2a2a2a;
            color: #ffffff;
            border: 1px solid #3a3a3a;
        }
        
        .message-content {
            flex: 1;
            max-width: 70%;
        }
        
        .message.user .message-content {
            background: rgba(66, 133, 244, 0.1);
            border: 1px solid rgba(66, 133, 244, 0.2);
            border-radius: 18px 18px 4px 18px;
        }
        
        .message.assistant .message-content {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 18px 18px 18px 4px;
        }
        
        .message-text {
            padding: 16px 20px;
            color: #ffffff;
            font-size: 15px;
            line-height: 1.5;
            word-wrap: break-word;
        }
        
        /* Input Area */
        .input-area {
            padding: 20px;
            border-top: 1px solid #2a2a2a;
            background: rgba(26, 26, 26, 0.8);
            backdrop-filter: blur(10px);
        }
        
        .input-container {
            max-width: 800px;
            margin: 0 auto;
            position: relative;
        }
        
        .input-box {
            width: 100%;
            background: #1a1a1a;
            border: 1px solid #3a3a3a;
            border-radius: 24px;
            padding: 16px 60px 16px 20px;
            color: #ffffff;
            font-size: 15px;
            resize: none;
            outline: none;
            transition: all 0.3s;
            min-height: 56px;
            max-height: 200px;
            font-family: inherit;
        }
        
        .input-box:focus {
            border-color: #4285f4;
            box-shadow: 0 0 0 3px rgba(66, 133, 244, 0.1);
        }
        
        .input-box::placeholder {
            color: #666666;
        }
        
        .send-button {
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4285f4 0%, #9c27b0 100%);
            border: none;
            color: white;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }
        
        .send-button:hover {
            transform: translateY(-50%) scale(1.05);
            box-shadow: 0 4px 12px rgba(66, 133, 244, 0.3);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Settings Panel */
        .settings-panel {
            position: fixed;
            right: 20px;
            top: 20px;
            width: 300px;
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            z-index: 1000;
        }
        
        .settings-title {
            font-size: 16px;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .setting-item {
            margin-bottom: 16px;
        }
        
        .setting-label {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
            color: #b0b0b0;
        }
        
        .setting-value {
            font-size: 12px;
            color: #666666;
        }
        
        .slider {
            width: 100%;
            height: 6px;
            background: #3a3a3a;
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: linear-gradient(135deg, #4285f4 0%, #9c27b0 100%);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(66, 133, 244, 0.3);
        }
        
        .slider::-moz-range-thumb {
            width: 18px;
            height: 18px;
            background: linear-gradient(135deg, #4285f4 0%, #9c27b0 100%);
            border-radius: 50%;
            cursor: pointer;
            border: none;
            box-shadow: 0 2px 6px rgba(66, 133, 244, 0.3);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .typing-indicator {
            display: none;
            padding: 16px 20px;
            color: #888888;
            font-style: italic;
        }
        
        .typing-dots {
            display: inline-block;
        }
        
        .typing-dots::after {
            content: "...";
            animation: typing 1.5s infinite;
        }
        
        @keyframes typing {
            0% { content: "."; }
            33% { content: ".."; }
            66% { content: "..."; }
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .sidebar {
                width: 100%;
                transform: translateX(-100%);
                transition: transform 0.3s ease;
                position: fixed;
                z-index: 1000;
            }
            
            .sidebar.open {
                transform: translateX(0);
            }
            
            .main-content {
                width: 100%;
            }
            
            .welcome-title {
                font-size: 32px;
            }
            
            .settings-panel {
                width: calc(100% - 40px);
                right: 20px;
                left: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-title">
                    <span>🤖</span>
                    AI Assistant
                </div>
                <button class="upgrade-btn">Upgrade</button>
            </div>
            
            <button class="new-chat-btn" onclick="newChat()">
                <span>+</span>
                New chat
            </button>
            
            <div class="chat-list">
                <div class="chat-item active">Build Your Own AI</div>
                <div class="chat-item">Privacy & Security Discussion</div>
                <div class="chat-item">Email Address Inquiry</div>
                <div class="chat-item">Game Ideas List</div>
                <div class="chat-item">Retro Game Nest Idea</div>
                <div class="chat-item">Tax Filing for Profit</div>
                <div class="chat-item">Partnership Proposal Email</div>
                <div class="chat-item">CyberSkies Project</div>
                <div class="chat-item">Custom Domain with Netlify</div>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <!-- Header -->
            <div class="main-header">
                <div class="model-selector">
                    <span>🤖</span>
                    <span>AI Assistant</span>
                    <span>▼</span>
                </div>
            </div>
            
            <!-- Chat Area -->
            <div class="chat-area">
                <div class="welcome-screen" id="welcomeScreen">
                    <h1 class="welcome-title">Good evening, Hassan</h1>
                    <p class="welcome-subtitle">How can I help you today?</p>
                    
                    <div class="quick-actions">
                        <div class="quick-action" onclick="quickAction('code')">
                            <span>💻</span>
                            <span>Code</span>
                        </div>
                        <div class="quick-action" onclick="quickAction('create')">
                            <span>🎨</span>
                            <span>Create</span>
                        </div>
                        <div class="quick-action" onclick="quickAction('write')">
                            <span>📝</span>
                            <span>Write</span>
                        </div>
                        <div class="quick-action" onclick="quickAction('learn')">
                            <span>🎓</span>
                            <span>Learn</span>
                        </div>
                        <div class="quick-action" onclick="quickAction('life stuff')">
                            <span>🌟</span>
                            <span>Life stuff</span>
                        </div>
                    </div>
                </div>
                
                <div class="messages-container" id="messagesContainer">
                    <!-- Messages will be added here -->
                </div>
            </div>
            
            <!-- Input Area -->
            <div class="input-area">
                <div class="input-container">
                    <textarea class="input-box" id="messageInput" placeholder="Ask anything..." rows="1"></textarea>
                    <button class="send-button" id="sendButton" onclick="sendMessage()">
                        <span>→</span>
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Settings Panel -->
        <div class="settings-panel">
            <div class="settings-title">
                <span>⚙️</span>
                Settings
            </div>
            
            <div class="setting-item">
                <div class="setting-label">
                    <span>🔥 Temperature</span>
                    <span class="setting-value" id="tempValue">0.8</span>
                </div>
                <input type="range" class="slider" id="tempSlider" min="0.1" max="2" step="0.1" value="0.8" oninput="updateSetting('temp', this.value)">
                <div style="font-size: 11px; color: #666; margin-top: 4px;">Higher = more creative</div>
            </div>
            
            <div class="setting-item">
                <div class="setting-label">
                    <span>🔄 Top-p</span>
                    <span class="setting-value" id="toppValue">0.9</span>
                </div>
                <input type="range" class="slider" id="toppSlider" min="0.1" max="1" step="0.1" value="0.9" oninput="updateSetting('topp', this.value)">
                <div style="font-size: 11px; color: #666; margin-top: 4px;">Higher = more diverse</div>
            </div>
            
            <div class="setting-item">
                <div class="setting-label">
                    <span>📏 Max Length</span>
                    <span class="setting-value" id="lengthValue">150</span>
                </div>
                <input type="range" class="slider" id="lengthSlider" min="20" max="300" step="10" value="150" oninput="updateSetting('length', this.value)">
                <div style="font-size: 11px; color: #666; margin-top: 4px;">Response length limit</div>
            </div>
        </div>
    </div>
    
    <script>
        // Global variables
        let messageHistory = [];
        
        // Auto-resize textarea
        const messageInput = document.getElementById('messageInput');
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });
        
        // Send message function
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Hide welcome screen and show messages
            document.getElementById('welcomeScreen').style.display = 'none';
            document.getElementById('messagesContainer').style.display = 'block';
            
            // Add user message
            addMessage(message, 'user');
            
            // Clear input
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            // Show typing indicator
            showTypingIndicator();
            
            try {
                // Send to backend
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                // Hide typing indicator
                hideTypingIndicator();
                
                if (data.success) {
                    addMessage(data.message, 'assistant');
                } else {
                    addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
                }
            } catch (error) {
                hideTypingIndicator();
                addMessage('Sorry, I encountered a connection error. Please try again.', 'assistant');
            }
        }
        
        function addMessage(content, type) {
            const messagesContainer = document.getElementById('messagesContainer');
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            const avatar = type === 'user' ? '👤' : '🤖';
            
            messageDiv.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">
                    <div class="message-text">${content}</div>
                </div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            messageHistory.push({ type, content, timestamp: new Date().toISOString() });
        }
        
        function showTypingIndicator() {
            const messagesContainer = document.getElementById('messagesContainer');
            
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message assistant';
            typingDiv.id = 'typingIndicator';
            
            typingDiv.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="typing-indicator">
                        AI is typing<span class="typing-dots"></span>
                    </div>
                </div>
            `;
            
            messagesContainer.appendChild(typingDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function hideTypingIndicator() {
            const typingIndicator = document.getElementById('typingIndicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }
        
        function quickAction(action) {
            messageInput.value = `Help me with ${action}`;
            sendMessage();
        }
        
        function newChat() {
            document.getElementById('messagesContainer').innerHTML = '';
            document.getElementById('messagesContainer').style.display = 'none';
            document.getElementById('welcomeScreen').style.display = 'flex';
            messageHistory = [];
        }
        
        function updateSetting(setting, value) {
            document.getElementById(setting + 'Value').textContent = value;
        }
        
        // Enter key to send
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Chat item selection
        document.querySelectorAll('.chat-item').forEach(item => {
            item.addEventListener('click', function() {
                document.querySelectorAll('.chat-item').forEach(i => i.classList.remove('active'));
                this.classList.add('active');
            });
        });
        
        // Update greeting based on time
        function updateGreeting() {
            const hour = new Date().getHours();
            let greeting;
            
            if (hour < 12) {
                greeting = 'Good morning, Hassan';
            } else if (hour < 17) {
                greeting = 'Good afternoon, Hassan';
            } else {
                greeting = 'Good evening, Hassan';
            }
            
            document.querySelector('.welcome-title').textContent = greeting;
        }
        
        // Initialize
        updateGreeting();
        
        // Focus input on load
        window.addEventListener('load', function() {
            messageInput.focus();
        });
    </script>
</body>
</html>"""

def start_server():
    """Start the HTTP server"""
    PORT = 8080
    
    with socketserver.TCPServer(("", PORT), AIAssistantHandler) as httpd:
        logger.info("Server starting at http://localhost:%d", PORT)
        logger.info("Opening browser...")
        
        # Open browser
        webbrowser.open(f'http://localhost:{PORT}')
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped")

def main():
    """Main function"""
    try:
        logger.info("Starting Pure HTML AI Assistant...")
        start_server()
    except Exception as e:
        logger.error("Failed to start server: %s", str(e))
        raise

if __name__ == "__main__":
    main()
