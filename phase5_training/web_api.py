"""
Simple Web API for Basic AI Assistant

This creates a simple Flask web API that the HTML interface can call
to get actual AI responses instead of just template responses.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from pathlib import Path

# Import our AI assistant
import sys
sys.path.append(str(Path(__file__).parent))

from basic_ai_interface import BasicAIAssistant

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Initialize the AI assistant
model_path = Path(__file__).parent.parent / "outputs" / "basic_ai_model.pth"
assistant = BasicAIAssistant(str(model_path) if model_path.exists() else None)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests from the web interface."""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Generate response using our AI assistant
        response = assistant.generate_response(user_message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Get the status of the AI assistant."""
    model_available = assistant.model is not None
    
    return jsonify({
        'model_loaded': model_available,
        'mode': 'neural_network' if model_available else 'rule_based',
        'status': 'ready'
    })


@app.route('/')
def index():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent.parent / "index.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "AI Assistant API is running. HTML interface not found."


if __name__ == '__main__':
    print("🚀 Starting Basic AI Assistant Web API...")
    print("📡 API will be available at: http://localhost:5000")
    print("🌐 Web interface will be available at: http://localhost:5000")
    print("📊 API Status: http://localhost:5000/api/status")
    print("💬 Chat API: POST http://localhost:5000/api/chat")
    print("-" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
