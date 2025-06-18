from flask import Flask, request, render_template, jsonify, session
import google.generativeai as genai
import os
from datetime import datetime, timedelta
import logging
from functools import wraps
import time
import json

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')  # Change this in production

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key from environment variable (more secure)
API_KEY = os.environ.get("GOOGLE_API_KEY", "your_api_key")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

genai.configure(api_key=API_KEY)

# Initialize the model with enhanced configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 150,  # Concise responses for voice
}

model = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config=generation_config
)

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 10  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds


def get_user_conversation_history():
    """Get or create conversation history for the current session"""
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    return session['conversation_history']


def rate_limit():
    """Simple rate limiting decorator"""

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Simple in-memory rate limiting (use Redis in production)
            client_id = request.remote_addr
            current_time = time.time()

            if not hasattr(rate_limit, 'requests'):
                rate_limit.requests = {}

            if client_id not in rate_limit.requests:
                rate_limit.requests[client_id] = []

            # Clean old requests
            rate_limit.requests[client_id] = [
                req_time for req_time in rate_limit.requests[client_id]
                if current_time - req_time < RATE_LIMIT_WINDOW
            ]

            # Check rate limit
            if len(rate_limit.requests[client_id]) >= RATE_LIMIT_REQUESTS:
                return jsonify({
                    'error': 'Rate limit exceeded. Please slow down.',
                    'retry_after': RATE_LIMIT_WINDOW
                }), 429

            # Add current request
            rate_limit.requests[client_id].append(current_time)

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def voice_assistance(user_input, conversation_history):
    """Enhanced voice assistance with context awareness"""
    try:
        # Build context from recent conversation
        context = ""
        if len(conversation_history) > 0:
            # Include last 3 exchanges for context
            recent_history = conversation_history[-3:]
            context = "Previous conversation:\n"
            for entry in recent_history:
                context += f"User: {entry['user']}\nAI: {entry['ai']}\n"
            context += "\n"

        prompt = f"""
        {context}You are a helpful AI voice assistant. Provide concise, conversational responses suitable for text-to-speech.
        Keep responses under 100 words and speak naturally as if having a conversation.

        Current user question: "{user_input}"

        Guidelines:
        - Be conversational and friendly
        - Give direct, practical answers
        - If you don't know something, say so briefly
        - For complex topics, offer to explain further
        - Use natural speech patterns
        """

        response = model.generate_content(prompt)

        if response and response.text:
            return response.text.strip()
        else:
            return "I'm sorry, I couldn't process that request. Could you try rephrasing?"

    except Exception as e:
        logger.error(f"Error in voice_assistance: {str(e)}")
        return "I'm experiencing some technical difficulties. Please try again in a moment."


@app.route("/")
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/process_voice', methods=['POST'])
@rate_limit()
def process_voice():
    """Process voice input with enhanced error handling and features"""
    try:
        # Validate input
        if not request.json or 'user_input' not in request.json:
            return jsonify({'error': 'Invalid request format'}), 400

        user_input = request.json.get('user_input', '').strip()

        if not user_input:
            return jsonify({'error': 'Empty input received'}), 400

        if len(user_input) > 500:  # Limit input length
            return jsonify({'error': 'Input too long. Please keep it under 500 characters.'}), 400

        # Get user's conversation history
        conversation_history = get_user_conversation_history()

        # Generate AI response
        ai_response = voice_assistance(user_input, conversation_history)

        # Create conversation entry with timestamp
        conversation_entry = {
            'user': user_input,
            'ai': ai_response,
            'timestamp': datetime.now().isoformat()
        }

        # Add to session history (keep only last 20 exchanges)
        conversation_history.append(conversation_entry)
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        session['conversation_history'] = conversation_history

        # Log the interaction (remove in production or use proper logging)
        logger.info(f"User input: {user_input[:50]}...")

        return jsonify({
            'response': ai_response,
            'conversation_history': conversation_history,
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Error in process_voice: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred. Please try again.',
            'status': 'error'
        }), 500


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    session['conversation_history'] = []
    return jsonify({'message': 'Conversation history cleared', 'status': 'success'})


@app.route('/get_history', methods=['GET'])
def get_history():
    """Get current conversation history"""
    conversation_history = get_user_conversation_history()
    return jsonify({
        'conversation_history': conversation_history,
        'status': 'success'
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@app.before_request
def before_request():
    """Set session timeout and other pre-request logic"""
    session.permanent = True
    app.permanent_session_lifetime = timedelta(hours=24)


if __name__ == "__main__":
    # Enhanced development server configuration
    app.run(
        debug=True,
        host='0.0.0.0',  # Allow external connections
        port=int(os.environ.get('PORT', 5000)),
        threaded=True
    )