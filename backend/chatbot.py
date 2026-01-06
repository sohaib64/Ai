"""
Enhanced Conversational Chatbot with Gemini API (UPDATED)
Using new google-genai package
"""

from google import genai
from google.genai import types
from datetime import datetime
import re
import os

class GeminiChatbot:
    """
    AI-powered chatbot using Gemini for natural conversations
    """
    
    def __init__(self, api_key=None):
        # Configure Gemini API
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.model_name = 'gemini-2.5-flash'  # Updated model name
                print(f"✅ Gemini API configured successfully! Using model: {self.model_name}")
            except Exception as e:
                print(f"❌ Gemini API configuration failed: {e}")
                self.client = None
        else:
            print("⚠️ Warning: GEMINI_API_KEY not set. Using rule-based fallback.")
            self.client = None
        
        self.sessions = {}
        
        # System prompt for Gemini
        self.system_prompt = """You are an AI assistant for a Pakistani car price prediction service. Your role:

1. Help users predict used car prices in Pakistan
2. Answer questions about car values, market trends, depreciation
3. Explain why certain factors affect price (brand, mileage, age, city, etc.)
4. Be friendly, conversational, and use simple Urdu/English mix if needed

When user wants to predict price, collect these details conversationally:
- Car brand (Honda, Toyota, Suzuki, etc.)
- Model (Civic, Corolla, Cultus, etc.)
- City (Karachi, Lahore, Islamabad, or any Pakistan city)
- Fuel type (Petrol, Diesel, Hybrid, CNG)
- Engine (cc, e.g., 1300, 1800)
- Transmission (Manual/Automatic)
- Registration year (e.g., 2018)
- Mileage (km, e.g., 50000)

Important guidelines:
- Don't ask all questions at once - collect one by one naturally
- If user asks "why is my car cheaper?", explain based on age, mileage, brand depreciation
- If asked about market trends, explain Pakistani car market insights
- Keep responses concise (2-3 sentences max unless explaining)
- Use emojis moderately 🚗💰

Pakistani Car Market Context:
- Economy segment: Under 20 Lacs
- Mid-Range: 20-40 Lacs  
- Premium: 40-70 Lacs
- Luxury: Above 70 Lacs
- Popular brands: Toyota (holds value best), Honda, Suzuki (affordable)
- Cities: Karachi/Lahore have more demand, better resale
- Depreciation: ~15-20% per year for first 5 years
"""
        
        self.steps = [
            'car_brand', 'car_model', 'city', 'fuel_type',
            'engine', 'transmission', 'registered_in', 'mileage'
        ]
    
    def create_session(self, session_id):
        """Create new chat session"""
        self.sessions[session_id] = {
            'step': 0,
            'data': {},
            'collecting': False,
            'history': [],
            'started': datetime.now()
        }
    
    def get_session(self, session_id):
        """Get or create session"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        return self.sessions[session_id]
    
    def detect_prediction_intent(self, message):
        """Check if user wants to predict price"""
        keywords = ['predict', 'price', 'value', 'kitni', 'kya', 'estimate', 'worth', 'check price', 'batao', 'bata']
        message_lower = message.lower()
        return any(kw in message_lower for kw in keywords)
    
    def extract_car_data(self, message, current_step):
        """Extract car information from natural language"""
        message = message.lower().strip()
        
        if current_step == 'car_brand':
            brands = ['honda', 'toyota', 'suzuki', 'daihatsu', 'nissan', 'mitsubishi', 'hyundai', 'kia']
            for brand in brands:
                if brand in message:
                    return brand
        
        elif current_step == 'car_model':
            models = ['civic', 'corolla', 'city', 'cultus', 'swift', 'vitz', 'yaris', 'fit', 'alto', 'mehran', 'bolan']
            for model in models:
                if model in message:
                    return model
        
        elif current_step == 'fuel_type':
            fuels = ['petrol', 'diesel', 'hybrid', 'cng', 'lpg']
            for fuel in fuels:
                if fuel in message:
                    return fuel
        
        elif current_step == 'transmission':
            if 'auto' in message:
                return 'automatic'
            elif 'manual' in message:
                return 'manual'
        
        elif current_step == 'engine':
            numbers = re.findall(r'\d+', message)
            if numbers:
                engine = int(numbers[0])
                if 500 <= engine <= 5000:
                    return str(engine)
        
        elif current_step == 'registered_in':
            numbers = re.findall(r'\b(19|20)\d{2}\b', message)
            if numbers:
                year = int(numbers[0])
                if 1990 <= year <= 2025:
                    return str(year)
        
        elif current_step == 'mileage':
            numbers = re.findall(r'\d+', message.replace(',', ''))
            if numbers:
                mileage = int(numbers[0])
                if 0 <= mileage <= 500000:
                    return str(mileage)
        
        elif current_step == 'city':
            return message
        
        return None
    
    def get_gemini_response(self, session, user_message):
        """Get response from Gemini API using new package"""
        if not self.client:
            return self.get_fallback_response(session, user_message)
        
        try:
            # Build conversation history
            conversation_history = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in session['history'][-10:]
            ])
            
            # Add context
            context = ""
            if session['collecting']:
                current_step = self.steps[session['step']]
                collected = list(session['data'].keys())
                context = f"\n\nCurrent task: Collecting {current_step}. Already collected: {collected}"
            
            # Generate response using new API
            prompt = f"{self.system_prompt}\n\nConversation:\n{conversation_history}\nUser: {user_message}{context}\n\nAssistant:"
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            return response.text
        
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return self.get_fallback_response(session, user_message)
    
    def get_fallback_response(self, session, message):
        """Simple rule-based fallback"""
        if self.detect_prediction_intent(message):
            return "Sure! I can help you predict the car price. Let's start - what's the car brand? (e.g., Honda, Toyota, Suzuki)"
        
        return "I'm here to help you predict car prices! Say 'predict' to start, or ask me anything about car values in Pakistan. 🚗"
    
    def process_message(self, session_id, message, model_info=None):
        """Process user message with Gemini AI"""
        session = self.get_session(session_id)
        
        # Add to history
        session['history'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check if user wants to predict
        if not session['collecting'] and self.detect_prediction_intent(message):
            session['collecting'] = True
            session['step'] = 0
            session['data'] = {}
            reply = "Great! Let's predict the price. What's the car brand? (e.g., Honda, Toyota, Suzuki) 🚗"
        
        # If collecting data
        elif session['collecting'] and session['step'] < len(self.steps):
            current_step = self.steps[session['step']]
            extracted_value = self.extract_car_data(message, current_step)
            
            if extracted_value:
                session['data'][current_step] = extracted_value
                session['step'] += 1
                
                # Check if done
                if session['step'] >= len(self.steps):
                    session['collecting'] = False
                    reply = "Perfect! I have all the details. Let me calculate the price... 💰"
                    action = 'predict'
                    
                    session['history'].append({
                        'role': 'assistant',
                        'content': reply,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    return {
                        'reply': reply,
                        'action': action,
                        'data': session['data']
                    }
                
                # Ask next question
                next_step = self.steps[session['step']]
                questions = {
                    'car_model': 'Got it! What model?',
                    'city': 'Which city?',
                    'fuel_type': 'Fuel type? (Petrol/Diesel/Hybrid)',
                    'engine': 'Engine capacity in cc? (e.g., 1300, 1800)',
                    'transmission': 'Manual or Automatic?',
                    'registered_in': 'Registration year? (e.g., 2018)',
                    'mileage': 'Mileage in km? (e.g., 50000)'
                }
                reply = questions.get(next_step, 'Next?')
            else:
                reply = self.get_gemini_response(session, message)
        
        # Normal conversation
        else:
            reply = self.get_gemini_response(session, message)
        
        # Add assistant reply to history
        session['history'].append({
            'role': 'assistant',
            'content': reply,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'reply': reply,
            'action': None
        }


# Initialize chatbot
chatbot = GeminiChatbot()