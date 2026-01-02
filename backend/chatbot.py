# # """
# # Conversational Chatbot for Car Price Prediction
# # Rule-based chatbot with context management and step-by-step data collection
# # """

# # from datetime import datetime
# # import re

# # class ConversationalChatbot:
# #     """
# #     Rule-based chatbot that guides users through car price prediction
# #     """
    
# #     def __init__(self):
# #         self.sessions = {}  # Store user sessions
        
# #         # Conversation steps
# #         self.steps = [
# #             'car_brand',
# #             'car_model', 
# #             'city',
# #             'fuel_type',
# #             'engine',
# #             'transmission',
# #             'registered_in',
# #             'mileage'
# #         ]
        
# #         # Valid options (loaded from feature_info)
# #         self.valid_options = {
# #             'car_brand': ['honda', 'toyota', 'suzuki', 'daihatsu', 'nissan', 'mitsubishi', 'hyundai', 'kia'],
# #             'car_model': ['civic', 'corolla', 'city', 'cultus', 'swift', 'vitz', 'yaris', 'fit'],
# #             'fuel_type': ['petrol', 'diesel', 'hybrid', 'cng', 'lpg'],
# #             'transmission': ['manual', 'automatic'],
# #         }
        
# #         # City mapping
# #         self.city_mapping = {
# #             'karachi': 'karachi', 'khi': 'karachi', 'hyderabad': 'karachi',
# #             'lahore': 'lahore', 'lhr': 'lahore', 'faisalabad': 'lahore', 'multan': 'lahore',
# #             'islamabad': 'islamabad', 'isb': 'islamabad', 'rawalpindi': 'islamabad', 'peshawar': 'islamabad'
# #         }
    
# #     def create_session(self, session_id):
# #         """Create new chat session"""
# #         self.sessions[session_id] = {
# #             'step': 0,
# #             'data': {},
# #             'started': datetime.now(),
# #             'last_message': None
# #         }
    
# #     def get_session(self, session_id):
# #         """Get or create session"""
# #         if session_id not in self.sessions:
# #             self.create_session(session_id)
# #         return self.sessions[session_id]
    
# #     def detect_intent(self, message):
# #         """Detect user intent from message"""
# #         message = message.lower().strip()
        
# #         intents = {
# #             'greeting': ['hello', 'hi', 'hey', 'start', 'salam'],
# #             'predict': ['predict', 'price', 'value', 'estimate', 'worth'],
# #             'help': ['help', 'how', 'what', 'guide'],
# #             'reset': ['reset', 'restart', 'start over', 'cancel'],
# #             'info': ['accuracy', 'model', 'about', 'info'],
# #             'thanks': ['thank', 'thanks', 'bye', 'goodbye']
# #         }
        
# #         for intent, keywords in intents.items():
# #             if any(kw in message for kw in keywords):
# #                 return intent
        
# #         return 'answer'  # Default: treat as answer to current question
    
# #     def validate_input(self, step_name, value):
# #         """Validate user input for current step"""
# #         value = value.lower().strip()
        
# #         if step_name == 'car_brand':
# #             return value in self.valid_options['car_brand'], self.valid_options['car_brand']
        
# #         elif step_name == 'car_model':
# #             return value in self.valid_options['car_model'], self.valid_options['car_model']
        
# #         elif step_name == 'city':
# #             return value in self.city_mapping or len(value) > 2, list(self.city_mapping.keys())[:10]
        
# #         elif step_name == 'fuel_type':
# #             return value in self.valid_options['fuel_type'], self.valid_options['fuel_type']
        
# #         elif step_name == 'engine':
# #             try:
# #                 engine = float(value)
# #                 return 500 <= engine <= 5000, "Engine should be between 500cc - 5000cc"
# #             except:
# #                 return False, "Please enter a valid number (e.g., 1300)"
        
# #         elif step_name == 'transmission':
# #             return value in self.valid_options['transmission'], self.valid_options['transmission']
        
# #         elif step_name == 'registered_in':
# #             try:
# #                 year = int(value)
# #                 return 1990 <= year <= 2025, "Year should be between 1990-2025"
# #             except:
# #                 return False, "Please enter a valid year (e.g., 2018)"
        
# #         elif step_name == 'mileage':
# #             try:
# #                 mileage = float(value)
# #                 return 0 <= mileage <= 500000, "Mileage should be between 0-500,000 km"
# #             except:
# #                 return False, "Please enter a valid number (e.g., 50000)"
        
# #         return False, "Invalid input"
    
# #     def get_question(self, step_name):
# #         """Get question for current step"""
# #         questions = {
# #             'car_brand': "🚗 What's the car brand?\n(e.g., Honda, Toyota, Suzuki)",
# #             'car_model': "🏷️ What's the model?\n(e.g., Civic, Corolla, Cultus)",
# #             'city': "📍 Which city?\n(e.g., Karachi, Lahore, Islamabad, or any Pakistan city)",
# #             'fuel_type': "⛽ Fuel type?\n(e.g., Petrol, Diesel, Hybrid)",
# #             'engine': "🔧 Engine capacity in cc?\n(e.g., 1300, 1500, 1800)",
# #             'transmission': "⚙️ Transmission type?\n(Manual or Automatic)",
# #             'registered_in': "📅 Registration year?\n(e.g., 2018, 2020)",
# #             'mileage': "📏 Mileage in km?\n(e.g., 50000, 80000)"
# #         }
# #         return questions.get(step_name, "Please provide the information")
    
# #     def process_message(self, session_id, message, model_info=None):
# #         """
# #         Process user message and return response
        
# #         Returns:
# #             dict: {
# #                 'reply': str,
# #                 'options': list (optional),
# #                 'action': str (optional),
# #                 'prediction': dict (optional)
# #             }
# #         """
# #         session = self.get_session(session_id)
# #         intent = self.detect_intent(message)
        
# #         # Handle intents
# #         if intent == 'greeting':
# #             return {
# #                 'reply': "👋 Hello! I'm your AI Car Price Assistant.\n\n"
# #                         "I can help you predict the price of used cars in Pakistan.\n\n"
# #                         "Just tell me the car details, and I'll give you an instant estimate!\n\n"
# #                         "Ready to start? 🚗",
# #                 'action': 'start'
# #             }
        
# #         elif intent == 'predict':
# #             session['step'] = 0
# #             session['data'] = {}
# #             question = self.get_question(self.steps[0])
# #             return {
# #                 'reply': f"Great! Let's predict the price. 💰\n\n{question}",
# #                 'action': 'collect_data'
# #             }
        
# #         elif intent == 'help':
# #             return {
# #                 'reply': "🤖 **How to use:**\n\n"
# #                         "1. Tell me you want to predict a price\n"
# #                         "2. I'll ask 8 simple questions about the car\n"
# #                         "3. Answer each question\n"
# #                         "4. I'll predict the price instantly!\n\n"
# #                         f"**Model Accuracy:** {model_info.get('test_r2', 0.94)*100:.1f}% ✨\n\n"
# #                         "Say 'predict' to start!",
# #                 'action': None
# #             }
        
# #         elif intent == 'reset':
# #             session['step'] = 0
# #             session['data'] = {}
# #             return {
# #                 'reply': "🔄 Session reset! Say 'predict' to start fresh.",
# #                 'action': 'reset'
# #             }
        
# #         elif intent == 'info':
# #             return {
# #                 'reply': f"📊 **Model Information:**\n\n"
# #                         f"• Algorithm: {model_info.get('model_name', 'Random Forest')}\n"
# #                         f"• Accuracy (R²): {model_info.get('test_r2', 0.94)*100:.1f}%\n"
# #                         f"• Average Error: ±{model_info.get('test_mae', 185000):,.0f} PKR\n"
# #                         f"• Features: Polynomial + K-Means Clustering\n"
# #                         f"• Dataset: 5,497 cars\n\n"
# #                         f"Want to predict a price? Say 'predict'! 🚗",
# #                 'action': None
# #             }
        
# #         elif intent == 'thanks':
# #             return {
# #                 'reply': "You're welcome! 😊\n\n"
# #                         "Feel free to predict more car prices anytime!\n\n"
# #                         "Have a great day! 🚗✨",
# #                 'action': 'end'
# #             }
        
# #         # Handle data collection (answering questions)
# #         elif intent == 'answer' and session['step'] < len(self.steps):
# #             current_step = self.steps[session['step']]
            
# #             # Validate input
# #             is_valid, options = self.validate_input(current_step, message)
            
# #             if not is_valid:
# #                 if isinstance(options, list):
# #                     return {
# #                         'reply': f"❌ Invalid {current_step.replace('_', ' ')}.\n\n"
# #                                 f"Please choose from: {', '.join(options[:5])}...\n\n"
# #                                 f"Or type 'help' for guidance.",
# #                         'options': options
# #                     }
# #                 else:
# #                     return {
# #                         'reply': f"❌ {options}\n\nPlease try again.",
# #                         'action': None
# #                     }
            
# #             # Store valid input
# #             if current_step == 'city':
# #                 session['data'][current_step] = self.city_mapping.get(message.lower(), message.lower())
# #             else:
# #                 session['data'][current_step] = message.lower().strip()
            
# #             session['step'] += 1
            
# #             # Check if all data collected
# #             if session['step'] >= len(self.steps):
# #                 return {
# #                     'reply': "✅ Got all the details!\n\n"
# #                             "📊 Let me calculate the price...\n\n"
# #                             "⏳ Processing...",
# #                     'action': 'predict',
# #                     'data': session['data']
# #                 }
            
# #             # Ask next question
# #             next_step = self.steps[session['step']]
# #             question = self.get_question(next_step)
            
# #             return {
# #                 'reply': f"✓ Got it!\n\n{question}",
# #                 'action': 'continue'
# #             }
        
# #         # Default response
# #         return {
# #             'reply': "I didn't understand that. 🤔\n\n"
# #                     "Say 'predict' to start, or 'help' for guidance.",
# #             'action': None
# #         }


# # # Global chatbot instance
# # chatbot = ConversationalChatbot()


# """
# Conversational Chatbot for Car Price Prediction
# Rule-based chatbot with context management and step-by-step data collection
# """

# from datetime import datetime
# import re

# class ConversationalChatbot:
#     """
#     Rule-based chatbot that guides users through car price prediction
#     """
    
#     def __init__(self):
#         self.sessions = {}  # Store user sessions
        
#         # Conversation steps
#         self.steps = [
#             'car_brand',
#             'car_model', 
#             'city',
#             'fuel_type',
#             'engine',
#             'transmission',
#             'registered_in',
#             'mileage'
#         ]
        
#         # Valid options (loaded from feature_info)
#         self.valid_options = {
#             'car_brand': ['honda', 'toyota', 'suzuki', 'daihatsu', 'nissan', 'mitsubishi', 'hyundai', 'kia'],
#             'car_model': ['civic', 'corolla', 'city', 'cultus', 'swift', 'vitz', 'yaris', 'fit'],
#             'fuel_type': ['petrol', 'diesel', 'hybrid', 'cng', 'lpg'],
#             'transmission': ['manual', 'automatic'],
#         }
        
#         # City mapping
#         self.city_mapping = {
#             'karachi': 'karachi', 'khi': 'karachi', 'hyderabad': 'karachi',
#             'lahore': 'lahore', 'lhr': 'lahore', 'faisalabad': 'lahore', 'multan': 'lahore',
#             'islamabad': 'islamabad', 'isb': 'islamabad', 'rawalpindi': 'islamabad', 'peshawar': 'islamabad'
#         }
    
#     def create_session(self, session_id):
#         """Create new chat session"""
#         self.sessions[session_id] = {
#             'step': 0,
#             'data': {},
#             'started': datetime.now(),
#             'last_message': None
#         }
    
#     def get_session(self, session_id):
#         """Get or create session"""
#         if session_id not in self.sessions:
#             self.create_session(session_id)
#         return self.sessions[session_id]
    
#     def detect_intent(self, message, session):
#         """
#         Detect user intent from message
#         Considers session context to avoid false positives
#         """
#         message = message.lower().strip()
        
#         # If in data collection mode, treat as answer unless explicit command
#         if session['step'] > 0 and session['step'] < len(self.steps):
#             # Check for explicit commands only
#             explicit_commands = {
#                 'reset': ['reset', 'restart', 'cancel'],
#                 'help': ['help me', 'i need help']
#             }
            
#             for intent, keywords in explicit_commands.items():
#                 if any(message == kw for kw in keywords):
#                     return intent
            
#             # Otherwise treat as answer
#             return 'answer'
        
#         # Not in collection mode - detect intent normally
#         intents = {
#             'greeting': ['hello', 'hi', 'hey', 'salam', 'start'],
#             'predict': ['predict', 'price', 'value', 'estimate', 'worth', 'check price'],
#             'help': ['help', 'how', 'guide', 'what can you do'],
#             'reset': ['reset', 'restart', 'start over', 'cancel'],
#             'info': ['accuracy', 'model', 'about', 'info', 'details'],
#             'thanks': ['thank', 'thanks', 'bye', 'goodbye']
#         }
        
#         # Exact matches first
#         for intent, keywords in intents.items():
#             if message in keywords:
#                 return intent
        
#         # Partial matches
#         for intent, keywords in intents.items():
#             if any(kw in message for kw in keywords):
#                 return intent
        
#         return 'answer'  # Default: treat as answer
    
#     def validate_input(self, step_name, value):
#         """Validate user input for current step"""
#         value = value.lower().strip()
        
#         if step_name == 'car_brand':
#             return value in self.valid_options['car_brand'], self.valid_options['car_brand']
        
#         elif step_name == 'car_model':
#             return value in self.valid_options['car_model'], self.valid_options['car_model']
        
#         elif step_name == 'city':
#             return value in self.city_mapping or len(value) > 2, list(self.city_mapping.keys())[:10]
        
#         elif step_name == 'fuel_type':
#             return value in self.valid_options['fuel_type'], self.valid_options['fuel_type']
        
#         elif step_name == 'engine':
#             try:
#                 engine = float(value)
#                 return 500 <= engine <= 5000, "Engine should be between 500cc - 5000cc"
#             except:
#                 return False, "Please enter a valid number (e.g., 1300)"
        
#         elif step_name == 'transmission':
#             return value in self.valid_options['transmission'], self.valid_options['transmission']
        
#         elif step_name == 'registered_in':
#             try:
#                 year = int(value)
#                 return 1990 <= year <= 2025, "Year should be between 1990-2025"
#             except:
#                 return False, "Please enter a valid year (e.g., 2018)"
        
#         elif step_name == 'mileage':
#             try:
#                 mileage = float(value)
#                 return 0 <= mileage <= 500000, "Mileage should be between 0-500,000 km"
#             except:
#                 return False, "Please enter a valid number (e.g., 50000)"
        
#         return False, "Invalid input"
    
#     def get_question(self, step_name):
#         """Get question for current step"""
#         questions = {
#             'car_brand': "🚗 What's the car brand?\n(e.g., Honda, Toyota, Suzuki)",
#             'car_model': "🏷️ What's the model?\n(e.g., Civic, Corolla, Cultus)",
#             'city': "📍 Which city?\n(e.g., Karachi, Lahore, Islamabad, or any Pakistan city)",
#             'fuel_type': "⛽ Fuel type?\n(e.g., Petrol, Diesel, Hybrid)",
#             'engine': "🔧 Engine capacity in cc?\n(e.g., 1300, 1500, 1800)",
#             'transmission': "⚙️ Transmission type?\n(Manual or Automatic)",
#             'registered_in': "📅 Registration year?\n(e.g., 2018, 2020)",
#             'mileage': "📏 Mileage in km?\n(e.g., 50000, 80000)"
#         }
#         return questions.get(step_name, "Please provide the information")
    
#     def process_message(self, session_id, message, model_info=None):
#         """
#         Process user message and return response
        
#         Returns:
#             dict: {
#                 'reply': str,
#                 'options': list (optional),
#                 'action': str (optional),
#                 'prediction': dict (optional),
#                 'session_info': dict (for debugging)
#             }
#         """
#         session = self.get_session(session_id)
#         intent = self.detect_intent(message, session)  # Pass session for context
        
#         # Debug info
#         session_info = {
#             'step': session['step'],
#             'total_steps': len(self.steps),
#             'current_field': self.steps[session['step']] if session['step'] < len(self.steps) else 'completed',
#             'collected_data': list(session['data'].keys())
#         }
        
#         # Handle intents
#         if intent == 'greeting':
#             return {
#                 'reply': "👋 Hello! I'm your AI Car Price Assistant.\n\n"
#                         "I can help you predict the price of used cars in Pakistan.\n\n"
#                         "Just tell me the car details, and I'll give you an instant estimate!\n\n"
#                         "Ready to start? 🚗",
#                 'action': 'start',
#                 'session_info': session_info
#             }
        
#         elif intent == 'predict':
#             session['step'] = 0
#             session['data'] = {}
#             question = self.get_question(self.steps[0])
#             return {
#                 'reply': f"Great! Let's predict the price. 💰\n\n{question}",
#                 'action': 'collect_data',
#                 'session_info': session_info
#             }
        
#         elif intent == 'help':
#             return {
#                 'reply': "🤖 **How to use:**\n\n"
#                         "1. Tell me you want to predict a price\n"
#                         "2. I'll ask 8 simple questions about the car\n"
#                         "3. Answer each question\n"
#                         "4. I'll predict the price instantly!\n\n"
#                         f"**Model Accuracy:** {model_info.get('test_r2', 0.94)*100:.1f}% ✨\n\n"
#                         "Say 'predict' to start!",
#                 'action': None
#             }
        
#         elif intent == 'reset':
#             session['step'] = 0
#             session['data'] = {}
#             return {
#                 'reply': "🔄 Session reset! Say 'predict' to start fresh.",
#                 'action': 'reset'
#             }
        
#         elif intent == 'info':
#             return {
#                 'reply': f"📊 **Model Information:**\n\n"
#                         f"• Algorithm: {model_info.get('model_name', 'Random Forest')}\n"
#                         f"• Accuracy (R²): {model_info.get('test_r2', 0.94)*100:.1f}%\n"
#                         f"• Average Error: ±{model_info.get('test_mae', 185000):,.0f} PKR\n"
#                         f"• Features: Polynomial + K-Means Clustering\n"
#                         f"• Dataset: 5,497 cars\n\n"
#                         f"Want to predict a price? Say 'predict'! 🚗",
#                 'action': None
#             }
        
#         elif intent == 'thanks':
#             return {
#                 'reply': "You're welcome! 😊\n\n"
#                         "Feel free to predict more car prices anytime!\n\n"
#                         "Have a great day! 🚗✨",
#                 'action': 'end'
#             }
        
#         # Handle data collection (answering questions)
#         elif intent == 'answer' and session['step'] < len(self.steps):
#             current_step = self.steps[session['step']]
            
#             # Validate input
#             is_valid, options = self.validate_input(current_step, message)
            
#             if not is_valid:
#                 if isinstance(options, list):
#                     return {
#                         'reply': f"❌ Invalid {current_step.replace('_', ' ')}.\n\n"
#                                 f"Please choose from: {', '.join(options[:5])}...\n\n"
#                                 f"Or type 'help' for guidance.",
#                         'options': options
#                     }
#                 else:
#                     return {
#                         'reply': f"❌ {options}\n\nPlease try again.",
#                         'action': None
#                     }
            
#             # Store valid input
#             if current_step == 'city':
#                 session['data'][current_step] = self.city_mapping.get(message.lower(), message.lower())
#             else:
#                 session['data'][current_step] = message.lower().strip()
            
#             session['step'] += 1
            
#             # Check if all data collected
#             if session['step'] >= len(self.steps):
#                 return {
#                     'reply': "✅ Got all the details!\n\n"
#                             "📊 Let me calculate the price...\n\n"
#                             "⏳ Processing...",
#                     'action': 'predict',
#                     'data': session['data']
#                 }
            
#             # Ask next question
#             next_step = self.steps[session['step']]
#             question = self.get_question(next_step)
            
#             return {
#                 'reply': f"✓ Got it!\n\n{question}",
#                 'action': 'continue'
#             }
        
#         # Default response
#         return {
#             'reply': "I didn't understand that. 🤔\n\n"
#                     "Say 'predict' to start, or 'help' for guidance.",
#             'action': None,
#             'session_info': session_info
#         }


# # Global chatbot instance
# chatbot = ConversationalChatbot()


# """
# Conversational Chatbot for Car Price Prediction
# Rule-based chatbot with context management and step-by-step data collection
# """

# from datetime import datetime
# import re

# class ConversationalChatbot:
#     """
#     Rule-based chatbot that guides users through car price prediction
#     """
    
#     def __init__(self):
#         self.sessions = {}  # Store user sessions
        
#         # Conversation steps
#         self.steps = [
#             'car_brand',
#             'car_model', 
#             'city',
#             'fuel_type',
#             'engine',
#             'transmission',
#             'registered_in',
#             'mileage'
#         ]
        
#         # Valid options (loaded from feature_info)
#         self.valid_options = {
#             'car_brand': ['honda', 'toyota', 'suzuki', 'daihatsu', 'nissan', 'mitsubishi', 'hyundai', 'kia'],
#             'car_model': ['civic', 'corolla', 'city', 'cultus', 'swift', 'vitz', 'yaris', 'fit'],
#             'fuel_type': ['petrol', 'diesel', 'hybrid', 'cng', 'lpg'],
#             'transmission': ['manual', 'automatic'],
#         }
        
#         # City mapping
#         self.city_mapping = {
#             'karachi': 'karachi', 'khi': 'karachi', 'hyderabad': 'karachi',
#             'lahore': 'lahore', 'lhr': 'lahore', 'faisalabad': 'lahore', 'multan': 'lahore',
#             'islamabad': 'islamabad', 'isb': 'islamabad', 'rawalpindi': 'islamabad', 'peshawar': 'islamabad'
#         }
    
#     def create_session(self, session_id):
#         """Create new chat session"""
#         self.sessions[session_id] = {
#             'step': 0,
#             'data': {},
#             'started': datetime.now(),
#             'last_message': None
#         }
    
#     def get_session(self, session_id):
#         """Get or create session"""
#         if session_id not in self.sessions:
#             self.create_session(session_id)
#         return self.sessions[session_id]
    
#     def detect_intent(self, message, session):
#         """
#         Detect user intent from message
#         Considers session context to avoid false positives
#         """
#         message = message.lower().strip()
        
#         # If in data collection mode, treat as answer unless explicit command
#         if session['step'] > 0 and session['step'] < len(self.steps):
#             # Check for explicit commands only
#             explicit_commands = {
#                 'reset': ['reset', 'restart', 'cancel'],
#                 'help': ['help me', 'i need help']
#             }
            
#             for intent, keywords in explicit_commands.items():
#                 if any(message == kw for kw in keywords):
#                     return intent
            
#             # Otherwise treat as answer
#             return 'answer'
        
#         # Not in collection mode - detect intent normally
#         intents = {
#             'greeting': ['hello', 'hi', 'hey', 'salam', 'start'],
#             'predict': ['predict', 'price', 'value', 'estimate', 'worth', 'check price'],
#             'help': ['help', 'how', 'guide', 'what can you do'],
#             'reset': ['reset', 'restart', 'start over', 'cancel'],
#             'info': ['accuracy', 'model', 'about', 'info', 'details'],
#             'thanks': ['thank', 'thanks', 'bye', 'goodbye']
#         }
        
#         # Exact matches first
#         for intent, keywords in intents.items():
#             if message in keywords:
#                 return intent
        
#         # Partial matches
#         for intent, keywords in intents.items():
#             if any(kw in message for kw in keywords):
#                 return intent
        
#         return 'answer'  # Default: treat as answer
    
#     def validate_input(self, step_name, value):
#         """Validate user input for current step"""
#         value = value.lower().strip()
        
#         if step_name == 'car_brand':
#             return value in self.valid_options['car_brand'], self.valid_options['car_brand']
        
#         elif step_name == 'car_model':
#             return value in self.valid_options['car_model'], self.valid_options['car_model']
        
#         elif step_name == 'city':
#             return value in self.city_mapping or len(value) > 2, list(self.city_mapping.keys())[:10]
        
#         elif step_name == 'fuel_type':
#             return value in self.valid_options['fuel_type'], self.valid_options['fuel_type']
        
#         elif step_name == 'engine':
#             try:
#                 engine = float(value)
#                 return 500 <= engine <= 5000, "Engine should be between 500cc - 5000cc"
#             except:
#                 return False, "Please enter a valid number (e.g., 1300)"
        
#         elif step_name == 'transmission':
#             return value in self.valid_options['transmission'], self.valid_options['transmission']
        
#         elif step_name == 'registered_in':
#             try:
#                 year = int(value)
#                 return 1990 <= year <= 2025, "Year should be between 1990-2025"
#             except:
#                 return False, "Please enter a valid year (e.g., 2018)"
        
#         elif step_name == 'mileage':
#             try:
#                 # Remove commas if present
#                 value = value.replace(',', '').strip()
#                 mileage = float(value)
#                 return 0 <= mileage <= 500000, "Mileage should be between 0-500,000 km"
#             except:
#                 return False, "Please enter a valid number (e.g., 50000)"
        
#         return False, "Invalid input"
    
#     def get_question(self, step_name):
#         """Get question for current step"""
#         questions = {
#             'car_brand': "🚗 What's the car brand?\n(e.g., Honda, Toyota, Suzuki)",
#             'car_model': "🏷️ What's the model?\n(e.g., Civic, Corolla, Cultus)",
#             'city': "📍 Which city?\n(e.g., Karachi, Lahore, Islamabad, or any Pakistan city)",
#             'fuel_type': "⛽ Fuel type?\n(e.g., Petrol, Diesel, Hybrid)",
#             'engine': "🔧 Engine capacity in cc?\n(e.g., 1300, 1500, 1800)",
#             'transmission': "⚙️ Transmission type?\n(Manual or Automatic)",
#             'registered_in': "📅 Registration year?\n(e.g., 2018, 2020)",
#             'mileage': "📏 Mileage in km?\n(e.g., 50000, 80000)"
#         }
#         return questions.get(step_name, "Please provide the information")
    
#     def process_message(self, session_id, message, model_info=None):
#         """
#         Process user message and return response
        
#         Returns:
#             dict: {
#                 'reply': str,
#                 'options': list (optional),
#                 'action': str (optional),
#                 'prediction': dict (optional),
#                 'session_info': dict (for debugging)
#             }
#         """
#         session = self.get_session(session_id)
#         intent = self.detect_intent(message, session)  # Pass session for context
        
#         # Debug info
#         session_info = {
#             'step': session['step'],
#             'total_steps': len(self.steps),
#             'current_field': self.steps[session['step']] if session['step'] < len(self.steps) else 'completed',
#             'collected_data': list(session['data'].keys())
#         }
        
#         # Handle intents
#         if intent == 'greeting':
#             return {
#                 'reply': "👋 Hello! I'm your AI Car Price Assistant.\n\n"
#                         "I can help you predict the price of used cars in Pakistan.\n\n"
#                         "Just tell me the car details, and I'll give you an instant estimate!\n\n"
#                         "Ready to start? 🚗",
#                 'action': 'start',
#                 'session_info': session_info
#             }
        
#         elif intent == 'predict':
#             session['step'] = 0
#             session['data'] = {}
#             question = self.get_question(self.steps[0])
#             return {
#                 'reply': f"Great! Let's predict the price. 💰\n\n{question}",
#                 'action': 'collect_data',
#                 'session_info': session_info
#             }
        
#         elif intent == 'help':
#             return {
#                 'reply': "🤖 **How to use:**\n\n"
#                         "1. Tell me you want to predict a price\n"
#                         "2. I'll ask 8 simple questions about the car\n"
#                         "3. Answer each question\n"
#                         "4. I'll predict the price instantly!\n\n"
#                         f"**Model Accuracy:** {model_info.get('test_r2', 0.94)*100:.1f}% ✨\n\n"
#                         "Say 'predict' to start!",
#                 'action': None
#             }
        
#         elif intent == 'reset':
#             session['step'] = 0
#             session['data'] = {}
#             return {
#                 'reply': "🔄 Session reset! Say 'predict' to start fresh.",
#                 'action': 'reset'
#             }
        
#         elif intent == 'info':
#             return {
#                 'reply': f"📊 **Model Information:**\n\n"
#                         f"• Algorithm: {model_info.get('model_name', 'Random Forest')}\n"
#                         f"• Accuracy (R²): {model_info.get('test_r2', 0.94)*100:.1f}%\n"
#                         f"• Average Error: ±{model_info.get('test_mae', 185000):,.0f} PKR\n"
#                         f"• Features: Polynomial + K-Means Clustering\n"
#                         f"• Dataset: 5,497 cars\n\n"
#                         f"Want to predict a price? Say 'predict'! 🚗",
#                 'action': None
#             }
        
#         elif intent == 'thanks':
#             return {
#                 'reply': "You're welcome! 😊\n\n"
#                         "Feel free to predict more car prices anytime!\n\n"
#                         "Have a great day! 🚗✨",
#                 'action': 'end'
#             }
        
#         # Handle data collection (answering questions)
#         elif intent == 'answer' and session['step'] < len(self.steps):
#             current_step = self.steps[session['step']]
            
#             # Validate input
#             is_valid, options = self.validate_input(current_step, message)
            
#             if not is_valid:
#                 if isinstance(options, list):
#                     return {
#                         'reply': f"❌ Invalid {current_step.replace('_', ' ')}.\n\n"
#                                 f"Please choose from: {', '.join(options[:5])}...\n\n"
#                                 f"Or type 'help' for guidance.",
#                         'options': options
#                     }
#                 else:
#                     return {
#                         'reply': f"❌ {options}\n\nPlease try again.",
#                         'action': None
#                     }
            
#             # Store valid input
#             if current_step == 'city':
#                 session['data'][current_step] = self.city_mapping.get(message.lower(), message.lower())
#             else:
#                 session['data'][current_step] = message.lower().strip()
            
#             session['step'] += 1
            
#             # Check if all data collected
#             if session['step'] >= len(self.steps):
#                 return {
#                     'reply': "✅ Got all the details!\n\n"
#                             "📊 Let me calculate the price...\n\n"
#                             "⏳ Processing...",
#                     'action': 'predict',
#                     'data': session['data']
#                 }
            
#             # Ask next question
#             next_step = self.steps[session['step']]
#             question = self.get_question(next_step)
            
#             return {
#                 'reply': f"✓ Got it!\n\n{question}",
#                 'action': 'continue'
#             }
        
#         # Default response
#         return {
#             'reply': "I didn't understand that. 🤔\n\n"
#                     "Say 'predict' to start, or 'help' for guidance.",
#             'action': None,
#             'session_info': session_info
#         }


# # Global chatbot instance
# chatbot = ConversationalChatbot()





# """
# Enhanced Conversational Chatbot with Gemini API
# Supports natural conversations + car price predictions
# """

# import google.generativeai as genai
# from datetime import datetime
# import re
# import os

# class GeminiChatbot:
#     """
#     AI-powered chatbot using Gemini for natural conversations
#     """
    
#     def __init__(self, api_key=None):
#         # Configure Gemini API
#         self.api_key = api_key or os.getenv('GEMINI_API_KEY')
#         if self.api_key:
#             genai.configure(api_key=self.api_key)
#             self.model = genai.GenerativeModel('gemini-pro')
#         else:
#             print("⚠️ Warning: GEMINI_API_KEY not set. Using rule-based fallback.")
#             self.model = None
        
#         self.sessions = {}
        
#         # System prompt for Gemini
#         self.system_prompt = """You are an AI assistant for a Pakistani car price prediction service. Your role:

# 1. Help users predict used car prices in Pakistan
# 2. Answer questions about car values, market trends, depreciation
# 3. Explain why certain factors affect price (brand, mileage, age, city, etc.)
# 4. Be friendly, conversational, and use simple Urdu/English mix if needed

# When user wants to predict price, collect these details conversationally:
# - Car brand (Honda, Toyota, Suzuki, etc.)
# - Model (Civic, Corolla, Cultus, etc.)
# - City (Karachi, Lahore, Islamabad, or any Pakistan city)
# - Fuel type (Petrol, Diesel, Hybrid, CNG)
# - Engine (cc, e.g., 1300, 1800)
# - Transmission (Manual/Automatic)
# - Registration year (e.g., 2018)
# - Mileage (km, e.g., 50000)

# Important guidelines:
# - Don't ask all questions at once - collect one by one naturally
# - If user asks "why is my car cheaper?", explain based on age, mileage, brand depreciation
# - If asked about market trends, explain Pakistani car market insights
# - Keep responses concise (2-3 sentences max unless explaining)
# - Use emojis moderately 🚗💰

# Pakistani Car Market Context:
# - Economy segment: Under 20 Lacs
# - Mid-Range: 20-40 Lacs  
# - Premium: 40-70 Lacs
# - Luxury: Above 70 Lacs
# - Popular brands: Toyota (holds value best), Honda, Suzuki (affordable)
# - Cities: Karachi/Lahore have more demand, better resale
# - Depreciation: ~15-20% per year for first 5 years
# """
        
#         self.steps = [
#             'car_brand', 'car_model', 'city', 'fuel_type',
#             'engine', 'transmission', 'registered_in', 'mileage'
#         ]
    
#     def create_session(self, session_id):
#         """Create new chat session"""
#         self.sessions[session_id] = {
#             'step': 0,
#             'data': {},
#             'collecting': False,  # Is actively collecting car details
#             'history': [],  # Conversation history for Gemini
#             'started': datetime.now()
#         }
    
#     def get_session(self, session_id):
#         """Get or create session"""
#         if session_id not in self.sessions:
#             self.create_session(session_id)
#         return self.sessions[session_id]
    
#     def detect_prediction_intent(self, message):
#         """Check if user wants to predict price"""
#         keywords = ['predict', 'price', 'value', 'kitni', 'kya', 'estimate', 'worth', 'check price']
#         message_lower = message.lower()
#         return any(kw in message_lower for kw in keywords)
    
#     def extract_car_data(self, message, current_step):
#         """Extract car information from natural language"""
#         message = message.lower().strip()
        
#         if current_step == 'car_brand':
#             brands = ['honda', 'toyota', 'suzuki', 'daihatsu', 'nissan', 'mitsubishi', 'hyundai', 'kia']
#             for brand in brands:
#                 if brand in message:
#                     return brand
        
#         elif current_step == 'car_model':
#             models = ['civic', 'corolla', 'city', 'cultus', 'swift', 'vitz', 'yaris', 'fit', 'alto', 'mehran', 'bolan']
#             for model in models:
#                 if model in message:
#                     return model
        
#         elif current_step == 'fuel_type':
#             fuels = ['petrol', 'diesel', 'hybrid', 'cng', 'lpg']
#             for fuel in fuels:
#                 if fuel in message:
#                     return fuel
        
#         elif current_step == 'transmission':
#             if 'auto' in message:
#                 return 'automatic'
#             elif 'manual' in message:
#                 return 'manual'
        
#         elif current_step == 'engine':
#             # Extract numbers (engine cc)
#             numbers = re.findall(r'\d+', message)
#             if numbers:
#                 engine = int(numbers[0])
#                 if 500 <= engine <= 5000:
#                     return str(engine)
        
#         elif current_step == 'registered_in':
#             # Extract year
#             numbers = re.findall(r'\b(19|20)\d{2}\b', message)
#             if numbers:
#                 year = int(numbers[0])
#                 if 1990 <= year <= 2025:
#                     return str(year)
        
#         elif current_step == 'mileage':
#             # Extract mileage
#             numbers = re.findall(r'\d+', message.replace(',', ''))
#             if numbers:
#                 mileage = int(numbers[0])
#                 if 0 <= mileage <= 500000:
#                     return str(mileage)
        
#         elif current_step == 'city':
#             # Accept any city name
#             return message
        
#         return None
    
#     def get_gemini_response(self, session, user_message):
#         """Get response from Gemini API"""
#         if not self.model:
#             return self.get_fallback_response(session, user_message)
        
#         try:
#             # Build conversation history
#             conversation_history = "\n".join([
#                 f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
#                 for msg in session['history'][-10:]  # Last 10 messages for context
#             ])
            
#             # Add context about data collection
#             context = ""
#             if session['collecting']:
#                 current_step = self.steps[session['step']]
#                 collected = list(session['data'].keys())
#                 context = f"\n\nCurrent task: Collecting {current_step}. Already collected: {collected}"
            
#             # Generate response
#             prompt = f"{self.system_prompt}\n\nConversation:\n{conversation_history}\nUser: {user_message}{context}\n\nAssistant:"
            
#             response = self.model.generate_content(prompt)
#             return response.text
        
#         except Exception as e:
#             print(f"Gemini API Error: {e}")
#             return self.get_fallback_response(session, user_message)
    
#     def get_fallback_response(self, session, message):
#         """Simple rule-based fallback if Gemini unavailable"""
#         if self.detect_prediction_intent(message):
#             return "Sure! I can help you predict the car price. Let's start - what's the car brand? (e.g., Honda, Toyota, Suzuki)"
        
#         return "I'm here to help you predict car prices! Say 'predict' to start, or ask me anything about car values in Pakistan. 🚗"
    
#     def process_message(self, session_id, message, model_info=None):
#         """
#         Process user message with Gemini AI
        
#         Returns response dict with reply and optional prediction action
#         """
#         session = self.get_session(session_id)
        
#         # Add to history
#         session['history'].append({
#             'role': 'user',
#             'content': message,
#             'timestamp': datetime.now().isoformat()
#         })
        
#         # Check if user wants to predict (start collection)
#         if not session['collecting'] and self.detect_prediction_intent(message):
#             session['collecting'] = True
#             session['step'] = 0
#             session['data'] = {}
#             reply = "Great! Let's predict the price. What's the car brand? (e.g., Honda, Toyota, Suzuki) 🚗"
        
#         # If collecting data
#         elif session['collecting'] and session['step'] < len(self.steps):
#             current_step = self.steps[session['step']]
            
#             # Try to extract data
#             extracted_value = self.extract_car_data(message, current_step)
            
#             if extracted_value:
#                 session['data'][current_step] = extracted_value
#                 session['step'] += 1
                
#                 # Check if done
#                 if session['step'] >= len(self.steps):
#                     session['collecting'] = False
#                     reply = "Perfect! I have all the details. Let me calculate the price... 💰"
#                     action = 'predict'
                    
#                     session['history'].append({
#                         'role': 'assistant',
#                         'content': reply,
#                         'timestamp': datetime.now().isoformat()
#                     })
                    
#                     return {
#                         'reply': reply,
#                         'action': action,
#                         'data': session['data']
#                     }
                
#                 # Ask next question
#                 next_step = self.steps[session['step']]
#                 questions = {
#                     'car_model': 'Got it! What model?',
#                     'city': 'Which city?',
#                     'fuel_type': 'Fuel type? (Petrol/Diesel/Hybrid)',
#                     'engine': 'Engine capacity in cc? (e.g., 1300, 1800)',
#                     'transmission': 'Manual or Automatic?',
#                     'registered_in': 'Registration year? (e.g., 2018)',
#                     'mileage': 'Mileage in km? (e.g., 50000)'
#                 }
#                 reply = questions.get(next_step, 'Next?')
#             else:
#                 # Couldn't extract, ask again with Gemini help
#                 reply = self.get_gemini_response(session, message)
        
#         # Normal conversation (not collecting)
#         else:
#             reply = self.get_gemini_response(session, message)
        
#         # Add assistant reply to history
#         session['history'].append({
#             'role': 'assistant',
#             'content': reply,
#             'timestamp': datetime.now().isoformat()
#         })
        
#         return {
#             'reply': reply,
#             'action': None
#         }


# # Initialize with API key from environment
# # Set your key: export GEMINI_API_KEY="your-key-here"
# chatbot = GeminiChatbot()


# # ============================================
# # Usage in Flask app.py:
# # ============================================
# """
# @app.route("/api/chatbot", methods=["POST"])
# def chatbot_endpoint():
#     try:
#         data = request.json
#         session_id = data.get("session_id", "default")
#         message = data.get("message", "")
        
#         response = chatbot.process_message(session_id=session_id, message=message, model_info=model_info)

#         if response.get('action') == 'predict' and 'data' in response:
#             car_data = response['data']
#             # Convert to proper types
#             car_data['engine'] = float(car_data['engine'])
#             car_data['registered_in'] = int(car_data['registered_in'])
#             car_data['mileage'] = float(car_data['mileage'])
            
#             prediction_result = make_prediction(car_data).json
#             if prediction_result['success']:
#                 pred = prediction_result
#                 response['reply'] = (
#                     f"🎉 Price Prediction\n"
#                     f"💰 Estimated: {pred['price_display']['formatted']}\n"
#                     f"📊 Range: {pred['price_range']['min_display']['lacs']} - {pred['price_range']['max_display']['lacs']} Lacs\n"
#                     f"🏷️ Segment: {pred['segment']}\n"
#                     f"✅ Confidence: {pred['model_performance']['confidence'].upper()}"
#                 )
#                 response['prediction'] = prediction_result
#             else:
#                 response['reply'] = f"❌ Prediction failed: {prediction_result.get('error')}"
        
#         response['timestamp'] = datetime.now().isoformat()
#         return jsonify(response)
    
#     except Exception as e:
#         return jsonify({
#             "error": str(e),
#             "reply": "Sorry, something went wrong.",
#             "timestamp": datetime.now().isoformat()
#         }), 500
# """


"""
Enhanced Conversational Chatbot with Gemini API (UPDATED)
Using new google-genai package
"""

from google.genai import Client
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