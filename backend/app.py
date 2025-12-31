# # """
# # Flask Backend for AI Car Price Predictor
# # Complete API with all endpoints
# # """

# # from flask import Flask, request, jsonify, send_file
# # from flask_cors import CORS
# # import pickle
# # import pandas as pd
# # import numpy as np
# # import os
# # from datetime import datetime

# # app = Flask(__name__)
# # CORS(app)  # Enable CORS for React frontend

# # # Paths
# # MODEL_DIR = os.path.join('..', 'models')
# # DATA_DIR = os.path.join('..', 'data')
# # VIZ_DIR = os.path.join('..', 'visualizations')

# # # Global variables for models
# # model = None
# # label_encoders = None
# # feature_info = None
# # model_info = None
# # poly = None
# # kmeans = None
# # cluster_scaler = None

# # # Load models at startup
# # def load_models():
# #     global model, label_encoders, feature_info, model_info, poly, kmeans, cluster_scaler
    
# #     print("\n" + "="*60)
# #     print("📦 Loading ML Models...")
# #     print("="*60)
    
# #     try:
# #         print("Loading best_model.pkl...", end=" ")
# #         with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
# #             model = pickle.load(f)
# #         print("✅")
        
# #         print("Loading label_encoders.pkl...", end=" ")
# #         with open(os.path.join(MODEL_DIR, 'label_encoders.pkl'), 'rb') as f:
# #             label_encoders = pickle.load(f)
# #         print("✅")
        
# #         print("Loading feature_info.pkl...", end=" ")
# #         with open(os.path.join(MODEL_DIR, 'feature_info.pkl'), 'rb') as f:
# #             feature_info = pickle.load(f)
# #         print("✅")
        
# #         print("Loading model_info.pkl...", end=" ")
# #         with open(os.path.join(MODEL_DIR, 'model_info.pkl'), 'rb') as f:
# #             model_info = pickle.load(f)
# #         print("✅")
        
# #         print("Loading poly_transformer.pkl...", end=" ")
# #         with open(os.path.join(MODEL_DIR, 'poly_transformer.pkl'), 'rb') as f:
# #             poly = pickle.load(f)
# #         print("✅")
        
# #         print("Loading kmeans_model.pkl...", end=" ")
# #         with open(os.path.join(MODEL_DIR, 'kmeans_model.pkl'), 'rb') as f:
# #             kmeans = pickle.load(f)
# #         print("✅")
        
# #         print("Loading cluster_scaler.pkl...", end=" ")
# #         with open(os.path.join(MODEL_DIR, 'cluster_scaler.pkl'), 'rb') as f:
# #             cluster_scaler = pickle.load(f)
# #         print("✅")
        
# #         print("\n✅ All models loaded successfully!")
# #         print("="*60 + "\n")
# #         return True
        
# #     except Exception as e:
# #         print(f"\n❌ Error loading models: {e}")
# #         return False

# # # City mapping (80+ Pakistan cities)
# # CITY_MAPPING = {
# #     'karachi': 'karachi', 'hyderabad': 'karachi', 'sukkur': 'karachi',
# #     'larkana': 'karachi', 'mirpurkhas': 'karachi', 'nawabshah': 'karachi',
# #     'lahore': 'lahore', 'faisalabad': 'lahore', 'multan': 'lahore',
# #     'gujranwala': 'lahore', 'sialkot': 'lahore', 'sargodha': 'lahore',
# #     'islamabad': 'islamabad', 'rawalpindi': 'islamabad', 'peshawar': 'islamabad',
# #     'abbottabad': 'islamabad', 'mardan': 'islamabad', 'quetta': 'islamabad'
# # }

# # def map_city(user_city):
# #     """Map user city to nearest training city"""
# #     city_lower = user_city.lower().strip()
# #     return CITY_MAPPING.get(city_lower, 'karachi'), city_lower in CITY_MAPPING


# # # ============================================
# # # API ENDPOINTS
# # # ============================================

# # @app.route('/', methods=['GET'])
# # def home():
# #     """Home endpoint"""
# #     return jsonify({
# #         'message': '🚗 AI Car Price Predictor API',
# #         'version': '1.0.0',
# #         'status': 'running',
# #         'endpoints': {
# #             'health': '/api/health',
# #             'model_info': '/api/model-info',
# #             'feature_options': '/api/feature-options',
# #             'predict': '/api/predict (POST)',
# #             'visualizations': '/api/visualizations/<type>',
# #             'analytics': '/api/analytics/stats',
# #             'chatbot': '/api/chatbot (POST)'
# #         }
# #     })


# # @app.route('/api/health', methods=['GET'])
# # def health_check():
# #     """Health check endpoint"""
# #     return jsonify({
# #         'status': 'healthy',
# #         'model_loaded': model is not None,
# #         'timestamp': datetime.now().isoformat(),
# #         'models': {
# #             'main_model': model is not None,
# #             'encoders': label_encoders is not None,
# #             'polynomial': poly is not None,
# #             'clustering': kmeans is not None
# #         }
# #     })


# # @app.route('/api/model-info', methods=['GET'])
# # def get_model_info():
# #     """Get model information and metadata"""
# #     if not model:
# #         return jsonify({'error': 'Model not loaded'}), 500
    
# #     return jsonify({
# #         'model_name': model_info.get('model_name', 'Random Forest'),
# #         'accuracy': float(model_info.get('test_r2', 0)),
# #         'avg_error': float(model_info.get('test_mae', 0)),
# #         'rmse': float(model_info.get('test_rmse', 0)),
# #         'features': {
# #             'categorical': feature_info['categorical_features'],
# #             'numerical': feature_info['numerical_features'],
# #             'total': len(feature_info['categorical_features']) + len(feature_info['numerical_features'])
# #         },
# #         'polynomial_degree': model_info.get('polynomial_degree', 2),
# #         'clustering_enabled': kmeans is not None
# #     })


# # @app.route('/api/feature-options', methods=['GET'])
# # def get_feature_options():
# #     """Get available options for all features"""
# #     try:
# #         df = pd.read_csv(os.path.join(DATA_DIR, 'pakwheels_cleaned.csv'))
        
# #         options = {}
        
# #         # Categorical features
# #         for feature in feature_info['categorical_features']:
# #             unique_values = sorted(df[feature].unique().tolist())
# #             options[feature] = unique_values
        
# #         # Numerical features
# #         for feature in feature_info['numerical_features']:
# #             if feature == 'engine':
# #                 options[feature] = [660, 800, 1000, 1200, 1300, 1500, 1600, 1800, 2000, 2400, 2500, 3000]
# #             elif feature == 'registered_in':
# #                 min_year = int(df[feature].min())
# #                 max_year = int(df[feature].max())
# #                 options[feature] = list(range(min_year, max_year + 1))
# #             else:
# #                 options[feature] = {
# #                     'min': float(df[feature].min()),
# #                     'max': float(df[feature].max()),
# #                     'mean': float(df[feature].mean())
# #                 }
        
# #         return jsonify({
# #             'options': options,
# #             'total_cars': len(df)
# #         })
    
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500


# # @app.route('/api/predict', methods=['POST'])
# # def predict_price():
# #     """Predict car price - Main prediction endpoint"""
# #     try:
# #         data = request.json
        
# #         # Validate required fields
# #         required_fields = feature_info['categorical_features'] + feature_info['numerical_features']
# #         for field in required_fields:
# #             if field not in data:
# #                 return jsonify({'error': f'Missing field: {field}'}), 400
        
# #         # Map city
# #         original_city = data.get('city', 'karachi')
# #         mapped_city, is_valid = map_city(original_city)
# #         data['city'] = mapped_city
        
# #         # Prepare features
# #         feature_vector = []
        
# #         # Encode categorical features
# #         for cat_feat in feature_info['categorical_features']:
# #             try:
# #                 value = str(data[cat_feat]).lower().strip()
# #                 encoded = label_encoders[cat_feat].transform([value])[0]
# #                 feature_vector.append(encoded)
# #             except:
# #                 feature_vector.append(0)
        
# #         # Get numerical features
# #         numerical_values = []
# #         for num_feat in feature_info['numerical_features']:
# #             numerical_values.append(float(data[num_feat]))
        
# #         # Apply polynomial features
# #         if poly:
# #             numerical_array = np.array([numerical_values])
# #             poly_features = poly.transform(numerical_array)
# #             poly_feature_names = poly.get_feature_names_out(feature_info['numerical_features'])
            
# #             all_features = feature_vector + poly_features[0].tolist()
            
# #             feature_columns = [f + '_encoded' for f in feature_info['categorical_features']]
# #             feature_columns.extend(poly_feature_names)
# #         else:
# #             all_features = feature_vector + numerical_values
# #             feature_columns = [f + '_encoded' for f in feature_info['categorical_features']]
# #             feature_columns.extend(feature_info['numerical_features'])
        
# #         # Create input dataframe
# #         input_df = pd.DataFrame([all_features], columns=feature_columns)
        
# #         # Align with training columns
# #         if 'feature_columns' in model_info:
# #             for col in model_info['feature_columns']:
# #                 if col not in input_df.columns:
# #                     input_df[col] = 0
# #             input_df = input_df[model_info['feature_columns']]
        
# #         # Predict price
# #         predicted_price = float(model.predict(input_df)[0])
        
# #         # Predict cluster/segment
# #         segment = 'Mid-Range'
# #         try:
# #             cluster_features = [predicted_price] + numerical_values
# #             expected_features = cluster_scaler.n_features_in_
            
# #             if len(cluster_features) == expected_features:
# #                 cluster_input = np.array([cluster_features])
# #                 cluster_scaled = cluster_scaler.transform(cluster_input)
# #                 cluster_id = int(kmeans.predict(cluster_scaled)[0])
                
# #                 cluster_map = {0: 'Economy', 1: 'Mid-Range', 2: 'Luxury'}
# #                 segment = cluster_map.get(cluster_id, 'Mid-Range')
# #         except:
# #             # Fallback segmentation
# #             if predicted_price < 1500000:
# #                 segment = 'Economy'
# #             elif predicted_price < 3000000:
# #                 segment = 'Mid-Range'
# #             elif predicted_price < 5000000:
# #                 segment = 'Premium'
# #             else:
# #                 segment = 'Luxury'
        
# #         # Calculate confidence based on segment and car details
# #         car_model = data.get('car_model', '').lower()
# #         popular_models = ['civic', 'corolla', 'city', 'swift', 'vitz', 'yaris']
        
# #         if any(model in car_model for model in popular_models):
# #             confidence = 0.95
# #         else:
# #             confidence = 0.85
        
# #         # Warning for edge cases
# #         warnings = []
# #         if segment in ['Economy', 'Luxury']:
# #             warnings.append('Limited training data for this segment. Prediction may vary ±20%.')
        
# #         if car_model in ['cultus', 'bolan', 'mehran']:
# #             warnings.append('Budget cars may have ±30% variance from market price.')
        
# #         if car_model in ['accord', 'camry', 'land cruiser']:
# #             warnings.append('Luxury cars may be undervalued. Consider adding 20-30% to estimate.')
        
# #         # Return prediction
# #         return jsonify({
# #             'success': True,
# #             'predicted_price': predicted_price,
# #             'price_in_lacs': predicted_price / 100000,
# #             'price_range': {
# #                 'min': predicted_price * 0.90,
# #                 'max': predicted_price * 1.10,
# #                 'variation': '±10%'
# #             },
# #             'segment': segment,
# #             'confidence': confidence,
# #             'warnings': warnings if warnings else None,
# #             'city_info': {
# #                 'original': original_city,
# #                 'mapped': mapped_city if not is_valid else None,
# #                 'was_mapped': not is_valid
# #             },
# #             'model_info': {
# #                 'name': model_info.get('model_name', 'Random Forest'),
# #                 'accuracy': float(model_info.get('test_r2', 0)),
# #                 'avg_error': float(model_info.get('test_mae', 0))
# #             },
# #             'timestamp': datetime.now().isoformat()
# #         })
    
# #     except Exception as e:
# #         return jsonify({
# #             'success': False,
# #             'error': str(e)
# #         }), 500


# # @app.route('/api/visualizations/<viz_type>', methods=['GET'])
# # def get_visualization(viz_type):
# #     """Get visualization images"""
# #     try:
# #         viz_files = {
# #             'actual_vs_predicted': 'actual_vs_predicted.png',
# #             'model_comparison': 'model_comparison.png',
# #             'feature_importance': 'feature_importance.png',
# #             'clusters': 'cluster_distributions.png',
# #             'elbow': 'elbow_curve.png'
# #         }
        
# #         if viz_type not in viz_files:
# #             return jsonify({'error': 'Visualization type not found'}), 404
        
# #         file_path = os.path.join(VIZ_DIR, viz_files[viz_type])
        
# #         if not os.path.exists(file_path):
# #             return jsonify({'error': f'File not found: {viz_files[viz_type]}'}), 404
        
# #         return send_file(file_path, mimetype='image/png')
    
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500


# # @app.route('/api/analytics/stats', methods=['GET'])
# # def get_analytics_stats():
# #     """Get dataset analytics and statistics"""
# #     try:
# #         df = pd.read_csv(os.path.join(DATA_DIR, 'pakwheels_cleaned.csv'))
        
# #         stats = {
# #             'dataset': {
# #                 'total_cars': len(df),
# #                 'date_range': f"{int(df['registered_in'].min())} - {int(df['registered_in'].max())}"
# #             },
# #             'price_stats': {
# #                 'min': float(df['price'].min()),
# #                 'max': float(df['price'].max()),
# #                 'mean': float(df['price'].mean()),
# #                 'median': float(df['price'].median()),
# #                 'std': float(df['price'].std())
# #             },
# #             'top_brands': dict(df['car_brand'].value_counts().head(10)),
# #             'top_models': dict(df['car_model'].value_counts().head(10)),
# #             'fuel_type': dict(df['fuel_type'].value_counts()),
# #             'transmission': dict(df['transmission'].value_counts()),
# #             'cities': dict(df['city'].value_counts()),
# #             'year_distribution': dict(df['registered_in'].value_counts().head(10).sort_index())
# #         }
        
# #         return jsonify(stats)
    
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500


# # @app.route('/api/chatbot', methods=['POST'])
# # def chatbot_response():
# #     """Simple rule-based chatbot"""
# #     try:
# #         data = request.json
# #         user_message = data.get('message', '').lower()
        
# #         # Simple responses
# #         responses = {
# #             'hello': "Hello! 👋 I'm your AI car price assistant. How can I help you?",
# #             'hi': "Hi there! 🚗 Ask me anything about car price predictions!",
# #             'how': f"I use {model_info['model_name']} with {model_info['test_r2']*100:.1f}% accuracy to predict prices!",
# #             'accuracy': f"Our model achieves {model_info['test_r2']*100:.1f}% accuracy with ±{model_info['test_mae']:,.0f} PKR error.",
# #             'price': "Fill the form with car details and I'll predict the price instantly! 💰",
# #             'best': "I work best with popular cars like Honda Civic, Toyota Corolla, and Suzuki Swift!",
# #             'features': f"I analyze {len(feature_info['categorical_features'])} categorical and {len(feature_info['numerical_features'])} numerical features.",
# #             'thank': "You're welcome! Feel free to ask anything else! 😊",
# #             'help': "I can predict prices, show analytics, explain model accuracy, or chat about cars!"
# #         }
        
# #         response = "I can help with car price predictions, model info, or analytics. What would you like to know?"
        
# #         for keyword, reply in responses.items():
# #             if keyword in user_message:
# #                 response = reply
# #                 break
        
# #         return jsonify({
# #             'response': response,
# #             'timestamp': datetime.now().isoformat()
# #         })
    
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 500


# # # Initialize models on startup
# # load_models()

# # if __name__ == '__main__':
# #     print("\n" + "="*60)
# #     print("🚀 FLASK BACKEND SERVER")
# #     print("="*60)
# #     print("📍 Server: http://localhost:5000")
# #     print("📚 API Documentation: http://localhost:5000/")
# #     print("🔗 Frontend will connect here!")
# #     print("="*60 + "\n")
    
# #     app.run(debug=True, host='0.0.0.0', port=5000)


# """
# Flask Backend for AI Car Price Predictor
# Complete API with all endpoints (JSON Safe)
# """

# # from flask import Flask, request, jsonify, send_file
# # from flask_cors import CORS
# # import pickle
# # import pandas as pd
# # import numpy as np
# # import os
# # from datetime import datetime

# # app = Flask(__name__)
# # CORS(app)

# # # ============================
# # # PATHS
# # # ============================
# # MODEL_DIR = os.path.join('..', 'models')
# # DATA_DIR = os.path.join('..', 'data')
# # VIZ_DIR = os.path.join('..', 'visualizations')

# # # ============================
# # # GLOBAL MODELS
# # # ============================
# # model = None
# # label_encoders = None
# # feature_info = None
# # model_info = None
# # poly = None
# # kmeans = None
# # cluster_scaler = None


# # # ============================
# # # LOAD MODELS
# # # ============================
# # def load_models():
# #     global model, label_encoders, feature_info, model_info, poly, kmeans, cluster_scaler

# #     print("\n" + "=" * 60)
# #     print("📦 Loading ML Models...")
# #     print("=" * 60)

# #     try:
# #         def load(file):
# #             with open(os.path.join(MODEL_DIR, file), "rb") as f:
# #                 return pickle.load(f)

# #         model = load("best_model.pkl")
# #         label_encoders = load("label_encoders.pkl")
# #         feature_info = load("feature_info.pkl")
# #         model_info = load("model_info.pkl")
# #         poly = load("poly_transformer.pkl")
# #         kmeans = load("kmeans_model.pkl")
# #         cluster_scaler = load("cluster_scaler.pkl")

# #         print("✅ All models loaded successfully!")
# #         print("=" * 60 + "\n")
# #         return True

# #     except Exception as e:
# #         print(f"❌ Error loading models: {e}")
# #         return False


# # # ============================
# # # CITY MAPPING
# # # ============================
# # CITY_MAPPING = {
# #     'karachi': 'karachi', 'hyderabad': 'karachi', 'sukkur': 'karachi',
# #     'lahore': 'lahore', 'faisalabad': 'lahore', 'multan': 'lahore',
# #     'islamabad': 'islamabad', 'rawalpindi': 'islamabad', 'peshawar': 'islamabad'
# # }


# # def map_city(city):
# #     city = city.lower().strip()
# #     return CITY_MAPPING.get(city, "karachi"), city in CITY_MAPPING


# # # ============================
# # # HOME
# # # ============================
# # @app.route("/", methods=["GET"])
# # def home():
# #     return jsonify({
# #         "message": "🚗 AI Car Price Predictor API",
# #         "status": "running",
# #         "endpoints": {
# #             "health": "/api/health",
# #             "model_info": "/api/model-info",
# #             "feature_options": "/api/feature-options",
# #             "predict": "/api/predict (POST)",
# #             "analytics": "/api/analytics/stats",
# #             "visualizations": "/api/visualizations/<type>",
# #             "chatbot": "/api/chatbot (POST)"
# #         }
# #     })


# # # ============================
# # # HEALTH
# # # ============================
# # @app.route("/api/health", methods=["GET"])
# # def health():
# #     return jsonify({
# #         "status": "healthy",
# #         "model_loaded": model is not None,
# #         "timestamp": datetime.now().isoformat()
# #     })


# # # ============================
# # # MODEL INFO
# # # ============================
# # @app.route("/api/model-info", methods=["GET"])
# # def model_details():
# #     return jsonify({
# #         "model_name": model_info.get("model_name"),
# #         "accuracy": float(model_info.get("test_r2", 0)),
# #         "mae": float(model_info.get("test_mae", 0)),
# #         "rmse": float(model_info.get("test_rmse", 0)),
# #         "features": {
# #             "categorical": feature_info["categorical_features"],
# #             "numerical": feature_info["numerical_features"]
# #         }
# #     })


# # # ============================
# # # FEATURE OPTIONS
# # # ============================
# # @app.route("/api/feature-options", methods=["GET"])
# # def feature_options():
# #     df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))

# #     options = {}

# #     for col in feature_info["categorical_features"]:
# #         options[col] = sorted(df[col].astype(str).unique().tolist())

# #     for col in feature_info["numerical_features"]:
# #         options[col] = {
# #             "min": float(df[col].min()),
# #             "max": float(df[col].max()),
# #             "mean": float(df[col].mean())
# #         }

# #     return jsonify({
# #         "options": options,
# #         "total_records": int(len(df))
# #     })


# # # ============================
# # # PREDICTION
# # # ============================
# # @app.route("/api/predict", methods=["POST"])
# # def predict():
# #     try:
# #         data = request.json

# #         city, _ = map_city(data.get("city", "karachi"))
# #         data["city"] = city

# #         encoded = []
# #         for col in feature_info["categorical_features"]:
# #             val = str(data[col]).lower()
# #             try:
# #                 encoded.append(label_encoders[col].transform([val])[0])
# #             except:
# #                 encoded.append(0)

# #         numerical = [float(data[col]) for col in feature_info["numerical_features"]]

# #         poly_features = poly.transform([numerical])[0].tolist()
# #         final_input = encoded + poly_features

# #         df_input = pd.DataFrame([final_input], columns=model_info["feature_columns"])
# #         prediction = float(model.predict(df_input)[0])

# #         return jsonify({
# #             "success": True,
# #             "predicted_price": prediction,
# #             "price_lacs": prediction / 100000,
# #             "timestamp": datetime.now().isoformat()
# #         })

# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500


# # # ============================
# # # ANALYTICS (FIXED)
# # # ============================
# # @app.route("/api/analytics/stats", methods=["GET"])
# # def analytics():
# #     df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))

# #     def clean(d):
# #         return {str(k): int(v) for k, v in d.items()}

# #     return jsonify({
# #         "dataset": {
# #             "total_cars": int(len(df)),
# #             "year_range": f"{int(df['registered_in'].min())} - {int(df['registered_in'].max())}"
# #         },
# #         "price_stats": {
# #             "min": float(df["price"].min()),
# #             "max": float(df["price"].max()),
# #             "mean": float(df["price"].mean()),
# #             "median": float(df["price"].median()),
# #             "std": float(df["price"].std())
# #         },
# #         "top_brands": clean(df["car_brand"].value_counts().head(10).to_dict()),
# #         "top_models": clean(df["car_model"].value_counts().head(10).to_dict()),
# #         "fuel_type": clean(df["fuel_type"].value_counts().to_dict()),
# #         "transmission": clean(df["transmission"].value_counts().to_dict()),
# #         "cities": clean(df["city"].value_counts().to_dict())
# #     })


# # # ============================
# # # VISUALIZATIONS
# # # ============================
# # @app.route("/api/visualizations/<name>")
# # def visualizations(name):
# #     files = {
# #         "actual_vs_predicted": "actual_vs_predicted.png",
# #         "feature_importance": "feature_importance.png"
# #     }

# #     if name not in files:
# #         return jsonify({"error": "Invalid visualization"}), 404

# #     path = os.path.join(VIZ_DIR, files[name])
# #     return send_file(path, mimetype="image/png")


# # # ============================
# # # CHATBOT
# # # ============================
# # @app.route("/api/chatbot", methods=["POST"])
# # def chatbot():
# #     msg = request.json.get("message", "").lower()

# #     replies = {
# #         "hello": "Hello! 👋 I'm your AI car price assistant.",
# #         "accuracy": f"Model accuracy is {model_info['test_r2'] * 100:.1f}%",
# #         "price": "Fill the form to predict car price 💰",
# #         "help": "I can predict prices and show analytics 📊"
# #     }

# #     for key in replies:
# #         if key in msg:
# #             return jsonify({"reply": replies[key]})

# #     return jsonify({"reply": "Ask me about car prices or model accuracy 🚗"})


# # # ============================
# # # START SERVER
# # # ============================
# # load_models()

# # if __name__ == "__main__":
# #     print("\n🚀 FLASK BACKEND RUNNING")
# #     print("📍 http://localhost:5000\n")
# #     app.run(debug=True, host="0.0.0.0", port=5000)



# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import pickle
# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime

# app = Flask(__name__)
# CORS(app)

# # ============================
# # PATHS (FIXED)
# # ============================
# # Adjust these paths based on your folder structure
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_DIR = os.path.join(BASE_DIR, 'models')
# DATA_DIR = os.path.join(BASE_DIR, 'data')
# VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')

# # ============================
# # GLOBAL MODELS
# # ============================
# model = None
# label_encoders = None
# feature_info = None
# model_info = None
# poly = None
# kmeans = None
# cluster_scaler = None


# # ============================
# # LOAD MODELS
# # ============================
# def load_models():
#     global model, label_encoders, feature_info, model_info, poly, kmeans, cluster_scaler

#     print("\n" + "=" * 60)
#     print("📦 Loading ML Models...")
#     print("=" * 60)

#     try:
#         def load(file):
#             path = os.path.join(MODEL_DIR, file)
#             print(f"   Loading: {file}")
#             with open(path, "rb") as f:
#                 return pickle.load(f)

#         model = load("best_model.pkl")
#         label_encoders = load("label_encoders.pkl")
#         feature_info = load("feature_info.pkl")
#         model_info = load("model_info.pkl")
#         poly = load("poly_transformer.pkl")
#         kmeans = load("kmeans_model.pkl")
#         cluster_scaler = load("cluster_scaler.pkl")

#         print("✅ All models loaded successfully!")
#         print(f"   Model: {model_info.get('model_name', 'Unknown')}")
#         print(f"   Accuracy: {model_info.get('test_r2', 0):.4f}")
#         print("=" * 60 + "\n")
#         return True

#     except Exception as e:
#         print(f"❌ Error loading models: {e}")
#         print(f"   MODEL_DIR: {MODEL_DIR}")
#         return False


# # ============================
# # CITY MAPPING
# # ============================
# CITY_MAPPING = {
#     # Sindh
#     'karachi': 'karachi', 'hyderabad': 'karachi', 'sukkur': 'karachi',
#     'larkana': 'karachi', 'mirpurkhas': 'karachi',
    
#     # Punjab
#     'lahore': 'lahore', 'faisalabad': 'lahore', 'multan': 'lahore',
#     'gujranwala': 'lahore', 'sialkot': 'lahore', 'bahawalpur': 'lahore',
#     'sargodha': 'lahore', 'sahiwal': 'lahore',
    
#     # Federal/KPK
#     'islamabad': 'islamabad', 'rawalpindi': 'islamabad', 'peshawar': 'islamabad',
#     'abbottabad': 'islamabad', 'mardan': 'islamabad', 'quetta': 'islamabad',
    
#     # Abbreviations
#     'khi': 'karachi', 'lhr': 'lahore', 'isb': 'islamabad', 'pindi': 'islamabad'
# }


# def map_city(city):
#     """Map user city to training city"""
#     city_lower = city.lower().strip()
#     mapped = CITY_MAPPING.get(city_lower, "karachi")
#     was_mapped = mapped != city_lower
#     return mapped, was_mapped


# # ============================
# # HOME
# # ============================
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({
#         "message": "🚗 AI Car Price Predictor API",
#         "version": "1.0.0",
#         "status": "running",
#         "endpoints": {
#             "health": "/api/health",
#             "model_info": "/api/model-info",
#             "feature_options": "/api/feature-options",
#             "predict": "/api/predict (POST)",
#             "analytics": "/api/analytics/stats",
#             "visualizations": "/api/visualizations/<type>",
#             "chatbot": "/api/chatbot (POST)"
#         },
#         "documentation": "http://localhost:5000/docs"
#     })


# # ============================
# # HEALTH
# # ============================
# @app.route("/api/health", methods=["GET"])
# def health():
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "poly_loaded": poly is not None,
#         "kmeans_loaded": kmeans is not None,
#         "timestamp": datetime.now().isoformat()
#     })


# # ============================
# # MODEL INFO
# # ============================
# @app.route("/api/model-info", methods=["GET"])
# def model_details():
#     if not model:
#         return jsonify({"error": "Model not loaded"}), 500
    
#     return jsonify({
#         "model_name": model_info.get("model_name"),
#         "accuracy_r2": float(model_info.get("test_r2", 0)),
#         "mae": float(model_info.get("test_mae", 0)),
#         "rmse": float(model_info.get("test_rmse", 0)),
#         "mape": float(model_info.get("test_mape", 0)) if "test_mape" in model_info else None,
#         "features": {
#             "categorical": feature_info["categorical_features"],
#             "numerical": feature_info["numerical_features"],
#             "total_features": len(model_info.get("feature_columns", []))
#         },
#         "polynomial_features": poly is not None,
#         "clustering": kmeans is not None
#     })


# # ============================
# # FEATURE OPTIONS
# # ============================
# @app.route("/api/feature-options", methods=["GET"])
# def feature_options():
#     try:
#         df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))

#         options = {}

#         # Categorical features
#         for col in feature_info["categorical_features"]:
#             options[col] = sorted(df[col].astype(str).unique().tolist())

#         # Numerical features with ranges
#         for col in feature_info["numerical_features"]:
#             options[col] = {
#                 "min": float(df[col].min()),
#                 "max": float(df[col].max()),
#                 "mean": float(df[col].mean()),
#                 "median": float(df[col].median())
#             }

#         return jsonify({
#             "options": options,
#             "total_records": int(len(df)),
#             "cities_supported": list(CITY_MAPPING.keys())
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ============================
# # PREDICTION (FIXED)
# # ============================
# @app.route("/api/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.json
        
#         # Validate required fields
#         required_fields = feature_info["categorical_features"] + feature_info["numerical_features"]
#         missing = [f for f in required_fields if f not in data]
#         if missing:
#             return jsonify({"error": f"Missing fields: {missing}"}), 400

#         # Map city
#         original_city = data.get("city", "karachi")
#         mapped_city, was_mapped = map_city(original_city)
#         data["city"] = mapped_city

#         # Encode categorical features
#         feature_vector = []
#         for col in feature_info["categorical_features"]:
#             val = str(data[col]).lower().strip()
#             try:
#                 encoded_val = label_encoders[col].transform([val])[0]
#                 feature_vector.append(encoded_val)
#             except (ValueError, KeyError):
#                 # Unknown value - use default
#                 feature_vector.append(0)
#                 print(f"   ⚠️ Unknown value '{val}' for {col}, using default")

#         # Get numerical features
#         numerical_values = [float(data[col]) for col in feature_info["numerical_features"]]

#         # Apply polynomial features
#         if poly is not None:
#             numerical_array = np.array([numerical_values])
#             poly_features = poly.transform(numerical_array)
#             poly_feature_names = poly.get_feature_names_out(feature_info["numerical_features"])
            
#             # Combine encoded categorical + polynomial features
#             all_features = feature_vector + poly_features[0].tolist()
            
#             # Create column names
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(poly_feature_names)
#         else:
#             # No polynomial - use original features
#             all_features = feature_vector + numerical_values
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(feature_info["numerical_features"])

#         # Create dataframe
#         df_input = pd.DataFrame([all_features], columns=feature_columns)
        
#         # Align with training columns
#         if 'feature_columns' in model_info:
#             training_columns = model_info['feature_columns']
#             # Add missing columns as 0
#             for col in training_columns:
#                 if col not in df_input.columns:
#                     df_input[col] = 0
#             # Reorder to match training
#             df_input = df_input[training_columns]

#         # Predict price
#         predicted_price = float(model.predict(df_input)[0])

#         # Calculate price range (±10%)
#         price_min = predicted_price * 0.90
#         price_max = predicted_price * 1.10

#         # Determine segment using K-Means
#         if kmeans is not None and cluster_scaler is not None:
#             cluster_input = np.array([[predicted_price, data['registered_in'], data['mileage']]])
#             cluster_scaled = cluster_scaler.transform(cluster_input)
#             cluster_id = kmeans.predict(cluster_scaled)[0]
            
#             cluster_map = feature_info.get('cluster_map', {0: 'Economy', 1: 'Mid-Range', 2: 'Luxury'})
#             segment = cluster_map.get(cluster_id, 'Unknown')
#         else:
#             # Fallback segment determination
#             if predicted_price < 1000000:
#                 segment = "Economy"
#             elif predicted_price < 3000000:
#                 segment = "Mid-Range"
#             elif predicted_price < 5000000:
#                 segment = "Premium"
#             else:
#                 segment = "Luxury"

#         return jsonify({
#             "success": True,
#             "predicted_price": predicted_price,
#             "price_range": {
#                 "min": price_min,
#                 "max": price_max
#             },
#             "price_lacs": round(predicted_price / 100000, 2),
#             "segment": segment,
#             "city": {
#                 "original": original_city,
#                 "mapped": mapped_city,
#                 "was_mapped": was_mapped
#             },
#             "model_accuracy": float(model_info.get("test_r2", 0)),
#             "average_error": float(model_info.get("test_mae", 0)),
#             "timestamp": datetime.now().isoformat()
#         })

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500


# # ============================
# # ANALYTICS (FIXED)
# # ============================
# @app.route("/api/analytics/stats", methods=["GET"])
# def analytics():
#     try:
#         df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))

#         def clean(d):
#             """Convert to JSON-serializable format"""
#             return {str(k): int(v) if isinstance(v, (np.integer, np.int64)) else float(v) for k, v in d.items()}

#         return jsonify({
#             "dataset": {
#                 "total_cars": int(len(df)),
#                 "year_range": f"{int(df['registered_in'].min())} - {int(df['registered_in'].max())}",
#                 "mileage_range": f"{int(df['mileage'].min())} - {int(df['mileage'].max())} km"
#             },
#             "price_stats": {
#                 "min": float(df["price"].min()),
#                 "max": float(df["price"].max()),
#                 "mean": float(df["price"].mean()),
#                 "median": float(df["price"].median()),
#                 "std": float(df["price"].std())
#             },
#             "top_brands": clean(df["car_brand"].value_counts().head(10).to_dict()),
#             "top_models": clean(df["car_model"].value_counts().head(10).to_dict()),
#             "fuel_type_distribution": clean(df["fuel_type"].value_counts().to_dict()),
#             "transmission_distribution": clean(df["transmission"].value_counts().to_dict()),
#             "city_distribution": clean(df["city"].value_counts().to_dict())
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ============================
# # VISUALIZATIONS
# # ============================
# @app.route("/api/visualizations/<name>")
# def visualizations(name):
#     files = {
#         "actual_vs_predicted": "actual_vs_predicted.png",
#         "feature_importance": "feature_importance.png",
#         "model_comparison": "model_comparison.png",
#         "error_distribution": "error_distribution.png"
#     }

#     if name not in files:
#         return jsonify({"error": "Invalid visualization name", "available": list(files.keys())}), 404

#     path = os.path.join(VIZ_DIR, files[name])
    
#     if not os.path.exists(path):
#         return jsonify({"error": f"File not found: {files[name]}"}), 404
    
#     return send_file(path, mimetype="image/png")


# # ============================
# # CONVERSATIONAL CHATBOT
# # ============================
# from chatbot import chatbot as ai_chatbot

# @app.route("/api/chatbot", methods=["POST"])
# def chatbot_endpoint():
#     """
#     Enhanced conversational chatbot endpoint
#     Handles step-by-step car detail collection
#     """
#     try:
#         data = request.json
#         session_id = data.get("session_id", "default")
#         message = data.get("message", "")
        
#         # Process message through chatbot
#         response = ai_chatbot.process_message(
#             session_id=session_id,
#             message=message,
#             model_info=model_info
#         )
        
#         # If chatbot says to predict, make prediction
#         if response.get('action') == 'predict' and 'data' in response:
#             try:
#                 car_data = response['data']
                
#                 # Convert types
#                 car_data['engine'] = float(car_data['engine'])
#                 car_data['registered_in'] = int(car_data['registered_in'])
#                 car_data['mileage'] = float(car_data['mileage'])
                
#                 # Make prediction (reuse predict logic)
#                 prediction_result = make_prediction(car_data)
                
#                 if prediction_result['success']:
#                     pred = prediction_result
#                     response['reply'] = (
#                         f"🎉 **Price Prediction:**\n\n"
#                         f"💰 **Estimated Price:** {pred['predicted_price']:,.0f} PKR\n"
#                         f"📊 **Price Range:** {pred['price_range']['min']:,.0f} - {pred['price_range']['max']:,.0f} PKR\n"
#                         f"🎯 **Segment:** {pred['segment']}\n"
#                         f"📍 **City:** {pred['city']['original'].title()}\n\n"
#                         f"✨ **Price in Lacs:** {pred['price_lacs']} Lacs\n\n"
#                         f"Want to predict another car? Say 'predict'!"
#                     )
#                     response['prediction'] = prediction_result
#                 else:
#                     response['reply'] = "❌ Prediction failed. Please try again."
            
#             except Exception as e:
#                 response['reply'] = f"❌ Error making prediction: {str(e)}"
        
#         response['timestamp'] = datetime.now().isoformat()
#         return jsonify(response)
    
#     except Exception as e:
#         return jsonify({"error": str(e), "reply": "Sorry, something went wrong. Please try again."}), 500


# def make_prediction(data):
#     """
#     Helper function to make prediction
#     Extracted from /api/predict for reuse
#     """
#     try:
#         # Map city
#         original_city = data.get("city", "karachi")
#         mapped_city, was_mapped = map_city(original_city)
#         data["city"] = mapped_city

#         # Encode categorical
#         feature_vector = []
#         for col in feature_info["categorical_features"]:
#             val = str(data[col]).lower().strip()
#             try:
#                 encoded_val = label_encoders[col].transform([val])[0]
#                 feature_vector.append(encoded_val)
#             except:
#                 feature_vector.append(0)

#         # Get numerical
#         numerical_values = [float(data[col]) for col in feature_info["numerical_features"]]

#         # Apply polynomial
#         if poly is not None:
#             numerical_array = np.array([numerical_values])
#             poly_features = poly.transform(numerical_array)
#             poly_feature_names = poly.get_feature_names_out(feature_info["numerical_features"])
#             all_features = feature_vector + poly_features[0].tolist()
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(poly_feature_names)
#         else:
#             all_features = feature_vector + numerical_values
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(feature_info["numerical_features"])

#         # Create dataframe
#         df_input = pd.DataFrame([all_features], columns=feature_columns)
        
#         if 'feature_columns' in model_info:
#             training_columns = model_info['feature_columns']
#             for col in training_columns:
#                 if col not in df_input.columns:
#                     df_input[col] = 0
#             df_input = df_input[training_columns]

#         # Predict
#         predicted_price = float(model.predict(df_input)[0])
#         price_min = predicted_price * 0.90
#         price_max = predicted_price * 1.10

#         # Segment
#         if kmeans is not None and cluster_scaler is not None:
#             cluster_input = np.array([[predicted_price, data['registered_in'], data['mileage']]])
#             cluster_scaled = cluster_scaler.transform(cluster_input)
#             cluster_id = kmeans.predict(cluster_scaled)[0]
#             cluster_map = feature_info.get('cluster_map', {0: 'Economy', 1: 'Mid-Range', 2: 'Luxury'})
#             segment = cluster_map.get(cluster_id, 'Unknown')
#         else:
#             if predicted_price < 1000000:
#                 segment = "Economy"
#             elif predicted_price < 3000000:
#                 segment = "Mid-Range"
#             else:
#                 segment = "Luxury"

#         return {
#             "success": True,
#             "predicted_price": predicted_price,
#             "price_range": {"min": price_min, "max": price_max},
#             "price_lacs": round(predicted_price / 100000, 2),
#             "segment": segment,
#             "city": {"original": original_city, "mapped": mapped_city, "was_mapped": was_mapped},
#             "model_accuracy": float(model_info.get("test_r2", 0)),
#             "average_error": float(model_info.get("test_mae", 0))
#         }
    
#     except Exception as e:
#         return {"success": False, "error": str(e)}


# # ============================
# # ERROR HANDLERS
# # ============================
# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({"error": "Endpoint not found"}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({"error": "Internal server error"}), 500


# # ============================
# # START SERVER
# # ============================
# if __name__ == "__main__":
#     print("\n" + "=" * 60)
#     print("🚀 FLASK BACKEND STARTING...")
#     print("=" * 60)
    
#     # Load models
#     success = load_models()
    
#     if not success:
#         print("⚠️ WARNING: Some models failed to load!")
#         print("   Server will start but predictions may fail.")
    
#     print("\n📍 Server running at: http://localhost:5000")
#     print("📖 API Documentation: http://localhost:5000")
#     print("\n💡 Test with:")
#     print("   curl http://localhost:5000/api/health")
#     print("\n" + "=" * 60 + "\n")
    
#     app.run(debug=True, host="0.0.0.0", port=5000)




# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import pickle
# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime

# app = Flask(__name__)
# CORS(app)

# # ============================
# # PATHS (FIXED)
# # ============================
# # Adjust these paths based on your folder structure
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_DIR = os.path.join(BASE_DIR, 'models')
# DATA_DIR = os.path.join(BASE_DIR, 'data')
# VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')

# # ============================
# # GLOBAL MODELS
# # ============================
# model = None
# label_encoders = None
# feature_info = None
# model_info = None
# poly = None
# kmeans = None
# cluster_scaler = None


# # ============================
# # LOAD MODELS
# # ============================
# def load_models():
#     global model, label_encoders, feature_info, model_info, poly, kmeans, cluster_scaler

#     print("\n" + "=" * 60)
#     print("📦 Loading ML Models...")
#     print("=" * 60)

#     try:
#         def load(file):
#             path = os.path.join(MODEL_DIR, file)
#             print(f"   Loading: {file}")
#             with open(path, "rb") as f:
#                 return pickle.load(f)

#         model = load("best_model.pkl")
#         label_encoders = load("label_encoders.pkl")
#         feature_info = load("feature_info.pkl")
#         model_info = load("model_info.pkl")
#         poly = load("poly_transformer.pkl")
#         kmeans = load("kmeans_model.pkl")
#         cluster_scaler = load("cluster_scaler.pkl")

#         print("✅ All models loaded successfully!")
#         print(f"   Model: {model_info.get('model_name', 'Unknown')}")
#         print(f"   Accuracy: {model_info.get('test_r2', 0):.4f}")
#         print("=" * 60 + "\n")
#         return True

#     except Exception as e:
#         print(f"❌ Error loading models: {e}")
#         print(f"   MODEL_DIR: {MODEL_DIR}")
#         return False


# # ============================
# # CITY MAPPING
# # ============================
# CITY_MAPPING = {
#     # Sindh
#     'karachi': 'karachi', 'hyderabad': 'karachi', 'sukkur': 'karachi',
#     'larkana': 'karachi', 'mirpurkhas': 'karachi',
    
#     # Punjab
#     'lahore': 'lahore', 'faisalabad': 'lahore', 'multan': 'lahore',
#     'gujranwala': 'lahore', 'sialkot': 'lahore', 'bahawalpur': 'lahore',
#     'sargodha': 'lahore', 'sahiwal': 'lahore',
    
#     # Federal/KPK
#     'islamabad': 'islamabad', 'rawalpindi': 'islamabad', 'peshawar': 'islamabad',
#     'abbottabad': 'islamabad', 'mardan': 'islamabad', 'quetta': 'islamabad',
    
#     # Abbreviations
#     'khi': 'karachi', 'lhr': 'lahore', 'isb': 'islamabad', 'pindi': 'islamabad'
# }


# def map_city(city):
#     """Map user city to training city"""
#     city_lower = city.lower().strip()
#     mapped = CITY_MAPPING.get(city_lower, "karachi")
#     was_mapped = mapped != city_lower
#     return mapped, was_mapped


# # ============================
# # HOME
# # ============================
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({
#         "message": "🚗 AI Car Price Predictor API",
#         "version": "1.0.0",
#         "status": "running",
#         "endpoints": {
#             "health": "/api/health",
#             "model_info": "/api/model-info",
#             "feature_options": "/api/feature-options",
#             "predict": "/api/predict (POST)",
#             "analytics": "/api/analytics/stats",
#             "visualizations": "/api/visualizations/<type>",
#             "chatbot": "/api/chatbot (POST)"
#         },
#         "documentation": "http://localhost:5000/docs"
#     })


# # ============================
# # HEALTH
# # ============================
# @app.route("/api/health", methods=["GET"])
# def health():
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "poly_loaded": poly is not None,
#         "kmeans_loaded": kmeans is not None,
#         "timestamp": datetime.now().isoformat()
#     })


# # ============================
# # MODEL INFO
# # ============================
# @app.route("/api/model-info", methods=["GET"])
# def model_details():
#     if not model:
#         return jsonify({"error": "Model not loaded"}), 500
    
#     return jsonify({
#         "model_name": model_info.get("model_name"),
#         "accuracy_r2": float(model_info.get("test_r2", 0)),
#         "mae": float(model_info.get("test_mae", 0)),
#         "rmse": float(model_info.get("test_rmse", 0)),
#         "mape": float(model_info.get("test_mape", 0)) if "test_mape" in model_info else None,
#         "features": {
#             "categorical": feature_info["categorical_features"],
#             "numerical": feature_info["numerical_features"],
#             "total_features": len(model_info.get("feature_columns", []))
#         },
#         "polynomial_features": poly is not None,
#         "clustering": kmeans is not None
#     })


# # ============================
# # FEATURE OPTIONS
# # ============================
# @app.route("/api/feature-options", methods=["GET"])
# def feature_options():
#     try:
#         df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))

#         options = {}

#         # Categorical features
#         for col in feature_info["categorical_features"]:
#             options[col] = sorted(df[col].astype(str).unique().tolist())

#         # Numerical features with ranges
#         for col in feature_info["numerical_features"]:
#             options[col] = {
#                 "min": float(df[col].min()),
#                 "max": float(df[col].max()),
#                 "mean": float(df[col].mean()),
#                 "median": float(df[col].median())
#             }

#         return jsonify({
#             "options": options,
#             "total_records": int(len(df)),
#             "cities_supported": list(CITY_MAPPING.keys())
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ============================
# # PREDICTION (FIXED)
# # ============================
# @app.route("/api/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.json
        
#         # Validate required fields
#         required_fields = feature_info["categorical_features"] + feature_info["numerical_features"]
#         missing = [f for f in required_fields if f not in data]
#         if missing:
#             return jsonify({"error": f"Missing fields: {missing}"}), 400

#         # Map city
#         original_city = data.get("city", "karachi")
#         mapped_city, was_mapped = map_city(original_city)
#         data["city"] = mapped_city

#         # Encode categorical features
#         feature_vector = []
#         for col in feature_info["categorical_features"]:
#             val = str(data[col]).lower().strip()
#             try:
#                 encoded_val = label_encoders[col].transform([val])[0]
#                 feature_vector.append(encoded_val)
#             except (ValueError, KeyError):
#                 # Unknown value - use default
#                 feature_vector.append(0)
#                 print(f"   ⚠️ Unknown value '{val}' for {col}, using default")

#         # Get numerical features
#         numerical_values = [float(data[col]) for col in feature_info["numerical_features"]]

#         # Apply polynomial features
#         if poly is not None:
#             numerical_array = np.array([numerical_values])
#             poly_features = poly.transform(numerical_array)
#             poly_feature_names = poly.get_feature_names_out(feature_info["numerical_features"])
            
#             # Combine encoded categorical + polynomial features
#             all_features = feature_vector + poly_features[0].tolist()
            
#             # Create column names
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(poly_feature_names)
#         else:
#             # No polynomial - use original features
#             all_features = feature_vector + numerical_values
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(feature_info["numerical_features"])

#         # Create dataframe
#         df_input = pd.DataFrame([all_features], columns=feature_columns)
        
#         # Align with training columns
#         if 'feature_columns' in model_info:
#             training_columns = model_info['feature_columns']
#             # Add missing columns as 0
#             for col in training_columns:
#                 if col not in df_input.columns:
#                     df_input[col] = 0
#             # Reorder to match training
#             df_input = df_input[training_columns]

#         # Predict price
#         predicted_price = float(model.predict(df_input)[0])

#         # Calculate price range (±10%)
#         price_min = predicted_price * 0.90
#         price_max = predicted_price * 1.10

#         # Determine segment using K-Means
#         if kmeans is not None and cluster_scaler is not None:
#             cluster_input = np.array([[predicted_price, data['registered_in'], data['mileage']]])
#             cluster_scaled = cluster_scaler.transform(cluster_input)
#             cluster_id = kmeans.predict(cluster_scaled)[0]
            
#             cluster_map = feature_info.get('cluster_map', {0: 'Economy', 1: 'Mid-Range', 2: 'Luxury'})
#             segment = cluster_map.get(cluster_id, 'Unknown')
#         else:
#             # Fallback segment determination
#             if predicted_price < 1000000:
#                 segment = "Economy"
#             elif predicted_price < 3000000:
#                 segment = "Mid-Range"
#             elif predicted_price < 5000000:
#                 segment = "Premium"
#             else:
#                 segment = "Luxury"

#         return jsonify({
#             "success": True,
#             "predicted_price": predicted_price,
#             "price_range": {
#                 "min": price_min,
#                 "max": price_max
#             },
#             "price_lacs": round(predicted_price / 100000, 2),
#             "segment": segment,
#             "city": {
#                 "original": original_city,
#                 "mapped": mapped_city,
#                 "was_mapped": was_mapped
#             },
#             "model_accuracy": float(model_info.get("test_r2", 0)),
#             "average_error": float(model_info.get("test_mae", 0)),
#             "timestamp": datetime.now().isoformat()
#         })

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500


# # ============================
# # ANALYTICS (FIXED)
# # ============================
# @app.route("/api/analytics/stats", methods=["GET"])
# def analytics():
#     try:
#         df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))

#         def clean(d):
#             """Convert to JSON-serializable format"""
#             return {str(k): int(v) if isinstance(v, (np.integer, np.int64)) else float(v) for k, v in d.items()}

#         return jsonify({
#             "dataset": {
#                 "total_cars": int(len(df)),
#                 "year_range": f"{int(df['registered_in'].min())} - {int(df['registered_in'].max())}",
#                 "mileage_range": f"{int(df['mileage'].min())} - {int(df['mileage'].max())} km"
#             },
#             "price_stats": {
#                 "min": float(df["price"].min()),
#                 "max": float(df["price"].max()),
#                 "mean": float(df["price"].mean()),
#                 "median": float(df["price"].median()),
#                 "std": float(df["price"].std())
#             },
#             "top_brands": clean(df["car_brand"].value_counts().head(10).to_dict()),
#             "top_models": clean(df["car_model"].value_counts().head(10).to_dict()),
#             "fuel_type_distribution": clean(df["fuel_type"].value_counts().to_dict()),
#             "transmission_distribution": clean(df["transmission"].value_counts().to_dict()),
#             "city_distribution": clean(df["city"].value_counts().to_dict())
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ============================
# # VISUALIZATIONS
# # ============================
# @app.route("/api/visualizations/<name>")
# def visualizations(name):
#     files = {
#         "actual_vs_predicted": "actual_vs_predicted.png",
#         "feature_importance": "feature_importance.png",
#         "model_comparison": "model_comparison.png",
#         "error_distribution": "error_distribution.png"
#     }

#     if name not in files:
#         return jsonify({"error": "Invalid visualization name", "available": list(files.keys())}), 404

#     path = os.path.join(VIZ_DIR, files[name])
    
#     if not os.path.exists(path):
#         return jsonify({"error": f"File not found: {files[name]}"}), 404
    
#     return send_file(path, mimetype="image/png")


# # ============================
# # CONVERSATIONAL CHATBOT
# # ============================
# from chatbot import chatbot as ai_chatbot

# @app.route("/api/chatbot", methods=["POST"])
# def chatbot_endpoint():
#     """
#     Enhanced conversational chatbot endpoint
#     Handles step-by-step car detail collection
#     """
#     try:
#         data = request.json
#         session_id = data.get("session_id", "default")
#         message = data.get("message", "")
        
#         print(f"\n📨 Chatbot request: session={session_id}, message='{message}'")
        
#         # Process message through chatbot
#         response = ai_chatbot.process_message(
#             session_id=session_id,
#             message=message,
#             model_info=model_info
#         )
        
#         print(f"🤖 Chatbot response action: {response.get('action')}")
        
#         # If chatbot says to predict, make prediction
#         if response.get('action') == 'predict' and 'data' in response:
#             try:
#                 car_data = response['data']
#                 print(f"📊 Making prediction with data: {car_data}")
                
#                 # Convert types with validation
#                 try:
#                     car_data['engine'] = float(str(car_data['engine']).replace(',', ''))
#                     car_data['registered_in'] = int(str(car_data['registered_in']).replace(',', ''))
#                     car_data['mileage'] = float(str(car_data['mileage']).replace(',', ''))
#                 except ValueError as ve:
#                     print(f"❌ Type conversion error: {ve}")
#                     response['reply'] = f"❌ Invalid data format: {str(ve)}\nPlease try again with 'predict'."
#                     response['timestamp'] = datetime.now().isoformat()
#                     return jsonify(response)
                
#                 # Make prediction (reuse predict logic)
#                 prediction_result = make_prediction(car_data)
                
#                 if prediction_result['success']:
#                     pred = prediction_result
#                     response['reply'] = (
#                         f"🎉 **Price Prediction:**\n\n"
#                         f"💰 **Estimated Price:** {pred['predicted_price']:,.0f} PKR\n"
#                         f"📊 **Price Range:** {pred['price_range']['min']:,.0f} - {pred['price_range']['max']:,.0f} PKR\n"
#                         f"🎯 **Segment:** {pred['segment']}\n"
#                         f"📍 **City:** {pred['city']['original'].title()}\n\n"
#                         f"✨ **Price in Lacs:** {pred['price_lacs']} Lacs\n\n"
#                         f"Want to predict another car? Say 'predict'!"
#                     )
#                     response['prediction'] = prediction_result
#                     print(f"✅ Prediction successful: {pred['predicted_price']:,.0f} PKR")
#                 else:
#                     response['reply'] = f"❌ Prediction failed: {prediction_result.get('error', 'Unknown error')}\nTry 'predict' to start over."
#                     print(f"❌ Prediction failed: {prediction_result.get('error')}")
            
#             except Exception as e:
#                 import traceback
#                 print(f"❌ Prediction exception: {e}")
#                 traceback.print_exc()
#                 response['reply'] = f"❌ Error making prediction: {str(e)}\nPlease try 'predict' to start fresh."
        
#         response['timestamp'] = datetime.now().isoformat()
#         return jsonify(response)
    
#     except Exception as e:
#         import traceback
#         print(f"❌ Chatbot endpoint error: {e}")
#         traceback.print_exc()
#         return jsonify({
#             "error": str(e), 
#             "reply": "Sorry, something went wrong. Please try again.",
#             "timestamp": datetime.now().isoformat()
#         }), 500


# def make_prediction(data):
#     """
#     Helper function to make prediction
#     Extracted from /api/predict for reuse
#     """
#     try:
#         # Map city
#         original_city = data.get("city", "karachi")
#         mapped_city, was_mapped = map_city(original_city)
#         data["city"] = mapped_city

#         # Encode categorical
#         feature_vector = []
#         for col in feature_info["categorical_features"]:
#             val = str(data[col]).lower().strip()
#             try:
#                 encoded_val = label_encoders[col].transform([val])[0]
#                 feature_vector.append(encoded_val)
#             except:
#                 feature_vector.append(0)

#         # Get numerical
#         numerical_values = [float(data[col]) for col in feature_info["numerical_features"]]

#         # Apply polynomial
#         if poly is not None:
#             numerical_array = np.array([numerical_values])
#             poly_features = poly.transform(numerical_array)
#             poly_feature_names = poly.get_feature_names_out(feature_info["numerical_features"])
#             all_features = feature_vector + poly_features[0].tolist()
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(poly_feature_names)
#         else:
#             all_features = feature_vector + numerical_values
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(feature_info["numerical_features"])

#         # Create dataframe
#         df_input = pd.DataFrame([all_features], columns=feature_columns)
        
#         if 'feature_columns' in model_info:
#             training_columns = model_info['feature_columns']
#             for col in training_columns:
#                 if col not in df_input.columns:
#                     df_input[col] = 0
#             df_input = df_input[training_columns]

#         # Predict
#         predicted_price = float(model.predict(df_input)[0])
#         price_min = predicted_price * 0.90
#         price_max = predicted_price * 1.10

#         # Segment
#         if kmeans is not None and cluster_scaler is not None:
#             cluster_input = np.array([[predicted_price, data['registered_in'], data['mileage']]])
#             cluster_scaled = cluster_scaler.transform(cluster_input)
#             cluster_id = kmeans.predict(cluster_scaled)[0]
#             cluster_map = feature_info.get('cluster_map', {0: 'Economy', 1: 'Mid-Range', 2: 'Luxury'})
#             segment = cluster_map.get(cluster_id, 'Unknown')
#         else:
#             if predicted_price < 1000000:
#                 segment = "Economy"
#             elif predicted_price < 3000000:
#                 segment = "Mid-Range"
#             else:
#                 segment = "Luxury"

#         return {
#             "success": True,
#             "predicted_price": predicted_price,
#             "price_range": {"min": price_min, "max": price_max},
#             "price_lacs": round(predicted_price / 100000, 2),
#             "segment": segment,
#             "city": {"original": original_city, "mapped": mapped_city, "was_mapped": was_mapped},
#             "model_accuracy": float(model_info.get("test_r2", 0)),
#             "average_error": float(model_info.get("test_mae", 0))
#         }
    
#     except Exception as e:
#         return {"success": False, "error": str(e)}


# # ============================
# # ERROR HANDLERS
# # ============================
# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({"error": "Endpoint not found"}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({"error": "Internal server error"}), 500


# # ============================
# # START SERVER
# # ============================
# if __name__ == "__main__":
#     print("\n" + "=" * 60)
#     print("🚀 FLASK BACKEND STARTING...")
#     print("=" * 60)
    
#     # Load models
#     success = load_models()
    
#     if not success:
#         print("⚠️ WARNING: Some models failed to load!")
#         print("   Server will start but predictions may fail.")
    
#     print("\n📍 Server running at: http://localhost:5000")
#     print("📖 API Documentation: http://localhost:5000")
#     print("\n💡 Test with:")
#     print("   curl http://localhost:5000/api/health")
#     print("\n" + "=" * 60 + "\n")
    
#     app.run(debug=True, host="0.0.0.0", port=5000)


# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import pickle
# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime

# app = Flask(__name__)
# CORS(app)

# # ============================
# # PATHS (FIXED)
# # ============================
# # Adjust these paths based on your folder structure
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_DIR = os.path.join(BASE_DIR, 'models')
# DATA_DIR = os.path.join(BASE_DIR, 'data')
# VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')

# # ============================
# # GLOBAL MODELS
# # ============================
# model = None
# label_encoders = None
# feature_info = None
# model_info = None
# poly = None
# kmeans = None
# cluster_scaler = None


# # ============================
# # LOAD MODELS
# # ============================
# def load_models():
#     global model, label_encoders, feature_info, model_info, poly, kmeans, cluster_scaler

#     print("\n" + "=" * 60)
#     print("📦 Loading ML Models...")
#     print("=" * 60)

#     try:
#         def load(file):
#             path = os.path.join(MODEL_DIR, file)
#             print(f"   Loading: {file}")
#             with open(path, "rb") as f:
#                 return pickle.load(f)

#         model = load("best_model.pkl")
#         label_encoders = load("label_encoders.pkl")
#         feature_info = load("feature_info.pkl")
#         model_info = load("model_info.pkl")
#         poly = load("poly_transformer.pkl")
#         kmeans = load("kmeans_model.pkl")
#         cluster_scaler = load("cluster_scaler.pkl")

#         print("✅ All models loaded successfully!")
#         print(f"   Model: {model_info.get('model_name', 'Unknown')}")
#         print(f"   Accuracy: {model_info.get('test_r2', 0):.4f}")
#         print("=" * 60 + "\n")
#         return True

#     except Exception as e:
#         print(f"❌ Error loading models: {e}")
#         print(f"   MODEL_DIR: {MODEL_DIR}")
#         return False


# # ============================
# # CITY MAPPING
# # ============================
# CITY_MAPPING = {
#     # Sindh
#     'karachi': 'karachi', 'hyderabad': 'karachi', 'sukkur': 'karachi',
#     'larkana': 'karachi', 'mirpurkhas': 'karachi',
    
#     # Punjab
#     'lahore': 'lahore', 'faisalabad': 'lahore', 'multan': 'lahore',
#     'gujranwala': 'lahore', 'sialkot': 'lahore', 'bahawalpur': 'lahore',
#     'sargodha': 'lahore', 'sahiwal': 'lahore',
    
#     # Federal/KPK
#     'islamabad': 'islamabad', 'rawalpindi': 'islamabad', 'peshawar': 'islamabad',
#     'abbottabad': 'islamabad', 'mardan': 'islamabad', 'quetta': 'islamabad',
    
#     # Abbreviations
#     'khi': 'karachi', 'lhr': 'lahore', 'isb': 'islamabad', 'pindi': 'islamabad'
# }


# def map_city(city):
#     """Map user city to training city"""
#     city_lower = city.lower().strip()
#     mapped = CITY_MAPPING.get(city_lower, "karachi")
#     was_mapped = mapped != city_lower
#     return mapped, was_mapped


# # ============================
# # HOME
# # ============================
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({
#         "message": "🚗 AI Car Price Predictor API",
#         "version": "1.0.0",
#         "status": "running",
#         "endpoints": {
#             "health": "/api/health",
#             "model_info": "/api/model-info",
#             "feature_options": "/api/feature-options",
#             "predict": "/api/predict (POST)",
#             "analytics": "/api/analytics/stats",
#             "visualizations": "/api/visualizations/<type>",
#             "chatbot": "/api/chatbot (POST)"
#         },
#         "documentation": "http://localhost:5000/docs"
#     })


# # ============================
# # HEALTH
# # ============================
# @app.route("/api/health", methods=["GET"])
# def health():
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "poly_loaded": poly is not None,
#         "kmeans_loaded": kmeans is not None,
#         "timestamp": datetime.now().isoformat()
#     })


# # ============================
# # MODEL INFO
# # ============================
# @app.route("/api/model-info", methods=["GET"])
# def model_details():
#     if not model:
#         return jsonify({"error": "Model not loaded"}), 500
    
#     return jsonify({
#         "model_name": model_info.get("model_name"),
#         "accuracy_r2": float(model_info.get("test_r2", 0)),
#         "mae": float(model_info.get("test_mae", 0)),
#         "rmse": float(model_info.get("test_rmse", 0)),
#         "mape": float(model_info.get("test_mape", 0)) if "test_mape" in model_info else None,
#         "features": {
#             "categorical": feature_info["categorical_features"],
#             "numerical": feature_info["numerical_features"],
#             "total_features": len(model_info.get("feature_columns", []))
#         },
#         "polynomial_features": poly is not None,
#         "clustering": kmeans is not None
#     })


# # ============================
# # FEATURE OPTIONS
# # ============================
# @app.route("/api/feature-options", methods=["GET"])
# def feature_options():
#     try:
#         df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))

#         options = {}

#         # Categorical features
#         for col in feature_info["categorical_features"]:
#             options[col] = sorted(df[col].astype(str).unique().tolist())

#         # Numerical features with ranges
#         for col in feature_info["numerical_features"]:
#             options[col] = {
#                 "min": float(df[col].min()),
#                 "max": float(df[col].max()),
#                 "mean": float(df[col].mean()),
#                 "median": float(df[col].median())
#             }

#         return jsonify({
#             "options": options,
#             "total_records": int(len(df)),
#             "cities_supported": list(CITY_MAPPING.keys())
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ============================
# # PREDICTION (FIXED)
# # ============================
# @app.route("/api/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.json
        
#         # Validate required fields
#         required_fields = feature_info["categorical_features"] + feature_info["numerical_features"]
#         missing = [f for f in required_fields if f not in data]
#         if missing:
#             return jsonify({"error": f"Missing fields: {missing}"}), 400

#         # Map city
#         original_city = data.get("city", "karachi")
#         mapped_city, was_mapped = map_city(original_city)
#         data["city"] = mapped_city

#         # Encode categorical features
#         feature_vector = []
#         for col in feature_info["categorical_features"]:
#             val = str(data[col]).lower().strip()
#             try:
#                 encoded_val = label_encoders[col].transform([val])[0]
#                 feature_vector.append(encoded_val)
#             except (ValueError, KeyError):
#                 # Unknown value - use default
#                 feature_vector.append(0)
#                 print(f"   ⚠️ Unknown value '{val}' for {col}, using default")

#         # Get numerical features
#         numerical_values = [float(data[col]) for col in feature_info["numerical_features"]]

#         # Apply polynomial features
#         if poly is not None:
#             numerical_array = np.array([numerical_values])
#             poly_features = poly.transform(numerical_array)
#             poly_feature_names = poly.get_feature_names_out(feature_info["numerical_features"])
            
#             # Combine encoded categorical + polynomial features
#             all_features = feature_vector + poly_features[0].tolist()
            
#             # Create column names
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(poly_feature_names)
#         else:
#             # No polynomial - use original features
#             all_features = feature_vector + numerical_values
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(feature_info["numerical_features"])

#         # Create dataframe
#         df_input = pd.DataFrame([all_features], columns=feature_columns)
        
#         # Align with training columns
#         if 'feature_columns' in model_info:
#             training_columns = model_info['feature_columns']
#             # Add missing columns as 0
#             for col in training_columns:
#                 if col not in df_input.columns:
#                     df_input[col] = 0
#             # Reorder to match training
#             df_input = df_input[training_columns]

#         # Predict price
#         predicted_price = float(model.predict(df_input)[0])

#         # Calculate price range (±10%)
#         price_min = predicted_price * 0.90
#         price_max = predicted_price * 1.10

#         # Determine segment using K-Means
#         if kmeans is not None and cluster_scaler is not None:
#             cluster_input = np.array([[predicted_price, data['registered_in'], data['mileage']]])
#             cluster_scaled = cluster_scaler.transform(cluster_input)
#             cluster_id = kmeans.predict(cluster_scaled)[0]
            
#             cluster_map = feature_info.get('cluster_map', {0: 'Economy', 1: 'Mid-Range', 2: 'Luxury'})
#             segment = cluster_map.get(cluster_id, 'Unknown')
#         else:
#             # Fallback segment determination
#             if predicted_price < 1000000:
#                 segment = "Economy"
#             elif predicted_price < 3000000:
#                 segment = "Mid-Range"
#             elif predicted_price < 5000000:
#                 segment = "Premium"
#             else:
#                 segment = "Luxury"

#         return jsonify({
#             "success": True,
#             "predicted_price": predicted_price,
#             "price_range": {
#                 "min": price_min,
#                 "max": price_max
#             },
#             "price_lacs": round(predicted_price / 100000, 2),
#             "segment": segment,
#             "city": {
#                 "original": original_city,
#                 "mapped": mapped_city,
#                 "was_mapped": was_mapped
#             },
#             "model_accuracy": float(model_info.get("test_r2", 0)),
#             "average_error": float(model_info.get("test_mae", 0)),
#             "timestamp": datetime.now().isoformat()
#         })

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500


# # ============================
# # ANALYTICS (FIXED)
# # ============================
# @app.route("/api/analytics/stats", methods=["GET"])
# def analytics():
#     try:
#         df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))

#         def clean(d):
#             """Convert to JSON-serializable format"""
#             return {str(k): int(v) if isinstance(v, (np.integer, np.int64)) else float(v) for k, v in d.items()}

#         return jsonify({
#             "dataset": {
#                 "total_cars": int(len(df)),
#                 "year_range": f"{int(df['registered_in'].min())} - {int(df['registered_in'].max())}",
#                 "mileage_range": f"{int(df['mileage'].min())} - {int(df['mileage'].max())} km"
#             },
#             "price_stats": {
#                 "min": float(df["price"].min()),
#                 "max": float(df["price"].max()),
#                 "mean": float(df["price"].mean()),
#                 "median": float(df["price"].median()),
#                 "std": float(df["price"].std())
#             },
#             "top_brands": clean(df["car_brand"].value_counts().head(10).to_dict()),
#             "top_models": clean(df["car_model"].value_counts().head(10).to_dict()),
#             "fuel_type_distribution": clean(df["fuel_type"].value_counts().to_dict()),
#             "transmission_distribution": clean(df["transmission"].value_counts().to_dict()),
#             "city_distribution": clean(df["city"].value_counts().to_dict())
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # ============================
# # VISUALIZATIONS
# # ============================
# @app.route("/api/visualizations/<name>")
# def visualizations(name):
#     files = {
#         "actual_vs_predicted": "actual_vs_predicted.png",
#         "feature_importance": "feature_importance.png",
#         "model_comparison": "model_comparison.png",
#         "error_distribution": "error_distribution.png"
#     }

#     if name not in files:
#         return jsonify({"error": "Invalid visualization name", "available": list(files.keys())}), 404

#     path = os.path.join(VIZ_DIR, files[name])
    
#     if not os.path.exists(path):
#         return jsonify({"error": f"File not found: {files[name]}"}), 404
    
#     return send_file(path, mimetype="image/png")


# # ============================
# # CONVERSATIONAL CHATBOT
# # ============================
# from chatbot import chatbot as ai_chatbot

# @app.route("/api/chatbot", methods=["POST"])
# def chatbot_endpoint():
#     """
#     Enhanced conversational chatbot endpoint
#     Handles step-by-step car detail collection
#     """
#     try:
#         data = request.json
#         session_id = data.get("session_id", "default")
#         message = data.get("message", "")
        
#         print(f"\n📨 Chatbot request: session={session_id}, message='{message}'")
        
#         # Process message through chatbot
#         response = ai_chatbot.process_message(
#             session_id=session_id,
#             message=message,
#             model_info=model_info
#         )
        
#         print(f"🤖 Chatbot response action: {response.get('action')}")
        
#         # If chatbot says to predict, make prediction
#         if response.get('action') == 'predict' and 'data' in response:
#             try:
#                 car_data = response['data']
#                 print(f"📊 Making prediction with data: {car_data}")
                
#                 # Convert types with validation
#                 try:
#                     car_data['engine'] = float(str(car_data['engine']).replace(',', ''))
#                     car_data['registered_in'] = int(str(car_data['registered_in']).replace(',', ''))
#                     car_data['mileage'] = float(str(car_data['mileage']).replace(',', ''))
#                 except ValueError as ve:
#                     print(f"❌ Type conversion error: {ve}")
#                     response['reply'] = f"❌ Invalid data format: {str(ve)}\nPlease try again with 'predict'."
#                     response['timestamp'] = datetime.now().isoformat()
#                     return jsonify(response)
                
#                 # Make prediction (reuse predict logic)
#                 prediction_result = make_prediction(car_data)
                
#                 if prediction_result['success']:
#                     pred = prediction_result
#                     response['reply'] = (
#                         f"🎉 **Price Prediction:**\n\n"
#                         f"💰 **Estimated Price:** {pred['predicted_price']:,.0f} PKR\n"
#                         f"📊 **Price Range:** {pred['price_range']['min']:,.0f} - {pred['price_range']['max']:,.0f} PKR\n"
#                         f"🎯 **Segment:** {pred['segment']}\n"
#                         f"📍 **City:** {pred['city']['original'].title()}\n\n"
#                         f"✨ **Price in Lacs:** {pred['price_lacs']} Lacs\n\n"
#                         f"Want to predict another car? Say 'predict'!"
#                     )
#                     response['prediction'] = prediction_result
#                     print(f"✅ Prediction successful: {pred['predicted_price']:,.0f} PKR")
#                 else:
#                     response['reply'] = f"❌ Prediction failed: {prediction_result.get('error', 'Unknown error')}\nTry 'predict' to start over."
#                     print(f"❌ Prediction failed: {prediction_result.get('error')}")
            
#             except Exception as e:
#                 import traceback
#                 print(f"❌ Prediction exception: {e}")
#                 traceback.print_exc()
#                 response['reply'] = f"❌ Error making prediction: {str(e)}\nPlease try 'predict' to start fresh."
        
#         response['timestamp'] = datetime.now().isoformat()
#         return jsonify(response)
    
#     except Exception as e:
#         import traceback
#         print(f"❌ Chatbot endpoint error: {e}")
#         traceback.print_exc()
#         return jsonify({
#             "error": str(e), 
#             "reply": "Sorry, something went wrong. Please try again.",
#             "timestamp": datetime.now().isoformat()
#         }), 500


# def make_prediction(data):
#     """
#     Helper function to make prediction
#     Extracted from /api/predict for reuse
#     """
#     try:
#         # Map city
#         original_city = data.get("city", "karachi")
#         mapped_city, was_mapped = map_city(original_city)
#         data["city"] = mapped_city

#         # Encode categorical
#         feature_vector = []
#         for col in feature_info["categorical_features"]:
#             val = str(data[col]).lower().strip()
#             try:
#                 encoded_val = label_encoders[col].transform([val])[0]
#                 feature_vector.append(encoded_val)
#             except:
#                 feature_vector.append(0)

#         # Get numerical
#         numerical_values = [float(data[col]) for col in feature_info["numerical_features"]]

#         # Apply polynomial
#         if poly is not None:
#             numerical_array = np.array([numerical_values])
#             poly_features = poly.transform(numerical_array)
#             poly_feature_names = poly.get_feature_names_out(feature_info["numerical_features"])
#             all_features = feature_vector + poly_features[0].tolist()
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(poly_feature_names)
#         else:
#             all_features = feature_vector + numerical_values
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(feature_info["numerical_features"])

#         # Create dataframe
#         df_input = pd.DataFrame([all_features], columns=feature_columns)
        
#         if 'feature_columns' in model_info:
#             training_columns = model_info['feature_columns']
#             for col in training_columns:
#                 if col not in df_input.columns:
#                     df_input[col] = 0
#             df_input = df_input[training_columns]

#         # Predict
#         predicted_price = float(model.predict(df_input)[0])
#         price_min = predicted_price * 0.90
#         price_max = predicted_price * 1.10

#         # Determine segment using K-Means
#         if kmeans is not None and cluster_scaler is not None:
#             try:
#                 # Check what features cluster_scaler expects
#                 expected_features = cluster_scaler.n_features_in_
                
#                 # Prepare cluster input based on expected features
#                 if expected_features == 3:
#                     cluster_input = np.array([[predicted_price, data['registered_in'], data['mileage']]])
#                 elif expected_features == 4:
#                     # If 4 features expected, add engine
#                     cluster_input = np.array([[predicted_price, data['registered_in'], data['mileage'], data['engine']]])
#                 else:
#                     # Fallback: try with 3 most common features
#                     cluster_input = np.array([[predicted_price, data['registered_in'], data['mileage']]])
                
#                 cluster_scaled = cluster_scaler.transform(cluster_input)
#                 cluster_id = kmeans.predict(cluster_scaled)[0]
                
#                 cluster_map = feature_info.get('cluster_map', {0: 'Economy', 1: 'Mid-Range', 2: 'Luxury'})
#                 segment = cluster_map.get(cluster_id, 'Unknown')
                
#             except Exception as cluster_error:
#                 print(f"⚠️ Clustering error: {cluster_error}, using price-based fallback")
#                 # Fallback to price-based segmentation
#                 if predicted_price < 1000000:
#                     segment = "Economy"
#                 elif predicted_price < 3000000:
#                     segment = "Mid-Range"
#                 elif predicted_price < 5000000:
#                     segment = "Premium"
#                 else:
#                     segment = "Luxury"
#         else:
#             # Fallback segment determination
#             if predicted_price < 1000000:
#                 segment = "Economy"
#             elif predicted_price < 3000000:
#                 segment = "Mid-Range"
#             elif predicted_price < 5000000:
#                 segment = "Premium"
#             else:
#                 segment = "Luxury"

#         return {
#             "success": True,
#             "predicted_price": predicted_price,
#             "price_range": {"min": price_min, "max": price_max},
#             "price_lacs": round(predicted_price / 100000, 2),
#             "segment": segment,
#             "city": {"original": original_city, "mapped": mapped_city, "was_mapped": was_mapped},
#             "model_accuracy": float(model_info.get("test_r2", 0)),
#             "average_error": float(model_info.get("test_mae", 0))
#         }
    
#     except Exception as e:
#         return {"success": False, "error": str(e)}


# # ============================
# # ERROR HANDLERS
# # ============================
# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({"error": "Endpoint not found"}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({"error": "Internal server error"}), 500


# # ============================
# # START SERVER
# # ============================
# if __name__ == "__main__":
#     print("\n" + "=" * 60)
#     print("🚀 FLASK BACKEND STARTING...")
#     print("=" * 60)
    
#     # Load models
#     success = load_models()
    
#     if not success:
#         print("⚠️ WARNING: Some models failed to load!")
#         print("   Server will start but predictions may fail.")
    
#     print("\n📍 Server running at: http://localhost:5000")
#     print("📖 API Documentation: http://localhost:5000")
#     print("\n💡 Test with:")
#     print("   curl http://localhost:5000/api/health")
#     print("\n" + "=" * 60 + "\n")
    
#     app.run(debug=True, host="0.0.0.0", port=5000)



# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import pickle
# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime

# app = Flask(__name__)
# CORS(app)

# # ============================
# # PATHS (FIXED)
# # ============================
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_DIR = os.path.join(BASE_DIR, 'models')
# DATA_DIR = os.path.join(BASE_DIR, 'data')
# VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')

# # ============================
# # GLOBAL MODELS
# # ============================
# model = None
# label_encoders = None
# feature_info = None
# model_info = None
# poly = None
# kmeans = None
# cluster_scaler = None

# # ============================
# # LOAD MODELS
# # ============================
# def load_models():
#     global model, label_encoders, feature_info, model_info, poly, kmeans, cluster_scaler
#     print("\n" + "=" * 60)
#     print("📦 Loading ML Models...")
#     print("=" * 60)
#     try:
#         def load(file):
#             path = os.path.join(MODEL_DIR, file)
#             print(f"   Loading: {file}")
#             with open(path, "rb") as f:
#                 return pickle.load(f)

#         model = load("best_model.pkl")
#         label_encoders = load("label_encoders.pkl")
#         feature_info = load("feature_info.pkl")
#         model_info = load("model_info.pkl")
#         poly = load("poly_transformer.pkl")
#         kmeans = load("kmeans_model.pkl")
#         cluster_scaler = load("cluster_scaler.pkl")

#         print("✅ All models loaded successfully!")
#         print(f"   Model: {model_info.get('model_name', 'Unknown')}")
#         print(f"   Accuracy: {model_info.get('test_r2', 0):.4f}")
#         print("=" * 60 + "\n")
#         return True

#     except Exception as e:
#         print(f"❌ Error loading models: {e}")
#         print(f"   MODEL_DIR: {MODEL_DIR}")
#         return False

# # ============================
# # CITY MAPPING
# # ============================
# CITY_MAPPING = {
#     'karachi': 'karachi', 'hyderabad': 'karachi', 'sukkur': 'karachi',
#     'larkana': 'karachi', 'mirpurkhas': 'karachi',
#     'lahore': 'lahore', 'faisalabad': 'lahore', 'multan': 'lahore',
#     'gujranwala': 'lahore', 'sialkot': 'lahore', 'bahawalpur': 'lahore',
#     'sargodha': 'lahore', 'sahiwal': 'lahore',
#     'islamabad': 'islamabad', 'rawalpindi': 'islamabad', 'peshawar': 'islamabad',
#     'abbottabad': 'islamabad', 'mardan': 'islamabad', 'quetta': 'islamabad',
#     'khi': 'karachi', 'lhr': 'lahore', 'isb': 'islamabad', 'pindi': 'islamabad'
# }

# def map_city(city):
#     city_lower = city.lower().strip()
#     mapped = CITY_MAPPING.get(city_lower, "karachi")
#     was_mapped = mapped != city_lower
#     return mapped, was_mapped

# # ============================
# # HOME
# # ============================
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({
#         "message": "🚗 AI Car Price Predictor API",
#         "version": "1.0.0",
#         "status": "running",
#         "endpoints": {
#             "health": "/api/health",
#             "model_info": "/api/model-info",
#             "feature_options": "/api/feature-options",
#             "predict": "/api/predict (POST)",
#             "analytics": "/api/analytics/stats",
#             "visualizations": "/api/visualizations/<type>",
#             "chatbot": "/api/chatbot (POST)"
#         },
#         "documentation": "http://localhost:5000/docs"
#     })

# # ============================
# # HEALTH
# # ============================
# @app.route("/api/health", methods=["GET"])
# def health():
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "poly_loaded": poly is not None,
#         "kmeans_loaded": kmeans is not None,
#         "timestamp": datetime.now().isoformat()
#     })

# # ============================
# # MODEL INFO
# # ============================
# @app.route("/api/model-info", methods=["GET"])
# def model_details():
#     if not model:
#         return jsonify({"error": "Model not loaded"}), 500
    
#     return jsonify({
#         "model_name": model_info.get("model_name"),
#         "accuracy_r2": float(model_info.get("test_r2", 0)),
#         "mae": float(model_info.get("test_mae", 0)),
#         "rmse": float(model_info.get("test_rmse", 0)),
#         "mape": float(model_info.get("test_mape", 0)) if "test_mape" in model_info else None,
#         "features": {
#             "categorical": feature_info["categorical_features"],
#             "numerical": feature_info["numerical_features"],
#             "total_features": len(model_info.get("feature_columns", []))
#         },
#         "polynomial_features": poly is not None,
#         "clustering": kmeans is not None
#     })

# # ============================
# # FEATURE OPTIONS
# # ============================
# @app.route("/api/feature-options", methods=["GET"])
# def feature_options():
#     try:
#         df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))
#         options = {}
#         for col in feature_info["categorical_features"]:
#             options[col] = sorted(df[col].astype(str).unique().tolist())
#         for col in feature_info["numerical_features"]:
#             options[col] = {
#                 "min": float(df[col].min()),
#                 "max": float(df[col].max()),
#                 "mean": float(df[col].mean()),
#                 "median": float(df[col].median())
#             }
#         return jsonify({
#             "options": options,
#             "total_records": int(len(df)),
#             "cities_supported": list(CITY_MAPPING.keys())
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # ============================
# # PREDICTION (FIXED)
# # ============================
# @app.route("/api/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.json
#         required_fields = feature_info["categorical_features"] + feature_info["numerical_features"]
#         missing = [f for f in required_fields if f not in data]
#         if missing:
#             return jsonify({"error": f"Missing fields: {missing}"}), 400

#         return make_prediction(data)

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500

# # ============================
# # HELPER FUNCTION
# # ============================
# def make_prediction(data):
#     try:
#         original_city = data.get("city", "karachi")
#         mapped_city, was_mapped = map_city(original_city)
#         data["city"] = mapped_city

#         feature_vector = []
#         for col in feature_info["categorical_features"]:
#             val = str(data[col]).lower().strip()
#             try:
#                 encoded_val = label_encoders[col].transform([val])[0]
#                 feature_vector.append(encoded_val)
#             except:
#                 feature_vector.append(0)

#         numerical_values = [float(data[col]) for col in feature_info["numerical_features"]]

#         if poly is not None:
#             numerical_array = np.array([numerical_values])
#             poly_features = poly.transform(numerical_array)
#             poly_feature_names = poly.get_feature_names_out(feature_info["numerical_features"])
#             all_features = feature_vector + poly_features[0].tolist()
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(poly_feature_names)
#         else:
#             all_features = feature_vector + numerical_values
#             feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
#             feature_columns.extend(feature_info["numerical_features"])

#         df_input = pd.DataFrame([all_features], columns=feature_columns)

#         if 'feature_columns' in model_info:
#             training_columns = model_info['feature_columns']
#             for col in training_columns:
#                 if col not in df_input.columns:
#                     df_input[col] = 0
#             df_input = df_input[training_columns]

#         predicted_price = float(model.predict(df_input)[0])
#         price_min = predicted_price * 0.90
#         price_max = predicted_price * 1.10

#         # ============================
#         # SEGMENT DETERMINATION (FIXED)
#         # ============================
#         segment = None
#         if kmeans is not None and cluster_scaler is not None:
#             try:
#                 expected_features = cluster_scaler.n_features_in_
#                 if expected_features == 3:
#                     cluster_input = np.array([[predicted_price, data['registered_in'], data['mileage']]])
#                 elif expected_features == 4:
#                     cluster_input = np.array([[predicted_price, data['registered_in'], data['mileage'], data['engine']]])
#                 else:
#                     cluster_input = np.array([[predicted_price, data['registered_in'], data['mileage']]])

#                 cluster_scaled = cluster_scaler.transform(cluster_input)
#                 cluster_id = kmeans.predict(cluster_scaled)[0]
#                 cluster_map = feature_info.get('cluster_map', {0: 'Economy', 1: 'Mid-Range', 2: 'Luxury'})
#                 segment = cluster_map.get(cluster_id, None)
#             except Exception as cluster_error:
#                 print(f"⚠️ Clustering error: {cluster_error}, using fallback")
#                 segment = None

#         if segment is None:
#             if predicted_price < 1000000:
#                 segment = "Economy"
#             elif predicted_price < 3000000:
#                 segment = "Mid-Range"
#             elif predicted_price < 5000000:
#                 segment = "Premium"
#             else:
#                 segment = "Luxury"

#         return jsonify({
#             "success": True,
#             "predicted_price": predicted_price,
#             "price_range": {"min": price_min, "max": price_max},
#             "price_lacs": round(predicted_price / 100000, 2),
#             "segment": segment,
#             "city": {"original": original_city, "mapped": mapped_city, "was_mapped": was_mapped},
#             "model_accuracy": float(model_info.get("test_r2", 0)),
#             "average_error": float(model_info.get("test_mae", 0)),
#             "timestamp": datetime.now().isoformat()
#         })
#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)})

# # ============================
# # ANALYTICS
# # ============================
# @app.route("/api/analytics/stats", methods=["GET"])
# def analytics():
#     try:
#         df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))
#         def clean(d):
#             return {str(k): int(v) if isinstance(v, (np.integer, np.int64)) else float(v) for k, v in d.items()}

#         return jsonify({
#             "dataset": {
#                 "total_cars": int(len(df)),
#                 "year_range": f"{int(df['registered_in'].min())} - {int(df['registered_in'].max())}",
#                 "mileage_range": f"{int(df['mileage'].min())} - {int(df['mileage'].max())} km"
#             },
#             "price_stats": {
#                 "min": float(df["price"].min()),
#                 "max": float(df["price"].max()),
#                 "mean": float(df["price"].mean()),
#                 "median": float(df["price"].median()),
#                 "std": float(df["price"].std())
#             },
#             "top_brands": clean(df["car_brand"].value_counts().head(10).to_dict()),
#             "top_models": clean(df["car_model"].value_counts().head(10).to_dict()),
#             "fuel_type_distribution": clean(df["fuel_type"].value_counts().to_dict()),
#             "transmission_distribution": clean(df["transmission"].value_counts().to_dict()),
#             "city_distribution": clean(df["city"].value_counts().to_dict())
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # ============================
# # VISUALIZATIONS
# # ============================
# @app.route("/api/visualizations/<name>")
# def visualizations(name):
#     files = {
#         "actual_vs_predicted": "actual_vs_predicted.png",
#         "feature_importance": "feature_importance.png",
#         "model_comparison": "model_comparison.png",
#         "error_distribution": "error_distribution.png"
#     }

#     if name not in files:
#         return jsonify({"error": "Invalid visualization name", "available": list(files.keys())}), 404

#     path = os.path.join(VIZ_DIR, files[name])
#     if not os.path.exists(path):
#         return jsonify({"error": f"File not found: {files[name]}"}), 404

#     return send_file(path, mimetype="image/png")

# # ============================
# # CHATBOT
# # ============================
# from chatbot import chatbot as ai_chatbot

# @app.route("/api/chatbot", methods=["POST"])
# def chatbot_endpoint():
#     try:
#         data = request.json
#         session_id = data.get("session_id", "default")
#         message = data.get("message", "")
#         response = ai_chatbot.process_message(session_id=session_id, message=message, model_info=model_info)

#         if response.get('action') == 'predict' and 'data' in response:
#             car_data = response['data']
#             try:
#                 car_data['engine'] = float(str(car_data['engine']).replace(',', ''))
#                 car_data['registered_in'] = int(str(car_data['registered_in']).replace(',', ''))
#                 car_data['mileage'] = float(str(car_data['mileage']).replace(',', ''))
#             except ValueError as ve:
#                 response['reply'] = f"❌ Invalid data format: {str(ve)}"
#                 response['timestamp'] = datetime.now().isoformat()
#                 return jsonify(response)

#             prediction_result = make_prediction(car_data).json
#             if prediction_result['success']:
#                 pred = prediction_result
#                 response['reply'] = (
#                     f"🎉 Price Prediction: {pred['predicted_price']:,.0f} PKR\n"
#                     f"Price Range: {pred['price_range']['min']:,.0f} - {pred['price_range']['max']:,.0f} PKR\n"
#                     f"Segment: {pred['segment']}\n"
#                     f"City: {pred['city']['original'].title()}\n"
#                     f"Price in Lacs: {pred['price_lacs']} Lacs"
#                 )
#                 response['prediction'] = prediction_result
#             else:
#                 response['reply'] = f"❌ Prediction failed: {prediction_result.get('error', 'Unknown')}"
#         response['timestamp'] = datetime.now().isoformat()
#         return jsonify(response)
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({
#             "error": str(e),
#             "reply": "Sorry, something went wrong. Please try again.",
#             "timestamp": datetime.now().isoformat()
#         }), 500

# # ============================
# # ERROR HANDLERS
# # ============================
# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({"error": "Endpoint not found"}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({"error": "Internal server error"}), 500

# # ============================
# # START SERVER
# # ============================
# if __name__ == "__main__":
#     print("\n" + "=" * 60)
#     print("🚀 FLASK BACKEND STARTING...")
#     print("=" * 60)
#     success = load_models()
#     if not success:
#         print("⚠️ WARNING: Some models failed to load!")
#     print("\n📍 Server running at: http://localhost:5000")
#     print("📖 API Documentation: http://localhost:5000")
#     print("\n💡 Test with:")
#     print("   curl http://localhost:5000/api/health")
#     print("\n" + "=" * 60 + "\n")
#     app.run(debug=True, host="0.0.0.0", port=5000)



from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle
import pandas as pd
from dotenv import load_dotenv

import numpy as np
import os
load_dotenv()
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ============================
# PATHS (FIXED)
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')

# ============================
# GLOBAL MODELS
# ============================
model = None
label_encoders = None
feature_info = None
model_info = None
poly = None
kmeans = None
cluster_scaler = None

# ============================
# LOAD MODELS
# ============================
def load_models():
    global model, label_encoders, feature_info, model_info, poly, kmeans, cluster_scaler
    print("\n" + "=" * 60)
    print("📦 Loading ML Models...")
    print("=" * 60)
    try:
        def load(file):
            path = os.path.join(MODEL_DIR, file)
            print(f"   Loading: {file}")
            with open(path, "rb") as f:
                return pickle.load(f)

        model = load("best_model.pkl")
        label_encoders = load("label_encoders.pkl")
        feature_info = load("feature_info.pkl")
        model_info = load("model_info.pkl")
        poly = load("poly_transformer.pkl")
        kmeans = load("kmeans_model.pkl")
        cluster_scaler = load("cluster_scaler.pkl")

        print("✅ All models loaded successfully!")
        print(f"   Model: {model_info.get('model_name', 'Unknown')}")
        print(f"   Accuracy: {model_info.get('test_r2', 0):.4f}")
        print("=" * 60 + "\n")
        return True

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print(f"   MODEL_DIR: {MODEL_DIR}")
        return False

# ============================
# CITY MAPPING
# ============================
CITY_MAPPING = {
    'karachi': 'karachi', 'hyderabad': 'karachi', 'sukkur': 'karachi',
    'larkana': 'karachi', 'mirpurkhas': 'karachi',
    'lahore': 'lahore', 'faisalabad': 'lahore', 'multan': 'lahore',
    'gujranwala': 'lahore', 'sialkot': 'lahore', 'bahawalpur': 'lahore',
    'sargodha': 'lahore', 'sahiwal': 'lahore',
    'islamabad': 'islamabad', 'rawalpindi': 'islamabad', 'peshawar': 'islamabad',
    'abbottabad': 'islamabad', 'mardan': 'islamabad', 'quetta': 'islamabad',
    'khi': 'karachi', 'lhr': 'lahore', 'isb': 'islamabad', 'pindi': 'islamabad'
}

def map_city(city):
    city_lower = city.lower().strip()
    mapped = CITY_MAPPING.get(city_lower, "karachi")
    was_mapped = mapped != city_lower
    return mapped, was_mapped

# ============================
# IMPROVED SEGMENT LOGIC
# ============================
def determine_segment(predicted_price, car_data):
    """
    Improved segment determination logic
    Fixed thresholds for Pakistani market + clustering support
    """
    segment = None
    method = "fallback"
    
    # Try clustering first (if available)
    if kmeans is not None and cluster_scaler is not None:
        try:
            expected_features = cluster_scaler.n_features_in_
            
            # Prepare cluster input based on scaler expectations
            if expected_features == 3:
                cluster_input = np.array([[
                    predicted_price, 
                    car_data['registered_in'], 
                    car_data['mileage']
                ]])
            elif expected_features == 4:
                cluster_input = np.array([[
                    predicted_price, 
                    car_data['registered_in'], 
                    car_data['mileage'], 
                    car_data['engine']
                ]])
            else:
                cluster_input = np.array([[
                    predicted_price, 
                    car_data['registered_in'], 
                    car_data['mileage']
                ]])

            cluster_scaled = cluster_scaler.transform(cluster_input)
            cluster_id = int(kmeans.predict(cluster_scaled)[0])
            
            # Get cluster mapping (with better defaults)
            cluster_map = feature_info.get('cluster_map', {
                0: 'Economy', 
                1: 'Mid-Range', 
                2: 'Premium',
                3: 'Luxury'
            })
            
            segment = cluster_map.get(cluster_id)
            method = "clustering"
            
            # Validate clustering result against price
            # If clustering gives unrealistic result, use fallback
            if segment == "Economy" and predicted_price > 3500000:
                print(f"⚠️ Clustering anomaly: {segment} for price {predicted_price:,.0f}")
                segment = None  # Force fallback
            elif segment == "Luxury" and predicted_price < 5000000:
                print(f"⚠️ Clustering anomaly: {segment} for price {predicted_price:,.0f}")
                segment = None  # Force fallback
                
        except Exception as cluster_error:
            print(f"⚠️ Clustering error: {cluster_error}")
            segment = None

    # Fallback to rule-based (FIXED THRESHOLDS for Pakistani market)
    if segment is None:
        method = "rule-based"
        if predicted_price < 2000000:  # Less than 20 Lacs
            segment = "Economy"
        elif predicted_price < 4000000:  # 20-40 Lacs
            segment = "Mid-Range"
        elif predicted_price < 7000000:  # 40-70 Lacs
            segment = "Premium"
        else:  # Above 70 Lacs
            segment = "Luxury"
    
    return segment, method

# ============================
# VALIDATION HELPERS
# ============================
def validate_prediction(predicted_price, input_data):
    """Sanity checks and warnings for predictions"""
    warnings = []
    
    # Calculate car age
    current_year = datetime.now().year
    car_age = current_year - input_data['registered_in']
    
    # Warning: Very high price for old car
    if predicted_price > 10000000 and car_age > 10:
        warnings.append("⚠️ Unusually high price for vehicle age")
    
    # Warning: Very low price for new car
    if predicted_price < 1000000 and car_age < 3:
        warnings.append("⚠️ Price seems low for recent model")
    
    # Warning: High mileage impact
    if input_data['mileage'] > 150000:
        warnings.append("⚠️ High mileage may affect actual resale value")
    
    # Warning: Very low mileage (suspiciously low)
    if input_data['mileage'] < 10000 and car_age > 5:
        warnings.append("⚠️ Unusually low mileage for vehicle age")
    
    return warnings

def format_price_display(price):
    """Format price in multiple representations"""
    return {
        "pkr": int(price),
        "lacs": round(price / 100000, 2),
        "crores": round(price / 10000000, 2) if price >= 10000000 else None,
        "formatted": f"{int(price):,} PKR"
    }

# ============================
# HOME
# ============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🚗 AI Car Price Predictor API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "model_info": "/api/model-info",
            "feature_options": "/api/feature-options",
            "predict": "/api/predict (POST)",
            "batch_predict": "/api/predict/batch (POST)",
            "analytics": "/api/analytics/stats",
            "visualizations": "/api/visualizations/<type>",
            "chatbot": "/api/chatbot (POST)"
        },
        "documentation": "http://localhost:5000/docs"
    })

# ============================
# HEALTH
# ============================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "poly_loaded": poly is not None,
        "kmeans_loaded": kmeans is not None,
        "timestamp": datetime.now().isoformat()
    })

# ============================
# MODEL INFO
# ============================
@app.route("/api/model-info", methods=["GET"])
def model_details():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_name": model_info.get("model_name"),
        "accuracy_r2": float(model_info.get("test_r2", 0)),
        "mae": float(model_info.get("test_mae", 0)),
        "rmse": float(model_info.get("test_rmse", 0)),
        "mape": float(model_info.get("test_mape", 0)) if "test_mape" in model_info else None,
        "features": {
            "categorical": feature_info["categorical_features"],
            "numerical": feature_info["numerical_features"],
            "total_features": len(model_info.get("feature_columns", []))
        },
        "polynomial_features": poly is not None,
        "clustering": kmeans is not None
    })

# ============================
# FEATURE OPTIONS
# ============================
@app.route("/api/feature-options", methods=["GET"])
def feature_options():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))
        options = {}
        for col in feature_info["categorical_features"]:
            options[col] = sorted(df[col].astype(str).unique().tolist())
        for col in feature_info["numerical_features"]:
            options[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median())
            }
        return jsonify({
            "options": options,
            "total_records": int(len(df)),
            "cities_supported": list(set(CITY_MAPPING.values())),
            "all_city_names": list(CITY_MAPPING.keys())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================
# PREDICTION (IMPROVED)
# ============================
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        required_fields = feature_info["categorical_features"] + feature_info["numerical_features"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        return make_prediction(data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ============================
# BATCH PREDICTION (NEW)
# ============================
@app.route("/api/predict/batch", methods=["POST"])
def batch_predict():
    """Predict prices for multiple cars at once"""
    try:
        data = request.json
        cars = data.get("cars", [])
        
        if not cars or not isinstance(cars, list):
            return jsonify({"error": "Expected 'cars' array in request"}), 400
        
        results = []
        for idx, car_data in enumerate(cars):
            try:
                prediction = make_prediction(car_data).json
                prediction['index'] = idx
                results.append(prediction)
            except Exception as e:
                results.append({
                    "index": idx,
                    "success": False,
                    "error": str(e)
                })
        
        # Summary statistics
        successful = [r for r in results if r.get('success')]
        if successful:
            prices = [r['predicted_price'] for r in successful]
            summary = {
                "total_predictions": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "price_range": {
                    "min": min(prices),
                    "max": max(prices),
                    "average": sum(prices) / len(prices)
                }
            }
        else:
            summary = {
                "total_predictions": len(results),
                "successful": 0,
                "failed": len(results)
            }
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ============================
# HELPER FUNCTION (IMPROVED)
# ============================
def make_prediction(data):
    try:
        original_city = data.get("city", "karachi")
        mapped_city, was_mapped = map_city(original_city)
        data["city"] = mapped_city

        feature_vector = []
        for col in feature_info["categorical_features"]:
            val = str(data[col]).lower().strip()
            try:
                encoded_val = label_encoders[col].transform([val])[0]
                feature_vector.append(encoded_val)
            except:
                feature_vector.append(0)

        numerical_values = [float(data[col]) for col in feature_info["numerical_features"]]

        if poly is not None:
            numerical_array = np.array([numerical_values])
            poly_features = poly.transform(numerical_array)
            poly_feature_names = poly.get_feature_names_out(feature_info["numerical_features"])
            all_features = feature_vector + poly_features[0].tolist()
            feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
            feature_columns.extend(poly_feature_names)
        else:
            all_features = feature_vector + numerical_values
            feature_columns = [f"{col}_encoded" for col in feature_info["categorical_features"]]
            feature_columns.extend(feature_info["numerical_features"])

        df_input = pd.DataFrame([all_features], columns=feature_columns)

        if 'feature_columns' in model_info:
            training_columns = model_info['feature_columns']
            for col in training_columns:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[training_columns]

        predicted_price = float(model.predict(df_input)[0])
        price_min = predicted_price * 0.90
        price_max = predicted_price * 1.10

        # FIXED SEGMENT DETERMINATION
        segment, segment_method = determine_segment(predicted_price, data)
        
        # Validation warnings
        warnings = validate_prediction(predicted_price, data)
        
        # Calculate car age
        car_age = datetime.now().year - data['registered_in']
        
        # Confidence score (based on model accuracy and warnings)
        confidence = "high" if model_info.get("test_r2", 0) > 0.90 and len(warnings) == 0 else "medium"
        if len(warnings) > 2:
            confidence = "low"

        return jsonify({
            "success": True,
            "predicted_price": predicted_price,
            "price_display": format_price_display(predicted_price),
            "price_range": {
                "min": price_min,
                "max": price_max,
                "min_display": format_price_display(price_min),
                "max_display": format_price_display(price_max)
            },
            "segment": segment,
            "segment_method": segment_method,
            "car_info": {
                "age": car_age,
                "age_category": "new" if car_age <= 3 else "used" if car_age <= 7 else "old",
                "original_city": original_city,
                "mapped_city": mapped_city,
                "city_mapped": was_mapped
            },
            "model_performance": {
                "accuracy_r2": float(model_info.get("test_r2", 0)),
                "average_error_pkr": float(model_info.get("test_mae", 0)),
                "confidence": confidence
            },
            "warnings": warnings,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

# ============================
# ANALYTICS (ENHANCED)
# ============================
@app.route("/api/analytics/stats", methods=["GET"])
def analytics():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "pakwheels_cleaned.csv"))
        def clean(d):
            return {str(k): int(v) if isinstance(v, (np.integer, np.int64)) else float(v) for k, v in d.items()}

        # Segment distribution
        segment_stats = []
        for threshold, name in [(2000000, "Economy"), (4000000, "Mid-Range"), (7000000, "Premium")]:
            count = len(df[df['price'] < threshold])
            segment_stats.append({
                "segment": name,
                "count": count,
                "percentage": round(count / len(df) * 100, 2)
            })
        luxury_count = len(df[df['price'] >= 7000000])
        segment_stats.append({
            "segment": "Luxury",
            "count": luxury_count,
            "percentage": round(luxury_count / len(df) * 100, 2)
        })

        return jsonify({
            "dataset": {
                "total_cars": int(len(df)),
                "year_range": f"{int(df['registered_in'].min())} - {int(df['registered_in'].max())}",
                "mileage_range": f"{int(df['mileage'].min())} - {int(df['mileage'].max())} km"
            },
            "price_stats": {
                "min": float(df["price"].min()),
                "max": float(df["price"].max()),
                "mean": float(df["price"].mean()),
                "median": float(df["price"].median()),
                "std": float(df["price"].std())
            },
            "segment_distribution": segment_stats,
            "top_brands": clean(df["car_brand"].value_counts().head(10).to_dict()),
            "top_models": clean(df["car_model"].value_counts().head(10).to_dict()),
            "fuel_type_distribution": clean(df["fuel_type"].value_counts().to_dict()),
            "transmission_distribution": clean(df["transmission"].value_counts().to_dict()),
            "city_distribution": clean(df["city"].value_counts().to_dict())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================
# VISUALIZATIONS
# ============================
@app.route("/api/visualizations/<name>")
def visualizations(name):
    files = {
        "actual_vs_predicted": "actual_vs_predicted.png",
        "feature_importance": "feature_importance.png",
        "model_comparison": "model_comparison.png",
        "error_distribution": "error_distribution.png"
    }

    if name not in files:
        return jsonify({"error": "Invalid visualization name", "available": list(files.keys())}), 404

    path = os.path.join(VIZ_DIR, files[name])
    if not os.path.exists(path):
        return jsonify({"error": f"File not found: {files[name]}"}), 404

    return send_file(path, mimetype="image/png")

# ============================
# CHATBOT (IMPROVED)
# ============================
from chatbot import chatbot as ai_chatbot

@app.route("/api/chatbot", methods=["POST"])
def chatbot_endpoint():
    try:
        data = request.json
        session_id = data.get("session_id", "default")
        message = data.get("message", "")
        response = ai_chatbot.process_message(session_id=session_id, message=message, model_info=model_info)

        if response.get('action') == 'predict' and 'data' in response:
            car_data = response['data']
            try:
                car_data['engine'] = float(str(car_data['engine']).replace(',', ''))
                car_data['registered_in'] = int(str(car_data['registered_in']).replace(',', ''))
                car_data['mileage'] = float(str(car_data['mileage']).replace(',', ''))
            except ValueError as ve:
                response['reply'] = f"❌ Invalid data format: {str(ve)}"
                response['timestamp'] = datetime.now().isoformat()
                return jsonify(response)

            prediction_result = make_prediction(car_data).json
            if prediction_result['success']:
                pred = prediction_result
                warning_text = "\n⚠️ " + "\n⚠️ ".join(pred.get('warnings', [])) if pred.get('warnings') else ""
                
                response['reply'] = (
                    f"🎉 **Price Prediction**\n"
                    f"💰 Estimated Price: **{pred['price_display']['formatted']}** ({pred['price_display']['lacs']} Lacs)\n"
                    f"📊 Price Range: {pred['price_range']['min_display']['formatted']} - {pred['price_range']['max_display']['formatted']}\n"
                    f"🏷️ Segment: **{pred['segment']}** (detected via {pred['segment_method']})\n"
                    f"📍 City: {pred['car_info']['original_city'].title()}\n"
                    f"🚗 Car Age: {pred['car_info']['age']} years ({pred['car_info']['age_category']})\n"
                    f"✅ Confidence: {pred['model_performance']['confidence'].upper()}"
                    f"{warning_text}"
                )
                response['prediction'] = prediction_result
            else:
                response['reply'] = f"❌ Prediction failed: {prediction_result.get('error', 'Unknown')}"
        response['timestamp'] = datetime.now().isoformat()
        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "reply": "Sorry, something went wrong. Please try again.",
            "timestamp": datetime.now().isoformat()
        }), 500

# ============================
# ERROR HANDLERS
# ============================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ============================
# START SERVER
# ============================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 FLASK BACKEND STARTING...")
    print("=" * 60)
    success = load_models()
    if not success:
        print("⚠️ WARNING: Some models failed to load!")
    print("\n📍 Server running at: http://localhost:5000")
    print("📖 API Documentation: http://localhost:5000")
    print("\n💡 Test with:")
    print("   curl http://localhost:5000/api/health")
    print("\n" + "=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)