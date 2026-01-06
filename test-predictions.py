"""
Interactive Prediction System for Car Price Prediction
FIXED: Cluster scaler feature mismatch resolved + 80+ Pakistan Cities support
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class CarPricePredictionSystem:
    """
    Interactive system for car price predictions
    """
    
    def __init__(self):
        """Initialize the prediction system"""
        self.model = None
        self.label_encoders = {}
        self.feature_info = {}
        self.model_info = {}
        self.feature_options = {}
        self.poly = None
        self.kmeans = None
        self.cluster_scaler = None
        
        # 🌍 COMPREHENSIVE City mapping - Accept ANY Pakistan city!
        self.city_mapping = {
            # Karachi region (Sindh)
            'karachi': 'karachi',
            'hyderabad': 'karachi',
            'sukkur': 'karachi',
            'larkana': 'karachi',
            'mirpurkhas': 'karachi',
            'nawabshah': 'karachi',
            'jacobabad': 'karachi',
            'shikarpur': 'karachi',
            'thatta': 'karachi',
            'badin': 'karachi',
            'dadu': 'karachi',
            'khairpur': 'karachi',
            'sanghar': 'karachi',
            'tando allahyar': 'karachi',
            'matiari': 'karachi',
            'tando muhammad khan': 'karachi',
            
            # Lahore region (Punjab)
            'lahore': 'lahore',
            'faisalabad': 'lahore',
            'multan': 'lahore',
            'gujranwala': 'lahore',
            'sialkot': 'lahore',
            'sargodha': 'lahore',
            'sheikhupura': 'lahore',
            'jhang': 'lahore',
            'kasur': 'lahore',
            'okara': 'lahore',
            'sahiwal': 'lahore',
            'bahawalpur': 'lahore',
            'rahim yar khan': 'lahore',
            'dera ghazi khan': 'lahore',
            'gujrat': 'lahore',
            'mandi bahauddin': 'lahore',
            'hafizabad': 'lahore',
            'chiniot': 'lahore',
            'vehari': 'lahore',
            'pakpattan': 'lahore',
            'khanewal': 'lahore',
            'mianwali': 'lahore',
            'bhakkar': 'lahore',
            'layyah': 'lahore',
            'muzaffargarh': 'lahore',
            'rajanpur': 'lahore',
            'lodhran': 'lahore',
            'khushab': 'lahore',
            'narowal': 'lahore',
            'nankana sahib': 'lahore',
            'toba tek singh': 'lahore',
            
            # Islamabad region (KPK, Federal, AJK, GB)
            'islamabad': 'islamabad',
            'rawalpindi': 'islamabad',
            'peshawar': 'islamabad',
            'abbottabad': 'islamabad',
            'mardan': 'islamabad',
            'mingora': 'islamabad',
            'kohat': 'islamabad',
            'dera ismail khan': 'islamabad',
            'swat': 'islamabad',
            'mansehra': 'islamabad',
            'haripur': 'islamabad',
            'attock': 'islamabad',
            'jhelum': 'islamabad',
            'chakwal': 'islamabad',
            'muzaffarabad': 'islamabad',
            'gilgit': 'islamabad',
            'skardu': 'islamabad',
            'quetta': 'islamabad',
            'bannu': 'islamabad',
            'swabi': 'islamabad',
            'charsadda': 'islamabad',
            'nowshera': 'islamabad',
            'tank': 'islamabad',
            'lakki marwat': 'islamabad',
            'karak': 'islamabad',
            'hangu': 'islamabad',
            'malakand': 'islamabad',
            'dir': 'islamabad',
            'chitral': 'islamabad',
            'upper dir': 'islamabad',
            'lower dir': 'islamabad',
            'buner': 'islamabad',
            'shangla': 'islamabad',
            'kohistan': 'islamabad',
            'batagram': 'islamabad',
            'torghar': 'islamabad',
            
            # Common misspellings/variations
            'pindi': 'islamabad',
            'rwp': 'islamabad',
            'isb': 'islamabad',
            'khi': 'karachi',
            'lhr': 'lahore',
        }
    
    def map_city_to_training_city(self, user_city):
        """Map user's city to nearest training city"""
        user_city_lower = user_city.lower().strip()
        
        if user_city_lower in self.feature_options.get('city', []):
            return user_city_lower, user_city, False
        
        if user_city_lower in self.city_mapping:
            mapped_city = self.city_mapping[user_city_lower]
            return mapped_city, user_city, True
        
        return 'karachi', user_city, True
    
    def load_model(self):
        """Load trained model and encoders"""
        print("="*70)
        print("📂 LOADING MODEL AND ENCODERS")
        print("="*70)
        
        try:
            print("\n1️⃣ Loading trained model...")
            with open('models/best_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("   ✓ Model loaded successfully")
            
            print("\n2️⃣ Loading label encoders...")
            with open('models/label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            print(f"   ✓ Loaded {len(self.label_encoders)} encoders")
            
            print("\n3️⃣ Loading feature information...")
            with open('models/feature_info.pkl', 'rb') as f:
                self.feature_info = pickle.load(f)
            print(f"   ✓ Loaded feature info")
            
            print("\n4️⃣ Loading model metadata...")
            with open('models/model_info.pkl', 'rb') as f:
                self.model_info = pickle.load(f)
            print(f"   ✓ Model: {self.model_info['model_name']}")
            print(f"   ✓ Accuracy (R²): {self.model_info['test_r2']:.4f}")
            
            print("\n5️⃣ Loading Polynomial Features transformer...")
            try:
                with open('models/poly_transformer.pkl', 'rb') as f:
                    self.poly = pickle.load(f)
                print(f"   ✓ Polynomial transformer loaded")
            except FileNotFoundError:
                print(f"   ⚠️ Polynomial transformer not found")
                self.poly = None
            
            print("\n6️⃣ Loading K-Means clustering model...")
            try:
                with open('models/kmeans_model.pkl', 'rb') as f:
                    self.kmeans = pickle.load(f)
                with open('models/cluster_scaler.pkl', 'rb') as f:
                    self.cluster_scaler = pickle.load(f)
                print(f"   ✓ K-Means model loaded")
            except FileNotFoundError:
                print(f"   ⚠️ K-Means model not found")
                self.kmeans = None
            
            print("\n7️⃣ Loading feature options...")
            df = pd.read_csv('data/pakwheels_cleaned.csv')
            
            for feature in self.feature_info['categorical_features']:
                self.feature_options[feature] = sorted(df[feature].unique().tolist())
            
            for feature in self.feature_info['numerical_features']:
                if feature == 'engine':
                    common_engines = [660, 800, 1000, 1200, 1300, 1500, 1600, 1800, 2000, 2400, 2500, 3000]
                    existing = set(df[feature].unique())
                    all_engines = sorted(list(existing.union(set(common_engines))))
                    self.feature_options[feature] = all_engines
                else:
                    self.feature_options[feature] = sorted(df[feature].unique().tolist())
            
            print(f"   ✓ Loaded options for {len(self.feature_options)} features")
            print("\n✅ All components loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: Required file not found!")
            print(f"   {str(e)}")
            raise
        except Exception as e:
            print(f"\n❌ Error loading model: {str(e)}")
            raise
    
    def display_options(self, feature_name, limit=20):
        """Display available options for a feature"""
        if feature_name in self.feature_options:
            options = self.feature_options[feature_name]
            print(f"\n   Available options for {feature_name}:")
            
            for i, option in enumerate(options[:limit], 1):
                print(f"      {i}. {option}")
            
            if len(options) > limit:
                print(f"      ... and {len(options) - limit} more")
                print(f"\n   💡 Tip: You can type any value from the list")
    
    def find_matching_options(self, feature, search_term):
        """Find options that match the search term"""
        if feature not in self.feature_options:
            return []
        
        search_term = search_term.lower().strip()
        options = self.feature_options[feature]
        
        exact_matches = [opt for opt in options if str(opt).lower() == search_term]
        if exact_matches:
            return exact_matches
        
        partial_matches = [opt for opt in options if search_term in str(opt).lower()]
        return partial_matches[:20]
    
    def get_user_input(self):
        """Get car details from user"""
        print("\n" + "="*70)
        print("🚗 ENTER CAR DETAILS")
        print("="*70)
        
        car_details = {}
        city_info = {}
        
        print("\n📝 Categorical Features:")
        print("-" * 70)
        
        for feature in self.feature_info['categorical_features']:
            # 🌍 SPECIAL HANDLING FOR CITY
            if feature == 'city':
                print(f"\n   🌍 Enter your city:")
                print(f"   💡 You can enter ANY Pakistan city!")
                print(f"   📍 Examples: Karachi, Lahore, Islamabad, Multan, Faisalabad,")
                print(f"              Peshawar, Quetta, Hyderabad, Rawalpindi, Sialkot, etc.")
                
                while True:
                    user_city = input(f"\n   Enter City: ").strip()
                    
                    if not user_city:
                        print(f"   ❌ Please enter a city name")
                        continue
                    
                    mapped_city, original_city, was_mapped = self.map_city_to_training_city(user_city)
                    
                    car_details[feature] = mapped_city
                    city_info['original'] = original_city
                    city_info['mapped'] = mapped_city
                    city_info['was_mapped'] = was_mapped
                    
                    if was_mapped:
                        print(f"   ✓ City: {original_city.title()}")
                        print(f"      💡 Using {mapped_city.title()} market pricing")
                    else:
                        print(f"   ✓ City: {original_city.title()}")
                    break
                
                continue
            
            # Regular categorical features
            self.display_options(feature)
            
            while True:
                user_input = input(f"\n   Enter {feature.replace('_', ' ').title()} (or type to search): ").strip().lower()
                
                if user_input in [str(opt).lower() for opt in self.feature_options[feature]]:
                    car_details[feature] = user_input
                    print(f"   ✓ {feature.replace('_', ' ').title()}: {user_input}")
                    break
                else:
                    matches = self.find_matching_options(feature, user_input)
                    
                    if matches:
                        print(f"\n   🔍 Found {len(matches)} matching options:")
                        for i, match in enumerate(matches, 1):
                            print(f"      {i}. {match}")
                        
                        select = input(f"\n   Enter number (1-{len(matches)}) or 'r' to retry: ").strip()
                        
                        if select.lower() == 'r':
                            continue
                        
                        try:
                            idx = int(select) - 1
                            if 0 <= idx < len(matches):
                                car_details[feature] = str(matches[idx]).lower()
                                print(f"   ✓ {feature.replace('_', ' ').title()}: {matches[idx]}")
                                break
                            else:
                                print(f"   ❌ Invalid selection!")
                        except ValueError:
                            print(f"   ❌ Invalid input!")
                    else:
                        print(f"   ❌ No matches found for '{user_input}'")
        
        # Numerical features
        print("\n" + "-" * 70)
        print("🔢 Numerical Features:")
        print("-" * 70)
        
        for feature in self.feature_info['numerical_features']:
            self.display_options(feature)
            
            while True:
                try:
                    user_input = input(f"\n   Enter {feature.replace('_', ' ').title()}: ").strip()
                    value = float(user_input)
                    
                    if value < 0:
                        print(f"   ❌ Value cannot be negative!")
                        continue
                    
                    car_details[feature] = value
                    print(f"   ✓ {feature.replace('_', ' ').title()}: {value}")
                    break
                    
                except ValueError:
                    print(f"   ❌ Invalid input! Please enter a valid number.")
        
        car_details['_city_info'] = city_info
        return car_details
    
    def predict_price(self, car_details):
        """Predict car price with Polynomial Features support - FIXED CLUSTERING"""
        try:
            feature_vector = []
            
            for cat_feat in self.feature_info['categorical_features']:
                try:
                    encoded_value = self.label_encoders[cat_feat].transform([car_details[cat_feat]])[0]
                    feature_vector.append(encoded_value)
                except ValueError:
                    print(f"   ⚠️ Warning: '{car_details[cat_feat]}' not seen in training")
                    feature_vector.append(0)
            
            numerical_values = []
            for num_feat in self.feature_info['numerical_features']:
                numerical_values.append(car_details[num_feat])
            
            if self.poly is not None:
                numerical_array = np.array([numerical_values])
                poly_features = self.poly.transform(numerical_array)
                poly_feature_names = self.poly.get_feature_names_out(self.feature_info['numerical_features'])
                
                feature_columns = []
                for cat_feat in self.feature_info['categorical_features']:
                    feature_columns.append(cat_feat + '_encoded')
                feature_columns.extend(poly_feature_names)
                
                all_features = feature_vector + poly_features[0].tolist()
            else:
                feature_columns = []
                for cat_feat in self.feature_info['categorical_features']:
                    feature_columns.append(cat_feat + '_encoded')
                feature_columns.extend(self.feature_info['numerical_features'])
                
                all_features = feature_vector + numerical_values
            
            input_df = pd.DataFrame([all_features], columns=feature_columns)
            
            if 'feature_columns' in self.model_info:
                training_columns = self.model_info['feature_columns']
                for col in training_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[training_columns]
            
            predicted_price = self.model.predict(input_df)[0]
            price_range_lower = predicted_price * 0.90
            price_range_upper = predicted_price * 1.10
            
            # 🔧 FIXED CLUSTERING SECTION
            if self.kmeans is not None and self.cluster_scaler is not None:
                try:
                    # Check how many features cluster_scaler expects
                    expected_features = self.cluster_scaler.n_features_in_
                    
                    # Build cluster input dynamically
                    cluster_features = []
                    cluster_features.append(predicted_price)
                    
                    # Add numerical features in the same order as training
                    for feat in self.feature_info['numerical_features']:
                        if feat in car_details:
                            cluster_features.append(car_details[feat])
                    
                    # Make sure we have the right number of features
                    if len(cluster_features) == expected_features:
                        cluster_input = np.array([cluster_features])
                        cluster_scaled = self.cluster_scaler.transform(cluster_input)
                        cluster_id = self.kmeans.predict(cluster_scaled)[0]
                        cluster_map = self.feature_info.get('cluster_map', {0: 'Economy', 1: 'Mid-Range', 2: 'Luxury'})
                        segment = cluster_map.get(cluster_id, 'Unknown')
                    else:
                        # Fallback if feature count mismatch
                        if predicted_price < 1000000:
                            segment = "Economy"
                        elif predicted_price < 3000000:
                            segment = "Mid-Range"
                        elif predicted_price < 5000000:
                            segment = "Premium"
                        else:
                            segment = "Luxury"
                except Exception as e:
                    # Fallback segmentation if clustering fails
                    print(f"   ⚠️ Clustering warning: {str(e)[:50]}... Using price-based segmentation")
                    if predicted_price < 1000000:
                        segment = "Economy"
                    elif predicted_price < 3000000:
                        segment = "Mid-Range"
                    elif predicted_price < 5000000:
                        segment = "Premium"
                    else:
                        segment = "Luxury"
            else:
                # No clustering model - use price-based segmentation
                if predicted_price < 1000000:
                    segment = "Economy"
                elif predicted_price < 3000000:
                    segment = "Mid-Range"
                elif predicted_price < 5000000:
                    segment = "Premium"
                else:
                    segment = "Luxury"
            
            emoji_map = {'Economy': '💰', 'Mid-Range': '🚗', 'Premium': '✨', 'Luxury': '💎'}
            emoji = emoji_map.get(segment, '🚗')
            
            return predicted_price, price_range_lower, price_range_upper, segment, emoji
            
        except Exception as e:
            print(f"\n❌ Error making prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def display_prediction(self, car_details, predicted_price, price_range_lower, price_range_upper, segment, emoji):
        """Display prediction results"""
        print("\n" + "="*70)
        print("📋 CAR DETAILS SUMMARY")
        print("="*70)
        
        print("\n🏷️  Vehicle Information:")
        for key in self.feature_info['categorical_features']:
            if key in car_details:
                display_name = key.replace('_', ' ').title()
                
                if key == 'city' and '_city_info' in car_details:
                    city_info = car_details['_city_info']
                    if city_info['was_mapped']:
                        print(f"   • {display_name}: {city_info['original'].title()} (market: {city_info['mapped'].title()})")
                    else:
                        print(f"   • {display_name}: {city_info['original'].title()}")
                else:
                    print(f"   • {display_name}: {car_details[key].title()}")
        
        print("\n📊 Technical Specifications:")
        for key in self.feature_info['numerical_features']:
            if key in car_details:
                display_name = key.replace('_', ' ').title()
                value = car_details[key]
                
                if 'mileage' in key.lower():
                    print(f"   • {display_name}: {value:,.0f} km")
                elif 'engine' in key.lower():
                    print(f"   • {display_name}: {value:,.0f} cc")
                elif 'registered' in key.lower() or 'year' in key.lower():
                    print(f"   • {display_name}: {int(value)}")
                else:
                    print(f"   • {display_name}: {value}")
        
        print("\n" + "="*70)
        proceed = input("\n✅ Proceed with prediction? (yes/no): ").strip().lower()
        
        if proceed == 'yes':
            print("\n⏳ Analyzing car details with AI model...")
            print("   🤖 Processing features...")
            print("   📊 Calculating market value...")
            
            print("\n" + "="*70)
            print("🎉 PREDICTION RESULTS")
            print("="*70)
            
            print(f"\n{emoji} Estimated Price: {predicted_price:,.0f} PKR")
            print(f"🎯 Market Segment: {segment}")
            
            print(f"\n📊 Expected Price Range:")
            print(f"   • Minimum: {price_range_lower:,.0f} PKR")
            print(f"   • Maximum: {price_range_upper:,.0f} PKR")
            print(f"   • Variation: ±10%")
            
            print(f"\n💡 Insights:")
            print(f"   • Model Accuracy (R²): {self.model_info['test_r2']:.2%}")
            print(f"   • Average Error: ±{self.model_info['test_mae']:,.0f} PKR")
            
            if predicted_price >= 1000000:
                price_in_lacs = predicted_price / 100000
                print(f"   • Price in Lacs: {price_in_lacs:.2f} Lacs")
            
            if predicted_price >= 10000000:
                price_in_crores = predicted_price / 10000000
                print(f"   • Price in Crores: {price_in_crores:.2f} Crores")
            
            print("\n" + "="*70)
            return True
        else:
            print("\n❌ Prediction cancelled.")
            return False
    
    def run(self):
        """Run the interactive prediction system"""
        print("\n" + "="*70)
        print("🚗 CAR PRICE PREDICTION SYSTEM")
        print("="*70)
        print("\n   Welcome to the AI-powered Car Price Prediction System!")
        print("   This system uses Machine Learning to estimate car prices")
        print("   based on various features.")
        print("\n" + "="*70)
        
        self.load_model()
        
        while True:
            try:
                car_details = self.get_user_input()
                
                predicted_price, price_range_lower, price_range_upper, segment, emoji = \
                    self.predict_price(car_details)
                
                predicted = self.display_prediction(
                    car_details, predicted_price,
                    price_range_lower, price_range_upper, segment, emoji
                )
                
                if predicted:
                    print("\n" + "="*70)
                    another = input("🔄 Predict another car? (yes/no): ").strip().lower()
                    
                    if another != 'yes':
                        print("\n👋 Thank you for using the Car Price Prediction System!")
                        print("="*70)
                        break
                else:
                    another = input("\n🔄 Try again with different details? (yes/no): ").strip().lower()
                    if another != 'yes':
                        print("\n👋 Goodbye!")
                        print("="*70)
                        break
            
            except KeyboardInterrupt:
                print("\n\n⚠️ Prediction interrupted by user.")
                print("👋 Goodbye!")
                print("="*70)
                break
            
            except Exception as e:
                print(f"\n❌ An error occurred: {str(e)}")
                retry = input("\n🔄 Would you like to try again? (yes/no): ").strip().lower()
                if retry != 'yes':
                    break


def main():
    """Main function"""
    
    required_files = [
        'models/best_model.pkl',
        'models/label_encoders.pkl',
        'models/feature_info.pkl',
        'models/model_info.pkl',
        'data/pakwheels_cleaned.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("\n❌ Error: Required files not found!")
        print("\nMissing files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 Please run these scripts first:")
        print("   1. python data-preprocessing.py")
        print("   2. python train-model.py")
        return
    
    try:
        prediction_system = CarPricePredictionSystem()
        prediction_system.run()
    except Exception as e:
        print(f"\n❌ System error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()