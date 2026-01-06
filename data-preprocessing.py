"""
FIXED Data Preprocessing Script
Properly extracts engine sizes and converts prices
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import re
import warnings
warnings.filterwarnings('ignore')


class ImprovedDataPreprocessor:
    """
    Enhanced preprocessor that extracts features from car names
    """
    
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.label_encoders = {}
        self.df = None
        self.categorical_features = []
        self.numerical_features = []
        
    def load_data(self):
        """Load the raw dataset"""
        print("="*70)
        print("📂 LOADING DATASET")
        print("="*70)
        
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"✅ Dataset loaded successfully!")
            print(f"   Rows: {self.df.shape[0]:,}")
            print(f"   Columns: {self.df.shape[1]}")
            print(f"\n📋 Column names:")
            for i, col in enumerate(self.df.columns, 1):
                print(f"   {i}. {col}")
            
            # Show sample car names
            if 'Car Name' in self.df.columns:
                print(f"\n📝 Sample car names:")
                for i, name in enumerate(self.df['Car Name'].head(5), 1):
                    print(f"   {i}. {name}")
            
            return self.df
            
        except FileNotFoundError:
            print(f"❌ Error: File '{self.input_file}' not found!")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def extract_car_brand_model(self, car_name):
        """Extract brand and model from car name"""
        if pd.isna(car_name):
            return 'unknown', 'unknown'
        
        car_name = str(car_name).lower().strip()
        
        # Common car brands in Pakistan
        brands = {
            'honda': ['city', 'civic', 'accord', 'br-v', 'vezel', 'fit', 'insight'],
            'toyota': ['corolla', 'yaris', 'prius', 'aqua', 'vitz', 'premio', 'fielder', 'camry'],
            'suzuki': ['cultus', 'swift', 'wagon r', 'alto', 'mehran', 'bolan', 'liana', 'jimny', 'every'],
            'daihatsu': ['mira', 'move', 'hijet', 'cuore', 'boon', 'terios'],
            'nissan': ['dayz', 'clipper', 'note', 'juke', 'x-trail'],
            'mitsubishi': ['lancer', 'pajero', 'outlander'],
            'kia': ['sportage', 'picanto', 'sorento'],
            'hyundai': ['santro', 'tucson', 'elantra'],
            'mg': ['hs', 'zs'],
            'changan': ['alsvin', 'karvaan'],
            'proton': ['saga'],
            'prince': ['pearl'],
            'united': ['bravo', 'alpha']
        }
        
        # Extract brand
        brand = 'other'
        model = 'unknown'
        
        for b, models in brands.items():
            if b in car_name:
                brand = b
                # Extract model
                for m in models:
                    if m in car_name:
                        model = m
                        break
                break
        
        return brand, model
    
    def extract_engine_from_name(self, car_name):
        """
        🔧 NEW: Extract engine size from car name
        Examples:
        - "1.0" → 1000cc
        - "1.3" → 1300cc
        - "1.5" → 1500cc
        - "1.8L" → 1800cc
        - "2.0 L" → 2000cc
        - "660cc" → 660cc
        """
        if pd.isna(car_name):
            return None
        
        car_name = str(car_name).lower()
        
        # Pattern 1: Direct cc mention (e.g., "660cc", "1300cc")
        match = re.search(r'(\d{3,4})\s*cc', car_name)
        if match:
            return float(match.group(1))
        
        # Pattern 2: Liter format (e.g., "1.0", "1.3", "1.5L")
        match = re.search(r'(\d\.\d)\s*l?(?:\s|$)', car_name)
        if match:
            liters = float(match.group(1))
            return liters * 1000  # Convert to cc
        
        # Pattern 3: Space-separated (e.g., "1 3", "1 5")
        match = re.search(r'\b1\s+([0358])\b', car_name)
        if match:
            return float(f"1{match.group(1)}00")
        
        return None
    
    def extract_year_from_name(self, car_name):
        """Extract year from car name if present"""
        if pd.isna(car_name):
            return None
        
        # Find 4-digit year (1980-2025)
        match = re.search(r'\b(19[8-9]\d|20[0-2]\d)\b', str(car_name))
        if match:
            return int(match.group(1))
        return None
    
    def convert_price_to_numeric(self):
        """
        🔧 FIXED: Properly convert price from lacs to full amount
        """
        print("\n💰 Converting price to numeric format...")
        
        if 'price' in self.df.columns:
            def clean_price(price):
                if pd.isna(price):
                    return np.nan
                
                price_str = str(price).lower().strip()
                
                # Remove common characters
                price_str = price_str.replace(',', '').replace('pkr', '').replace('rs', '').strip()
                
                # Check for lakh/crore
                multiplier = 1
                if 'crore' in price_str:
                    multiplier = 10000000
                    price_str = price_str.replace('crore', '').strip()
                elif 'lakh' in price_str or 'lac' in price_str:
                    multiplier = 100000
                    price_str = price_str.replace('lakh', '').replace('lac', '').strip()
                
                # Extract numeric value
                price_str = re.sub(r'[^\d.]', '', price_str)
                
                try:
                    price_value = float(price_str)
                    return price_value * multiplier
                except:
                    return np.nan
            
            original_prices = self.df['price'].copy()
            self.df['price'] = self.df['price'].apply(clean_price)
            converted_count = self.df['price'].notna().sum()
            
            print(f"   ✓ Successfully converted: {converted_count:,} values")
            
            # Show conversion examples
            print(f"\n   📊 Conversion examples:")
            for i in range(min(5, len(self.df))):
                if pd.notna(original_prices.iloc[i]) and pd.notna(self.df['price'].iloc[i]):
                    print(f"      {original_prices.iloc[i]} → {self.df['price'].iloc[i]:,.0f} PKR")
    
    def convert_numerical_columns(self):
        """Convert numerical columns"""
        print("\n🔢 Converting numerical columns...")
        
        numerical_cols = ['engine', 'registered_in', 'mileage']
        
        for col in numerical_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_numeric(
                        self.df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                        errors='coerce'
                    )
                    print(f"   ✓ {col}: Converted to numeric")
                except Exception as e:
                    print(f"   ⚠️  {col}: Could not convert - {str(e)}")
    
    def clean_and_extract_features(self):
        """Clean data and extract features from car names"""
        print("\n" + "="*70)
        print("🧹 DATA CLEANING AND FEATURE EXTRACTION")
        print("="*70)
        
        # Clean column names
        print("\n1️⃣ Cleaning column names...")
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
        print("   ✓ Column names standardized")
        
        # Convert data types
        print("\n2️⃣ Converting data types...")
        self.convert_price_to_numeric()
        self.convert_numerical_columns()
        
        # Remove duplicates
        print("\n3️⃣ Removing duplicates...")
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"   ✓ Removed {initial_rows - len(self.df):,} duplicate rows")
        
        # Handle missing price
        print("\n4️⃣ Handling missing prices...")
        if 'price' in self.df.columns:
            price_missing = self.df['price'].isnull().sum()
            self.df = self.df.dropna(subset=['price'])
            self.df = self.df[self.df['price'] > 100000]  # At least 1 lakh
            print(f"   ✓ Removed {price_missing:,} rows with invalid price")
        
        # Extract brand and model from car_name
        print("\n5️⃣ Extracting brand and model from car names...")
        if 'car_name' in self.df.columns:
            self.df[['car_brand', 'car_model']] = self.df['car_name'].apply(
                lambda x: pd.Series(self.extract_car_brand_model(x))
            )
            print(f"   ✓ Extracted car_brand and car_model")
            print(f"   ✓ Unique brands: {self.df['car_brand'].nunique()}")
            print(f"   ✓ Unique models: {self.df['car_model'].nunique()}")
            
            # Show brand distribution
            print(f"\n   📊 Top 5 brands:")
            top_brands = self.df['car_brand'].value_counts().head(5)
            for brand, count in top_brands.items():
                print(f"      • {brand}: {count:,} cars")
        
        # 🔧 NEW: Extract engine from car name
        print("\n5.5️⃣ Extracting engine sizes from car names...")
        if 'car_name' in self.df.columns:
            self.df['extracted_engine'] = self.df['car_name'].apply(self.extract_engine_from_name)
            
            # If engine column exists, fill missing values with extracted
            if 'engine' in self.df.columns:
                original_engine = self.df['engine'].copy()
                self.df['engine'] = self.df['engine'].fillna(self.df['extracted_engine'])
                filled = self.df['engine'].notna().sum() - original_engine.notna().sum()
                print(f"   ✓ Filled {filled:,} missing engine values from car names")
            else:
                self.df['engine'] = self.df['extracted_engine']
                print(f"   ✓ Created engine column from car names")
            
            # Show engine distribution
            print(f"\n   🔧 Engine size distribution:")
            engine_counts = self.df['engine'].value_counts().sort_index().head(10)
            for engine, count in engine_counts.items():
                print(f"      • {engine:.0f}cc: {count:,} cars")
            
            print(f"   ✓ Total unique engine sizes: {self.df['engine'].nunique()}")
        
        # Extract model year from car name if not present
        print("\n6️⃣ Extracting/validating model year...")
        if 'registered_in' in self.df.columns:
            # If registered_in is missing, try to extract from car_name
            missing_year = self.df['registered_in'].isnull().sum()
            if missing_year > 0:
                print(f"   ⚠️  {missing_year:,} missing years, attempting extraction...")
                self.df['extracted_year'] = self.df['car_name'].apply(self.extract_year_from_name)
                self.df['registered_in'] = self.df['registered_in'].fillna(self.df['extracted_year'])
                filled = self.df['registered_in'].notnull().sum() - (len(self.df) - missing_year)
                print(f"   ✓ Filled {filled:,} years from car names")
            
            # Fill remaining with median
            median_year = self.df['registered_in'].median()
            self.df['registered_in'] = self.df['registered_in'].fillna(median_year)
        
        # Define features
        print("\n7️⃣ Defining features...")
        
        # Categorical features (using extracted features)
        possible_categorical = [
            'car_brand', 'car_model', 'city', 'fuel_type', 'transmission'
        ]
        
        # Numerical features
        possible_numerical = [
            'engine', 'registered_in', 'mileage'
        ]
        
        # Check which features exist
        for col in possible_categorical:
            if col in self.df.columns:
                self.categorical_features.append(col)
        
        for col in possible_numerical:
            if col in self.df.columns:
                self.numerical_features.append(col)
        
        print(f"\n   Categorical features ({len(self.categorical_features)}):")
        for feat in self.categorical_features:
            unique_count = self.df[feat].nunique()
            print(f"      ✓ {feat} ({unique_count} unique values)")
        
        print(f"\n   Numerical features ({len(self.numerical_features)}):")
        for feat in self.numerical_features:
            non_null = self.df[feat].notna().sum()
            print(f"      ✓ {feat} ({non_null:,} non-null values)")
        
        # Handle missing values in categorical features
        print("\n8️⃣ Handling missing categorical values...")
        for col in self.categorical_features:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                self.df[col] = self.df[col].fillna('unknown')
                print(f"   ✓ {col}: Filled {missing_count:,} missing values")
            
            self.df[col] = self.df[col].astype(str).str.lower().str.strip()
        
        # Handle missing values in numerical features
        print("\n9️⃣ Handling missing numerical values...")
        for col in self.numerical_features:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                median_value = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_value)
                print(f"   ✓ {col}: Filled {missing_count:,} missing values with median ({median_value:.0f})")
        
        # Remove price outliers
        print("\n🔟 Removing price outliers...")
        if 'price' in self.df.columns:
            initial_count = len(self.df)
            Q1 = self.df['price'].quantile(0.01)
            Q3 = self.df['price'].quantile(0.99)
            self.df = self.df[(self.df['price'] >= Q1) & (self.df['price'] <= Q3)]
            outliers_removed = initial_count - len(self.df)
            print(f"   ✓ Removed {outliers_removed:,} price outliers")
            print(f"   ✓ Price range: {Q1:,.0f} - {Q3:,.0f} PKR")
        
        print(f"\n✅ Data cleaning completed!")
        print(f"   Final dataset: {len(self.df):,} rows")
        
        return self.df
    
    def encode_features(self):
        """Encode categorical features"""
        print("\n" + "="*70)
        print("🔧 FEATURE ENCODING")
        print("="*70)
        
        print("\n📝 Encoding categorical features...")
        
        for feature in self.categorical_features:
            try:
                self.label_encoders[feature] = LabelEncoder()
                self.df[feature + '_encoded'] = self.label_encoders[feature].fit_transform(
                    self.df[feature]
                )
                unique_count = len(self.label_encoders[feature].classes_)
                print(f"   ✓ {feature}: {unique_count} unique values encoded")
                
            except Exception as e:
                print(f"   ❌ Error encoding {feature}: {str(e)}")
        
        print(f"\n✅ Feature encoding completed!")
        
        return self.df
    
    def save_data(self):
        """Save processed data"""
        print("\n" + "="*70)
        print("💾 SAVING PROCESSED DATA")
        print("="*70)
        
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            os.makedirs('models', exist_ok=True)
            
            # Save cleaned dataset
            print("\n1️⃣ Saving cleaned dataset...")
            self.df.to_csv(self.output_file, index=False)
            print(f"   ✓ Saved to: {self.output_file}")
            print(f"   ✓ Size: {os.path.getsize(self.output_file) / 1024:.2f} KB")
            
            # Save label encoders
            print("\n2️⃣ Saving label encoders...")
            with open('models/label_encoders.pkl', 'wb') as f:
                pickle.dump(self.label_encoders, f)
            print(f"   ✓ Saved to: models/label_encoders.pkl")
            
            # Save feature information
            print("\n3️⃣ Saving feature information...")
            feature_info = {
                'categorical_features': self.categorical_features,
                'numerical_features': self.numerical_features,
                'all_features': self.categorical_features + self.numerical_features
            }
            with open('models/feature_info.pkl', 'wb') as f:
                pickle.dump(feature_info, f)
            print(f"   ✓ Saved to: models/feature_info.pkl")
            
            print("\n✅ All files saved successfully!")
            
        except Exception as e:
            print(f"\n❌ Error saving files: {str(e)}")
            raise
    
    def run(self):
        """Run the complete preprocessing pipeline"""
        print("\n" + "="*70)
        print("🚀 IMPROVED DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        try:
            self.load_data()
            self.clean_and_extract_features()
            self.encode_features()
            self.save_data()
            
            # Summary
            print("\n" + "="*70)
            print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\n📊 Summary:")
            print(f"   ✓ Final rows: {len(self.df):,}")
            print(f"   ✓ Categorical features: {len(self.categorical_features)}")
            print(f"   ✓ Numerical features: {len(self.numerical_features)}")
            print(f"\n🎯 Key Improvements:")
            print(f"   ✓ Fixed price conversion (lacs → full amount)")
            print(f"   ✓ Extracted engine sizes from car names")
            print(f"   ✓ Now supports multiple engine sizes!")
            print("\n🎯 Next step: Run train-model.py to train the model!")
            print("="*70)
            
            return self.df
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function"""
    
    INPUT_FILE = 'data/pak_wheel.csv'
    OUTPUT_FILE = 'data/pakwheels_cleaned.csv'
    
    if not os.path.exists(INPUT_FILE):
        print("\n❌ Error: Input file not found!")
        print(f"   Looking for: {INPUT_FILE}")
        return
    
    preprocessor = ImprovedDataPreprocessor(INPUT_FILE, OUTPUT_FILE)
    preprocessor.run()


if __name__ == "__main__":
    main()