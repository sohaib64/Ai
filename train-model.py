"""
Complete Model Training Script with Polynomial Features
Fulfills all project requirements:
- Supervised Learning: Linear Regression
- Feature Engineering: One-Hot Encoding + Polynomial Features
- Model Selection: Compare multiple algorithms
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class CompleteModelTrainer:
    """
    Complete model trainer with Polynomial Features
    """
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.results = {}
        
    def load_data(self):
        """Load preprocessed data"""
        print("="*70)
        print("📂 LOADING DATA")
        print("="*70)
        
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Data loaded successfully!")
            print(f"   Rows: {self.df.shape[0]:,}")
            print(f"   Columns: {self.df.shape[1]}")
            
            # Load feature info
            with open('models/feature_info.pkl', 'rb') as f:
                self.feature_info = pickle.load(f)
            
            print(f"\n📋 Features:")
            print(f"   Categorical: {len(self.feature_info['categorical_features'])}")
            print(f"   Numerical: {len(self.feature_info['numerical_features'])}")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: File not found!")
            print(f"   {str(e)}")
            print("\n💡 Make sure you ran data-preprocessing.py first!")
            raise
        except Exception as e:
            print(f"\n❌ Error loading data: {str(e)}")
            raise
    
    def prepare_features_with_polynomial(self):
        """Prepare features WITH Polynomial transformation"""
        print("\n" + "="*70)
        print("🔧 PREPARING FEATURES WITH POLYNOMIAL TRANSFORMATION")
        print("="*70)
        
        # Get encoded categorical features
        cat_feature_columns = []
        for cat_feat in self.feature_info['categorical_features']:
            encoded_col = cat_feat + '_encoded'
            if encoded_col in self.df.columns:
                cat_feature_columns.append(encoded_col)
        
        # Get numerical features
        num_feature_columns = self.feature_info['numerical_features']
        
        print(f"\n📝 Original features:")
        print(f"   Categorical (encoded): {len(cat_feature_columns)}")
        print(f"   Numerical: {len(num_feature_columns)}")
        
        # Extract categorical and numerical features separately
        X_cat = self.df[cat_feature_columns].copy()
        X_num = self.df[num_feature_columns].copy()
        
        # Apply Polynomial Features to NUMERICAL features only
        print(f"\n🔄 Applying Polynomial Features (degree=2) to numerical features...")
        X_num_poly = self.poly.fit_transform(X_num)
        
        # Get polynomial feature names
        poly_feature_names = self.poly.get_feature_names_out(num_feature_columns)
        
        print(f"   ✓ Original numerical features: {len(num_feature_columns)}")
        print(f"   ✓ Polynomial features generated: {len(poly_feature_names)}")
        print(f"   ✓ Examples: {list(poly_feature_names[:5])}")
        
        # Combine categorical (encoded) + polynomial numerical features
        X_num_poly_df = pd.DataFrame(X_num_poly, columns=poly_feature_names, index=self.df.index)
        self.X = pd.concat([X_cat, X_num_poly_df], axis=1)
        self.y = self.df['price'].copy()
        
        # Store feature columns for prediction
        self.feature_columns = list(self.X.columns)
        
        print(f"\n📊 Final feature set:")
        print(f"   Total features: {self.X.shape[1]}")
        print(f"   Target (price) samples: {self.y.shape[0]}")
        
        # Split data
        print(f"\n✂️  Splitting data (80% train, 20% test)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"   ✓ Training set: {self.X_train.shape[0]:,} samples")
        print(f"   ✓ Test set: {self.X_test.shape[0]:,} samples")
        
    def train_models(self):
        """Train multiple models and compare"""
        print("\n" + "="*70)
        print("🤖 TRAINING MODELS")
        print("="*70)
        
        # Define models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
                verbose=0
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                verbose=0
            )
        }
        
        print(f"\n🎯 Training {len(models_to_train)} models with polynomial features...\n")
        
        for i, (name, model) in enumerate(models_to_train.items(), 1):
            print(f"{i}. Training {name}...")
            print("   " + "-"*60)
            
            try:
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Predictions
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
                
                # Calculate metrics
                train_r2 = r2_score(self.y_train, y_train_pred)
                test_r2 = r2_score(self.y_test, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
                test_mae = mean_absolute_error(self.y_test, y_test_pred)
                test_mape = np.mean(np.abs((self.y_test - y_test_pred) / self.y_test)) * 100
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_mape': test_mape,
                    'y_pred': y_test_pred
                }
                
                # Display results
                print(f"   📊 Training Performance:")
                print(f"      R² Score: {train_r2:.4f}")
                print(f"      RMSE: {train_rmse:,.0f} PKR")
                
                print(f"\n   📊 Testing Performance:")
                print(f"      R² Score: {test_r2:.4f}")
                print(f"      RMSE: {test_rmse:,.0f} PKR")
                print(f"      MAE: {test_mae:,.0f} PKR")
                print(f"      MAPE: {test_mape:.2f}%")
                
                # Quality rating
                if test_r2 >= 0.90:
                    quality = "⭐⭐⭐⭐⭐ Excellent"
                elif test_r2 >= 0.80:
                    quality = "⭐⭐⭐⭐ Very Good"
                elif test_r2 >= 0.70:
                    quality = "⭐⭐⭐ Good"
                else:
                    quality = "⭐⭐ Needs Improvement"
                
                print(f"   🎯 Model Quality: {quality}")
                print()
                
                self.models[name] = model
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}")
                print()
        
        # Select best model
        if self.results:
            self.best_model_name = max(self.results, key=lambda x: self.results[x]['test_r2'])
            self.best_model = self.results[self.best_model_name]['model']
            
            print("="*70)
            print(f"🏆 BEST MODEL: {self.best_model_name}")
            print("="*70)
            print(f"   Test R² Score: {self.results[self.best_model_name]['test_r2']:.4f}")
            print(f"   Test RMSE: {self.results[self.best_model_name]['test_rmse']:,.0f} PKR")
            print(f"   Test MAE: {self.results[self.best_model_name]['test_mae']:,.0f} PKR")
            print(f"   Test MAPE: {self.results[self.best_model_name]['test_mape']:.2f}%")
        
        return self.results
    
    def save_model(self):
        """Save trained model and transformers"""
        print("\n" + "="*70)
        print("💾 SAVING MODEL AND TRANSFORMERS")
        print("="*70)
        
        try:
            os.makedirs('models', exist_ok=True)
            
            # Save best model
            print("\n1️⃣ Saving best model...")
            with open('models/best_model.pkl', 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"   ✓ Saved: {self.best_model_name}")
            
            # Save Polynomial transformer (IMPORTANT!)
            print("\n2️⃣ Saving Polynomial Features transformer...")
            with open('models/poly_transformer.pkl', 'wb') as f:
                pickle.dump(self.poly, f)
            print(f"   ✓ Saved polynomial transformer (degree=2)")
            
            # Save model info
            print("\n3️⃣ Saving model metadata...")
            model_info = {
                'model_name': self.best_model_name,
                'feature_columns': self.feature_columns,
                'test_r2': self.results[self.best_model_name]['test_r2'],
                'test_rmse': self.results[self.best_model_name]['test_rmse'],
                'test_mae': self.results[self.best_model_name]['test_mae'],
                'test_mape': self.results[self.best_model_name]['test_mape'],
                'uses_polynomial': True,
                'polynomial_degree': 2
            }
            with open('models/model_info.pkl', 'wb') as f:
                pickle.dump(model_info, f)
            print(f"   ✓ Saved model metadata")
            
            # Save comparison results
            print("\n4️⃣ Saving model comparison...")
            comparison_results = {
                name: {
                    'train_r2': res['train_r2'],
                    'test_r2': res['test_r2'],
                    'test_rmse': res['test_rmse'],
                    'test_mae': res['test_mae'],
                    'test_mape': res['test_mape']
                }
                for name, res in self.results.items()
            }
            with open('models/model_comparison.pkl', 'wb') as f:
                pickle.dump(comparison_results, f)
            print(f"   ✓ Saved comparison results")
            
            print("\n✅ All models and transformers saved!")
            
        except Exception as e:
            print(f"\n❌ Error saving models: {str(e)}")
            raise
    
    def create_visualizations(self):
        """Create visualizations"""
        print("\n" + "="*70)
        print("📊 CREATING VISUALIZATIONS")
        print("="*70)
        
        try:
            os.makedirs('visualizations', exist_ok=True)
            
            y_pred = self.results[self.best_model_name]['y_pred']
            
            # 1. Actual vs Predicted
            print("\n1️⃣ Creating Actual vs Predicted plot...")
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.scatter(self.y_test, y_pred, alpha=0.5, s=20)
            plt.plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction')
            plt.xlabel('Actual Price (PKR)', fontsize=12)
            plt.ylabel('Predicted Price (PKR)', fontsize=12)
            plt.title(f'Actual vs Predicted\n{self.best_model_name} with Polynomial Features', 
                     fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 2. Residual Plot
            plt.subplot(1, 2, 2)
            residuals = self.y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.5, s=20)
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            plt.xlabel('Predicted Price (PKR)', fontsize=12)
            plt.ylabel('Residuals (PKR)', fontsize=12)
            plt.title('Residual Plot', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('visualizations/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ Saved: visualizations/actual_vs_predicted.png")
            
            # 3. Feature Importance (if available)
            if hasattr(self.best_model, 'feature_importances_'):
                print("\n2️⃣ Creating Feature Importance plot...")
                
                feature_importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(12, 8))
                top_n = min(15, len(feature_importance))
                
                sns.barplot(
                    data=feature_importance.head(top_n), 
                    x='importance', 
                    y='feature',
                    palette='viridis'
                )
                plt.xlabel('Importance Score', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.title(f'Top {top_n} Feature Importance\n{self.best_model_name}', 
                         fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("   ✓ Saved: visualizations/feature_importance.png")
            
            # 4. Model Comparison
            print("\n3️⃣ Creating Model Comparison plot...")
            
            comparison_df = pd.DataFrame({
                'Model': list(self.results.keys()),
                'R² Score': [res['test_r2'] for res in self.results.values()],
                'RMSE': [res['test_rmse'] for res in self.results.values()]
            })
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # R² Score
            sns.barplot(data=comparison_df, x='Model', y='R² Score', ax=axes[0], palette='Set2')
            axes[0].set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
            axes[0].set_ylim(0, 1)
            axes[0].grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(comparison_df['R² Score']):
                axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
            
            # RMSE
            sns.barplot(data=comparison_df, x='Model', y='RMSE', ax=axes[1], palette='Set2')
            axes[1].set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('RMSE (PKR)')
            axes[1].grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(comparison_df['RMSE']):
                axes[1].text(i, v + 10000, f'{v:,.0f}', ha='center', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ Saved: visualizations/model_comparison.png")
            
            print("\n✅ All visualizations created!")
            
        except Exception as e:
            print(f"\n❌ Error creating visualizations: {str(e)}")
            print("   Continuing anyway...")
    
    def run(self):
        """Run complete training pipeline"""
        print("\n" + "="*70)
        print("🚀 COMPLETE MODEL TRAINING WITH POLYNOMIAL FEATURES")
        print("="*70)
        
        try:
            self.load_data()
            self.prepare_features_with_polynomial()
            self.train_models()
            self.save_model()
            self.create_visualizations()
            
            # Final summary
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*70)
            
            print(f"\n🏆 Best Model: {self.best_model_name}")
            print(f"\n📊 Performance:")
            print(f"   ✓ R² Score: {self.results[self.best_model_name]['test_r2']:.4f}")
            print(f"   ✓ RMSE: {self.results[self.best_model_name]['test_rmse']:,.0f} PKR")
            print(f"   ✓ MAE: {self.results[self.best_model_name]['test_mae']:,.0f} PKR")
            print(f"   ✓ MAPE: {self.results[self.best_model_name]['test_mape']:.2f}%")
            
            print(f"\n🔧 Feature Engineering:")
            print(f"   ✓ Polynomial Features: Yes (degree=2)")
            print(f"   ✓ Total Features: {len(self.feature_columns)}")
            
            print(f"\n📁 Generated Files:")
            print(f"   ✓ models/best_model.pkl")
            print(f"   ✓ models/poly_transformer.pkl  ← IMPORTANT!")
            print(f"   ✓ models/model_info.pkl")
            print(f"   ✓ visualizations/*.png")
            
            print("\n🎯 Next: Run add-clustering.py for unsupervised learning!")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function"""
    
    DATA_FILE = 'data/pakwheels_cleaned.csv'
    
    if not os.path.exists(DATA_FILE):
        print("\n❌ Error: Cleaned data not found!")
        print(f"   Looking for: {DATA_FILE}")
        print("\n💡 Run data-preprocessing.py first!")
        return
    
    if not os.path.exists('models/feature_info.pkl'):
        print("\n❌ Error: Feature info not found!")
        print("   Looking for: models/feature_info.pkl")
        print("\n💡 Run data-preprocessing.py first!")
        return
    
    trainer = CompleteModelTrainer(DATA_FILE)
    trainer.run()


if __name__ == "__main__":
    main()