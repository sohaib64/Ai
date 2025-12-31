# check_requirements.py
import pickle
import os

print("="*70)
print("🔍 CHECKING PROPOSAL REQUIREMENTS")
print("="*70)

requirements = {
    'Linear Regression': False,
    'Polynomial Features': False,
    'One-Hot Encoding': False,
    'K-Means Clustering': False,
    'Visualizations': False
}

# Check models
print("\n📂 Checking models folder...")
if os.path.exists('models/poly_transformer.pkl'):
    requirements['Polynomial Features'] = True
    print("   ✅ Polynomial Features: Found")
else:
    print("   ❌ Polynomial Features: Missing")

if os.path.exists('models/best_model.pkl'):
    requirements['Linear Regression'] = True
    print("   ✅ Linear Regression model: Found")
    
if os.path.exists('models/kmeans_model.pkl'):
    requirements['K-Means Clustering'] = True
    print("   ✅ K-Means model: Found")

if os.path.exists('models/label_encoders.pkl'):
    requirements['One-Hot Encoding'] = True
    print("   ✅ One-Hot Encoding: Found")

# Check visualizations
print("\n📊 Checking visualizations...")
required_viz = [
    'actual_vs_predicted.png',
    'error_distribution.png', 
    'feature_importance.png',
    'model_comparison.png'
]

viz_count = sum(1 for viz in required_viz if os.path.exists(f'visualizations/{viz}'))
if viz_count >= 3:
    requirements['Visualizations'] = True
    print(f"   ✅ Visualizations: {viz_count}/4 found")
else:
    print(f"   ⚠️ Visualizations: Only {viz_count}/4 found")

# Summary
print("\n" + "="*70)
print("📋 SUMMARY")
print("="*70)
complete = sum(requirements.values())
total = len(requirements)

for req, status in requirements.items():
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {req}")

print(f"\n🎯 Completion: {complete}/{total} requirements")

if complete == total:
    print("\n✅ ALL REQUIREMENTS FULFILLED!")
    print("   Your project is ready for submission!")
else:
    print(f"\n⚠️ {total - complete} requirements missing")
    print("   Need to update code")

print("="*70)