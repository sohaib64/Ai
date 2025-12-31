"""
K-Means Clustering Analysis
Adds unsupervised learning component to your project
Run this AFTER training your model

Steps before running this:
1. ✅ data-preprocessing-FINAL-FIX.py (completed)
2. ✅ train-model.py (completed)
3. ▶️ add-clustering.py (run this now)
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')


class CarMarketClusterAnalysis:
    """
    Performs K-Means clustering on car data for market segmentation
    """
    
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.optimal_k = 3
        
    def load_data(self):
        """Load cleaned data"""
        print("="*70)
        print("📂 LOADING DATA FOR CLUSTERING")
        print("="*70)
        
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Data loaded: {len(self.df):,} rows")
            return self.df
        except FileNotFoundError:
            print(f"❌ Error: {self.data_file} not found!")
            print("💡 Please run data-preprocessing-FINAL-FIX.py first!")
            raise
    
    def prepare_clustering_features(self):
        """Prepare features for clustering"""
        print("\n" + "="*70)
        print("🔧 PREPARING FEATURES FOR CLUSTERING")
        print("="*70)
        
        # Select numerical features for clustering
        clustering_features = ['price', 'engine', 'registered_in', 'mileage']
        
        # Check which features exist
        available_features = [f for f in clustering_features if f in self.df.columns]
        
        print(f"\n📊 Using features: {available_features}")
        
        # Create feature matrix
        self.X_cluster = self.df[available_features].copy()
        
        # Remove any rows with missing values
        initial_count = len(self.X_cluster)
        self.X_cluster = self.X_cluster.dropna()
        
        print(f"   ✓ Feature matrix: {self.X_cluster.shape}")
        print(f"   ✓ Removed {initial_count - len(self.X_cluster)} rows with missing values")
        print(f"   ✓ Features: {list(self.X_cluster.columns)}")
        
        return self.X_cluster
    
    def find_optimal_clusters(self, max_k=10):
        """Find optimal number of clusters using elbow method"""
        print("\n" + "="*70)
        print("🔍 FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*70)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(self.X_cluster)
        
        inertias = []
        K_range = range(2, max_k + 1)
        
        print("\n📊 Calculating inertia for different K values...")
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            print(f"   K={k}: Inertia={kmeans.inertia_:.2f}")
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Inertia', fontsize=12)
        plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/elbow_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n   ✓ Elbow curve saved: visualizations/elbow_curve.png")
        print(f"   💡 Recommended K: 3 clusters (Economy, Mid-Range, Luxury)")
        
        self.optimal_k = 3  # Based on car market segments
        
    def perform_clustering(self, n_clusters=3):
        """Perform K-Means clustering"""
        print("\n" + "="*70)
        print(f"🤖 PERFORMING K-MEANS CLUSTERING (K={n_clusters})")
        print("="*70)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(self.X_cluster)
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        self.df.loc[self.X_cluster.index, 'cluster'] = clusters
        
        print(f"\n✅ Clustering completed!")
        print(f"\n📊 Cluster distribution:")
        for i in range(n_clusters):
            count = np.sum(clusters == i)
            percentage = (count / len(clusters)) * 100
            print(f"   Cluster {i}: {count:,} cars ({percentage:.1f}%)")
        
        return clusters
    
    def analyze_clusters(self):
        """Analyze and interpret clusters"""
        print("\n" + "="*70)
        print("📊 CLUSTER ANALYSIS")
        print("="*70)
        
        # Remove rows without cluster assignment
        df_clustered = self.df.dropna(subset=['cluster'])
        
        cluster_summary = df_clustered.groupby('cluster').agg({
            'price': ['mean', 'median', 'min', 'max', 'count'],
            'engine': ['mean', 'median'],
            'registered_in': ['mean', 'min', 'max'],
            'mileage': ['mean', 'median']
        }).round(0)
        
        print("\n📋 Cluster Statistics:")
        print(cluster_summary)
        
        # Interpret clusters
        print("\n" + "="*70)
        print("🎯 MARKET SEGMENTATION INTERPRETATION")
        print("="*70)
        
        for cluster_id in sorted(df_clustered['cluster'].unique()):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            avg_price = cluster_data['price'].mean()
            avg_year = cluster_data['registered_in'].mean()
            avg_engine = cluster_data['engine'].mean()
            count = len(cluster_data)
            
            # Determine segment name
            if avg_price < 1500000:
                segment = "Economy Segment"
                emoji = "💰"
            elif avg_price < 3500000:
                segment = "Mid-Range Segment"
                emoji = "🚗"
            else:
                segment = "Premium/Luxury Segment"
                emoji = "✨"
            
            print(f"\n{emoji} Cluster {cluster_id}: {segment}")
            print(f"   • Average Price: {avg_price:,.0f} PKR")
            print(f"   • Average Year: {avg_year:.0f}")
            print(f"   • Average Engine: {avg_engine:.0f} cc")
            print(f"   • Number of Cars: {count:,} ({count/len(df_clustered)*100:.1f}%)")
            
            # Top brands in this cluster
            if 'car_brand' in cluster_data.columns:
                top_brands = cluster_data['car_brand'].value_counts().head(3)
                print(f"   • Top Brands: {', '.join(top_brands.index.tolist())}")
    
    def visualize_clusters(self):
        """Create cluster visualizations"""
        print("\n" + "="*70)
        print("📊 CREATING VISUALIZATIONS")
        print("="*70)
        
        os.makedirs('visualizations', exist_ok=True)
        
        # Get data with clusters only
        df_viz = self.df.dropna(subset=['cluster']).copy()
        
        # 1. 2D PCA Visualization
        print("\n1️⃣ Creating 2D PCA visualization...")
        X_scaled = self.scaler.transform(self.X_cluster.loc[df_viz.index])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=df_viz['cluster'], 
                            cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        plt.title('Car Market Clusters (PCA Visualization)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/cluster_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: visualizations/cluster_pca.png")
        
        # 2. Cluster Distribution by Features
        print("\n2️⃣ Creating cluster distribution plots...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Price distribution by cluster
        df_viz.boxplot(column='price', by='cluster', ax=axes[0, 0])
        axes[0, 0].set_title('Price Distribution by Cluster', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Price (PKR)', fontsize=10)
        axes[0, 0].set_xlabel('Cluster', fontsize=10)
        
        # Engine distribution by cluster
        df_viz.boxplot(column='engine', by='cluster', ax=axes[0, 1])
        axes[0, 1].set_title('Engine Size by Cluster', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Engine (cc)', fontsize=10)
        axes[0, 1].set_xlabel('Cluster', fontsize=10)
        
        # Year distribution by cluster
        df_viz.boxplot(column='registered_in', by='cluster', ax=axes[1, 0])
        axes[1, 0].set_title('Model Year by Cluster', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Year', fontsize=10)
        axes[1, 0].set_xlabel('Cluster', fontsize=10)
        
        # Mileage distribution by cluster
        df_viz.boxplot(column='mileage', by='cluster', ax=axes[1, 1])
        axes[1, 1].set_title('Mileage by Cluster', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Mileage (km)', fontsize=10)
        axes[1, 1].set_xlabel('Cluster', fontsize=10)
        
        plt.suptitle('Cluster Feature Distributions', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('visualizations/cluster_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: visualizations/cluster_distributions.png")
        
        # 3. Cluster Size Pie Chart
        print("\n3️⃣ Creating cluster size visualization...")
        cluster_counts = df_viz['cluster'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 8))
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        plt.pie(cluster_counts.values, 
                labels=[f'Cluster {i}' for i in cluster_counts.index],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 12, 'fontweight': 'bold'})
        plt.title('Market Segment Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/cluster_pie_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: visualizations/cluster_pie_chart.png")
        
        print("\n✅ All visualizations created!")
    
    def save_cluster_model(self):
        """Save clustering model and data"""
        print("\n" + "="*70)
        print("💾 SAVING CLUSTER MODEL AND DATA")
        print("="*70)
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Save K-Means model
        with open('models/kmeans_model.pkl', 'wb') as f:
            pickle.dump(self.kmeans, f)
        print("   ✓ Saved: models/kmeans_model.pkl")
        
        # Save scaler
        with open('models/cluster_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("   ✓ Saved: models/cluster_scaler.pkl")
        
        # Save clustered data
        self.df.to_csv('data/pakwheels_with_clusters.csv', index=False)
        print("   ✓ Saved: data/pakwheels_with_clusters.csv")
        
        file_size = os.path.getsize('data/pakwheels_with_clusters.csv') / 1024
        print(f"   ✓ File size: {file_size:.2f} KB")
    
    def run(self):
        """Run complete clustering analysis"""
        print("\n" + "="*70)
        print("🚀 K-MEANS CLUSTERING ANALYSIS")
        print("="*70)
        
        try:
            self.load_data()
            self.prepare_clustering_features()
            self.find_optimal_clusters()
            self.perform_clustering(n_clusters=3)
            self.analyze_clusters()
            self.visualize_clusters()
            self.save_cluster_model()
            
            print("\n" + "="*70)
            print("✅ CLUSTERING ANALYSIS COMPLETED!")
            print("="*70)
            print("\n📁 Generated Files:")
            print("   ✓ models/kmeans_model.pkl")
            print("   ✓ models/cluster_scaler.pkl")
            print("   ✓ data/pakwheels_with_clusters.csv")
            print("   ✓ visualizations/elbow_curve.png")
            print("   ✓ visualizations/cluster_pca.png")
            print("   ✓ visualizations/cluster_distributions.png")
            print("   ✓ visualizations/cluster_pie_chart.png")
            print("\n🎯 Next Step: Run Flask backend (python app.py)")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Clustering failed: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    
    DATA_FILE = 'data/pakwheels_cleaned.csv'
    
    if not os.path.exists(DATA_FILE):
        print("\n❌ Error: Cleaned data not found!")
        print(f"   Looking for: {DATA_FILE}")
        print("\n💡 Please run data-preprocessing-FINAL-FIX.py first!")
        print("\n📋 Correct order:")
        print("   1. python data-preprocessing-FINAL-FIX.py")
        print("   2. python train-model.py")
        print("   3. python add-clustering.py  ← You are here")
        return
    
    # Check if model is trained
    if not os.path.exists('models/best_model.pkl'):
        print("\n⚠️  Warning: Prediction model not found!")
        print("   You should run train-model.py first, but clustering can proceed.")
        proceed = input("\n   Continue anyway? (yes/no): ").strip().lower()
        if proceed != 'yes':
            return
    
    analyzer = CarMarketClusterAnalysis(DATA_FILE)
    analyzer.run()


if __name__ == "__main__":
    main()