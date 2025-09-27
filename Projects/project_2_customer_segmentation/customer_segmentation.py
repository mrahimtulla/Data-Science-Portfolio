
```python
#!/usr/bin/env python3
"""
Customer Segmentation using K-means Clustering
Complete implementation with visualization and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self, data_path='datasets/mall_customers.csv'):
        """
        Initialize the Customer Segmentation analysis
        
        Args:
            data_path (str): Path to the customer data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.cluster_labels = None
        
    def load_data(self):
        """Load and inspect the customer data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print("✅ Data loaded successfully!")
            print(f"📊 Dataset shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print("❌ Data file not found. Creating sample data...")
            self._create_sample_data()
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _create_sample_data(self):
        """Create sample customer data if file not found"""
        np.random.seed(42)
        n_customers = 200
        
        data = {
            'CustomerID': range(1, n_customers + 1),
            'Gender': np.random.choice(['Male', 'Female'], n_customers),
            'Age': np.random.normal(35, 10, n_customers).astype(int),
            'Annual Income (k$)': np.random.normal(60, 15, n_customers).astype(int),
            'Spending Score (1-100)': np.random.normal(50, 20, n_customers).astype(int)
        }
        
        # Ensure positive values
        data['Age'] = np.clip(data['Age'], 18, 70)
        data['Annual Income (k$)'] = np.clip(data['Annual Income (k$)'], 15, 140)
        data['Spending Score (1-100)'] = np.clip(data['Spending Score (1-100)'], 1, 100)
        
        self.df = pd.DataFrame(data)
        self.df.to_csv('datasets/mall_customers.csv', index=False)
        print("✅ Sample data created and saved!")
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("\n🔍 Exploratory Data Analysis")
        print("=" * 50)
        
        # Basic information
        print("Dataset Info:")
        print(self.df.info())
        
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        # Create visualizations
        self._create_eda_visualizations()
    
    def _create_eda_visualizations(self):
        """Create EDA visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Data Analysis', fontsize=16, fontweight='bold')
        
        # Age distribution
        axes[0,0].hist(self.df['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Age Distribution')
        axes[0,0].set_xlabel('Age')
        axes[0,0].set_ylabel('Frequency')
        
        # Income distribution
        axes[0,1].hist(self.df['Annual Income (k$)'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].set_title('Annual Income Distribution')
        axes[0,1].set_xlabel('Annual Income (k$)')
        axes[0,1].set_ylabel('Frequency')
        
        # Spending score distribution
        axes[0,2].hist(self.df['Spending Score (1-100)'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0,2].set_title('Spending Score Distribution')
        axes[0,2].set_xlabel('Spending Score (1-100)')
        axes[0,2].set_ylabel('Frequency')
        
        # Gender distribution
        gender_counts = self.df['Gender'].value_counts()
        axes[1,0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                     colors=['lightblue', 'lightpink'])
        axes[1,0].set_title('Gender Distribution')
        
        # Income vs Spending
        colors = {'Male': 'blue', 'Female': 'red'}
        for gender in ['Male', 'Female']:
            gender_data = self.df[self.df['Gender'] == gender]
            axes[1,1].scatter(gender_data['Annual Income (k$)'], 
                            gender_data['Spending Score (1-100)'],
                            c=colors[gender], label=gender, alpha=0.6)
        axes[1,1].set_title('Income vs Spending Score')
        axes[1,1].set_xlabel('Annual Income (k$)')
        axes[1,1].set_ylabel('Spending Score (1-100)')
        axes[1,1].legend()
        
        # Correlation heatmap
        numeric_df = self.df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
        axes[1,2].set_title('Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('project_2_customer_segmentation/images/eda_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """Preprocess data for clustering"""
        print("\n⚙️ Data Preprocessing")
        print("=" * 50)
        
        # Select features for clustering
        X = self.df[self.features].copy()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        print("✅ Data preprocessing completed!")
        return X_scaled
    
    def find_optimal_clusters(self, X_scaled, max_k=10):
        """Find optimal number of clusters using elbow method and silhouette scores"""
        print("\n📈 Finding Optimal Number of Clusters")
        print("=" * 50)
        
        wcss = []  # Within-Cluster Sum of Square
        silhouette_scores = []
        calinski_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            
            # Silhouette Score
            silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
            
            # Calinski-Harabasz Score
            calinski_avg = calinski_harabasz_score(X_scaled, kmeans.labels_)
            calinski_scores.append(calinski_avg)
            
            print(f"K={k}: WCSS={kmeans.inertia_:.2f}, "
                  f"Silhouette={silhouette_avg:.3f}, "
                  f"Calinski={calinski_avg:.2f}")
        
        # Plot results
        self._plot_optimal_k(k_range, wcss, silhouette_scores, calinski_scores)
        
        # Find optimal k (you can modify this logic)
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        print(f"\n🎯 Optimal number of clusters: {optimal_k}")
        
        return optimal_k
    
    def _plot_optimal_k(self, k_range, wcss, silhouette_scores, calinski_scores):
        """Plot metrics for different k values"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Elbow curve
        ax1.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('WCSS')
        ax1.set_title('Elbow Method for Optimal K')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Scores for Different K')
        ax2.grid(True, alpha=0.3)
        
        # Calinski-Harabasz scores
        ax3.plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('Calinski-Harabasz Score')
        ax3.set_title('Calinski-Harabasz Scores for Different K')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('project_2_customer_segmentation/images/optimal_k_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def apply_kmeans(self, X_scaled, n_clusters=5):
        """Apply K-means clustering"""
        print(f"\n🎯 Applying K-means with {n_clusters} clusters")
        print("=" * 50)
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        self.df['Cluster'] = self.cluster_labels
        
        # Calculate metrics
        silhouette_avg = silhouette_score(X_scaled, self.cluster_labels)
        calinski_avg = calinski_harabasz_score(X_scaled, self.cluster_labels)
        
        print(f"✅ K-means clustering completed!")
        print(f"📊 Silhouette Score: {silhouette_avg:.3f}")
        print(f"📊 Calinski-Harabasz Score: {calinski_avg:.2f}")
        
        return self.cluster_labels
    
    def visualize_clusters(self, X_scaled):
        """Visualize the clustering results"""
        print("\n🎨 Visualizing Clusters")
        print("=" * 50)
        
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create visualization dataframe
        viz_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        viz_df['Cluster'] = self.cluster_labels
        viz_df['Annual Income'] = self.df['Annual Income (k$)']
        viz_df['Spending Score'] = self.df['Spending Score (1-100)']
        viz_df['Age'] = self.df['Age']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Segmentation Results', fontsize=16, fontweight='bold')
        
        # PCA visualization
        scatter = axes[0,0].scatter(viz_df['PC1'], viz_df['PC2'], 
                                   c=viz_df['Cluster'], cmap='viridis', 
                                   alpha=0.7, s=60)
        axes[0,0].set_xlabel('Principal Component 1')
        axes[0,0].set_ylabel('Principal Component 2')
        axes[0,0].set_title('PCA - Customer Segments')
        plt.colorbar(scatter, ax=axes[0,0])
        
        # Income vs Spending with clusters
        scatter = axes[0,1].scatter(self.df['Annual Income (k$)'], 
                                   self.df['Spending Score (1-100)'],
                                   c=self.df['Cluster'], cmap='viridis',
                                   alpha=0.7, s=60)
        axes[0,1].set_xlabel('Annual Income (k$)')
        axes[0,1].set_ylabel('Spending Score (1-100)')
        axes[0,1].set_title('Income vs Spending Score by Cluster')
        plt.colorbar(scatter, ax=axes[0,1])
        
        # Age vs Spending with clusters
        scatter = axes[1,0].scatter(self.df['Age'], 
                                   self.df['Spending Score (1-100)'],
                                   c=self.df['Cluster'], cmap='viridis',
                                   alpha=0.7, s=60)
        axes[1,0].set_xlabel('Age')
        axes[1,0].set_ylabel('Spending Score (1-100)')
        axes[1,0].set_title('Age vs Spending Score by Cluster')
        plt.colorbar(scatter, ax=axes[1,0])
        
        # Cluster sizes
        cluster_sizes = self.df['Cluster'].value_counts().sort_index()
        bars = axes[1,1].bar(cluster_sizes.index, cluster_sizes.values, 
                            color=plt.cm.viridis(np.linspace(0, 1, len(cluster_sizes))))
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Number of Customers')
        axes[1,1].set_title('Cluster Sizes')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('project_2_customer_segmentation/images/cluster_visualizations.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_clusters(self):
        """Analyze and profile each cluster"""
        print("\n📊 Cluster Analysis and Profiling")
        print("=" * 50)
        
        # Cluster profiles
        cluster_profile = self.df.groupby('Cluster').agg({
            'Age': ['mean', 'std', 'min', 'max'],
            'Annual Income (k$)': ['mean', 'std', 'min', 'max'],
            'Spending Score (1-100)': ['mean', 'std', 'min', 'max'],
            'Gender': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
            'CustomerID': 'count'
        }).round(2)
        
        # Flatten column names
        cluster_profile.columns = ['_'.join(col).strip() for col in cluster_profile.columns.values]
        cluster_profile = cluster_profile.rename(columns={'CustomerID_count': 'Count'})
        
        print("Cluster Profiles:")
        print(cluster_profile)
        
        # Create detailed analysis
        self._create_cluster_analysis_plots(cluster_profile)
        
        return cluster_profile
    
    def _create_cluster_analysis_plots(self, cluster_profile):
        """Create detailed cluster analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cluster Profile Analysis', fontsize=16, fontweight='bold')
        
        # Average metrics by cluster
        metrics = ['Age_mean', 'Annual Income (k$)_mean', 'Spending Score (1-100)_mean']
        metric_names = ['Average Age', 'Average Income (k$)', 'Average Spending Score']
        
        x = np.arange(len(cluster_profile))
        width = 0.25
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            axes[0,0].bar(x + i*width, cluster_profile[metric], width, label=name)
        
        axes[0,0].set_xlabel('Cluster')
        axes[0,0].set_ylabel('Value')
        axes[0,0].set_title('Average Metrics by Cluster')
        axes[0,0].set_xticks(x + width)
        axes[0,0].set_xticklabels(cluster_profile.index)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Gender distribution by cluster
        cluster_gender = pd.crosstab(self.df['Cluster'], self.df['Gender'])
        cluster_gender.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Gender Distribution by Cluster')
        axes[0,1].set_xlabel('Cluster')
        axes[0,1].set_ylabel('Count')
        axes[0,1].legend(title='Gender')
        
        # Age distribution by cluster
        for cluster in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster]
            axes[1,0].hist(cluster_data['Age'], alpha=0.6, 
                          label=f'Cluster {cluster}', bins=15, density=True)
        axes[1,0].set_title('Age Distribution by Cluster')
        axes[1,0].set_xlabel('Age')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        
        # Spending pattern by cluster
        spending_income = self.df.groupby('Cluster').agg({
            'Annual Income (k$)': 'mean',
            'Spending Score (1-100)': 'mean'
        })
        scatter = axes[1,1].scatter(spending_income['Annual Income (k$)'], 
                                   spending_income['Spending Score (1-100)'],
                                   s=cluster_profile['Count']*10,  # Size by cluster size
                                   c=spending_income.index, cmap='viridis', alpha=0.7)
        
        # Annotate points with cluster numbers
        for i, (x, y) in enumerate(zip(spending_income['Annual Income (k$)'], 
                                      spending_income['Spending Score (1-100)'])):
            axes[1,1].annotate(f'Cluster {i}', (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=10)
        
        axes[1,1].set_xlabel('Average Annual Income (k$)')
        axes[1,1].set_ylabel('Average Spending Score')
        axes[1,1].set_title('Spending Patterns by Cluster')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('project_2_customer_segmentation/images/cluster_profiles.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_business_recommendations(self, cluster_profile):
        """Generate business recommendations based on clusters"""
        print("\n💡 Business Recommendations")
        print("=" * 50)
        
        # Define cluster interpretations
        cluster_interpretations = {
            0: "Budget-Conscious Shoppers",
            1: "Balanced Spenders", 
            2: "Young Trendsetters",
            3: "High-Value Targets",
            4: "Conservative Shoppers"
        }
        
        self.df['Segment'] = self.df['Cluster'].map(cluster_interpretations)
        
        recommendations = []
        
        for cluster_num in sorted(self.df['Cluster'].unique()):
            segment_data = self.df[self.df['Cluster'] == cluster_num]
            segment_name = cluster_interpretations[cluster_num]
            
            avg_age = segment_data['Age'].mean()
            avg_income = segment_data['Annual Income (k$)'].mean()
            avg_spending = segment_data['Spending Score (1-100)'].mean()
            segment_size = len(segment_data)
            gender_ratio = segment_data['Gender'].value_counts(normalize=True)
            
            print(f"\n🎯 {segment_name} (Cluster {cluster_num}):")
            print(f"   • Size: {segment_size} customers ({segment_size/len(self.df)*100:.1f}%)")
            print(f"   • Average Age: {avg_age:.1f} years")
            print(f"   • Average Income: ${avg_income:.1f}k")
            print(f"   • Average Spending Score: {avg_spending:.1f}")
            print(f"   • Gender Ratio: {gender_ratio.get('Male', 0)*100:.1f}% Male, "
                  f"{gender_ratio.get('Female', 0)*100:.1f}% Female")
            
            # Generate specific recommendations
            if cluster_num == 3:  # High-Value Targets
                rec = "💡 Premium loyalty program, exclusive events, personalized shopping"
            elif cluster_num == 2:  # Young Trendsetters
                rec = "💡 Social media campaigns, trendy products, influencer collaborations"
            elif cluster_num == 0:  # Budget-Conscious
                rec = "💡 Value deals, discount programs, budget-friendly options"
            elif cluster_num == 1:  # Balanced Spenders
                rec = "💡 Family packages, seasonal promotions, loyalty rewards"
            else:  # Conservative Shoppers
                rec = "💡 Quality assurance, trust-building campaigns, reliable products"
            
            print(f"   {rec}")
            recommendations.append(rec)
        
        return recommendations
    
    def save_results(self):
        """Save all results to files"""
        print("\n💾 Saving Results")
        print("=" * 50)
        
        # Save clustered data
        self.df.to_csv('project_2_customer_segmentation/data/customers_clustered.csv', index=False)
        
        # Save cluster profiles
        cluster_profile = self.df.groupby('Cluster').agg({
            'Age': ['mean', 'std'],
            'Annual Income (k$)': ['mean', 'std'],
            'Spending Score (1-100)': ['mean', 'std'],
            'CustomerID': 'count'
        }).round(2)
        cluster_profile.to_csv('project_2_customer_segmentation/data/cluster_profiles.csv')
        
        # Save model information
        model_info = {
            'n_clusters': self.kmeans_model.n_clusters,
            'features_used': self.features,
            'silhouette_score': silhouette_score(self.scaler.transform(self.df[self.features]), 
                                               self.cluster_labels),
            'inertia': self.kmeans_model.inertia_
        }
        
        import json
        with open('project_2_customer_segmentation/data/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("✅ Results saved successfully!")
    
    def run_complete_analysis(self):
        """Run the complete customer segmentation analysis"""
        print("🎯 Starting Customer Segmentation Analysis")
        print("=" * 60)
        
        # Step 1: Load data
        if not self.load_data():
            return
        
        # Step 2: Exploratory Data Analysis
        self.exploratory_data_analysis()
        
        # Step 3: Preprocess data
        X_scaled = self.preprocess_data()
        
        # Step 4: Find optimal clusters
        optimal_k = self.find_optimal_clusters(X_scaled)
        
        # Step 5: Apply K-means
        self.apply_kmeans(X_scaled, optimal_k)
        
        # Step 6: Visualize results
        self.visualize_clusters(X_scaled)
        
        # Step 7: Analyze clusters
        cluster_profile = self.analyze_clusters()
        
        # Step 8: Generate recommendations
        recommendations = self.generate_business_recommendations(cluster_profile)
        
        # Step 9: Save results
        self.save_results()
        
        print("\n🎉 Customer Segmentation Analysis Completed Successfully!")
        print("=" * 60)
        print(f"📊 Total customers analyzed: {len(self.df)}")
        print(f"🎯 Number of segments identified: {optimal_k}")
        print(f"📈 Analysis results saved in project_2_customer_segmentation/")

def main():
    """Main function to run the customer segmentation analysis"""
    # Create necessary directories
    import os
    os.makedirs('project_2_customer_segmentation/images', exist_ok=True)
    os.makedirs('project_2_customer_segmentation/data', exist_ok=True)
    
    # Initialize and run analysis
    segmentation = CustomerSegmentation()
    segmentation.run_complete_analysis()

if __name__ == "__main__":
    main()