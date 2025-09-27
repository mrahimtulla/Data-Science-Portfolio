import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

class CustomerSegmentation:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.labels_ = None
        
    def preprocess_data(self, df, features):
        """Preprocess the data for clustering"""
        X = df[features].copy()
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def find_optimal_k(self, X, max_k=10):
        """Find optimal number of clusters using elbow method and silhouette scores"""
        wcss = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            
        return wcss, silhouette_scores, k_range
    
    def fit_kmeans(self, X, n_clusters=5):
        """Fit K-means clustering"""
        self.model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.labels_ = self.model.fit_predict(X)
        return self.labels_
    
    def fit_dbscan(self, X, eps=0.5, min_samples=5):
        """Fit DBSCAN clustering"""
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = self.model.fit_predict(X)
        return self.labels_
    
    def evaluate_clustering(self, X, labels):
        """Evaluate clustering performance"""
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(X, labels)
            calinski_score = calinski_harabasz_score(X, labels)
            return silhouette_avg, calinski_score
        return None, None
    
    def plot_elbow_curve(self, wcss, k_range, save_path=None):
        """Plot elbow curve for K-means"""
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, wcss, 'bo-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title('Elbow Method for Optimal K')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_clusters(self, X, labels, feature_names, save_path=None):
        """Visualize clusters using first two features"""
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('Customer Clusters')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def analyze_cluster_profiles(df, cluster_labels, numeric_features):
    """Analyze and profile each cluster"""
    df_cluster = df.copy()
    df_cluster['Cluster'] = cluster_labels
    
    profile = df_cluster.groupby('Cluster').agg({
        **{feature: ['mean', 'std', 'count'] for feature in numeric_features},
        'Cluster': 'count'
    }).round(2)
    
    return profile

# Example usage
if __name__ == "__main__":
    # Sample data creation for testing
    np.random.seed(42)
    data = {
        'Age': np.random.normal(35, 10, 200),
        'Annual Income': np.random.normal(60, 15, 200),
        'Spending Score': np.random.normal(50, 20, 200)
    }
    sample_df = pd.DataFrame(data)
    
    # Initialize segmentation
    seg = CustomerSegmentation()
    X_scaled = seg.preprocess_data(sample_df, ['Age', 'Annual Income', 'Spending Score'])
    
    # Find optimal K
    wcss, silhouette_scores, k_range = seg.find_optimal_k(X_scaled)
    seg.plot_elbow_curve(wcss, k_range)
    
    # Fit with optimal K
    labels = seg.fit_kmeans(X_scaled, n_clusters=5)
    
    # Evaluate
    silhouette_avg, calinski_score = seg.evaluate_clustering(X_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Calinski-Harabasz Score: {calinski_score:.3f}")