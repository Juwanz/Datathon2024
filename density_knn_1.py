from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN

# Load your dataset as shown previously
df = pd.read_csv('data_set_1.csv', header=None)
X = df[[1, 2]].values

# Fit NearestNeighbors to find distances
neighbors = NearestNeighbors(n_neighbors=15)  # 15 can vary; adjust for density of clusters
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# Sort and plot distances of the k-nearest neighbors
distances = np.sort(distances[:, -1], axis=0)  # Sort distances to the 15th nearest neighbor
plt.figure(figsize=(8, 6))
plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel("15th Nearest Neighbor Distance")
plt.title("KNN Distance Plot to Estimate DBSCAN `eps`")
plt.grid(True)
plt.show()


# Use the `eps` value estimated from the KNN plot and set min_samples
eps = 0.0018  # Replace with the elbow point from the plot
min_samples = 12

# Apply DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X)

# Count clusters and noise
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")


# Convert labels to DataFrame for easier filtering
df['Cluster'] = dbscan_labels


# Plot a larger number of clusters
plt.figure(figsize=(12, 10))

# Generate a colormap to assign different colors to each cluster
num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
colors = cm.get_cmap('tab20', num_clusters)

# Plot each cluster
for cluster_label in range(num_clusters):
    cluster_data = X[dbscan_labels == cluster_label]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                label=f'Cluster {cluster_label}', s=10, color=colors(cluster_label))

# Plot noise points in a different color (labeled as -1 in DBSCAN)
noise_data = X[dbscan_labels == -1]
plt.scatter(noise_data[:, 0], noise_data[:, 1], color='gray', label='Noise', s=10, alpha=0.5)

# Customize plot
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Scatter Plot of DBSCAN Clusters and Noise Points')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot for clarity
plt.grid(True)
plt.show()


