import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the CSV file, skipping the first column and the header
data = np.genfromtxt('data selection hack/Selection/data_set_1.csv', delimiter=',', skip_header=1, usecols=(1, 2))

# Verify the shape of the data
print("Shape of data:", data.shape)

# Parameters
n_samples_target = 2500  # Select 2500 data points for the selection
n_clusters = 10  # Number of clusters for KMeans

def kmeans_uniform_selection(data, n_clusters=10):
    """
    Selects data points with uniform coverage using KMeans.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # Sample one point from each cluster to ensure uniform coverage
    selected_points = []
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        selected_points.extend(random.sample(list(cluster_points), k=1))  # k=1 for uniform coverage
    
    # If fewer points than target, repeat sampling within clusters
    while len(selected_points) < n_samples_target:
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            selected_points.extend(random.sample(list(cluster_points), k=1))
            if len(selected_points) >= n_samples_target:
                break

    return np.array(selected_points[:n_samples_target])

def kmeans_density_selection(data, n_clusters=10):
    """
    Selects data points with density-based coverage using KMeans.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # Calculate the size of each cluster
    cluster_sizes = {i: len(data[labels == i]) for i in range(n_clusters)}
    total_size = sum(cluster_sizes.values())
    
    selected_points = []
    
    # Proportionally sample points based on the cluster size
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        num_points = max(1, int((cluster_sizes[i] / total_size) * n_samples_target))
        selected_points.extend(random.sample(list(cluster_points), k=num_points))
    
    return np.array(selected_points[:n_samples_target])

# Usage:
# Uniform coverage selection
selected_uniform = kmeans_uniform_selection(data, n_clusters=n_clusters)
print("Uniformly selected data points:\n", selected_uniform)
print("Number of selected uniform points:", len(selected_uniform))

# Density-based selection
selected_density = kmeans_density_selection(data, n_clusters=n_clusters)
print("Density-based selected data points:\n", selected_density)
print("Number of selected density points:", len(selected_density))

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot Uniform Selection
plt.subplot(1, 2, 1)
plt.scatter(selected_uniform[:, 0], selected_uniform[:, 1], color='blue', marker='o', s=30, edgecolor='k')
plt.title("Uniformly Selected 2500 Data Points")
plt.xlabel("Normalized Frequency (Hz) - X1")
plt.ylabel("Normalized Power (Watts) - X2")
plt.grid(True)

# Plot Density Selection
plt.subplot(1, 2, 2)
plt.scatter(selected_density[:, 0], selected_density[:, 1], color='red', marker='o', s=30, edgecolor='k')
plt.title("Density-based Selected 2500 Data Points")
plt.xlabel("Normalized Frequency (Hz) - X1")
plt.ylabel("Normalized Power (Watts) - X2")
plt.grid(True)

# Display the plots
plt.tight_layout()
plt.show()