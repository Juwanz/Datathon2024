import numpy as np
from sklearn.cluster import DBSCAN
import random
import matplotlib.pyplot as plt

# Load the CSV file, skipping the first column and the header
data = np.genfromtxt('data selection hack/Selection/data_set_1.csv', delimiter=',', skip_header=1, usecols=(1, 2))

# Verify the shape of the data
print("Shape of data:", data.shape)

# Extract X1 and X2 values for plotting
X1 = data[:, 0]
X2 = data[:, 1]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X1, X2, color='blue', marker='o', s=30, edgecolor='k')
plt.title("Scatter Plot of X1 vs X2")
plt.xlabel("Normalized Frequency (Hz) - X1")
plt.ylabel("Normalized Power (Watts) - X2")
plt.grid(True)
# Display the plot
plt.show()

# Parameters
n_samples_target = 2500  # Select 2500 data points for the selection

def dbscan_uniform_selection(data, eps=0.001, min_samples=2):
    """
    Selects data points with uniform coverage using DBSCAN.
    """
    # Run DBSCAN with larger eps and smaller min_samples for broad clusters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    
    # Get unique clusters (ignoring noise points marked by -1)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    # Sample one or a few points from each cluster to achieve uniform coverage
    selected_points = []
    for label in unique_labels:
        cluster_points = data[labels == label]
        selected_points.extend(random.sample(list(cluster_points), k=1))  # k can be adjusted
    
    # If fewer points than target, repeat sampling within clusters
    while len(selected_points) < n_samples_target:
        for label in unique_labels:
            cluster_points = data[labels == label]
            selected_points.extend(random.sample(list(cluster_points), k=1))
            if len(selected_points) >= n_samples_target:
                break

    # Return selected points as numpy array
    return np.array(selected_points[:n_samples_target])

def dbscan_density_selection(data, eps=0.0005, min_samples=2):
    """
    Selects data points with density-based coverage using DBSCAN.
    """
    # Run DBSCAN with smaller eps and higher min_samples for tighter clusters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    
    # Get unique clusters (ignoring noise points marked by -1)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    # Select more points from denser clusters
    selected_points = []
    cluster_sizes = {label: len(data[labels == label]) for label in unique_labels}
    total_size = sum(cluster_sizes.values())
    
    for label in unique_labels:
        cluster_points = data[labels == label]
        # Proportionally sample based on cluster size
        num_points = max(1, int((cluster_sizes[label] / total_size) * n_samples_target))
        selected_points.extend(random.sample(list(cluster_points), k=num_points))
    
    # Ensure exactly n_samples_target points
    return np.array(selected_points[:n_samples_target])

# Usage:
# Uniform coverage selection
selected_uniform = dbscan_uniform_selection(data)
print("Uniformly selected data points:\n", selected_uniform)
print(len(selected_uniform))

# Density-based selection
selected_density = dbscan_density_selection(data)
print("Density-based selected data points:\n", selected_density)
print(len(selected_density))

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
