import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

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
    
    selected_points = []
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        selected_points.extend(random.sample(list(cluster_points), k=1))  # k=1 for uniform coverage
    
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
    
    cluster_sizes = {i: len(data[labels == i]) for i in range(n_clusters)}
    total_size = sum(cluster_sizes.values())
    
    selected_points = []
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        num_points = max(1, int((cluster_sizes[i] / total_size) * n_samples_target))
        selected_points.extend(random.sample(list(cluster_points), k=num_points))
    
    return np.array(selected_points[:n_samples_target])

# Usage:
selected_uniform = kmeans_uniform_selection(data, n_clusters=n_clusters)
selected_density = kmeans_density_selection(data, n_clusters=n_clusters)

# 3D Histogram (Uniform Selection)
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')

# Define histogram bin edges for uniform selection
x_edges = np.linspace(selected_uniform[:, 0].min(), selected_uniform[:, 0].max(), 20)
y_edges = np.linspace(selected_uniform[:, 1].min(), selected_uniform[:, 1].max(), 20)
hist, x_edges, y_edges = np.histogram2d(selected_uniform[:, 0], selected_uniform[:, 1], bins=[x_edges, y_edges])

# Filter values where the height is less than 1
xpos, ypos = np.meshgrid(x_edges[:-1] + 0.5 * (x_edges[1] - x_edges[0]), 
                         y_edges[:-1] + 0.5 * (y_edges[1] - y_edges[0]), indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)
dz = hist.ravel()

# Only keep bars with dz >= 1
mask = dz >= 1
xpos, ypos, dz = xpos[mask], ypos[mask], dz[mask]
dx = dy = 0.02

ax1.bar3d(xpos, ypos, zpos[mask], dx, dy, dz, zsort='average')
ax1.set_title("3D Histogram (Uniform Selection)")
ax1.set_xlabel("Normalized Frequency (Hz) - X1")
ax1.set_ylabel("Normalized Power (Watts) - X2")
ax1.set_zlabel("Number of Points")

# 3D Histogram (Density-based Selection)
ax2 = fig.add_subplot(122, projection='3d')

# Define histogram bin edges for density-based selection
x_edges = np.linspace(selected_density[:, 0].min(), selected_density[:, 0].max(), 20)
y_edges = np.linspace(selected_density[:, 1].min(), selected_density[:, 1].max(), 20)
hist, x_edges, y_edges = np.histogram2d(selected_density[:, 0], selected_density[:, 1], bins=[x_edges, y_edges])

# Filter values where the height is less than 1
xpos, ypos = np.meshgrid(x_edges[:-1] + 0.5 * (x_edges[1] - x_edges[0]), 
                         y_edges[:-1] + 0.5 * (y_edges[1] - y_edges[0]), indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
dz = hist.ravel()

# Only keep bars with dz >= 1
mask = dz >= 1
xpos, ypos, dz = xpos[mask], ypos[mask], dz[mask]

ax2.bar3d(xpos, ypos, zpos[mask], dx, dy, dz, zsort='average')
ax2.set_title("3D Histogram (Density-based Selection)")
ax2.set_xlabel("Normalized Frequency (Hz) - X1")
ax2.set_ylabel("Normalized Power (Watts) - X2")
ax2.set_zlabel("Number of Points")

# Display the 3D histograms
plt.tight_layout()
plt.show()
