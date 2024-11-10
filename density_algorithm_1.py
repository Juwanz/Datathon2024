import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial.distance import cdist

# Step 1: Load the dataset (replace 'your_file.csv' with the actual file path)
df = pd.read_csv('data_set_1.csv', header=None)  # Assuming no header, so `header=None`

# Extract the columns for power and frequency (columns are 1-based and 2-based: 0=Index, 1=Frequency, 2=Power)
X = df[[1, 2]].values  # Extract only 'frequency' and 'power'
original_index = df[0].values  # Extract the original index values (first column)

# Step 2: Apply DBSCAN to find dense regions (adjust parameters based on your data)
dbscan = DBSCAN(eps=0.0018, min_samples=12)  # Adjust `eps` and `min_samples` based on data
dbscan_labels = dbscan.fit_predict(X)

# Step 3: Extract points that are not labeled as noise (label = -1)
dense_points = X[dbscan_labels != -1]
dense_indices = original_index[dbscan_labels != -1]  # Extract the corresponding indices of dense points

# Step 4: Perform K-means clustering on the dense points to find 2500 clusters
kmeans = KMeans(n_clusters=2500, random_state=42)  # Number of clusters to select is 2500
kmeans_labels = kmeans.fit_predict(dense_points)

# Step 5: Get the centroids of the clusters (these are the representative points)
centroids = kmeans.cluster_centers_

# Step 6: Find the closest dense points to each of the K-means centroids
# Calculate distances from each dense point to each centroid
distances = cdist(dense_points, centroids, 'euclidean')

# Find the closest dense point to each centroid
closest_points_indices = np.argmin(distances, axis=0)

# Get the selected points from the original data based on the closest dense points
selected_dense_points = dense_points[closest_points_indices]
selected_indices = dense_indices[closest_points_indices]

# Step 10: Save the selected 2500 points with original index to a CSV
final_df = pd.DataFrame(np.column_stack((selected_indices, selected_dense_points)), columns=['Original Index', 'Frequency', 'Power'])
final_df.to_csv('selected_points_with_index1.csv', index=False)

# To make sure you have the original points that are closest to the centroids
# You can extract these rows from the original dataframe and save them as well
final_original_points = df.iloc[closest_points_indices]
final_original_points.to_csv('selected_points_original_data1.csv', index=False)
