import sys
import numpy as np
import csv
import simpy 
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


#coordinates for datapoints imported
x1_vals =[]
x2_vals = []
index = []
domains = {}
with open('Challenge Info\data_set_1.csv', 'r', newline = '') as data:
    map = csv.reader(data, delimiter = ',')
    for rows in map:
        x1_vals.append(float(rows[1]))
        x2_vals.append(float(rows[2]))
        index.append(int(rows[0]))

# Step 1: Define data and calculate the Convex Hull
data = np.column_stack((x1_vals, x2_vals))
hull = ConvexHull(data)
hull_points = data[hull.vertices]  # Hull vertices

# Step 2: Define the bounding box of the Convex Hull
x_min, y_min = hull_points[:, 0].min(), hull_points[:, 1].min()
x_max, y_max = hull_points[:, 0].max(), hull_points[:, 1].max()

# Step 3: Determine grid dimensions for ~2500 cells within the bounding box
approx_cells = 9220
grid_size = int(np.sqrt(approx_cells))  # Aim for a roughly square grid
cell_width = (x_max - x_min) / grid_size
cell_height = (y_max - y_min) / grid_size

# Step 4: Initialize a list for storing active grid cells and filter by Convex Hull
hull_path = mpath.Path(hull_points)  # Convert hull to a path for easy point-in-polygon checks
active_grid = []

for row in range(grid_size):
    for col in range(grid_size):
        # Calculate the center point of each cell
        cell_center_x = x_min + (col + 0.5) * cell_width
        cell_center_y = y_min + (row + 0.5) * cell_height
        cell_center = np.array([cell_center_x, cell_center_y])

        # Check if the cell center is within the convex hull
        if hull_path.contains_point(cell_center):
            # Only add cells that are within the convex hull region
            active_grid.append((row, col))

# Step 5: Map data points to active grid cells
# Initialize a dictionary for storing points in each active cell
grid_cells = {cell: [] for cell in active_grid}

for point in data:
    x, y = point
    col = min(grid_size - 1, max(0, int((x - x_min) / cell_width)))
    row = min(grid_size - 1, max(0, int((y - y_min) / cell_height)))
    
    
    if (row, col) in grid_cells:
        grid_cells[(row, col)].append(point)

# Step 6: Check the number of filled cells and visualize
num_active_cells = len(grid_cells)
print(f"Total active grid cells: {num_active_cells} (approx 2500)")

cluster_centers = []
for (row, col) in active_grid:
    center_x = x_min + (col + 0.5) * cell_width
    center_y = y_min + (row + 0.5) * cell_height
    cluster_centers.append([center_x, center_y])

# Step 2: Use K-means with the specified centers
kmeans = KMeans(n_clusters=len(cluster_centers), n_init=10, init='k-means++')
kmeans.fit(data)

# Step 3: Select one representative point per cluster
unique_points = []
for i in range(len(cluster_centers)):
    # Get points in the cluster and find the one closest to the centroid
    cluster_points = data[kmeans.labels_ == i]
    centroid = kmeans.cluster_centers_[i]
    closest_point = cluster_points[np.argmin(np.linalg.norm(cluster_points - centroid, axis=1))]
    unique_points.append(closest_point)
unique_points = np.array(unique_points)

with open('Challenge Info\hybrid_set_1.csv', 'w') as csvfile:
   writer = csv.writer(csvfile)
   for i in range(len(unique_points)):
        # Get the closest point in the original data to the representative point
        cluster_points = data[kmeans.labels_ == i]
        centroid = kmeans.cluster_centers_[i]
        
        # Find the closest point in the cluster to the centroid
        closest_point_index = np.argmin(np.linalg.norm(cluster_points - centroid, axis=1))
        closest_point = cluster_points[closest_point_index]
        
        # The corresponding index from the original data
        original_index = index[np.where((data == closest_point).all(axis=1))[0][0]]
        
        # Write the original index and the unique point to the CSV
        writer.writerow([original_index, closest_point[0], closest_point[1]])
# Optional: Visualize the active cells within the Convex Hull
plt.scatter(unique_points[:, 0], unique_points[:, 1], s=1, alpha=0.8, label="Data Points")
plt.ylim([0, 1])  # Setting y-axis limits
plt.xlim([0.2, 1.2])  # Setting x-axis limits
#plt.legend()
#plt.show()
# for (row, col) in grid_cells.keys():
#     plt.plot(
#         [x_min + col * cell_width, x_min + (col + 1) * cell_width],
#         [y_min + row * cell_height, y_min + row * cell_height],
#         color='grey', alpha=0.3
# )
#plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', lw=2, label="Convex Hull")
#plt.legend()
plt.show()

# Define the grid
# grid_bins = 50

# # Compute the 2D histogram to get the frequencies
# hist, x_edges, y_edges = np.histogram2d(unique_points[:,0], unique_points[:,1], bins=[grid_bins, grid_bins], range=[[0.2,1.2], [0,1]])

# Prepare data for 3D bar plot
# x_pos, y_pos = np.meshgrid(x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2, y_edges[:-1] + (y_edges[1] - y_edges[0]) / 2, indexing="ij")
# x_pos = x_pos.ravel()
# y_pos = y_pos.ravel()
# z_pos = np.zeros_like(x_pos)
# dx = dy = (x_edges[1] - x_edges[0])  # Width of each bar
# dz = hist.ravel()  # Frequency counts as the height of each bar

# non_zero_mask = dz > 0  # Create a mask for non-zero frequencies
# x_pos, y_pos = np.meshgrid(
#     x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2,  # Center positions on x-axis
#     y_edges[:-1] + (y_edges[1] - y_edges[0]) / 2,  # Center positions on y-axis
#     indexing="ij"
# )
# x_pos = x_pos.ravel()[non_zero_mask]
# y_pos = y_pos.ravel()[non_zero_mask]
# z_pos = np.zeros_like(x_pos)  # Start bars at z=0

# # Filter dx, dy, and dz based on the non-zero mask
# dx = dy = (x_edges[1] - x_edges[0])  # Width and depth of each bar
# dz = dz [non_zero_mask]  # Filtered heights with only non-zero values

# # Step 3: Create the 3D histogram plot without zero-frequency bars
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Plot only non-zero frequency bars
# ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, shade=True)

# # Set labels and title
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('Relative Frequency')
# ax.set_title('3D Histogram with Non-Zero Frequencies Only')

# plt.show()

# # Calculate the total number of points being plotted
# total_points_plotted = np.sum(dz)
# print(f"Total number of points being plotted: {total_points_plotted}")