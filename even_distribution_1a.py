import numpy as np
import csv
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
# Coordinates for datapoints imported
x1_vals =[]
x2_vals = []
index = []
domains = {}
with open('Challenge Info\data_set_1.csv', 'r', newline='') as data:
    map = csv.reader(data, delimiter=',')
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
approx_cells = 19200
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
kmeans = KMeans(n_clusters=len(cluster_centers), init=np.array(cluster_centers), n_init=1)
kmeans.fit(data)
# Step 3: Select one representative point per cluster
grid_box_size = cell_width * 0.99  # Adjust 0.9 to your preference, ensuring it's within bounds
print(f"Grid box size: {grid_box_size}")
# Step 5: Adjust the selection of unique points for grid cells
unique_points = []
for (row, col) in active_grid:
    points_in_cell = grid_cells[(row, col)]
    if points_in_cell:
        # Randomly select a point from this cell
        selected_point = points_in_cell[np.random.choice(len(points_in_cell))]
        unique_points.append(selected_point)
# Convert to numpy array
unique_points = np.array(unique_points)
# Define the grid
grid_bins = 50
# Compute the 2D histogram to get the frequencies
hist, x_edges, y_edges = np.histogram2d(unique_points[:,0], unique_points[:,1], bins=[grid_bins, grid_bins], range=[[0.2,1.2], [0,1]])
# Prepare data for 3D bar plot
x_pos, y_pos = np.meshgrid(x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2, y_edges[:-1] + (y_edges[1] - y_edges[0]) / 2, indexing="ij")
x_pos = x_pos.ravel()
y_pos = y_pos.ravel()
z_pos = np.zeros_like(x_pos)
dx = dy = (x_edges[1] - x_edges[0])  # Width of each bar
dz = hist.ravel()  # Frequency counts as the height of each bar
# Step 7: Select only points in populated bins (non-zero frequencies)
non_zero_indices = np.where(dz > 0)[0]
selected_points = []
for i in non_zero_indices:
    x_center = x_pos[i]
    y_center = y_pos[i]
    # Find points close to this bin center
    mask = (np.abs(unique_points[:, 0] - x_center) < dx/2) & (np.abs(unique_points[:, 1] - y_center) < dy/2)
    selected_points.extend(unique_points[mask])
# Trim selected_points to exactly 2500 if necessary
selected_points = np.array(selected_points)
# Step 8: Save the selected points to 'normal_set_2.csv'
with open('Challenge Info\density_set_1.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    for point in selected_points:
        # Find the original index in the dataset
        original_index = index[np.where((data == point).all(axis=1))[0][0]]
        writer.writerow([original_index, point[0], point[1]])
print(f"Total number of points saved in CSV: {len(selected_points)}")
plt.title("K-means Clustering with Unique Points")
plt.scatter(unique_points[:, 0], unique_points[:, 1], s=1, alpha=0.8, label="Data Points")
plt.ylim([0, 1])  # Setting y-axis limits
plt.xlim([0.2, 1.2])  # Setting x-axis limits
plt.show()
# Visualize the active cells within the Convex Hull
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Plot only non-zero frequency bars
ax.bar3d(x_pos[non_zero_indices], y_pos[non_zero_indices], z_pos[non_zero_indices], dx, dy, dz[non_zero_indices], shade=True)
# Set labels and title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('')
ax.set_title('Dataset 1 Uniform Selection')
plt.show()
# Calculate the total number of points being plotted
total_points_plotted = np.sum(dz)
print(f"Total number of points being plotted: {total_points_plotted}")