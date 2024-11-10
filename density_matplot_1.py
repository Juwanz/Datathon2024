import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the selected 2500 points from the CSV file, excluding the header
df_selected = pd.read_csv('selected_points_original_data1.csv', header=0)  # Skip the first row as header is '0,1,2'

# Extract frequency and power columns (assuming columns 1 and 2 correspond to frequency and power)
frequency_data = df_selected.iloc[:, 1]  # Frequency column (2nd column)
power_data = df_selected.iloc[:, 2]      # Power column (3rd column)

# Set up the bin edges for frequency and power (adjust as needed based on your data range)
frequency_bins = np.linspace(frequency_data.min(), frequency_data.max(), 10)  # 10 bins for frequency
power_bins = np.linspace(power_data.min(), power_data.max(), 10)  # 10 bins for power

# Digitize the data to determine the bin for each point in frequency and power
frequency_indices = np.digitize(frequency_data, frequency_bins) - 1
power_indices = np.digitize(power_data, power_bins) - 1

# Initialize a 2D array to store counts for each bin
histogram_counts = np.zeros((len(frequency_bins) - 1, len(power_bins) - 1))

# Count the number of points in each bin
for freq_idx, power_idx in zip(frequency_indices, power_indices):
    if 0 <= freq_idx < histogram_counts.shape[0] and 0 <= power_idx < histogram_counts.shape[1]:
        histogram_counts[freq_idx, power_idx] += 1

# Get the center points of each bin for plotting
x_centers = (frequency_bins[:-1] + frequency_bins[1:]) / 2
y_centers = (power_bins[:-1] + power_bins[1:]) / 2
x_pos, y_pos = np.meshgrid(x_centers, y_centers)
x_pos = x_pos.flatten()
y_pos = y_pos.flatten()
z_pos = np.zeros_like(x_pos)  # Start bars from z=0

# Set the height (z) of each bar based on the count of points in each bin
dz = histogram_counts.flatten()

# Define narrower width and depth for each bar in the histogram
dx = (frequency_bins[1] - frequency_bins[0]) * 0.4  # Make bars narrower in x direction
dy = (power_bins[1] - power_bins[0]) * 0.4          # Make bars narrower in y direction

# Filter out the bins where dz is zero
nonzero_indices = dz > 0
x_pos = x_pos[nonzero_indices]
y_pos = y_pos[nonzero_indices]
z_pos = z_pos[nonzero_indices]
dz = dz[nonzero_indices]

# Plot the 3D histogram
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D bar chart
ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color='b', zsort='average')

# Set labels and title
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (Watts)')
ax.set_zlabel('Number of Points')
ax.set_title('3D Histogram of 2500 Selected Points (Non-zero Counts Only)')

# Show plot
plt.show()

# Check total points count in histogram for validation
print("Total points in histogram:", int(dz.sum()))
