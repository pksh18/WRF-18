import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Metadata
tile_x = 4320
tile_y = 2160
wordsize = 1  # Byte size of each data point (e.g., 1 byte = 8 bits)
scale_factor = 0.5
units = "percent"
description = "irrigated land percentage"

# File paths
input_file_path = '/Users/prashant/WRF/irrigation/fao/00001-04320.00001-02160'  # Replace with the correct file path
output_file_path = '/Users/prashant/WRF/irrigation/fao.np'  # Replace with your desired output path

# Read the binary file
data = np.fromfile(input_file_path, dtype=np.uint8)  # Assuming wordsize=1 corresponds to np.uint8

# Reshape the data to match the grid size (tile_x by tile_y)
data = data.reshape((tile_y, tile_x))

# Apply the scale factor
data = data * scale_factor

# Save the processed data
np.save(output_file_path, data)

# Plot 1: Full Data Visualization
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='viridis', extent=[-179.95833, 179.95833, -89.95833, 89.95833])
plt.colorbar(label=f'Irrigated Land ({units})')
plt.title(description)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Plot 2: Histogram of Data Values
plt.figure(figsize=(10, 6))
plt.hist(data.ravel(), bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Irrigated Land Percentages')
plt.xlabel(f'Irrigated Land ({units})')
plt.ylabel('Frequency')
plt.show()

# Plot 3: Zoomed-In Section of the Map (Example: Lat: -20 to 20, Lon: -60 to 60)
zoom_lat_min, zoom_lat_max = -20, 20
zoom_lon_min, zoom_lon_max = -60, 60
zoomed_data = data[int((zoom_lat_min + 90) / 180 * tile_y):int((zoom_lat_max + 90) / 180 * tile_y),
                   int((zoom_lon_min + 180) / 360 * tile_x):int((zoom_lon_max + 180) / 360 * tile_x)]

plt.figure(figsize=(10, 8))
plt.imshow(zoomed_data, cmap='viridis', extent=[zoom_lon_min, zoom_lon_max, zoom_lat_min, zoom_lat_max])
plt.colorbar(label=f'Irrigated Land ({units})')
plt.title(f'Zoomed-In: {description} (Lat: {zoom_lat_min} to {zoom_lat_max}, Lon: {zoom_lon_min} to {zoom_lon_max})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Plot 4: 3D Surface Plot of a Subsection (Reducing data size for performance)
subsample_factor = 10  # Adjust to balance resolution and performance
x = np.linspace(-179.95833, 179.95833, tile_x)[::subsample_factor]
y = np.linspace(-89.95833, 89.95833, tile_y)[::subsample_factor]
X, Y = np.meshgrid(x, y)
Z = data[::subsample_factor, ::subsample_factor]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title('3D Surface Plot of Irrigated Land')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel(f'Irrigated Land ({units})')

plt.show()

print(f"Data processed and saved to {output_file_path}.")
