import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Load WRF output file
wrf_file = 'wrfout_d01_2020-08-29_00:00:00'
nc = Dataset(wrf_file)

# Extract variables
times = nc.variables['Times'][:]  # Time array
temp = nc.variables['T2'][:]      # 2-meter temperature
lats = nc.variables['XLAT'][0, :, :]  # Latitude array
lons = nc.variables['XLONG'][0, :, :] # Longitude array

# Select a specific time step to plot
time_idx = 0

# Plot the 2-meter temperature
plt.figure(figsize=(12, 8))
plt.contourf(lons, lats, temp[time_idx, :, :] - 273.15, cmap='coolwarm')
plt.colorbar(label='2m Temperature (°C)')
plt.title(f'WRF 2m Temperature - {str(times[time_idx], "utf-8")}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
# Close the NetCDF file
nc.close()
