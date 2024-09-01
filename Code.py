import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Path to the NetCDF file
file_path = '/Users/prashant/WRF/6c38965082355c2710a2a877134d5750/C3S-312bL1-L3C-MONTHLY-CLOUD-ATSR2_ORAC_ERS2_200308_fv3.0.nc'

try:
    # Open the NetCDF file
    nc = Dataset(file_path, mode='r')
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

print(nc.variables.keys())  # Print all available variables in the NetCDF file

try:
    # Extract the correct latitude, longitude, and time variables based on your inspection
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    # There is no 'time' variable, so we'll skip it for now
except KeyError as e:
    print(f"Error: Variable not found - {e}")
    exit()

try:
    # Extract the data variables
    land_mask = nc.variables['pixel_count'][:]  # Assuming pixel_count is the land mask
    surface_pressure = nc.variables['ctp'][:]
    surface_temperature = nc.variables['ctt'][:]
except KeyError as e:
    print(f"Error: Variable not found - {e}")
    exit()

# Close the NetCDF file
nc.close()

# Plot the land mask
plt.figure(figsize=(10, 6))
plt.contourf(lon, lat, land_mask, cmap='gray')
plt.title('Land Mask')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Land (1) / Water (0)')
plt.show()

# Plot the surface temperature
plt.figure(figsize=(10, 6))
plt.contourf(lon, lat, surface_temperature, cmap='coolwarm')  # Assuming ctt is in Kelvin
plt.title('Surface Temperature (K)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Temperature (K)')
plt.show()

# Plot the surface pressure
plt.figure(figsize=(10, 6))
plt.contourf(lon, lat, surface_pressure, cmap='viridis')  # Assuming ctp is in Pa
plt.title('Surface Pressure (Pa)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Pressure (Pa)')
plt.show()