import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Path to the NetCDF file
file_path = '/Users/prashant/WRF/6c38965082355c2710a2a877134d5750/C3S-312bL1-L3C-MONTHLY-CLOUD-ATSR2_ORAC_ERS2_200308_fv3.0.nc'

# Open the NetCDF file
nc = Dataset(file_path, mode='r')


print(nc.variables.keys())  # Print all available variables in the NetCDF file

try:
    # Extract the data variables
    land_mask = nc.variables['pixel_count'][:]  # Assuming pixel_count is the land mask
    surface_pressure = nc.variables['ctp'][:]
    surface_temperature = nc.variables['ctt'][:]
except KeyError as e:
    print(f"Error: Variable not found - {e}")
    exit()