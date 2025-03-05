# -*- coding: utf-8 -*-
"""
@Time    : 3/3/2025 2:15 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import time

import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Utilities import is_cell_occupied
from matplotlib.path import Path
from scipy.io import loadmat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend for pop-up windows
# Load the .mat file
data = loadmat('building.mat')
building_coord = data['Buildings']

# Access individual fields in the struct
positions = building_coord[0]['pos']
heights = building_coord[0]['height']
# Input boundaries (longitude and latitude)
x1, y1 = 103.7460, 1.3230  # Reference point
x2, y2 = 103.7513, 1.3265  # Boundary point

# Approximate scaling factors
lat_to_m = 111320  # 1 degree latitude in meters
lon_to_m = np.cos(np.radians(y1)) * 111320  # 1 degree longitude in meters

# Convert coordinates to meters relative to (x1, y1)
x_m = (x2 - x1) * lon_to_m
y_m = (y2 - y1) * lat_to_m

x_min = 0  # x1 is the reference point
x_max = (x2 - x1) * lon_to_m
y_min = 0  # y1 is the reference point
y_max = (y2 - y1) * lat_to_m
x_limit = (x_min - 10, x_max + 10)
y_limit = (y_min - 10, y_max + 10)

# Convert building_coord to the building_polygons format
building_polygons = []
# Dictionary to store the maximum height for each unique 2D polygon
unique_polygons = {}

for b_coords, b_height in zip(positions, heights):  # Iterate over buildings (assuming the outermost structure is a list/array)
    # Convert to a list of lists with the format [x, y, z]
    building_polygon = []
    base_2d = []  # To store the 2D base coordinates of the polygon
    for xy_coord in b_coords.T:
        if np.isnan(xy_coord).any():
            continue
        lon = xy_coord[1]
        lat = xy_coord[0]
        x_meter = (lon - x1) * lon_to_m
        y_meter = (lat - y1) * lat_to_m
        if x_meter > x_max or y_meter > y_max or x_meter < x_min or y_meter < y_min:
            break
        base_2d.append((x_meter, y_meter))  # Add to the base coordinates
        building_polygon.append([x_meter, y_meter, b_height[0, 0]])

    # Convert the 2D base to a tuple for use as a dictionary key
    base_2d_tuple = tuple(sorted(base_2d))  # Sort to ensure consistency

    # Check if this base exists and keep only the polygon with the highest height
    if base_2d_tuple not in unique_polygons or unique_polygons[base_2d_tuple][0] < b_height[0, 0]:
        unique_polygons[base_2d_tuple] = (b_height[0, 0], building_polygon)


# Extract the final list of polygons with the highest height
building_polygons = [v[1] for v in unique_polygons.values()]
building_polygons = [poly for poly in building_polygons if poly]  # remove empty list

# ====================================================
# Precomputed grid parameters (from your code)
# ====================================================
dx, dy, dz = 12, 8, 0.5
nx, ny, nz = 50, 50, 100

# Create 1D arrays for lower corner coordinates along each axis.
x_coords = np.arange(0, nx * dx, dx)  # e.g., 0, 12, 24, ... 588
y_coords = np.arange(0, ny * dy, dy)  # e.g., 0, 8, 16, ... 392
z_coords = np.arange(0, nz * dz, dz)    # e.g., 0, 0.5, 1.0, ... 49.5

# Precompute cell centroids along each axis.
x_centers = x_coords + dx / 2.0  # shape: (nx,)
y_centers = y_coords + dy / 2.0  # shape: (ny,)
z_centers = z_coords + dz / 2.0  # shape: (nz,)

# ====================================================
# Create an occupancy array to hold 1 (occupied) or 0 (empty)
# ====================================================
occupancy_array = np.zeros((nx, ny, nz), dtype=np.int8)

start_time = time.time()
# ====================================================
# Loop over each building polygon and mark cells as occupied.
# ====================================================
# Assume building_polygons is a list where each element is a list of vertices:
# each vertex is [x, y, building_height]
for poly in building_polygons:
    # Get the building height from the first vertex.
    building_height = poly[0][2]
    # Create a 2D polygon (only x and y) for the building footprint.
    poly_xy = [(pt[0], pt[1]) for pt in poly]
    path = Path(poly_xy)

    # Determine the bounding box of the polygon.
    poly_x = [pt[0] for pt in poly_xy]
    poly_y = [pt[1] for pt in poly_xy]
    min_x, max_x = min(poly_x), max(poly_x)
    min_y, max_y = min(poly_y), max(poly_y)

    # Find the x and y indices whose centers fall within the bounding box.
    i_indices = np.where((x_centers >= min_x) & (x_centers <= max_x))[0]
    j_indices = np.where((y_centers >= min_y) & (y_centers <= max_y))[0]

    # Loop through the candidate (i, j) cells.
    for i in i_indices:
        for j in j_indices:
            # Check if the cell's center is inside the building's footprint.
            if path.contains_point((x_centers[i], y_centers[j])):
                # Find the maximum k index for which the cell's center is below the building height.
                k_max = np.searchsorted(z_centers, building_height, side='right')
                # Mark all cells in the vertical direction up to k_max as occupied.
                occupancy_array[i, j, :k_max] = 1

# ====================================================
# Convert the occupancy array into a dictionary for quick lookup.
# Key: (x_center, y_center, z_center) of the cell (tuple); Value: 1 (occupied) or 0 (empty)
occupancy_dict = {}
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            key = (x_centers[i], y_centers[j], z_centers[k])
            occupancy_dict[key] = int(occupancy_array[i, j, k])

computational_time = time.time()-start_time
print('end')