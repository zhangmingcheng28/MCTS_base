# -*- coding: utf-8 -*-
"""
@Time    : 2/9/2025 4:13 PM
@Author  : Mingcheng
@FileName:
@Description:
@Package dependency:
"""
import pyvista as pv
import matplotlib.pyplot as plt
from Utilities import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from Utilities import is_cell_occupied
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

# Define the grid increments
dx, dy, dz = 12, 8, 0.5

# Create arrays of grid points including the last point (endpoint)
x_points = np.arange(0, 600 + dx, dx)  # 0 to 600, step 12 -> 51 points
y_points = np.arange(0, 400 + dy, dy)  # 0 to 400, step 8  -> 51 points
z_points = np.arange(0, 50 + dz, dz)     # 0 to 50, step 0.5 -> 101 points

# Total number of cells (boxes) will be one less in each direction:
num_cells_x = len(x_points) - 1  # 50
num_cells_y = len(y_points) - 1  # 50
num_cells_z = len(z_points) - 1  # 100


# Number of cells in each direction (computed previously)
nx = 50  # number of cells in x
ny = 50  # number of cells in y
nz = 100 # number of cells in z

# Create coordinate arrays for the lower corner of each cell
x_coords = np.arange(0, nx * dx, dx)  # From 0 to 600 (exclusive of endpoint 600 because 600 is the upper bound of the grid points)
y_coords = np.arange(0, ny * dy, dy)  # From 0 to 400
z_coords = np.arange(0, nz * dz, dz)    # From 0 to 50

# Use np.meshgrid to create a grid of coordinates; 'ij' indexing ensures the order is (x, y, z)
X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

# Stack the coordinate arrays so each cell holds the coordinate triplet (x, y, z)
# grid [0,0,0,0], retrieves the x-coordinate of the lower corner of that grid cell. In our case, since the grid starts at [0, 0, 0], the value is 0.
# grid [0,0,0,1], accesses the y-coordinate of the lower corner of the grid cell at indices (0, 0, 0)
# grid [0,0,0,:], access the exact coordinate values, of the grid cell at indices (0,0,0)
grid = np.stack((X, Y, Z), axis=-1)  # grid.shape will be (50, 50, 100, 3)  grid[:,:,:,0] extracts the x-coordinate of the point where each cell begins along the x-axis.

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

actionSpace_path_gama = [-10, 0, 10]  # in degree
actionSpace_heading_chi = [-25, -15, 0, 15, 25]  # in degree
all_actions = [(gama, chi) for gama in actionSpace_path_gama for chi in actionSpace_heading_chi]
pt = [402.0, 108.0, 35.25]
pt_v = 10  # m/s
dt = 3
collection_of_next_point = []
for action_path_gama, action_heading_chi in all_actions:
    host_vx, host_vy, host_vz = decompose_speed_to_three_axis(pt_v, action_path_gama, action_heading_chi)
    n_x = pt[0]+host_vx*dt
    n_y = pt[1]+host_vy*dt
    n_z = pt[2]+host_vz*dt
    collection_of_next_point.append([n_x, n_y, n_z])

# # Create the figure and 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Draw each building
for poly in building_polygons:
    # Create the walls of the building
    for i in range(len(poly)):
        x = [poly[i][0], poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][0], poly[i][0]]
        y = [poly[i][1], poly[(i + 1) % len(poly)][1], poly[(i + 1) % len(poly)][1], poly[i][1]]
        z = [0, 0, poly[(i + 1) % len(poly)][2], poly[i][2]]  # Extend from ground (z=0) to building height

        ax.add_collection3d(Poly3DCollection([list(zip(x, y, z))], color='green', alpha=0.3, edgecolor='none'))

        # Add the wall as a 3D polygon
        # ax.add_collection3d(Poly3DCollection([list(zip(x, y, z))], color='green', alpha=0.1, edgecolors='black', linewidths=1))

    # # Create the roof of the building
    roof = [[point[0], point[1], point[2]] for point in poly]
    ax.add_collection3d(Poly3DCollection([roof], color='green', alpha=0.1))

    # Scatter the points
    ax.scatter(pt[0], pt[1], pt[2], color='red')
    for pts in collection_of_next_point:
        ax.scatter(pts[0], pts[1], pts[2], color='blue')

# Set limits in the 3D plot
ax.set_xlim(x_limit)
ax.set_ylim(y_limit)
ax.set_zlim(0, max([max(poly, key=lambda p: p[2])[2] for poly in building_polygons]) + 10)

# # Set viewing angle
# ax.view_init(elev=30, azim=45)  # Adjust angles for better visibility

# Labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Height')

plt.show()
print("end")


