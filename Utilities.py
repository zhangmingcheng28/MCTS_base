# -*- coding: utf-8 -*-
"""
@Time    : 3/3/2025 2:05 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from shapely.geometry import Point, LineString
import numpy as np
import math
import itertools
from matplotlib.path import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def generate_start_end_positions(candidate_points, min_distance=400, boundary_margin=7.5, boundaries=None):
    """
    Efficiently generates a random start and end position for the host UAV.

    Requirements:
      - The two positions are at least 'min_distance' meters apart (Euclidean 3D distance).
      - Both positions must lie in free cells (occupancy == 0).
      - Both positions must be at least 'boundary_margin' away from the airspace boundaries.
      - No fixed altitude is assumed; all free cells across altitudes are candidates.

    Parameters:
      candidate_points (list): free points
      min_distance (float): Minimum required Euclidean distance between start and end positions.
      boundary_margin (float): Minimum required distance from any airspace boundary.
      boundaries (tuple): A tuple defining the airspace boundaries as
                          (x_min, x_max, y_min, y_max, z_min, z_max).
                          If None, no boundary filtering is applied.

    Returns:
      tuple: (start_point, end_point) where each is a tuple (x, y, z)

    Raises:
      ValueError: If no candidate free points exist or if no valid end point can be found.
    """
    # If boundaries are provided, filter out candidate points too close to boundaries.
    if boundaries is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = boundaries
        valid_mask = (
                (candidate_points[:, 0] >= x_min + boundary_margin) &
                (candidate_points[:, 0] <= x_max - boundary_margin) &
                (candidate_points[:, 1] >= y_min + boundary_margin) &
                (candidate_points[:, 1] <= y_max - boundary_margin) &
                (candidate_points[:, 2] >= z_min + boundary_margin) &
                (candidate_points[:, 2] <= z_max - boundary_margin)
        )
        candidate_points = candidate_points[valid_mask]

    if candidate_points.shape[0] == 0:
        raise ValueError("No candidate free points remain after boundary filtering.")

    # Randomly choose one candidate as the start point.
    idx_start = np.random.choice(candidate_points.shape[0])
    start_point = candidate_points[idx_start]

    # Compute Euclidean distances (in 3D) from start_point to all candidate points.
    diffs = candidate_points - start_point
    dists = np.linalg.norm(diffs, axis=1)

    # Find indices where distance is at least min_distance.
    valid_indices = np.where(dists >= min_distance)[0]
    if valid_indices.size == 0:
        raise ValueError(
            "No valid end point found that is at least {} meters away from the start point.".format(min_distance))

    # Randomly select one valid candidate as the end point.
    idx_end = np.random.choice(valid_indices)
    end_point = candidate_points[idx_end]

    return tuple(start_point), tuple(end_point)


def plot_trajectories(building_polygons, host_trajectory, intruder_trajectory, x_limit, y_limit):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw each building
    for poly in building_polygons:
        # Create the walls of the building
        for i in range(len(poly)):
            x = [poly[i][0], poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][0], poly[i][0]]
            y = [poly[i][1], poly[(i + 1) % len(poly)][1], poly[(i + 1) % len(poly)][1], poly[i][1]]
            z = [0, 0, poly[(i + 1) % len(poly)][2], poly[i][2]]  # Extend from ground (z=0) to building height

            ax.add_collection3d(
                Poly3DCollection([list(zip(x, y, z))], color='green', alpha=0.3, edgecolor='none'))

        # Create the roof of the building
        roof = [[point[0], point[1], point[2]] for point in poly]
        ax.add_collection3d(Poly3DCollection([roof], color='green', alpha=0.1))

    # Plot the host UAV's historical trajectory
    if host_trajectory:
        host_traj = np.array(host_trajectory)
        ax.plot(host_traj[:, 0], host_traj[:, 1], host_traj[:, 2], color='blue', marker='o',
                label='Host Trajectory')

        # Mark first and last points for the host UAV
        ax.scatter(host_traj[0, 0], host_traj[0, 1], host_traj[0, 2], color='cyan', s=100, label='Host Start',
                   marker='D')
        ax.scatter(host_traj[-1, 0], host_traj[-1, 1], host_traj[-1, 2], color='darkblue', s=100,
                   label='Host End', marker='X')

    # Plot the intruder UAV's historical trajectory
    if intruder_trajectory:
        intruder_traj = np.array(intruder_trajectory)
        ax.plot(intruder_traj[:, 0], intruder_traj[:, 1], intruder_traj[:, 2], color='red', marker='o',
                label='Intruder Trajectory')

        # Mark first and last points for the intruder UAV
        ax.scatter(intruder_traj[0, 0], intruder_traj[0, 1], intruder_traj[0, 2], color='orange', s=100,
                   label='Intruder Start', marker='D')
        ax.scatter(intruder_traj[-1, 0], intruder_traj[-1, 1], intruder_traj[-1, 2], color='maroon', s=100,
                   label='Intruder End', marker='X')

    # Set limits in the 3D plot
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)
    ax.set_zlim(0, max([max(poly, key=lambda p: p[2])[2] for poly in building_polygons]) + 10)
    # Labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    # Add legend to distinguish trajectories
    ax.legend()
    plt.show()


def max_velocity_difference(V, V_desired, gamma_chi_combination):
    """
    Finds the maximum possible velocity deviation given discrete action spaces.

    Parameters:
        V (float): UAV speed (constant)
        V_desired (tuple): Desired velocity (Vx', Vy', Vz')
        actionSpace_path_gama (list): Discrete possible path angles (degrees)
        actionSpace_heading_chi (list): Discrete possible heading angles (degrees)

    Returns:
        max_delta_V (float): Maximum velocity deviation
        best_gamma (float): Best path angle achieving max deviation
        best_chi (float): Best heading angle achieving max deviation
    """
    max_delta_V = 0
    best_gamma = None
    best_chi = None
    largest_vel_diff = None

    # Iterate through all possible (gamma, chi) pairs
    for gamma_degree, chi_degree in gamma_chi_combination:
        # Compute UAV velocity for this (gamma, chi)
        V_current = decompose_speed_to_three_axis(V, gamma_degree, chi_degree)

        # Compute deviation
        delta_V = np.linalg.norm(np.array(V_desired) - np.array(V_current))
        vel_diff = np.array(V_desired) - np.array(V_current)

        # Check if it's the maximum deviation
        if delta_V > max_delta_V:
            max_delta_V = delta_V
            largest_vel_diff = vel_diff
            best_gamma = gamma_degree
            best_chi = chi_degree

    return largest_vel_diff


def normalize_dot_product(A, B):
    """
    Computes the normalized dot product between two vectors A and B, ensuring the result is in the range [0,1].

    Parameters:
        A (numpy array): First vector
        B (numpy array): Second vector

    Returns:
        normalized_dot (float): Normalized dot product in [0,1]
    """
    # Compute the dot product
    dot_product = np.dot(A, B)

    # Compute the magnitudes (norms) of the vectors
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    # Compute cosine of the angle
    if norm_A == 0 or norm_B == 0:
        return 0  # Avoid division by zero; return 0 if any vector is zero

    cos_theta = dot_product / (norm_A * norm_B)

    # Normalize to [0,1]
    normalized_dot = (1 + cos_theta) / 2

    return normalized_dot


def check_collision_with_grid(position, radius, occupancy_dict, dx, dy, dz):
    """
    Efficiently checks if a UAV's sphere intersects an occupied grid cell.

    Instead of looping over all 27 neighbors, we only check cells that are within the UAV's sphere.

    Parameters:
        position: (x, y, z) UAV's position
        radius: UAV's collision sphere radius (7.5m)
        occupancy_dict: Dictionary with grid occupancy status
        dx, dy, dz: Grid cell sizes

    Returns:
        True if UAV collides with an occupied grid, otherwise False.
    """
    x, y, z = position  # actual uav position
    is_occupied, cell_centre_encloses_actual_position = is_point_in_occupied_cell(position, dx, dy, dz, occupancy_dict)
    if is_occupied:
        return True  # the current position of the UAV is already occupied
    # Compute the number of cells that need to be checked in each axis
    x_range = int(np.ceil(radius / dx))  # Number of grid steps to cover UAV radius in x
    y_range = int(np.ceil(radius / dy))  # Number of grid steps to cover UAV radius in y
    z_range = int(np.ceil(radius / dz))  # Number of grid steps to cover UAV radius in z

    # Iterate only through the required grid cells within the UAV's sphere
    for i in range(-x_range, x_range + 1):
        for j in range(-y_range, y_range + 1):
            for k in range(-z_range, z_range + 1):
                # Compute the center of the neighboring grid cell
                cell_x = cell_centre_encloses_actual_position[0]
                cell_y = cell_centre_encloses_actual_position[1]
                cell_z = cell_centre_encloses_actual_position[2]

                # Check if the cell is occupied
                if (cell_x, cell_y, cell_z) in occupancy_dict and occupancy_dict[(cell_x, cell_y, cell_z)] == 1:
                    # Compute distance from UAV center to this occupied cell
                    distance_to_cell = math.sqrt((x - cell_x) ** 2 + (y - cell_y) ** 2 + (z - cell_z) ** 2)

                    # If the occupied cell is within the UAV's sphere, return True
                    if distance_to_cell < radius:
                        return True  # Collision detected

    return False  # No collision


def check_if_reach_goal(host_position, original_host_position, target_x, target_y, target_z):
    host_travel_line = LineString([host_position[0:2], original_host_position[0:2]])
    # Buffer the LineString with a radius of 7.5
    travel_buffered_area = host_travel_line.buffer(7.5)
    # Define a goal point (x3, y3)
    target_point = Point(target_x, target_y)
    # Check if the third point is within the buffered area
    is_intersecting = travel_buffered_area.intersects(target_point)
    if is_intersecting:
        if target_z >= original_host_position[2] and target_z <= host_position[2]:
            # height of the target is above original position
            return 1
        elif target_z <= original_host_position[2] and target_z >= host_position[2]:
            # height of the target is below original position
            return 1
    else:
        return 0


def get_original_position(current_pos, speed, dt, gamma_deg, chi_deg):
    """
    Given the current position of the UAV, the constant speed, the time step,
    and the discrete action angles (path angle gamma and heading angle chi),
    this function computes the original position before the UAV took the action.

    Parameters:
        current_pos (tuple): Current UAV position as (x, y, z).
        speed (float): Constant speed of the UAV.
        dt (float): Time step duration.
        gamma_deg (float): Path angle in degrees.
        chi_deg (float): Heading angle in degrees.

    Returns:
        tuple: The original position (x, y, z) before the action.
    """
    # Convert angles from degrees to radians
    gamma_rad = math.radians(gamma_deg)
    chi_rad = math.radians(chi_deg)

    # Compute the velocity components based on the given formulas
    Vx = speed * math.cos(gamma_rad) * math.cos(chi_rad)
    Vy = speed * math.cos(gamma_rad) * math.sin(chi_rad)
    Vz = speed * math.sin(gamma_rad)

    # Compute the displacement over the time step
    dx = Vx * dt
    dy = Vy * dt
    dz = Vz * dt

    # Compute and return the original position (by subtracting the displacement)
    original_x = current_pos[0] - dx
    original_y = current_pos[1] - dy
    original_z = current_pos[2] - dz

    return (original_x, original_y, original_z)


def decompose_speed_to_three_axis(speed, action_path_gama_degree, action_heading_chi_degree):
    action_path_gama_rad = deg2rad(action_path_gama_degree)
    action_heading_chi_rad = deg2rad(action_heading_chi_degree)
    Vx = speed * math.cos(action_path_gama_rad) * math.cos(action_heading_chi_rad)
    Vy = speed * math.cos(action_path_gama_rad) * math.sin(action_heading_chi_rad)
    Vz = speed * math.sin(action_path_gama_rad)
    return Vx, Vy, Vz


def deviation_from_path(host_position, path_start, path_end):
    """
    Calculate the deviation (perpendicular distance) between the host's current position and
    its desired straight-line path, defined by path_start and path_end. Also returns the
    closest point on the path.

    Parameters:
      host_position : tuple or list of (x, y, z) for the host's current continuous position.
      path_start    : tuple or list of (x, y, z) for the start point of the desired path.
      path_end      : tuple or list of (x, y, z) for the end point of the desired path.

    Returns:
      deviation   : The perpendicular distance from host_position to the line (in meters).
      closest_pt  : The (x, y, z) coordinate on the desired path that is closest to host_position.

    Note:
      If the start and end points are the same, the deviation is simply the distance between
      host_position and path_start.
    """
    host = np.array(host_position, dtype=float)
    start = np.array(path_start, dtype=float)
    end = np.array(path_end, dtype=float)

    # Vector along the desired path.
    AB = end - start
    AB_norm_sq = np.dot(AB, AB)

    if AB_norm_sq == 0:
        # Path start and end are the same.
        closest_pt = start
        deviation = np.linalg.norm(host - start)
        return deviation, tuple(closest_pt)

    # Projection factor (t) along AB.
    t = np.dot(host - start, AB) / AB_norm_sq

    # Compute the closest point on the infinite line.
    closest_pt = start + t * AB

    # Perpendicular distance from host to the line.
    deviation = np.linalg.norm(host - closest_pt)

    return deviation, tuple(closest_pt)


def relative_distance(host_pos, intruder_pos):
    """
    Calculate the relative distance along each axis (x, y, z) between host and intruder UAV.

    Parameters:
        host_pos: A tuple or list (x, y, z) representing the host UAV's continuous position.
        intruder_pos: A tuple or list (x, y, z) representing the intruder UAV's continuous position.

    Returns:
        A tuple (dx, dy, dz) where:
            dx = host_pos[0] - intruder_pos[0]
            dy = host_pos[1] - intruder_pos[1]
            dz = host_pos[2] - intruder_pos[2]
    """
    dx = host_pos[0] - intruder_pos[0]
    dy = host_pos[1] - intruder_pos[1]
    dz = host_pos[2] - intruder_pos[2]
    return (dx, dy, dz)


def deg2rad(deg):
    """Convert degrees to radians."""
    return deg * math.pi / 180.0


def desired_velocity(current, goal, speed):
    """
    Compute the desired velocity vector given the current position, goal, and constant speed.
    Both `current` and `goal` are lists or arrays of [x, y, z].
    """
    current = np.array(current)
    goal = np.array(goal)
    diff = goal - current
    norm = np.linalg.norm(diff)
    if norm == 0:
        return np.zeros_like(diff)
    return (speed / norm) * diff


def get_cell_index_from_point(point, dx, dy, dz):
    """
    Compute the cell index (i, j, k) by rounding based on the grid cell sizes.
    """
    x, y, z = point
    i = round((x - dx/2.0) / dx)
    j = round((y - dy/2.0) / dy)
    k = round((z - dz/2.0) / dz)
    return (i, j, k)


def cell_center_from_index(i, j, k, dx, dy, dz):
    """
    Compute the cell centre given cell indices (i, j, k).
    """
    cx = dx/2.0 + i * dx
    cy = dy/2.0 + j * dy
    cz = dz/2.0 + k * dz
    return (cx, cy, cz)


def point_belongs_to_cell(point, cell_center, dx, dy, dz):
    """
    Check if the continuous point falls within the cell boundaries defined by its center.
    """
    x, y, z = point
    cx, cy, cz = cell_center
    in_x = (cx - dx/2.0) <= x < (cx + dx/2.0)
    in_y = (cy - dy/2.0) <= y < (cy + dy/2.0)
    in_z = (cz - dz/2.0) <= z < (cz + dz/2.0)
    return in_x and in_y and in_z


def is_point_in_occupied_cell(point, dx, dy, dz, occupancy_dict):
    """
    Determine whether a continuous point belongs to an occupied cell and return the cell center.

    First, compute the cell index (base_index) using rounding.
    Then, check if the point falls within the cell corresponding to that index.
    If it does, return a tuple (occupancy, cell_center) for that cell.
    Otherwise, loop through the 27-cell neighborhood around base_index and return
    the occupancy and cell center for the first cell that contains the point.

    occupancy_dict: a dictionary with keys as cell centers (tuples) and values as occupancy (1 = occupied, 0 = free).
    """
    base_index = get_cell_index_from_point(point, dx, dy, dz)
    base_center = cell_center_from_index(*base_index, dx, dy, dz)

    # First, check if the point belongs to the base cell.
    if point_belongs_to_cell(point, base_center, dx, dy, dz):
        occ = occupancy_dict.get(base_center, 0) == 1
        return occ, base_center

    # If not, loop through the 27 cells in the neighborhood (including base_index).
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                candidate_index = (base_index[0] + di, base_index[1] + dj, base_index[2] + dk)
                candidate_center = cell_center_from_index(*candidate_index, dx, dy, dz)
                if point_belongs_to_cell(point, candidate_center, dx, dy, dz):
                    occ = occupancy_dict.get(candidate_center, 0) == 1
                    return occ, candidate_center
    # If no cell is found to contain the point (unlikely if the grid is complete), return False and None.
    return False, None


def is_cell_occupied(cell_coord, building_polygons, dx=12, dy=8):
    """
    Check if the cell at the given coordinate is occupied by a building.

    Parameters:
      cell_coord: The lower corner coordinate of the grid cell, e.g. grid[i,j,k,:]
      building_polygons: A list of building polygons, where each polygon is a list of [x, y, z] points.
      dx, dy: Grid spacing in x and y directions.

    Returns:
      True if the cell is occupied, False otherwise.
    """
    # Use the center of the cell for the horizontal (x,y) check.
    x_center = cell_coord[0] + dx / 2.0
    y_center = cell_coord[1] + dy / 2.0
    # For vertical occupancy, you can use the lower z or cell center; here we use the lower corner.
    z_val = cell_coord[2]

    # Loop over all building polygons.
    for poly in building_polygons:
        # Assume all vertices in poly share the same building height.
        building_height = poly[0][2]
        # Create a 2D polygon (only x and y) from the building's base.
        poly_xy = [(pt[0], pt[1]) for pt in poly]
        path = Path(poly_xy)
        # Check if the cell's horizontal center is inside the polygon.
        if path.contains_point((x_center, y_center)):
            # Check vertical occupancy: if the cell is below the building's height.
            if z_val < building_height:
                return True
    return False