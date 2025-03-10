# -*- coding: utf-8 -*-
"""
@Time    : 3/3/2025 2:05 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from shapely.geometry import Point, LineString
from shapely.vectorized import contains
import matplotlib.ticker as ticker
import numba
import numpy as np
import heapq
import random
from shapely.geometry import Point, Polygon
import math
import itertools
from matplotlib.path import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def pick_candidate_point_for_intruders(candidate_points, alt_range=(20, 50),
                                        min_distance=400, boundary_margin=7.5, boundaries=None):
    """
    Generates intruder start and end positions using the same rule as the host UAV.

    This version removes the requirement that candidate points be near the host UAV path.
    Instead, it only filters candidate points based on the altitude range (if provided),
    and then uses generate_start_end_positions to ensure that the two positions are
    at least 'min_distance' apart and maintain a safe distance from boundaries.

    Parameters:
        candidate_points (np.ndarray): Array of candidate free points (shape (N,3)).
        host_start (tuple): (x,y,z) start of host path (not used in filtering here).
        host_end (tuple): (x,y,z) end of host path (not used in filtering here).
        max_offset (float): Not used in this version.
        alt_range (tuple): (min_alt, max_alt) for candidate altitude.
        min_distance (float): Minimum required 3D distance between start and end.
        boundary_margin (float): Minimum distance from airspace boundaries.
        boundaries (tuple): Airspace boundaries as (x_min, x_max, y_min, y_max, z_min, z_max).

    Returns:
        tuple: (start_point, end_point) for the intruder UAV.
    """
    # Filter candidate points only by altitude.
    valid_mask = (candidate_points[:, 2] >= alt_range[0]) & (candidate_points[:, 2] <= alt_range[1])
    filtered = candidate_points[valid_mask]
    if filtered.shape[0] == 0:
        # If no candidates remain after altitude filtering, use all candidate points.
        filtered = candidate_points

    # Use the host UAV rule to pick a start and end point.
    start_point, end_point = generate_start_end_positions(filtered, min_distance, boundary_margin, boundaries)
    return start_point, end_point


def intruder_generate_end_positions(candidate_points, start_point, min_distance=400, boundary_margin=7.5, boundaries=None):
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

    start_point = np.array(start_point)

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

def filter_points_near_buildings_vectorized(candidate_points, buffered_polys):
    """
    Filters candidate points (as a NumPy array of shape (N,3)) to those within
    any of the buffered building polygons using vectorized operations.

    Parameters:
        candidate_points (np.array): Array of candidate points (x,y,z).
        buffered_polys (list): List of Shapely Polygon objects (buffered buildings).

    Returns:
        np.array: Filtered candidate points (subset of input) that are within any buffered polygon.
    """
    # Extract x and y coordinates from candidate_points
    xs = candidate_points[:, 0]
    ys = candidate_points[:, 1]

    # Initialize a boolean mask for all points
    mask = np.zeros(candidate_points.shape[0], dtype=bool)

    # For each buffered polygon, use vectorized "contains" to update the mask
    for buff in buffered_polys:
        mask |= contains(buff, xs, ys)

    # Filter the candidate_points using the combined mask
    filtered_points = candidate_points[mask]
    return filtered_points


def filter_points_near_buildings(candidate_points, building_polygons, buffer_distance=30):
    """
    Filters candidate points to those that are within buffer_distance (in meters)
    of any building.

    Parameters:
        candidate_points (list of tuples): List of (x, y, z) points.
        building_polygons (list of lists): Each element is a list of [x, y, z] vertices of a building.
        buffer_distance (float): Buffer distance in meters (default 30).

    Returns:
        np.array: Filtered candidate points (as an array of (x,y,z)).
    """
    filtered = []
    # Precompute buffered 2D polygons for each building (using only x,y)
    buffered_polys = []
    for poly in building_polygons:
        # Extract (x,y) coordinates
        poly_xy = [(pt[0], pt[1]) for pt in poly]
        # Create a Shapely Polygon and buffer it
        building_poly = Polygon(poly_xy)
        buffered_polys.append(building_poly.buffer(buffer_distance))

    filtered_points = filter_points_near_buildings_vectorized(candidate_points, buffered_polys)

    return filtered_points


def quad_to_triangles(quad):
    """
    Given a quadrilateral defined by 4 points (v0, v1, v2, v3) in order,
    returns two triangles: (v0, v1, v2) and (v0, v2, v3).

    quad: list/array of four points, each [x, y, z]
    Returns: vertices (list of points), and triangle indices (i, j, k lists)
    """
    # vertices: v0, v1, v2, v3 in order
    # Triangles: (0,1,2) and (0,2,3)
    vertices = quad
    i = [0, 0]
    j = [1, 0]
    k = [2, 2]
    # second triangle: (0,2,3)
    i[1] = 0
    j[1] = 2
    k[1] = 3
    return vertices, i, j, k


def fan_triangulation(polygon):
    """
    For a polygon defined by n vertices (assumed convex and ordered),
    perform a fan triangulation using the first vertex as the pivot.

    Returns the vertices (as is) and the triangle indices lists i, j, k.
    """
    n = len(polygon)
    i = []
    j = []
    k = []
    for t in range(1, n - 1):
        i.append(0)
        j.append(t)
        k.append(t + 1)
    return polygon, i, j, k


def plot_trajectories(building_polygons, host_trajectory, intruder_trajectory, x_limit, y_limit, host_start, host_end):
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
        ax.scatter(host_start[0], host_start[1], host_start[2], color='cyan', s=10, label='Host Start',
                   marker='D')
        ax.scatter(host_end[0], host_end[1], host_end[2], color='red', s=10,
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


def extend_endpoint(start, end, extension=15):
    """
    Given a start and end point (each a 3D coordinate), compute a new end point that
    is extended by 'extension' meters along the line from start to end.

    Parameters:
        start (list or tuple): The start coordinate, e.g. [323.9, 112.2, 175.7].
        end (list or tuple): The original end coordinate, e.g. [352.02, 120.94, 35].
        extension (float): Distance (in meters) to extend the line.

    Returns:
        tuple: The new end coordinate (3D tuple).
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    vec = end - start
    norm = np.linalg.norm(vec)
    if norm == 0:
        return tuple(end)  # if start and end coincide, return the same end.
    unit_vec = vec / norm
    new_end = end + extension * unit_vec
    return tuple(new_end)


def redesign_intruder_end_for_conflict(host_start, host_end, intruder_start,
                                       speed_host=10.0, speed_intruder=15.0,
                                       fraction=0.5):
    """
    Redesigns an intruder's end point so that the intruder
    will definitely conflict with the host if both fly straight
    at constant speeds (host_speed=10, intruder_speed=15).

    The conflict is forced at time t_conf = fraction * host_total_time.
    The intruder is placed exactly at the host's position at t_conf.

    Parameters:
      host_start      : (x,y,z) host start
      host_end        : (x,y,z) host end
      intruder_start  : (x,y,z) intruder start
      speed_host      : float, host speed in m/s (default=10)
      speed_intruder  : float, intruder speed in m/s (default=15)
      fraction        : fraction in [0,1], controlling the conflict time
                        as fraction * total host flight time.

    Returns:
      intruder_end    : (x,y,z) new intruder end point guaranteeing conflict
    """
    # Convert to NumPy arrays
    host_start = np.array(host_start, dtype=float)
    host_end = np.array(host_end, dtype=float)
    intru_start = np.array(intruder_start, dtype=float)

    # 1. Compute host total distance & time
    host_vec = host_end - host_start
    dist_host = np.linalg.norm(host_vec)
    if dist_host < 1e-6:
        # If host start & end are effectively the same, can't force conflict meaningfully
        return tuple(intru_start)
    host_time = dist_host / speed_host

    # 2. Compute the conflict time t_conf for the intruder
    t_conf = fraction * host_time

    # 3. Host position at t_conf
    #    param: host(t) = host_start + (speed_host * t) * (host_vec / dist_host)
    #    or simply fraction of the entire vector:
    host_conf_pos = host_start + (host_vec * fraction)  # same as fraction * host_vec

    # 4. We want intruder(t_conf) = host_conf_pos
    #    intruder(t) = intru_start + unit_dir_intru * (speed_intruder * t)
    #    => intru_start + unit_dir_intru*(speed_intruder*t_conf) = host_conf_pos
    #    => unit_dir_intru = (host_conf_pos - intru_start)/norm(...)
    #    => but we need the intruder end to be intru_start + unit_dir_intru*(some total distance)
    #    We want the intruder to reach host_conf_pos at time t_conf.
    dir_vec = host_conf_pos - intru_start
    dist_to_conf = np.linalg.norm(dir_vec)
    if dist_to_conf < 1e-6:
        # If intruder start is already at the conflict point,
        # can't define a new line meaningfully, just return intruder_start.
        return tuple(intru_start)

    # The intruder must travel dist_to_conf in time t_conf => speed_intruder * t_conf = dist_to_conf
    # If dist_to_conf != speed_intruder * t_conf, we forcibly define it:
    #   speed_intruder * t_conf = dist_to_conf
    # If dist_to_conf is bigger or smaller, we forcibly set fraction so it matches or we keep fraction
    # but the intruder might not meet the host exactly at t_conf.
    #
    # Instead, we can forcibly define a new end so that the intruder line is exactly dist_to_conf away
    # from intruder_start. That means the total intruder flight time is t_conf, so:
    #   total_intru_dist = speed_intruder * t_conf
    #   if total_intru_dist < dist_to_conf => we can't place conflict
    #   so we require total_intru_dist >= dist_to_conf => we can place conflict exactly.
    # We'll place the intruder end further along that same direction so the total distance is speed_intruder * host_time
    # or we do the minimal distance so that conflict is exactly at t_conf.
    #
    # Let's do minimal so that at t_conf they coincide, after that intruder continues in the same direction:
    # intruder_end = intru_start + unit_dir_intru*(speed_intruder * total_intru_time).
    # But we want t_conf <= total_intru_time. Let's define total_intru_time = t_conf + 20% for example, or just t_conf
    # if we want them to land exactly at conflict. We'll pick total_intru_time = t_conf + 1 (some margin).

    # For simplicity, define total_intru_time = t_conf. Then the intruder is done exactly at conflict moment.
    # That means intruder_end is exactly host_conf_pos.
    # If you'd prefer the intruder to keep flying, pick total_intru_time = 2 * t_conf, etc.

    intruder_end = host_conf_pos  # intruder finishes exactly at the conflict point

    return tuple(intruder_end)


# --------------------------------------------------------------------

def clamp_intruder_start_altitude(intruders, z_max=70.0):
    """
    Given a list of intruders, each defined by (start, end),
    force the intruder's start altitude to be at most z_max.

    intruders: list of tuples [(start, end), (start, end), ...],
               where start = [x, y, z], end = [x, y, z].
    z_max: float, the maximum altitude allowed for the intruder start.

    Returns a new list of intruders with clamped start altitude.
    """
    new_intruders = []
    for (start, end) in intruders:
        # Convert start to a list so we can modify it
        start_mod = list(start)
        if start_mod[2] > z_max:
            start_mod[2] = z_max
        new_intruders.append((tuple(start_mod), end))
    return new_intruders


def plot_relative_distances(episode_relative_dist_record, num_intruders, delta_t=3.0, text_offset_x=0, text_offset_y=0):
    """
    Plots the distance between the host and each intruder over time, based on the
    'episode_relative_dist_record' list, adds a dotted horizontal line at 15 m, and
    labels the lowest point for each intruder using the same color as its line.

    Parameters:
      episode_relative_dist_record (list): A list (length = # steps) where each element
         is a list (length = # intruders) storing the distance to each intruder.
      num_intruders (int): Number of intruders.
      delta_t (float): Time (seconds) between each decision step.
      text_offset_x (float): Horizontal offset for the label of the lowest point.
      text_offset_y (float): Vertical offset for the label of the lowest point.
    """
    import matplotlib.pyplot as plt

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define a list of colors to cycle through (one per intruder)
    colors = ['blue', 'red', 'green', 'magenta', 'orange']

    # Number of time steps
    num_steps = len(episode_relative_dist_record)
    # X-axis: time in seconds
    time_values = [step_idx * delta_t for step_idx in range(num_steps)]

    # Plot each intruder's distance and annotate the lowest point
    for intr_idx in range(num_intruders):
        # Gather distance values for this intruder
        dist_values = [episode_relative_dist_record[step][intr_idx]
                       for step in range(num_steps)]
        color = colors[intr_idx % len(colors)]
        ax.plot(time_values, dist_values, marker='o', color=color,
                label=f'Intruder {intr_idx + 1}')

        # Find the lowest point (minimum distance) and its time index
        min_val = min(dist_values)
        min_index = dist_values.index(min_val)
        min_time = time_values[min_index]

        # Annotate the lowest point with its value
        if intr_idx == 0 or intr_idx == 1 or intr_idx == 2:
            ax.text(min_time + text_offset_x, min_val - 2.3,
                    f"{min_val:.1f} m", fontsize=10, ha='center', va='top', color=color)
        elif intr_idx == 3:
            ax.text(min_time - 1, min_val - 2.5,
                    f"{min_val:.1f} m", fontsize=10, ha='center', va='top', color=color)
        elif intr_idx == 4:
            ax.text(min_time + 1, min_val - 2.5,
                    f"{min_val:.1f} m", fontsize=10, ha='center', va='top', color=color)

    # Add a dotted horizontal line at y=15 m
    ax.axhline(y=15, color='red', linestyle=':', label='15 m threshold')

    # Set labels, title, legend, and grid
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Distance (m)")
    ax.set_title("Host–Intruder Distances Over Time")
    ax.legend()
    ax.grid(True)

    # Finally, show the plot
    plt.show()



def plot_one_intru_head_on(building_polygons, host_trajectory, intruder_trajectories, x_limit, y_limit, host_start, host_end, delta_t):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw buildings
    for poly_idx, poly in enumerate(building_polygons):
        if poly_idx == 2 or poly_idx == 6 or poly_idx == 8 or poly_idx == 10 or poly_idx == 11 or poly_idx == 12 or poly_idx == 13 or poly_idx == 6:
            continue  # for one and two intru case
        if poly_idx == 3 or poly_idx == 5 or poly_idx == 9:
            continue  # for 5 intru case
        # Draw walls of the building
        for i in range(len(poly)):
            x = [poly[i][0], poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][0], poly[i][0]]
            y = [poly[i][1], poly[(i + 1) % len(poly)][1], poly[(i + 1) % len(poly)][1], poly[i][1]]
            z = [0, 0, poly[(i + 1) % len(poly)][2], poly[i][2]]
            ax.add_collection3d(
                Poly3DCollection([list(zip(x, y, z))], color='green', alpha=0.3, edgecolor='none')
            )
        # Draw roof
        roof = [[pt[0], pt[1], pt[2]] for pt in poly]
        ax.add_collection3d(Poly3DCollection([roof], color='green', alpha=0.1))

    # Plot host trajectory
    if host_trajectory:
        host_traj = np.array(host_trajectory)
        ax.plot(host_traj[:, 0], host_traj[:, 1], host_traj[:, 2],
                color='blue', marker='o', lw=2, label='Host Trajectory')
        # Mark start and end of host trajectory
        ax.scatter(host_start[0], host_start[1], host_start[2],
                   color='cyan', s=10, marker='D', label='Host Start')

        # Label the host's time stamps at each marker
        for step, (x, y, z) in enumerate(host_trajectory):
            t_label = f"{step * delta_t:.1f}s"  # e.g. "0.0s", "3.0s", ...
            ax.text(x+4, y+3, z+3, t_label, color='blue', fontsize=8)

        # Plot blue dotted line from host_start to host_end
        ax.plot([host_start[0], host_end[0]],
                [host_start[1], host_end[1]],
                [host_start[2], host_end[2]],
                color='blue', ls=':', lw=2, label='Host Nominal Path')

    # Colors to cycle through for intruders
    colors = ['magenta', 'orange', 'purple', 'brown', 'black']

    # Plot each intruder's trajectory
    for idx, traj in enumerate(intruder_trajectories):
        traj_arr = np.array(traj)
        color = colors[idx % len(colors)]
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2],
                color=color, marker='o', ls='--', lw=1, label=f'Intruder {idx + 1} Trajectory')
        # ax.scatter(traj_arr[0, 0], traj_arr[0, 1], traj_arr[0, 2],
        #            color=color, s=10, marker='D', label=f'Intruder {idx + 1} Start')

        # Add a time-stamp label for each waypoint in this trajectory
        for step, (x, y, z) in enumerate(traj_arr):
            if idx == 0:
                if step == 0 or step == 1 or step == len(traj_arr) - 1:
                    t_label = f"{step * delta_t:.1f}s"  # e.g. 0.0s, 3.0s, 6.0s, ...
                    ax.text(x - 5, y - 5, z - 5, t_label, color=color, fontsize=8)
            elif idx == 2:
                if step == 0 or step == 1 or step == len(traj_arr) - 1:
                    t_label = f"{step * delta_t:.1f}s"  # e.g. 0.0s, 3.0s, 6.0s, ...
                    ax.text(x - 5, y - 5, z - 5, t_label, color=color, fontsize=8)
            elif idx == 4:
                if step == 0 or step == 1 or step == len(traj_arr) - 1:
                    t_label = f"{step * delta_t:.1f}s"  # e.g. 0.0s, 3.0s, 6.0s, ...
                    ax.text(x - 5, y - 5, z - 5, t_label, color=color, fontsize=8)
            elif idx == 1:
                if step == 0 or step == 1 or step == 2 or step == 3 or step == len(traj_arr) - 1:
                    t_label = f"{step * delta_t:.1f}s"  # e.g. 0.0s, 3.0s, 6.0s, ...
                    ax.text(x - 5, y - 5, z - 5, t_label, color=color, fontsize=8)
            elif step == 0 or step == 1 or step == 2 or step == len(traj_arr) - 1:
                t_label = f"{step * delta_t:.1f}s"  # e.g. 0.0s, 3.0s, 6.0s, ...
                ax.text(x-5, y-5, z-5, t_label, color=color, fontsize=8)

    # Set plot limits and labels
    # ax.set_xlim((200, x_limit[1]))  # for one or two intru
    ax.set_xlim((200, 450))  # for five intru case
    ax.set_ylim((100, y_limit[1]))
    # Set z limit from 0 up to max building height plus some margin.
    z_max_building = max([max(poly, key=lambda p: p[2])[2] for poly in building_polygons])
    ax.set_zlim(0, z_max_building + 10)

    ax.set_xlabel('E-W (m)')  # x
    ax.set_ylabel('N-S (m)')  # y
    ax.set_zlabel("Altitude (m)")
    ax.legend()
    plt.show()


def plot_multi_intruder_trajectories(building_polygons, host_trajectory, intruder_trajectories, x_limit, y_limit,
                                     host_start, host_end):
    """
    Plots the host UAV trajectory and the trajectories of multiple intruders.

    Parameters:
        building_polygons (list): List of building polygons (each as a list of [x,y,z] vertices).
        host_trajectory (list): List of [x,y,z] positions for the host UAV.
        intruder_trajectories (list): List of trajectories, where each trajectory is a list of [x,y,z] positions for an intruder.
        x_limit (tuple): x-axis limits for the plot.
        y_limit (tuple): y-axis limits for the plot.
        host_start (list): Starting position of the host UAV.
        host_end (list): End (target) position of the host UAV.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw buildings
    for poly in building_polygons:
        # Draw walls of the building
        for i in range(len(poly)):
            x = [poly[i][0], poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][0], poly[i][0]]
            y = [poly[i][1], poly[(i + 1) % len(poly)][1], poly[(i + 1) % len(poly)][1], poly[i][1]]
            z = [0, 0, poly[(i + 1) % len(poly)][2], poly[i][2]]
            ax.add_collection3d(
                Poly3DCollection([list(zip(x, y, z))], color='green', alpha=0.3, edgecolor='none')
            )
        # Draw roof
        roof = [[pt[0], pt[1], pt[2]] for pt in poly]
        ax.add_collection3d(Poly3DCollection([roof], color='green', alpha=0.1))

    # Plot host trajectory
    if host_trajectory:
        host_traj = np.array(host_trajectory)
        ax.plot(host_traj[:, 0], host_traj[:, 1], host_traj[:, 2],
                color='blue', marker='o', label='Host Trajectory')
        # Mark start and end of host trajectory
        ax.scatter(host_start[0], host_start[1], host_start[2],
                   color='cyan', s=50, marker='D', label='Host Start')
        ax.scatter(host_end[0], host_end[1], host_end[2],
                   color='red', s=50, marker='X', label='Host End')

    # Colors to cycle through for intruders
    colors = ['red', 'orange', 'magenta', 'purple', 'brown', 'black']

    # Plot each intruder's trajectory
    for idx, traj in enumerate(intruder_trajectories):
        traj_arr = np.array(traj)
        color = colors[idx % len(colors)]
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2],
                color=color, marker='o', label=f'Intruder {idx + 1} Trajectory')
        ax.scatter(traj_arr[0, 0], traj_arr[0, 1], traj_arr[0, 2],
                   color=color, s=50, marker='D', label=f'Intruder {idx + 1} Start')
        ax.scatter(traj_arr[-1, 0], traj_arr[-1, 1], traj_arr[-1, 2],
                   color=color, s=50, marker='X', label=f'Intruder {idx + 1} End')

    # Set plot limits and labels
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)
    # Set z limit from 0 up to max building height plus some margin.
    z_max_building = max([max(poly, key=lambda p: p[2])[2] for poly in building_polygons])
    ax.set_zlim(0, z_max_building + 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.legend()
    plt.show()


def relative_distance_multiUAV(host_position, intruder_positions):
    """
    Computes the relative distance vector between the host UAV and the nearest intruder.

    The relative distance is defined as:
        (rx, ry, rz) = (intruder_x - host_x, intruder_y - host_y, intruder_z - host_z)
    for the intruder that is nearest to the host.

    Args:
        host_position (list or tuple): The (x, y, z) coordinates of the host UAV.
        intruder_positions (list): A list of (x, y, z) coordinates for each intruder.

    Returns:
        tuple: (rx, ry, rz) the relative distance components for the nearest intruder.
    """
    best_distance = float('inf')
    best_diff = (0, 0, 0)

    for pos in intruder_positions:
        diff_x = pos[0] - host_position[0]
        diff_y = pos[1] - host_position[1]
        diff_z = pos[2] - host_position[2]
        distance = math.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)
        if distance < best_distance:
            best_distance = distance
            best_diff = (diff_x, diff_y, diff_z)

    return best_diff

# @numba.njit
# def relative_distance_multiUAV(host_position, intruder_positions):
#     best_distance = 1e12
#     best_diff0 = 0.0
#     best_diff1 = 0.0
#     best_diff2 = 0.0
#     for i in range(intruder_positions.shape[0]):
#         diff0 = intruder_positions[i, 0] - host_position[0]
#         diff1 = intruder_positions[i, 1] - host_position[1]
#         diff2 = intruder_positions[i, 2] - host_position[2]
#         dist = math.sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2)
#         if dist < best_distance:
#             best_distance = dist
#             best_diff0 = diff0
#             best_diff1 = diff1
#             best_diff2 = diff2
#     return best_diff0, best_diff1, best_diff2


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


def interpolate_trajectory(start, end, speed=15):
    """
    Interpolate intermediate 3D points between start and end
    so that the spacing between points is approximately 'speed' meters.

    Parameters:
        start (list or tuple): Starting 3D coordinate, e.g. [500, 103.93, 40].
        end (list or tuple): Ending 3D coordinate, e.g. [212.0, 158.36, 30].
        speed (float): Spacing in meters between successive points (default = 15).

    Returns:
        list: A list of 3D tuples representing the trajectory from start to end.
              The list includes both the start and end points.
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    diff = end - start
    total_distance = np.linalg.norm(diff)

    # Compute the number of segments needed (ceil to ensure spacing <= speed)
    num_segments = int(np.ceil(total_distance / speed))
    # Number of points is segments + 1 (including start and end)
    num_points = num_segments + 1

    # Generate interpolation parameter values from 0 to 1.
    ts = np.linspace(0, 1, num_points)
    # Compute interpolated points
    trajectory = [tuple(start + t * diff) for t in ts]
    return trajectory


def shorten_line(start, end, cut_length):
    """
    Shortens the line from start to end by cutting off 'cut_length' from the end.

    Parameters:
        start (list or tuple): The start coordinate, e.g. [x, y, z].
        end (list or tuple): The original end coordinate.
        cut_length (float): The distance to remove from the end.

    Returns:
        tuple: The new end coordinate after shortening the line.

    Note: If the cut_length is greater than or equal to the length of the line,
          the function returns the start point.
    """
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    line_vector = end - start
    total_length = np.linalg.norm(line_vector)

    if total_length <= cut_length:
        # If the cut_length is greater than or equal to the line length,
        # return the start coordinate (or raise an error, depending on your needs)
        return tuple(start)

    # Compute the unit vector in the direction from start to end
    unit_vector = line_vector / total_length
    # Subtract 'cut_length' from the end point along the unit vector
    new_end = end - cut_length * unit_vector
    return tuple(new_end)


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


def compute_initial_path_heading_angle(p_init, p_dest, speed=10.0):
    """
    Computes the path angle (gamma) and heading angle (chi) that orient
    a UAV from p_init to p_dest at the given speed. Angles are in degrees.

    Parameters:
        p_init (tuple): Initial position (x_init, y_init, z_init).
        p_dest (tuple): Destination position (x_dest, y_dest, z_dest).
        speed (float): UAV speed in m/s (default=10).

    Returns:
        (gamma_deg, chi_deg): Path angle (gamma) and heading angle (chi),
                              both in degrees.
    """
    # 1. Direction vector from p_init to p_dest
    dx = p_dest[0] - p_init[0]
    dy = p_dest[1] - p_init[1]
    dz = p_dest[2] - p_init[2]

    # 2. Magnitude of the direction vector
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist == 0:
        raise ValueError("Initial and destination positions are identical.")

    # 3. Unit direction vector
    dx_hat = dx / dist
    dy_hat = dy / dist
    dz_hat = dz / dist

    # 4. Velocity components (assuming UAV speed = 'speed')
    vx = speed * dx_hat
    vy = speed * dy_hat
    vz = speed * dz_hat

    # 5. Compute heading angle chi (in degrees)
    #    heading angle = atan2(vy, vx)
    chi_rad = math.atan2(vy, vx)
    chi_deg = math.degrees(chi_rad)

    # 6. Compute path angle gamma (in degrees)
    #    path angle = asin(vz / speed)
    gamma_rad = math.asin(vz / speed)
    gamma_deg = math.degrees(gamma_rad)

    return (gamma_deg, chi_deg)


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


# def decompose_speed_to_three_axis(speed, action_path_gama_degree, action_heading_chi_degree):
#     action_path_gama_rad = deg2rad(action_path_gama_degree)
#     action_heading_chi_rad = deg2rad(action_heading_chi_degree)
#     Vx = speed * math.cos(action_path_gama_rad) * math.cos(action_heading_chi_rad)
#     Vy = speed * math.cos(action_path_gama_rad) * math.sin(action_heading_chi_rad)
#     Vz = speed * math.sin(action_path_gama_rad)
#     return Vx, Vy, Vz

@numba.njit
def decompose_speed_to_three_axis(speed, path_gama_deg, heading_chi_deg):
    # Convert degrees to radians
    gamma = path_gama_deg * math.pi / 180.0
    chi = heading_chi_deg * math.pi / 180.0
    Vx = speed * math.cos(gamma) * math.cos(chi)
    Vy = speed * math.cos(gamma) * math.sin(chi)
    Vz = speed * math.sin(gamma)
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


# def relative_distance(host_pos, intruder_pos):
#     """
#     Calculate the relative distance along each axis (x, y, z) between host and intruder UAV.
#
#     Parameters:
#         host_pos: A tuple or list (x, y, z) representing the host UAV's continuous position.
#         intruder_pos: A tuple or list (x, y, z) representing the intruder UAV's continuous position.
#
#     Returns:
#         A tuple (dx, dy, dz) where:
#             dx = host_pos[0] - intruder_pos[0]
#             dy = host_pos[1] - intruder_pos[1]
#             dz = host_pos[2] - intruder_pos[2]
#     """
#     dx = host_pos[0] - intruder_pos[0]
#     dy = host_pos[1] - intruder_pos[1]
#     dz = host_pos[2] - intruder_pos[2]
#     return (dx, dy, dz)


@numba.njit
def relative_distance(host_pos, intruder_pos):
    dx_val = host_pos[0] - intruder_pos[0]
    dy_val = host_pos[1] - intruder_pos[1]
    dz_val = host_pos[2] - intruder_pos[2]
    return dx_val, dy_val, dz_val


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


# -----------------------------
# Helper: Conversion between real-world coordinates and grid indices
# -----------------------------
def to_grid_coords(x, y, z, dx, dy, dz):
    i = int(x // dx)
    j = int(y // dy)
    k = int(z // dz)
    return (i, j, k)

def from_grid_coords(i, j, k, dx, dy, dz):
    # Assuming cell centers are used.
    x = (i + 0.5)*dx
    y = (j + 0.5)*dy
    z = (k + 0.5)*dz
    return (x, y, z)

# -----------------------------
# A* 3D Path Planning Functions
# -----------------------------
def heuristic(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def reconstruct_path(came_from, current, dx, dy, dz):
    path = []
    while current is not None:
        path.append(from_grid_coords(*current, dx, dy, dz))
        current = came_from.get(current)
    path.reverse()
    return path

def a_star_3d(start_xyz, goal_xyz, occupancy_dict, dx, dy, dz, x_min, x_max, y_min, y_max, z_min, z_max):
    start = to_grid_coords(*start_xyz, dx, dy, dz)
    goal  = to_grid_coords(*goal_xyz, dx, dy, dz)
    # If start or goal are not free, return None.
    if occupancy_dict.get(from_grid_coords(*start, dx, dy, dz), 1) == 1:
        return None
    if occupancy_dict.get(from_grid_coords(*goal, dx, dy, dz), 1) == 1:
        return None

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    # 6-connected neighbors
    # neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    # 26-connected neighbors (all adjacent grid cells except the center)
    neighbors = [(i, j, k)
                 for i in [-1, 0, 1]
                 for j in [-1, 0, 1]
                 for k in [-1, 0, 1]
                 if not (i == 0 and j == 0 and k == 0)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current, dx, dy, dz)
        for d in neighbors:
            nxt = (current[0]+d[0], current[1]+d[1], current[2]+d[2])
            # Check bounds using real coordinates
            rx, ry, rz = from_grid_coords(*nxt, dx, dy, dz)
            if rx < x_min or rx > x_max or ry < y_min or ry > y_max or rz < z_min or rz > z_max:
                continue
            # Check occupancy: using cell center as key
            if occupancy_dict.get((rx, ry, rz), 1) == 1:
                continue
            new_cost = cost_so_far[current] + 1  # uniform cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(open_set, (priority, nxt))
                came_from[nxt] = current
    return None

# -----------------------------
# Helper: Pick candidate points from candidate_points that are near the host path
# -----------------------------

def filter_candidate_points(candidate_points, host_start, host_end, max_offset, alt_range):
    """
    Filters candidate_points (NumPy array shape (N,3)) for points that are within max_offset (in meters)
    of the line from host_start to host_end in the XY plane and whose altitude is within alt_range.

    Args:
        candidate_points (np.ndarray): Array of shape (N,3) with [x, y, z].
        host_start (tuple): (x,y,z) start of host path.
        host_end (tuple): (x,y,z) end of host path.
        max_offset (float): Maximum allowed perpendicular distance in XY.
        alt_range (tuple): (min_alt, max_alt) for the candidate altitude.

    Returns:
        np.ndarray: Filtered candidate points.
    """
    # Extract A and B (host start and end) in XY:
    A = np.array(host_start[:2])
    B = np.array(host_end[:2])
    d = B - A
    line_len = np.linalg.norm(d)
    if line_len < 1e-6:
        line_len = 1  # Avoid division by zero
    # Get all candidate points' XY components:
    pts_xy = candidate_points[:, :2]  # shape (N,2)
    # Compute the vector from A to each candidate point:
    diff = pts_xy - A  # shape (N,2)
    # Compute 2D cross product magnitude:
    cross = np.abs(d[0] * diff[:, 1] - d[1] * diff[:, 0])
    distances = cross / line_len  # shape (N,)
    # Create a mask: candidate must have distance <= max_offset and altitude within alt_range.
    mask = (distances <= max_offset) & (candidate_points[:, 2] >= alt_range[0]) & (
                candidate_points[:, 2] <= alt_range[1])
    return candidate_points[mask]


# def optimized_find_target_wp_along_path(traj, curr_pos, dist_to_move, tol=1e-6):
#     """
#     Given a 3D trajectory (as an (N,3) numpy array) and a current position,
#     return the point along the trajectory that is dist_to_move (in meters)
#     further along the path (measured as arc-length) from the projection of curr_pos.
#
#     This function is fully vectorized for efficiency.
#
#     Parameters:
#       traj       : np.ndarray of shape (N,3) representing the trajectory waypoints.
#       curr_pos   : np.ndarray or tuple (x,y,z) representing the current position.
#       dist_to_move : scalar, desired arc-length to move ahead along the trajectory.
#       tol        : tolerance to avoid division by zero.
#
#     Returns:
#       best_wp    : a tuple (x, y, z) of the computed target waypoint on the path.
#       best_wp_idx: the index of the segment where the target lies (for reference).
#     """
#     # Ensure inputs are numpy arrays
#     traj = np.array(traj)  # shape (N,3)
#     curr_pos = np.array(curr_pos)  # shape (3,)
#     N = traj.shape[0]
#     if N < 2:
#         return tuple(traj[0]), 0
#
#     # 1. Compute cumulative arc-length along the trajectory.
#     segs = np.diff(traj, axis=0)  # shape (N-1,3)
#     seg_lens = np.linalg.norm(segs, axis=1)  # shape (N-1,)
#     cum_dist = np.concatenate(([0], np.cumsum(seg_lens)))  # shape (N,)
#     total_length = cum_dist[-1]
#
#     # 2. Vectorized projection of curr_pos onto each segment.
#     # For each segment from traj[i] to traj[i+1]:
#     p0 = traj[:-1]  # shape (N-1, 3)
#     p1 = traj[1:]  # shape (N-1, 3)
#     segs = p1 - p0  # (recompute to be sure)
#     seg_lens_sq = np.sum(segs ** 2, axis=1)  # shape (N-1,)
#     diff = curr_pos - p0  # shape (N-1, 3)
#     t = np.sum(diff * segs, axis=1) / (seg_lens_sq + tol)
#     t_clamped = np.clip(t, 0.0, 1.0)
#     proj_points = p0 + segs * t_clamped[:, None]  # shape (N-1,3)
#
#     # Compute the perpendicular (Euclidean) distance from curr_pos to each projected point.
#     perp_dists = np.linalg.norm(proj_points - curr_pos, axis=1)
#     best_seg_idx = np.argmin(perp_dists)
#
#     # 3. Compute the cumulative distance of the projection.
#     proj_dist = cum_dist[best_seg_idx] + t_clamped[best_seg_idx] * seg_lens[best_seg_idx]
#
#     # 4. Define the target cumulative distance along the path.
#     target_cum = proj_dist + dist_to_move
#     if target_cum >= total_length:
#         return tuple(traj[-1]), N - 1
#
#     # 5. Use searchsorted to find the segment where target_cum falls.
#     wp_idx = np.searchsorted(cum_dist, target_cum)
#     if wp_idx == 0:
#         best_wp = traj[0]
#     else:
#         # Interpolate between traj[wp_idx-1] and traj[wp_idx]
#         d0 = cum_dist[wp_idx - 1]
#         d1 = cum_dist[wp_idx]
#         # Compute interpolation factor:
#         alpha = (target_cum - d0) / (d1 - d0 + tol)
#         best_wp = traj[wp_idx - 1] + alpha * (traj[wp_idx] - traj[wp_idx - 1])
#
#     return tuple(best_wp), wp_idx - 1

@numba.njit
def optimized_find_target_wp_along_path(traj, curr_pos, dist_to_move, tol=1e-6):
    """
    JIT-compiled version of your optimized waypoint finder.
    traj : 2D np.ndarray of shape (N,3)
    curr_pos : 1D np.ndarray of shape (3,)
    dist_to_move : scalar
    """
    N = traj.shape[0]
    if N < 2:
        return traj[0], 0

    # Compute cumulative arc-length along trajectory
    cum_dist = np.empty(N, dtype=np.float64)
    cum_dist[0] = 0.0
    for i in range(1, N):
        seg0 = traj[i, 0] - traj[i-1, 0]
        seg1 = traj[i, 1] - traj[i-1, 1]
        seg2 = traj[i, 2] - traj[i-1, 2]
        seg_len = math.sqrt(seg0*seg0 + seg1*seg1 + seg2*seg2)
        cum_dist[i] = cum_dist[i-1] + seg_len
    total_length = cum_dist[-1]

    best_error = 1e12
    best_seg_idx = 0
    best_t = 0.0
    # Find projection of curr_pos onto each segment
    for i in range(N - 1):
        p0 = traj[i]
        p1 = traj[i+1]
        seg0 = p1[0] - p0[0]
        seg1 = p1[1] - p0[1]
        seg2 = p1[2] - p0[2]
        seg_len_sq = seg0*seg0 + seg1*seg1 + seg2*seg2
        t = ((curr_pos[0]-p0[0])*seg0 + (curr_pos[1]-p0[1])*seg1 + (curr_pos[2]-p0[2])*seg2) / (seg_len_sq + tol)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        proj0 = p0[0] + t * seg0
        proj1 = p0[1] + t * seg1
        proj2 = p0[2] + t * seg2
        err = math.sqrt((curr_pos[0]-proj0)**2 + (curr_pos[1]-proj1)**2 + (curr_pos[2]-proj2)**2)
        if err < best_error:
            best_error = err
            best_seg_idx = i
            best_t = t

    proj_dist = cum_dist[best_seg_idx] + best_t * (cum_dist[best_seg_idx+1] - cum_dist[best_seg_idx])
    target_cum = proj_dist + dist_to_move

    if target_cum >= total_length:
        return traj[N-1], N-1

    for i in range(1, N):
        if cum_dist[i] >= target_cum:
            alpha = (target_cum - cum_dist[i-1]) / ((cum_dist[i] - cum_dist[i-1]) + tol)
            best_wp = traj[i-1] + alpha * (traj[i] - traj[i-1])
            return best_wp, i-1
    return traj[N-1], N-1


def find_target_wp_along_path(path, curr_pos, desired, tol=1e-6):
    """
    Given a piecewise-linear 3D path (a list/array of waypoints) and a current position,
    return the waypoint along the path that is approximately 'desired' meters ahead of the
    projection of curr_pos onto the path. The distance is measured along the path.

    The function satisfies:
      1. Among the waypoints that come after the projection of curr_pos,
         the chosen waypoint's arc-length difference from the projection is as close as possible to 'desired'.
      2. If multiple candidates have nearly the same error, choose the one furthest along the path (i.e.
         the one with the smallest remaining distance to the final waypoint).

    Parameters:
      path      : list or np.ndarray of shape (N, 3) representing the 3D waypoints.
      curr_pos  : tuple or np.ndarray representing the current position (x,y,z).
      desired   : desired distance (in meters) to move ahead along the path.
      tol       : tolerance for numerical comparisons.

    Returns:
      best_wp   : a tuple (x,y,z) corresponding to the selected waypoint.
      best_idx  : the index in the path of that waypoint.
    """
    # Convert inputs to NumPy arrays.
    path = np.array(path)  # shape (N,3)
    curr_pos = np.array(curr_pos)  # shape (3,)
    N = path.shape[0]
    if N < 2:
        return tuple(path[0]), 0

    # 1. Compute cumulative distances along the path.
    segs = np.diff(path, axis=0)  # Differences between consecutive waypoints, shape (N-1, 3)
    seg_lengths = np.linalg.norm(segs, axis=1)  # Lengths of each segment, shape (N-1,)
    cum_dist = np.concatenate(([0], np.cumsum(seg_lengths)))  # cumulative distance at each waypoint, shape (N,)
    total_length = cum_dist[-1]

    # 2. Project curr_pos onto the path.
    best_proj_dist = None
    best_proj_error = np.inf
    proj_index = None
    proj_t = None  # parameter along the segment

    for i in range(N - 1):
        p0 = path[i]
        p1 = path[i + 1]
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)
        if seg_len < tol:
            continue
        # Compute parameter t such that projection = p0 + t*(p1 - p0)
        t = np.dot(curr_pos - p0, seg) / (seg_len ** 2)
        t_clamped = np.clip(t, 0.0, 1.0)
        proj = p0 + t_clamped * seg
        # Use Euclidean distance as error for projection.
        err = np.linalg.norm(curr_pos - proj)
        if err < best_proj_error:
            best_proj_error = err
            proj_index = i
            proj_t = t_clamped

    if proj_index is None:
        # Fallback: assume curr_pos is at the beginning
        proj_dist = 0.0
    else:
        proj_dist = cum_dist[proj_index] + proj_t * seg_lengths[proj_index]

    # 3. Define the target cumulative distance along the path.
    target_cum = proj_dist + desired
    if target_cum >= total_length:
        return tuple(path[-1]), N - 1

    # 4. Among waypoints after proj_index, find the candidate with cumulative distance
    #    closest to target_cum.
    # We only consider indices from proj_index+1 to N-1.
    candidate_indices = np.arange(proj_index + 1, N)
    candidate_errors = np.abs(cum_dist[candidate_indices] - target_cum)
    min_err = np.min(candidate_errors)
    # Get all indices where error is within tolerance of min_err.
    tied = candidate_indices[candidate_errors - min_err < tol]
    if tied.size > 0:
        # Choose the one with the largest cumulative distance (i.e. furthest along the path).
        best_idx = int(np.max(tied))
    else:
        best_idx = int(candidate_indices[np.argmin(candidate_errors)])

    best_wp = tuple(path[best_idx])
    return best_wp, best_idx



def pick_candidate_point_near_host_path(candidate_points, host_start, host_end, max_offset=50.0, alt_range=(20,50)):
    """
    Filters candidate_points (each as a numpy array [x,y,z]) for those whose 2D (xy) distance
    to the line (host_start to host_end) is less than max_offset, and whose altitude is within alt_range.
    Returns one randomly chosen candidate point as a tuple.
    """
    # Convert host_start and host_end to numpy arrays (for convenience)
    Hs = np.array(host_start[:2])
    He = np.array(host_end[:2])
    # Create a LineString in 2D:
    host_line = LineString([tuple(Hs), tuple(He)])
    # Filter candidate_points:

    filtered = filter_candidate_points(candidate_points, host_start, host_end, max_offset, alt_range)
    if len(filtered) == 0:
        # if none found, just pick any candidate with altitude in range
        filtered = [pt for pt in candidate_points if alt_range[0] <= pt[2] <= alt_range[1]]
    if len(filtered) == 0:
        # fallback to random candidate
        return tuple(random.choice(candidate_points))
    return tuple(random.choice(filtered))


def plot_intruder_normal_trajectories(host_uav_start, host_uav_end, intruder_norminal_trajectories,
                                      x_limit, y_limit, building_polygons=None):
    """
    Plots the host UAV's straight-line path (from host_uav_start to host_uav_end)
    and the full planned A* trajectories for each intruder.

    Parameters:
      host_uav_start (tuple): (x,y,z) start position for the host UAV.
      host_uav_end   (tuple): (x,y,z) end position for the host UAV.
      intruder_norminal_trajectories (list): List of trajectories, one per intruder.
           Each trajectory is a list of waypoints (tuples (x,y,z)).
      x_limit, y_limit, z_limit: Plot limits in x, y, and z.
      building_polygons (optional): A list of building polygons to be drawn in the background.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # # Plot building polygons if provided.
    # if building_polygons is not None:
    #     for poly in building_polygons:
    #         # Draw walls: iterate over polygon vertices
    #         for i in range(len(poly)):
    #             x_vals = [poly[i][0], poly[(i + 1) % len(poly)][0],
    #                       poly[(i + 1) % len(poly)][0], poly[i][0]]
    #             y_vals = [poly[i][1], poly[(i + 1) % len(poly)][1],
    #                       poly[(i + 1) % len(poly)][1], poly[i][1]]
    #             z_vals = [0, 0, poly[(i + 1) % len(poly)][2], poly[i][2]]
    #             wall = Poly3DCollection([list(zip(x_vals, y_vals, z_vals))], color='gray', alpha=0.3)
    #             ax.add_collection3d(wall)
    #         # Roof:
    #         roof = [[pt[0], pt[1], pt[2]] for pt in poly]
    #         roof_poly = Poly3DCollection([roof], color='gray', alpha=0.2)
    #         ax.add_collection3d(roof_poly)

    # Plot host UAV's nominal path (straight line)
    host_line_x = [host_uav_start[0], host_uav_end[0]]
    host_line_y = [host_uav_start[1], host_uav_end[1]]
    host_line_z = [host_uav_start[2], host_uav_end[2]]
    ax.plot(host_line_x, host_line_y, host_line_z, 'b--', lw=2, label="Host Nominal Path")
    ax.scatter(host_uav_start[0], host_uav_start[1], host_uav_start[2], color='cyan', marker='o', s=50,
               label="Host Start")
    ax.scatter(host_uav_end[0], host_uav_end[1], host_uav_end[2], color='red', marker='^', s=50, label="Host End")

    # Plot each intruder's planned A* trajectory.
    for idx, traj in enumerate(intruder_norminal_trajectories):
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], marker='o', label=f"Intruder {idx + 1}")

    # Set limits and labels.
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)
    # Set z limit from 0 up to max building height plus some margin.
    z_max_building = max([max(poly, key=lambda p: p[2])[2] for poly in building_polygons])
    ax.set_zlim(0, z_max_building + 10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.legend(loc='best')
    plt.title("Intruder A* Planned Trajectories and Host Nominal Path")
    plt.show()


# -----------------------------
# Helper: Cyclic (Ping-Pong) Position Getter
# -----------------------------
def get_cyclic_position(path, t):
    """
    Given a list of waypoints 'path' and a time step t (an integer),
    returns a waypoint in a ping-pong (cyclic) manner.
    """
    L = len(path)
    if L == 0:
        return None
    if L == 1:
        return path[0]
    period = 2*(L - 1)
    idx = t % period
    if idx < L:
        return path[idx]
    else:
        return path[period - idx]

# -----------------------------
# Generate Intruder Trajectories using A*
# -----------------------------

def generate_intruder_trajectories_Astar_given_OD(start, goal, num_intruders=20,
                                         occupancy_dict=None, dx=12, dy=8, dz=0.5,
                                         x_min=0, x_max=600, y_min=0, y_max=400, z_min=0, z_max=50):
    """
    For each intruder, pick a start and end point from candidate_points (free grid cells)
    that are near the host UAV's straight-line path. Then use A* to compute a collision-free path.
    Returns a list (length num_intruders) of paths (each a list of (x,y,z) waypoints).
    """
    trajectories = []
    for i in range(num_intruders):
        # Run A* planning on the 3D grid
        path = a_star_3d(start, goal, occupancy_dict, dx, dy, dz, x_min, x_max, y_min, y_max, z_min, z_max)
        if path is None:
            # If A* fails, fallback to a straight-line interpolation with 10 waypoints
            path = [tuple(np.linspace(s, g, num=10)) for s, g in zip(start, goal)]
            # Transpose to list of waypoints:
            path = list(zip(*path))
        trajectories.append(path)
    return trajectories


def generate_intruder_trajectories_Astar(host_start, host_end, candidate_points, num_intruders=20,
                                         occupancy_dict=None, dx=12, dy=8, dz=0.5,
                                         x_min=0, x_max=600, y_min=0, y_max=400, z_min=0, z_max=50,
                                         max_offset=50.0, alt_range=(20,50)):
    """
    For each intruder, pick a start and end point from candidate_points (free grid cells)
    that are near the host UAV's straight-line path. Then use A* to compute a collision-free path.
    Returns a list (length num_intruders) of paths (each a list of (x,y,z) waypoints).
    """
    trajectories = []
    for i in range(num_intruders):
        # Pick start and goal points from candidate_points
        start, goal = pick_candidate_point_for_intruders(candidate_points, alt_range, min_distance=300, boundary_margin=7.5, boundaries=(x_min, x_max, y_min, y_max, z_min, z_max))
        # Run A* planning on the 3D grid
        path = a_star_3d(start, goal, occupancy_dict, dx, dy, dz, x_min, x_max, y_min, y_max, z_min, z_max)
        if path is None:
            # If A* fails, fallback to a straight-line interpolation with 10 waypoints
            path = [tuple(np.linspace(s, g, num=10)) for s, g in zip(start, goal)]
            # Transpose to list of waypoints:
            path = list(zip(*path))
        trajectories.append(path)
    return trajectories