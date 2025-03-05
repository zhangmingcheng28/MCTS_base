# -*- coding: utf-8 -*-
"""
@Time    : 3/3/2025 2:29 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
# !/usr/bin/env python
from shapely.geometry import Point, LineString
import random
import math
import hashlib
import logging
import time
from Utilities import *
from matplotlib.path import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse


# ------------------------------
# Define your simulation state
# ------------------------------
class AirspaceState():
    """
    This class represents a state in your discrete airspace.
    Modify the __init__, next_state, terminal, and reward functions as needed.
    """

    def __init__(self, state_vector=None, moves=[], time_step=1, position_state=[]):
        # If no state_vector is provided, we default to (0,0,0) and an empty moves list.
        if state_vector is None:
            self.state_vector = [0, 0, 0]  # default value
        else:
            self.state_vector = state_vector
        self.moves = moves  # Store actions taken
        self.time_step = time_step
        self.host_position = position_state[0]
        self.intruder_position = position_state[1:]
        # TODO: Add any additional state variables relevant to your simulation

    def get_possible_actions(self):
        """
        Returns the possible actions for the host UAV: a combination of path angle and heading angle adjustments.
        """
        return [(gama, chi) for gama in actionSpace_path_gama for chi in actionSpace_heading_chi]

    def next_state(self, action):
        """
        Computes the next state after applying the selected action.

        - Updates host UAV's path and heading angles based on the action.
        - Computes new host velocities.
        - Computes the new relative distances.
        - Computes the new deviation from the desired path.

        The intruder UAV maintains constant speed and moves toward its goal.
        """
        gama_action, chi_action = action  # Extract chosen action
        # Extract current state variables

        intru_vx, intru_vy, intru_vz = self.state_vector[:3]  # Intruder velocity
        # host_vx, host_vy, host_vz = self.state_vector[3:6]  # Host velocity
        # host_desired = self.state_vector[6:9]  # Host's desired velocity
        # action_path_gama = self.state_vector[9]  # Previous path angle
        # action_heading_chi = self.state_vector[10]  # Previous heading angle
        # rx, ry, rz = self.state_vector[11:14]  # Relative distance
        # dev_x, dev_y, dev_z = self.state_vector[14:]  # Deviation components

        # Update the host UAV angles using the chosen action
        new_path_gama = gama_action
        new_heading_chi = chi_action

        # Recalculate the host UAV’s new velocity components
        host_vx, host_vy, host_vz = decompose_speed_to_three_axis(host_speed, new_path_gama, new_heading_chi)

        # Update host position based on velocity
        new_host_position = (
            self.host_position[0] + host_vx * self.time_step,
            self.host_position[1] + host_vy * self.time_step,
            self.host_position[2] + host_vz * self.time_step
        )

        # Compute new intruder position using constant velocity motion
        new_intruder_position = (
            self.intruder_position[0][0] + intru_vx * self.time_step,
            self.intruder_position[0][1] + intru_vy * self.time_step,
            self.intruder_position[0][2] + intru_vz * self.time_step
        )
        new_position_state = [new_host_position, new_intruder_position]

        # Compute new relative distances
        rx, ry, rz = relative_distance(new_host_position, new_intruder_position)

        # Compute new desired velocity (host moving toward intruder's start position)
        host_desired = desired_velocity(new_host_position, intruder_uav_simulation_start, host_speed)

        # Compute new deviation from path
        dev_norm, closest_point = deviation_from_path(new_host_position, host_uav_simulation_start,
                                                      intruder_uav_simulation_start)
        dev_x, dev_y, dev_z = relative_distance(new_host_position, closest_point)

        # New state vector after taking the action
        new_state_vector = [
            intru_vx, intru_vy, intru_vz,  # Intruder velocity
            host_vx, host_vy, host_vz,  # Host velocity
            host_desired[0], host_desired[1], host_desired[2],  # Desired velocity
            new_path_gama, new_heading_chi,  # New angles
            rx, ry, rz,  # Relative distances
            dev_x, dev_y, dev_z  # Deviations
        ]

        return AirspaceState(state_vector=new_state_vector, moves=self.moves+[action], position_state=new_position_state)  # return new_state_vector

    def terminal(self):
        """
        Define termination conditions for the simulation:
        - Host UAV collides with an occupied grid (sphere intersection check).
        - Host UAV and intruder UAV are within 15m of each other (collision).
        - Host UAV is within 10m of its target.
        - Host UAV has collied with the boundary
        """
        # Unpack the host and intruder UAV positions
        host_x, host_y, host_z = self.host_position
        intruder_x, intruder_y, intruder_z = self.intruder_position[0]
        original_host_position = get_original_position(self.host_position, host_speed, delta_t, self.state_vector[9], self.state_vector[10])

        # 1. Check collision with an occupied grid (sphere radius 7.5m)
        if check_collision_with_grid(self.host_position, 7.5, occupancy_dict, dx, dy, dz):
            # print("UAV hit obstacles")
            return -1  # UAV has hit an obstacle
        # 2. Check if host UAV and intruder UAV collide (distance < 15m)
        distance_between_uavs = math.sqrt((host_x - intruder_x) ** 2 +
                                          (host_y - intruder_y) ** 2 +
                                          (host_z - intruder_z) ** 2)
        if distance_between_uavs < 15:
            # print("UAV collides with other UAVs")
            return -1  # UAVs have collided
        # 3. Check if host UAV has reached its target (distance < 10m)
        target_x, target_y, target_z = intruder_uav_simulation_start  # Target is the intruder's start position
        distance_to_target = math.sqrt((host_x - target_x) ** 2 +
                                       (host_y - target_y) ** 2 +
                                       (host_z - target_z) ** 2)
        check_goal_result = check_if_reach_goal(self.host_position, original_host_position, target_x, target_y, target_z)
        if check_goal_result:
            # print("Terminate due to reach goal")
            return 1
        # if distance_to_target < 7.5:
        #     # print("UAV reaches the goal")
        #     return True  # Host UAV has reached the goal

        # 4. Check if host UAV's sphere is too close to any boundary.
        # For a UAV modeled as a sphere with centroid at (host_x, host_y, host_z) radius=7.5m,
        # if the distance from the centroid to any boundary wall is less than 15 m, we consider it a conflict.
        if (host_x - x_min < 7.5) or (x_max - host_x < 7.5) or \
                (host_y - y_min < 7.5) or (y_max - host_y < 7.5) or \
                (host_z < 7.5) or (z_max - host_z < 7.5):
            # print("UAV collided with boundary")
            return -1  # Host UAV has collided with the boundary

        return 0  # Simulation continues

    def reward(self, alpha=0, R_A=50):
        """
        Define the reward function for your simulation.
        For example, reward might be based on the distance to a target cell.
        """
        """
        Computes the reward function based on Eq. (4.6) from the reference.

        Parameters:
            alpha (float): Weighting factor (default = 0.5).
            R_A (float): Alert zone radius (default = 50m).

        Returns:
            float: Reward value.
        """

        # Extract UAV position & intruder position
        host_x, host_y, host_z = self.host_position
        intruder_x, intruder_y, intruder_z = self.intruder_position[0]
        host_vel_current = self.state_vector[3:6]  # Host velocity
        host_vel_desired = self.state_vector[6:9]  # Host's desired velocity
        deviation_desired = (np.array(host_vel_desired) - np.array(host_vel_current))*delta_t

        original_host_position = get_original_position(self.host_position, host_speed, delta_t, self.state_vector[9], self.state_vector[10])

        # Compute relative distances
        R_x = intruder_x - host_x
        R_y = intruder_y - host_y
        R_z = intruder_z - host_z

        # Terminal Conditions
        terminate_states = self.terminal()
        if terminate_states == 1:  # Local Subgoal Arrival Condition
            return 1  # goal arrival give reward = 1
        elif terminate_states == 0:  # Normal operation reward
            prevent_UAV_conflict_portion = 1 / (
                        R_x ** 2 + R_y ** 2 + R_z ** 2 + 1e-6)  # Small term prevents division by zero
            # deviation_desired_portion = (deviation_desired[0]**2+deviation_desired[1]**2+deviation_desired[2]**2) / R_A  # zn version, not working at all

            # reward_second = normalize_dot_product(np.array(host_vel_desired), np.array(host_vel_current))  # V1.1 sometime works, success rate not high.

            # V2 use deviation to norminal path when solely use this V2 can host UAV is able to find it goal within 5 episodes, v2 works, when alpha set to 0
            dev_norminal_x, dev_norminal_y, dev_norminal_z = self.state_vector[14:]  # Deviation components
            norm_dev = np.linalg.norm(np.array([dev_norminal_x, dev_norminal_y, dev_norminal_z]))
            reward_second = max(0, 1 - (norm_dev / R_A))

            reward_normal = (prevent_UAV_conflict_portion * alpha) + (reward_second * (1 - alpha))

            if reward_normal > 1:
                print("debug, check")
            return reward_normal  # Otherwise, return the normal operation reward
        elif terminate_states == -1:  # collision happens
            return 0


    def __hash__(self):  # make the self.current_index can be stored within a dict
        # For uniqueness, we use the current index (and optionally moves) to compute the hash.
        return hash(self.current_index)

    def __eq__(self, other):  # make index can be compared
        return self.current_index == other.current_index

    def __repr__(self):  # make index can be printed, so all above make easier for debug
        return f"AirspaceState(index={self.current_index}, moves={self.moves})"


# ------------------------------
# MCTS Node Class
# ------------------------------
class Node():
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward_val):
        self.reward = self.reward + reward_val
        self.visits = self.visits + 1

    def fully_expanded(self, num_moves_lambda=None):
        # For this simulation, full expansion means all possible actions from the current state have been tried.
        num_moves = len(self.state.get_possible_actions())
        if num_moves_lambda is not None:
            num_moves = num_moves_lambda(self)
        return len(self.children) == num_moves

    def __repr__(self):
        return f"Node(state={self.state}, visits={self.visits}, reward={self.reward}, children={len(self.children)})"


# ------------------------------
# MCTS Core Functions
# ------------------------------
def UCTSEARCH(budget, root, num_moves_lambda=None):
    for iter in range(int(budget)):
        # if iter % 10000 == 9999:
            # logger.info("simulation: %d" % iter)
            # logger.info(root)
        front = TREEPOLICY(root, num_moves_lambda)
        reward_val = DEFAULTPOLICY(front.state)
        BACKUP(front, reward_val)
    return BESTCHILD(root, 0)  # The exploration bonus is unnecessary at this stage because we are no longer searching—we are choosing the best action at current depth.


def TREEPOLICY(node, num_moves_lambda):
    while not node.state.terminal():
        if len(node.children) == 0:
            return EXPAND(node)
        elif random.uniform(0, 1) < 0.5:
            node = BESTCHILD(node, SCALAR)
        else:
            if not node.fully_expanded(num_moves_lambda):
                return EXPAND(node)
            else:
                node = BESTCHILD(node, SCALAR)
    return node


def EXPAND(node):
    """Expands the node by adding a new child corresponding to an untried action."""

    tried_actions = {c.state.moves[-1] for c in node.children}  # Actions already tried
    possible_actions = node.state.get_possible_actions()  # All possible actions

    # Select an action that has not been tried yet
    untried_actions = [a for a in possible_actions if a not in tried_actions]

    if not untried_actions:
        return random.choice(node.children) if node.children else node  # Ensure consistency, return a random child  # may be can optimize more, i.e. return the most explored child

    # Pick the first untried action (or random untried action if needed)
    chosen_action = random.choice(untried_actions)

    # Generate new state by applying chosen action
    new_state = node.state.next_state(chosen_action)

    # Add this as a new child node
    node.add_child(new_state)

    return node.children[-1]  # Return the newly expanded node


def BESTCHILD(node, scalar):
    bestscore = -float('inf')
    bestchildren = []
    for c in node.children:
        exploit = c.reward / c.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
        score = exploit + scalar * explore
        if score == bestscore:
            bestchildren.append(c)
        elif score > bestscore:
            bestchildren = [c]
            bestscore = score
    if not bestchildren:
        if node.state.terminal():
            print("Debug: The node's state is terminal.")
        print("No best child found!")
        return None
    return random.choice(bestchildren)


def DEFAULTPOLICY(state, max_depth=3):
    roll_out_depth = 0
    while not state.terminal() and roll_out_depth < max_depth:
        possible_actions = state.get_possible_actions()  # Get all possible actions
        random_action = random.choice(possible_actions)  # Choose a random action
        state = state.next_state(random_action)
        roll_out_depth = roll_out_depth + 1
    return state.reward()


def BACKUP(node, reward_val):
    while node is not None:
        node.visits = node.visits + 1
        node.reward = node.reward + reward_val
        node = node.parent


# ------------------------------
# Main: Running the MCTS
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCTS for Discrete Airspace Simulation')
    parser.add_argument('--num_sims', action="store", type=int, default=100,
                        help="Number of simulations to run at each level (default: 100)")
    parser.add_argument('--levels', action="store", type=int, default=3,
                        help="Number of levels (depth) to search (default: 3)")
    parser.add_argument('--steps', action="store", type=int, default=100,
                        help="Number of episodes to run (default: 100)")
    parser.add_argument('--episodes', action="store", type=int, default=1000,
                        help="Number of episodes to run (default: 1000)")
    args = parser.parse_args()

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

    x_min = 0  # x1 is the reference point
    x_max = (x2 - x1) * lon_to_m
    y_min = 0  # y1 is the reference point
    y_max = (y2 - y1) * lat_to_m
    x_limit = (x_min - 10, x_max + 10)
    y_limit = (y_min - 10, y_max + 10)
    z_min = 0
    z_max = round(max(heights)[0,0], -1)+10

    # Convert building_coord to the building_polygons format
    building_polygons = []
    # Dictionary to store the maximum height for each unique 2D polygon
    unique_polygons = {}

    for b_coords, b_height in zip(positions,
                                  heights):  # Iterate over buildings (assuming the outermost structure is a list/array)
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

    # ==========================================================
    # Define the discrete airspace grid
    # ==========================================================
    # Grid parameters from your previous code:
    dx, dy, dz = 12, 8, 0.5
    nx, ny, nz = 50, 50, 100  # number of cells in x, y, and z respectively

    # Create grid coordinates for the lower corners of each cell
    x_coords = np.arange(0, nx * dx, dx)  # 0 to 600
    y_coords = np.arange(0, ny * dy, dy)  # 0 to 400
    z_coords = np.arange(0, nz * dz, dz)  # 0 to 50

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
    occupancy_dict_computational_time = time.time() - start_time

    # extract free points inside occupancy_dict
    candidate_points = []
    for key, occ in occupancy_dict.items():
        if occ == 0:
            candidate_points.append(np.array(key))
    candidate_points = np.array(candidate_points)
    if candidate_points.shape[0] == 0:
        raise ValueError("No candidate free points found in occupancy_dict.")
    airspace_boundaries = (x_min, x_max, y_min, y_max, z_min, z_max)
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    airspace_grid = np.stack((X, Y, Z), axis=-1)  # grid.shape is (50, 50, 100, 3)

    # ==========================================================
    # MCTS Implementation with a Placeholder for Simulation
    # ==========================================================

    # MCTS scalar: Larger scalar increases exploitation, smaller increases exploration.
    SCALAR = 1 / (2 * math.sqrt(2.0))
    # logging.basicConfig(level=logging.WARNING)
    # logger = logging.getLogger('MCTSLogger')

    # start condition:
    host_uav_start = [259.47, 145.66, 35]
    start, end = generate_start_end_positions(candidate_points, min_distance=400, boundary_margin=7.5,
                                              boundaries=airspace_boundaries)
    intruder_uav_start = [396.39, 109.09, 35]
    # simulation_host_centre
    # map_simulation_position_to_grid_start = time.time()
    occupied_host, host_uav_simulation_start = is_point_in_occupied_cell(host_uav_start, dx, dy, dz, occupancy_dict)
    occupied_intruder, intruder_uav_simulation_start = is_point_in_occupied_cell(intruder_uav_start, dx, dy, dz,
                                                                                 occupancy_dict)
    # end_map_simulation_position_to_grid = time.time() - map_simulation_position_to_grid_start

    rx, ry, rz = relative_distance(host_uav_simulation_start, intruder_uav_simulation_start)

    # Drone speeds:
    host_speed = 10  # m/s
    intruder_speed = 15  # m/s

    host_desired = desired_velocity(host_uav_simulation_start, intruder_uav_simulation_start, host_speed)
    intruder_desired = desired_velocity(intruder_uav_simulation_start, host_uav_simulation_start, intruder_speed)

    action_path_gama = 0  # in degree
    action_heading_chi = 0  # in degree

    actionSpace_path_gama = [-10, 0, 10]  # in degree
    actionSpace_heading_chi = [-25, -15, 0, 15, 25]  # in degree

    delta_t = 3 # in seconds

    host_vx, host_vy, host_vz = decompose_speed_to_three_axis(host_speed, action_path_gama, action_heading_chi)
    intru_vx, intru_vy, intru_vz = intruder_desired[0], intruder_desired[1], intruder_desired[2]  # intruder just follow its desired velocity
    dev_norm, closest_point = deviation_from_path(host_uav_simulation_start, host_uav_simulation_start,
                                                  intruder_uav_simulation_start)
    dev_x, dev_y, dev_z = relative_distance(host_uav_simulation_start, closest_point)
    ini_position_state = [host_uav_simulation_start, intruder_uav_simulation_start]

    root_state = [intru_vx, intru_vy, intru_vz, host_vx, host_vy, host_vz, host_desired[0], host_desired[1],
                  host_desired[2], action_path_gama, action_heading_chi, rx, ry, rz, dev_x, dev_y, dev_z]
    root_state_obj = AirspaceState(state_vector=root_state, time_step=delta_t, position_state=ini_position_state)
    current_node = Node(root_state_obj)

    # Initialize historical trajectories
    host_trajectory = []
    intruder_trajectory = []
    # Store the initial positions
    host_trajectory.append(current_node.state.host_position)
    intruder_trajectory.append(current_node.state.intruder_position[0])
    for step in range(args.steps):
        print(f"step {step + 1}")
        decision_thinking_start = time.time()
        for l in range(args.levels):
            UCTSEARCH(args.num_sims / (l + 1), current_node)  # Just expand the MCTS tree, don't update `current_node`
            # print("end for search of 1 depth")
        # Find the best immediate action after MCTS search
        # Now choose the best immediate child at the root
        best_child = BESTCHILD(current_node, 0)  # Pure exploitation at the root, since we are just choose the best child
        decision_time = time.time() - decision_thinking_start
        print("time taken for making one decision is {} milliseconds".format(decision_time*1000))
        if best_child is not None:
            best_next_action = best_child.state.moves[0]  # Get the last action taken
            reward_for_current_action = best_child.state.reward()
            distance_to_goal = np.linalg.norm(np.array(best_child.state.host_position) - np.array(intruder_uav_simulation_start))
            print(f"Best Next Action (after {args.levels} lookahead steps): {best_next_action}")
            print("The distance to goal is {}".format(distance_to_goal))
            print('The reward produce by the best action is {}'.format(reward_for_current_action))
        else:
            print("No valid best action found. Terminating simulation.")
            # Record last position upon termination for debug via trajectory history
            host_trajectory.append(current_node.state.host_position)
            intruder_trajectory.append(current_node.state.intruder_position[0])
            if check_collision_with_grid(current_node.state.host_position, 7.5, occupancy_dict, dx, dy, dz):
                print("Termination Reason: Host UAV collided with an obstacle.")
            elif math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(current_node.state.host_position, current_node.state.intruder_position[0]))) < 15:
                print("Termination Reason: Host UAV collided with the intruder.")
            else:
                host_x, host_y, host_z = current_node.state.host_position
                if (host_x - x_min < 7.5) or (x_max - host_x < 7.5) or \
                        (host_y - y_min < 7.5) or (y_max - host_y < 7.5) or \
                        (host_z < 7.5) or (z_max - host_z < 7.5):
                    print("Termination Reason: The host UAV has collided with the boundary.")
            plot_trajectories(building_polygons, host_trajectory, intruder_trajectory, x_limit, y_limit)
            break  # Exit all episodes

        # Record the new positions in trajectory history
        host_trajectory.append(best_child.state.host_position)
        intruder_trajectory.append(best_child.state.intruder_position[0])

        # Update current node to best child, but start with a new root
        new_ini_position_state = [best_child.state.host_position, best_child.state.intruder_position[0]]
        new_root_state_obj = AirspaceState(state_vector=best_child.state.state_vector, time_step=delta_t, position_state=new_ini_position_state)

        current_node = Node(new_root_state_obj)

        # Check if the next state leads to termination
        if current_node.state.terminal():
            print("Simulation Terminated.")
            if check_collision_with_grid(current_node.state.host_position, 7.5, occupancy_dict, dx, dy, dz):
                print("Termination Reason: Host UAV collided with an obstacle.")
            elif math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(current_node.state.host_position, current_node.state.intruder_position[0]))) < 15:
                print("Termination Reason: Host UAV collided with the intruder.")
            elif math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(current_node.state.host_position, intruder_uav_simulation_start))) < 10:
                print("Termination Reason: Host UAV reached its target.")
            else:
                host_x, host_y, host_z = current_node.state.host_position
                if (host_x - x_min < 7.5) or (x_max - host_x < 7.5) or \
                        (host_y - y_min < 7.5) or (y_max - host_y < 7.5) or \
                        (host_z < 7.5) or (z_max - host_z < 7.5):
                    print("Termination Reason: The host UAV has collided with the boundary.")
            plot_trajectories(building_polygons, host_trajectory, intruder_trajectory, x_limit, y_limit)
            break  # Stop running further episodes
        print("================================\n")