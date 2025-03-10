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
import plotly.graph_objs as go
import pickle
import math
import hashlib
import logging
import time
import numba
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

    def __init__(self, state_vector=None, moves=[], time_step=1, position_state=[],intruder_time_indices=None, prev_intruder_positions=None):
        # If no state_vector is provided, we default to (0,0,0) and an empty moves list.
        if state_vector is None:
            self.state_vector = [0, 0, 0]  # default value
        else:
            self.state_vector = state_vector
        self.moves = moves  # Store actions taken
        self.time_step = time_step
        self.host_position = position_state[0]
        # Store intruder positions as a list
        self.intruder_positions = position_state[1:]
        # Initialize intruder_time_indices if not provided.
        if intruder_time_indices is None:
            # Start each intruder at index 0.
            self.intruder_time_indices = [0 for _ in self.intruder_positions]
        else:
            self.intruder_time_indices = intruder_time_indices
        # If not provided, we default to the current intruder positions.
        if prev_intruder_positions is None:
            self.prev_intruder_positions = list(self.intruder_positions)
        else:
            self.prev_intruder_positions = prev_intruder_positions
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
        previous_action_path_gama = self.state_vector[9]  # Previous path angle
        previous_action_heading_chi = self.state_vector[10]  # Previous heading angle
        # rx, ry, rz = self.state_vector[11:14]  # Relative distance
        # dev_x, dev_y, dev_z = self.state_vector[14:]  # Deviation components

        # Update the host UAV angles using the chosen action
        new_path_gama = previous_action_path_gama + gama_action
        new_heading_chi = previous_action_heading_chi + chi_action

        # Recalculate the host UAV’s new velocity components
        host_vx, host_vy, host_vz = decompose_speed_to_three_axis(host_speed, new_path_gama, new_heading_chi)

        # Update host position based on velocity
        new_host_position = (
            self.host_position[0] + host_vx * self.time_step,
            self.host_position[1] + host_vy * self.time_step,
            self.host_position[2] + host_vz * self.time_step
        )

        # Before computing new intruder positions, store the current ones as previous.
        old_intruder_positions = list(self.intruder_positions)

        # Update each intruder's position (they fly with constant velocity)
        # new_intruder_positions = []
        # for idx, intruder_pos in enumerate(self.intruder_positions):
        #     ivx, ivy, ivz = intruder_velocities[idx]  # intruder_velocities is a global list computed in main
        #     new_pos = (
        #         intruder_pos[0] + ivx * self.time_step,
        #         intruder_pos[1] + ivy * self.time_step,
        #         intruder_pos[2] + ivz * self.time_step
        #     )
        #     new_intruder_positions.append(new_pos)
        # Update each intruder's position along its pre-planned A* trajectory:
        new_intruder_positions = []
        new_intruder_time_indices = []

        for idx, curr_pos in enumerate(self.intruder_positions):
            path = intruder_norminal_trajectories[idx]  # Intruder’s A* path
            curr_index = self.intruder_time_indices[idx]
            direction = intruder_directions[idx]  # +1 or -1

            # If path has 0 or 1 waypoints, do nothing
            if len(path) <= 1:
                new_intruder_positions.append(curr_pos)
                new_intruder_time_indices.append(curr_index)
                continue

            dist_to_move = intruder_speed * self.time_step
            Astar_traj = np.array(intruder_norminal_trajectories[idx])

            to_cur_goal = np.array(curr_pos) - Astar_traj[-1]
            dist_to_cur_traj_goal = np.linalg.norm(to_cur_goal)
            if dist_to_move > dist_to_cur_traj_goal:
                dist_to_move = dist_to_cur_traj_goal
            if dist_to_cur_traj_goal < 7.5:
                # TO DO: may be increase the intruder destination, but computational time is too expandsive, once number of intrudedr increase
                # start_point_after_intruder_reach = intruder_norminal_trajectories[idx][-1]
                # start_point_after_intruder_reach, endpt_after_intruder_reach = intruder_generate_end_positions(candidate_points, start_point_after_intruder_reach, min_distance=300, boundary_margin=7.5, boundaries=airspace_boundaries)
                #
                # traj_to_concate = generate_intruder_trajectories_Astar_given_OD(
                #         start_point_after_intruder_reach, endpt_after_intruder_reach, candidate_points,
                #         num_intruders=1,
                #         occupancy_dict=occupancy_dict,
                #         dx=dx, dy=dy, dz=dz,
                #         x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)[0]
                # we just make the intruder to stop at that point.
                new_pos_arr = np.array(curr_pos)
            else:
                # comuting_time = time.time()
                # best_wp, best_Astar_idx = find_target_wp_along_path(Astar_traj, curr_pos, dist_to_move)
                best_wp, best_Astar_idx = optimized_find_target_wp_along_path(Astar_traj, curr_pos, dist_to_move)
                # end_computing_time = time.time() - comuting_time
                # print("computing time is {} seconds".format(end_computing_time))

                # Compute next waypoint index
                next_index = curr_index + direction
                # Ping-pong if out of bounds
                if next_index >= len(path):
                    direction = -1
                    intruder_directions[idx] = direction
                    next_index = curr_index + direction
                elif next_index < 0:
                    direction = 1
                    intruder_directions[idx] = direction
                    next_index = curr_index + direction

                curr_wp = np.array(curr_pos)
                next_wp = best_wp
                to_next = next_wp - curr_wp
                dist_to_next = np.linalg.norm(to_next)
                if dist_to_next == 0:
                    dist_to_next = dist_to_next + 1e-6  # prevent division by zero

                direction_unit = to_next / dist_to_next
                step_vec = direction_unit * dist_to_move
                step_len = np.linalg.norm(step_vec)
                # actual next step position
                new_pos_arr = curr_wp + step_vec

            new_intruder_positions.append(tuple(new_pos_arr))
            new_intruder_time_indices.append(curr_index)

        new_position_state = [new_host_position] + new_intruder_positions

        # Compute new relative distances using the multi-UAV function:
        rx, ry, rz = relative_distance_multiUAV(new_host_position, new_intruder_positions)

        # Compute new desired velocity (host moving toward intruder's start position)
        host_desired = desired_velocity(new_host_position, host_uav_simulation_end, host_speed)
        # Determine the nearest intruder (by Euclidean distance) and compute its desired velocity:
        distances = [math.sqrt((new_host_position[0]-pos[0])**2 + (new_host_position[1]-pos[1])**2 + (new_host_position[2]-pos[2])**2)
                     for pos in new_intruder_positions]
        nearest_idx = distances.index(min(distances))
        nearest_intruder_desired = desired_velocity(new_intruder_positions[nearest_idx],
                                                     intruder_simulation_end_list[nearest_idx],
                                                     intruder_speed)
        # Compute new deviation from path
        dev_norm, closest_point = deviation_from_path(new_host_position, host_uav_simulation_start,
                                                      host_uav_simulation_end)
        dev_x, dev_y, dev_z = relative_distance(new_host_position, closest_point)

        new_state_vector = [
            nearest_intruder_desired[0], nearest_intruder_desired[1], nearest_intruder_desired[2],  # Intruder velocity
            host_vx, host_vy, host_vz,  # Host velocity
            host_desired[0], host_desired[1], host_desired[2],  # Desired velocity
            new_path_gama, new_heading_chi,  # New angles
            rx, ry, rz,  # Relative distances
            dev_x, dev_y, dev_z  # Deviations
        ]

        return AirspaceState(state_vector=new_state_vector, moves=self.moves+[action], position_state=new_position_state, prev_intruder_positions=old_intruder_positions)  # return new_state_vector

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

        # 1. Check collision with an occupied grid (sphere radius 7.5m)
        if check_collision_with_grid(self.host_position, 7.5, occupancy_dict, dx, dy, dz):
            # print("UAV hit obstacles")
            return -1  # UAV has hit an obstacle

        # 2. Check if host UAV and intruder UAV collide (distance < 15m)
        original_host_position = get_original_position(self.host_position, host_speed, delta_t, self.state_vector[9], self.state_vector[10])
        host_line = LineString([tuple(self.host_position[0:2]), tuple(original_host_position[0:2])])
        host_buffered_area = host_line.buffer(7.5)
        # For each intruder:
        for idx, intruder_pos in enumerate(self.intruder_positions):
            # # Use the global intruder_velocities list for intruder's constant velocity.
            # ivx, ivy, ivz = intruder_velocities[idx]
            # # Compute intruder's previous position (assumed linear motion)
            # intruder_original = (
            #     intruder_pos[0] - ivx * delta_t,
            #     intruder_pos[1] - ivy * delta_t,
            #     intruder_pos[2] - ivz * delta_t
            # )
            # intruder_line = LineString([tuple(intruder_pos[0:2]), tuple(intruder_original[0:2])])
            # intruder_buffered_area = intruder_line.buffer(7.5)

            # Instead of using intruder_velocities, use the precomputed trajectory:
            # Retrieve the current time index for this intruder from the state.
            curr_idx = self.intruder_time_indices[idx]
            intruder_original = self.prev_intruder_positions[idx]
            # Create the intruder's swept 2D area:
            intruder_line = LineString([tuple(intruder_pos[0:2]), tuple(intruder_original[0:2])])
            intruder_buffered_area = intruder_line.buffer(7.5)

            # Check if the 2D swept areas intersect.
            if host_buffered_area.intersects(intruder_buffered_area):
                # Now check vertical (z) overlap.
                host_z_min = min(host_z, original_host_position[2])
                host_z_max = max(host_z, original_host_position[2])
                intruder_z_min = min(intruder_pos[2], intruder_original[2])
                intruder_z_max = max(intruder_pos[2], intruder_original[2])
                # If the vertical intervals overlap, we consider it a collision.
                if not (host_z_max < intruder_z_min or intruder_z_max < host_z_min):
                    return -1  # Conflict (collision) detected.

        # 3. Check if host UAV has reached its target (distance < 10m)
        target_x, target_y, target_z = host_uav_simulation_end
        distance_to_target = math.sqrt((host_x - target_x) ** 2 +
                                       (host_y - target_y) ** 2 +
                                       (host_z - target_z) ** 2)
        check_goal_result = check_if_reach_goal(self.host_position, original_host_position, target_x, target_y, target_z)
        if check_goal_result or distance_to_target<=10:
            # print("Terminate due to reach goal")
            return 1

        # 4. Check if host UAV's sphere is too close to any boundary.
        # For a UAV modeled as a sphere with centroid at (host_x, host_y, host_z) radius=7.5m,
        # if the distance from the centroid to any boundary wall is less than 15 m, we consider it a conflict.
        if (host_x - x_min < 7.5) or (x_max - host_x < 7.5) or \
                (host_y - y_min < 7.5) or (y_max - host_y < 7.5) or \
                (host_z < 7.5) or (z_max - host_z < 7.5):
            # print("UAV collided with boundary")
            return -1  # Host UAV has collided with the boundary

        return 0  # Simulation continues

    def reward(self, alpha=0, R_A=30):
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
        alpha = reward_alpha
        R_A = reward_R_A
        # Extract UAV position & intruder position
        host_x, host_y, host_z = self.host_position

        host_vel_current = self.state_vector[3:6]  # Host velocity
        host_vel_desired = self.state_vector[6:9]  # Host's desired velocity
        deviation_desired = (np.array(host_vel_desired) - np.array(host_vel_current))*delta_t

        # original_host_position = get_original_position(self.host_position, host_speed, delta_t, self.state_vector[9], self.state_vector[10])

        # Terminal Conditions
        terminate_states = self.terminal()
        if terminate_states == 1:  # Local Subgoal Arrival Condition
            return 1  # goal arrival give reward = 1
        elif terminate_states == 0:  # Normal operation reward

            # desired path and heading angle (leads to a better search directions)
            # supposed_current_gamma_deg, supposed_current_chi_deg = compute_initial_path_heading_angle(self.host_position, host_uav_simulation_end,
            #                                                         host_speed)

            # deviation_desired_portion = (deviation_desired[0]**2+deviation_desired[1]**2+deviation_desired[2]**2) / R_A  # zn version, not working at all

            # V2 use deviation to norminal path when solely use this V2 can host UAV is able to find it goal within 5 episodes, v2 works, when alpha set to 0
            dev_norminal_x, dev_norminal_y, dev_norminal_z = self.state_vector[14:]  # Deviation components
            norm_dev = np.linalg.norm(np.array([dev_norminal_x, dev_norminal_y, dev_norminal_z]))
            reward_second = max(0, 1 - (norm_dev / R_A))  #
            # V2.1 include reward_first and enables alpha usage, initial is 0.5
            reward_first = normalize_dot_product(np.array(host_vel_desired), np.array(host_vel_current))

            reward_normal = (reward_first * alpha) + (reward_second * (1 - alpha))

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
    parser.add_argument('--episodes', action="store", type=int, default=100,
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
    # Assume building_polygons is a list where each element is a list of vertices: each vertex is [x, y, building_height]
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
    reward_alpha = 0.5
    reward_R_A = 60

    # reaching_analysis
    episode_record = {}
    reach_count = 0

    args.episodes = 1000
    # with open('eps_to_debug.pickle', 'rb') as handle:
    #     b = pickle.load(handle)
    # look_into = []
    # for eps, eps_data in b.items():
    #     if eps_data['reach_status'] == 0:
    #         look_into.append(eps)
    # eps_to_check = 22

    # with open('host_traj_1_intru.pickle', 'rb') as handle:
    #     host_traj = pickle.load(handle)
    # with open('intru_traj_1_intru.pickle', 'rb') as handle:
    #     intru_traj = pickle.load(handle)
    # diff_vec = list(np.array(host_traj) - np.array(intru_traj[0]))
    #
    # diff_list = []
    # for i in diff_vec:
    #     dist_diff = np.linalg.norm(i)
    #     diff_list.append([dist_diff])
    #
    # plot_relative_distances(diff_list, 1)

    for ea_eps in range(args.episodes):
        print(f"start of episode {ea_eps + 1}")
        # host_uav_start = b[eps_to_check]['host_start']
        # host_uav_end = b[eps_to_check]['host_target']

        pre_episode_process_start = time.time()
        host_uav_start, host_uav_end = generate_start_end_positions(candidate_points, min_distance=300, boundary_margin=7.5, boundaries=airspace_boundaries)

        # host_uav_start = [258.0, 204.0, 35.25]  # for one intru case
        # # host_uav_start = [259.47, 145.66, 35]  # for two intru case
        # host_uav_end = [396.39, 109.09, 35]

        # # head-on intruder,one intruder  # one intruder case
        # intruder_uav_start = [396.39, 109.09, 35]
        # intruder_uav_end = [258.0, 204.0, 35.25]

        # two intruder, cross-track
        # intruder1_uav_start = [236.69, 74.64, 50]
        # intruder1_uav_end = [419.17, 180.11, 20]
        #
        # intruder2_uav_start = [500, 103.93, 40]
        # intruder2_uav_end = [212.0, 158.36, 30]


        # intruder_norminal_trajectories = generate_intruder_trajectories_Astar_given_OD(intruder_uav_start, intruder_uav_end, num_intruders=1,
        #                                  occupancy_dict=occupancy_dict, dx=dx, dy=dy, dz=dz,
        #                                  x_min=x_min, x_max=x_max, y_min=x_min, y_max=y_max, z_min=z_min, z_max=z_max)

        # Generate A* planned trajectories for 20 intruders
        num_intruders = 1
        # host_uav_start = (258.0, 348.0, 40.25)
        # host_uav_end = (90.0, 44.0, 41.75)

        # with open('Intruder_norminal_traj.pickle', 'rb') as handle:
        #     intruder_norminal_trajectories = pickle.load(handle)

        intruder_norminal_trajectories = generate_intruder_trajectories_Astar(
            host_uav_start, host_uav_end, candidate_points,
            num_intruders=num_intruders,
            occupancy_dict=occupancy_dict,
            dx=dx, dy=dy, dz=dz,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
            max_offset=100.0,
            alt_range=(20, 70)
        )
        # plot_intruder_normal_trajectories(host_uav_start, host_uav_end, intruder_norminal_trajectories,
        #                               x_limit, y_limit, building_polygons)
        pre_episode_process_time = time.time() - pre_episode_process_start
        print("pre-episode process need {} seconds".format(pre_episode_process_time))

        intruder_indices = [0 for _ in range(num_intruders)]
        intruder_directions = [1 for _ in range(num_intruders)]

        # two intruder, cross-track
        # intruder1_uav_start = [236.69, 74.64, 50]
        # intruder1_uav_end = [419.17, 180.11, 20]
        #
        # intruder2_uav_start = [500, 103.93, 40]
        # intruder2_uav_end = [212.0, 158.36, 30]


        # simulation_host_centre
        # map_simulation_position_to_grid_start = time.time()
        occupied_host_start, host_uav_simulation_start = is_point_in_occupied_cell(host_uav_start, dx, dy, dz, occupancy_dict)
        occupied_host_end, host_uav_simulation_end = is_point_in_occupied_cell(host_uav_end, dx, dy, dz, occupancy_dict)

        intruder_simulation_start_list = [traj[0] for traj in intruder_norminal_trajectories]
        intruder_simulation_end_list = [traj[1] for traj in intruder_norminal_trajectories]  # just for initial step

        # Drone speeds:
        host_speed = 10  # m/s
        intruder_speed = 15  # m/s

        host_desired = desired_velocity(host_uav_simulation_start, host_uav_simulation_end, host_speed)

        # Compute relative distance from host to its nearest intruder (using relative_distance_multiUAV)
        rx, ry, rz = relative_distance_multiUAV(host_uav_simulation_start, intruder_simulation_start_list)

        # Determine the desired velocity for the nearest intruder:
        distances = [math.sqrt((host_uav_simulation_start[0] - pos[0]) ** 2 +
                               (host_uav_simulation_start[1] - pos[1]) ** 2 +
                               (host_uav_simulation_start[2] - pos[2]) ** 2)
                     for pos in intruder_simulation_start_list]
        nearest_idx = distances.index(min(distances))
        nearest_intruder_desired = desired_velocity(intruder_simulation_start_list[nearest_idx],
                                                    intruder_simulation_end_list[nearest_idx],
                                                    intruder_speed)

        # initial path and heading angle
        gamma_deg, chi_deg = compute_initial_path_heading_angle(host_uav_simulation_start, host_uav_simulation_end, host_speed)
        action_path_gama = gamma_deg  # in degree
        action_heading_chi = chi_deg  # in degree

        actionSpace_path_gama = [-10, 0, 10]  # in degree
        # actionSpace_path_gama = [-30, 0, 30]  # in degree
        actionSpace_heading_chi = [-25, -15, 0, 15, 25]  # in degree

        delta_t = 3  # in seconds

        host_vx, host_vy, host_vz = decompose_speed_to_three_axis(host_speed, action_path_gama, action_heading_chi)

        dev_norm, closest_point = deviation_from_path(host_uav_simulation_start, host_uav_simulation_start,
                                                      host_uav_simulation_end)
        dev_x, dev_y, dev_z = relative_distance(host_uav_simulation_start, closest_point)


        root_state = [nearest_intruder_desired[0], nearest_intruder_desired[1], nearest_intruder_desired[2], host_vx, host_vy, host_vz, host_desired[0], host_desired[1],
                      host_desired[2], action_path_gama, action_heading_chi, rx, ry, rz, dev_x, dev_y, dev_z]

        ini_position_state = [host_uav_simulation_start] + intruder_simulation_start_list

        root_state_obj = AirspaceState(state_vector=root_state, time_step=delta_t, position_state=ini_position_state, intruder_time_indices=intruder_indices)
        current_node = Node(root_state_obj)

        # Initialize historical trajectories
        host_trajectory = []
        host_action_series = []
        reward_series = []
        episode_relative_dist_record = []
        host_nodes = [current_node]
        # Before starting the simulation loop, initialize the history container:
        intruder_positions_history = [[] for _ in range(num_intruders)]
        reach = 0
        episode_decision_time = []

        # Store the initial positions
        host_trajectory.append(current_node.state.host_position)

        # For each intruder, append its current position to its history.
        for idx, pos in enumerate(current_node.state.intruder_positions):
            intruder_positions_history[idx].append(pos)


        for step in range(args.steps):
            print(f"step {step + 1}")
            ini_relative_distances = []
            # Record relative distances into a list for future analysis
            if len(episode_relative_dist_record) == 0:
                # Use the root state's intruder positions (i.e. the initial state) to compute initial distances.
                for idx, intruder_pos in enumerate(current_node.state.intruder_positions):
                    init_dist = np.linalg.norm(np.array(current_node.state.host_position) - np.array(intruder_pos))
                    ini_relative_distances.append(init_dist)
                episode_relative_dist_record.append(ini_relative_distances)
            decision_thinking_start = time.time()
            H = sum([1.0 / (l + 1) for l in range(args.levels)])
            for l in range(args.levels):
                level_budget = args.num_sims * (1.0 / (l + 1)) / H
                UCTSEARCH(level_budget, current_node)  # Just expand the MCTS tree, don't update `current_node`
                # print("end for search of 1 depth")
            # Find the best immediate action after MCTS search
            # Now choose the best immediate child at the root
            best_child = BESTCHILD(current_node, 0)  # Pure exploitation at the root, since we are just choose the best child
            decision_time = time.time() - decision_thinking_start
            print("time taken for making one decision is {} milliseconds".format(decision_time*1000))
            # record decision time in second
            episode_decision_time.append(decision_time)
            if best_child is not None:
                best_next_action = best_child.state.moves[0]  # Get the last action taken
                host_action_series.append(best_next_action)
                reward_for_current_action = best_child.state.reward()
                reward_series.append(reward_for_current_action)
                host_nodes.append(best_child)
                distance_to_goal = np.linalg.norm(np.array(best_child.state.host_position) - np.array(host_uav_simulation_end))

                relative_distances = []
                for idx, intruder_pos in enumerate(best_child.state.intruder_positions):
                    rel_distance = np.linalg.norm(np.array(best_child.state.host_position) - np.array(intruder_pos))
                    relative_distances.append(rel_distance)
                episode_relative_dist_record.append(relative_distances)

                print(f"Best Next Action (after {args.levels} lookahead steps): {best_next_action}")
                print("The distance to goal is {}".format(distance_to_goal))
                print('The reward produce by the best action is {}'.format(reward_for_current_action))
            else:
                print("No valid best action found. Terminating simulation.")
                # # Record last position upon termination for debug via trajectory history
                # host_trajectory.append(current_node.state.host_position)
                # intruder_trajectory.append(current_node.state.intruder_positions)
                # if check_collision_with_grid(current_node.state.host_position, 7.5, occupancy_dict, dx, dy, dz):
                #     print("Termination Reason: Host UAV collided with an obstacle.")
                # elif math.sqrt(
                #         sum((a - b) ** 2 for a, b in zip(current_node.state.host_position, current_node.state.intruder_position[0]))) < 15:
                #     print("Termination Reason: Host UAV collided with the intruder.")
                # else:
                #     host_x, host_y, host_z = current_node.state.host_position
                #     if (host_x - x_min < 7.5) or (x_max - host_x < 7.5) or \
                #             (host_y - y_min < 7.5) or (y_max - host_y < 7.5) or \
                #             (host_z < 7.5) or (z_max - host_z < 7.5):
                #         print("Termination Reason: The host UAV has collided with the boundary.")
                # # plot_trajectories(building_polygons, host_trajectory, intruder_trajectory, x_limit, y_limit, host_uav_simulation_start, host_uav_simulation_end)
                break  # Exit all episodes

            # Record the new positions in trajectory history
            host_trajectory.append(best_child.state.host_position)


            # For each intruder, append its current position to its history.
            for idx, pos in enumerate(best_child.state.intruder_positions):
                intruder_positions_history[idx].append(pos)

            # Update current node to best child, but start with a new root
            new_ini_position_state = [best_child.state.host_position] + best_child.state.intruder_positions
            new_root_state_obj = AirspaceState(state_vector=best_child.state.state_vector, time_step=delta_t, position_state=new_ini_position_state, intruder_time_indices=best_child.state.intruder_time_indices, prev_intruder_positions=best_child.state.prev_intruder_positions)

            current_node = Node(new_root_state_obj)

            # Check if the next state leads to termination
            terminate_state = current_node.state.terminal()
            if terminate_state:
                print("Simulation Terminated.")
                if terminate_state == -1:
                    host_x, host_y, host_z = current_node.state.host_position
                    if (host_x - x_min < 7.5) or (x_max - host_x < 7.5) or \
                            (host_y - y_min < 7.5) or (y_max - host_y < 7.5) or \
                            (host_z < 7.5) or (z_max - host_z < 7.5):
                        print("Termination Reason: The host UAV has collided with the boundary.")
                    elif check_collision_with_grid(current_node.state.host_position, 7.5, occupancy_dict, dx, dy, dz):
                        print("Termination Reason: Host UAV collided with an obstacle.")
                    else:
                        print("Termination Reason: Host UAV collided with the intruder.")
                elif current_node.state.terminal() == 1:
                    reach_count = reach_count + 1
                    reach = 1
                    print("Termination Reason: Host UAV reached its target.")
                # plot_multi_intruder_trajectories(building_polygons, host_trajectory, intruder_positions_history,
                #                                  x_limit, y_limit, host_uav_simulation_start, host_uav_simulation_end)
                # plot_one_intru_head_on(building_polygons, host_trajectory, intruder_positions_history,
                #                                  x_limit, y_limit, host_uav_simulation_start, host_uav_simulation_end, delta_t)
                # plot_relative_distances(episode_relative_dist_record, num_intruders, delta_t=delta_t)
                break  # Stop running further episodes
            print("================================\n")

        # ==========================================================
        # At end of one single episode
        # ==========================================================
        # store the episode number
        episode_record[ea_eps] = {'eps_traj': [host_trajectory, intruder_positions_history],
                                  'host_start': host_uav_start,
                                  'host_target': host_uav_end,
                                  'reach_status': reach,
                                  'host_action_series': host_action_series,
                                  'step_used': step,
                                  'reward_series': reward_series,
                                  # Add your recorded distances
                                  'intruder_relative_distances': episode_relative_dist_record,
                                  'decision_time_per_step': episode_decision_time
                                  }

    # ==========================================================
    # End of all episode
    # ==========================================================
    reach_percentage = (reach_count / args.episodes)*100
    print("There are a total of {} scenario reach out of total {} scenario".format(reach_count, args.episodes))
    output_filename = f"simulation_intruder_{num_intruders}_results_episodes_{args.episodes}_alpha_{reward_alpha}_R_A_{reward_R_A}_.pickle"
    with open(output_filename, "wb") as handle:
        pickle.dump([episode_record, reach_count], handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved to {output_filename}")
