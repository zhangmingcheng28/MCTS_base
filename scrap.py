np.linalg.norm(np.array(a) - np.array(b))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
host_start = host_uav_simulation_start
host_end = host_uav_simulation_end
# Plot the host UAV's historical trajectory
if host_trajectory:
    host_traj = np.array(host_trajectory)
    ax.plot(host_traj[:, 0], host_traj[:, 1], host_traj[:, 2], color='blue', marker='o',
            label='Host Trajectory')

    ax.scatter(host_traj[-1, 0], host_traj[-1, 1], host_traj[-1, 2], color='yellow', marker='o',
               label='last_traj')

    # Mark first and last points for the host UAV
    ax.scatter(host_start[0], host_start[1], host_start[2], color='cyan', s=10, label='Host Start',
               marker='D')
    ax.scatter(host_end[0], host_end[1], host_end[2], color='red', s=10,
               label='Host End', marker='X')
for ea_node in host_nodes:
    ori_x = ea_node.state.host_position[0]
    ori_y = ea_node.state.host_position[1]
    ori_z = ea_node.state.host_position[2]
    ori_gamma = ea_node.state.state_vector[9]
    ori_chi = ea_node.state.state_vector[10]
    all_avil_act = [(gama, chi) for gama in actionSpace_path_gama for chi in actionSpace_heading_chi]
    for gamma_deg, chi_deg in all_avil_act:
        action_path_gama_rad = deg2rad(ori_gamma+gamma_deg)
        action_heading_chi_rad = deg2rad(ori_chi+chi_deg)
        Vx = 10 * math.cos(action_path_gama_rad) * math.cos(action_heading_chi_rad)
        Vy = 10 * math.cos(action_path_gama_rad) * math.sin(action_heading_chi_rad)
        Vz = 10 * math.sin(action_path_gama_rad)
        new_pt_x = ori_x+ Vx * 3
        new_pt_y = ori_y+ Vy * 3
        new_pt_z = ori_z+ Vz * 3
        ax.scatter(new_pt_x, new_pt_y, new_pt_z, color='black', s=10,
                   label='all_pts', marker='X')

# Set limits in the 3D plot
ax.set_xlim(x_limit)
ax.set_ylim(y_limit)
ax.set_zlim(0, 50)
# Labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Height')
# Add legend to distinguish trajectories
ax.legend()
plt.show()


-------------------------------------------------------------------------------
plotly

fig = go.Figure()

# --- Plot each building ---
for poly in building_polygons:
    # For each building, we want to plot both the walls and the roof.
    n = len(poly)

    # --- Walls ---
    # For each edge in the polygon, create a quadrilateral representing a wall.
    for i_pt in range(n):
        j_pt = (i_pt + 1) % n
        # For each wall, assume the wall goes from ground (z=0) to the building height at that edge.
        # We create a quadrilateral with vertices:
        # v0: (poly[i_pt][0], poly[i_pt][1], 0)
        # v1: (poly[j_pt][0], poly[j_pt][1], 0)
        # v2: (poly[j_pt][0], poly[j_pt][1], poly[j_pt][2])
        # v3: (poly[i_pt][0], poly[i_pt][1], poly[i_pt][2])
        v0 = [poly[i_pt][0], poly[i_pt][1], 0]
        v1 = [poly[j_pt][0], poly[j_pt][1], 0]
        v2 = [poly[j_pt][0], poly[j_pt][1], poly[j_pt][2]]
        v3 = [poly[i_pt][0], poly[i_pt][1], poly[i_pt][2]]
        quad = [v0, v1, v2, v3]
        vertices, tri_i, tri_j, tri_k = quad_to_triangles(quad)
        # Unzip vertices into x, y, z lists
        x_vals = [v[0] for v in vertices]
        y_vals = [v[1] for v in vertices]
        z_vals = [v[2] for v in vertices]

        fig.add_trace(go.Mesh3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            i=tri_i,
            j=tri_j,
            k=tri_k,
            opacity=0.3,
            color='green',
            name='Building Wall',
            showscale=False
        ))

    # --- Roof ---
    # Use fan triangulation for the roof.
    # Roof vertices are taken directly from poly.
    vertices, tri_i, tri_j, tri_k = fan_triangulation(poly)
    x_roof = [v[0] for v in vertices]
    y_roof = [v[1] for v in vertices]
    z_roof = [v[2] for v in vertices]
    fig.add_trace(go.Mesh3d(
        x=x_roof,
        y=y_roof,
        z=z_roof,
        i=tri_i,
        j=tri_j,
        k=tri_k,
        opacity=0.1,
        color='green',
        name='Building Roof',
        showscale=False
    ))

# --- Plot candidate points ---
fig.add_trace(go.Scatter3d(
    x=candidate_points_filter[:, 0],
    y=candidate_points_filter[:, 1],
    z=candidate_points_filter[:, 2],
    mode='markers',
    marker=dict(size=0.1, color='blue'),
    name='Candidate Points'
))

# Set scene limits and labels
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X (m)', range=[x_limit[0], x_limit[1]]),
        yaxis=dict(title='Y (m)', range=[y_limit[0], y_limit[1]]),
        zaxis=dict(title='Height (m)',
                   range=[0, max([max(poly, key=lambda p: p[2])[2] for poly in building_polygons]) + 10])
    ),
    title="3D Visualization of Buildings and Candidate Points"
)

fig.show()

# can_px = candidate_points[:, 0]
# can_py = candidate_points[:, 1]
# can_pz = candidate_points[:, 2]
# trace = go.Scatter3d(
#     x=can_px,
#     y=can_py,
#     z=can_pz,
#     mode='markers',
#     marker=dict(
#         size=0.1,
#         color='blue'
#     )
# )
#
# data = [trace]
# layout = go.Layout(
#     scene=dict(
#         xaxis=dict(title='X'),
#         yaxis=dict(title='Y'),
#         zaxis=dict(title='Z')
#     ),
#     title='3D Scatter Plot of 230,000 Points'
# )
# fig = go.Figure(data=data, layout=layout)
# fig.show()

------------------------- # for plot velocity vectors:
# Create figure and 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot vectors from the origin (0,0,0)
ax.quiver(0, 0, 0, host_vel_desired[0], host_vel_desired[1], host_vel_desired[2],
          color='r', label='Desired Velocity', linewidth=2)
ax.quiver(0, 0, 0, host_vel_current[0], host_vel_current[1], host_vel_current[2],
          color='b', label='Current Velocity', linewidth=2)

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Velocity Vectors in 3D Space')

# Set limits dynamically based on vector magnitudes
max_range = max(np.linalg.norm(host_vel_desired), np.linalg.norm(host_vel_current)) * 1.2
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

# Add legend
ax.legend()

# Display the plot
plt.show()


from geopy.distance import geodesic

def parse_dms(dms_str):
    parts = dms_str.replace('°', ' ').replace('′', ' ').replace('″', ' ').split()
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    direction = parts[3]  # 'N', 'S', 'E', 'W'

    decimal_degrees = degrees + minutes / 60 + seconds / 3600

    # Apply negative sign for South and West coordinates
    if direction in ['S', 'W']:
        decimal_degrees *= -1

    return decimal_degrees

def compute_holding_fix_distances(wsss_coord, holding_fixes):
    wsss_lat = parse_dms("1° 21′ 33″ N")
    wsss_lon = parse_dms("103° 59′ 22″ E")
    wsss_pos = (wsss_lat, wsss_lon)

    holding_fix_distances = {}

    for fix, (lat_dms, lon_dms) in holding_fixes.items():
        lat = parse_dms(lat_dms)
        lon = parse_dms(lon_dms)

        fix_pos = (lat, lon)
        distance_nm = geodesic(wsss_pos, fix_pos).nautical

        holding_fix_distances[fix] = (lat, lon, distance_nm)

    return holding_fix_distances

# Example data from the table
holding_fixes = {
    "SAMKO_Holding_Area": ("1° 5′ 30″ N", "103° 52′ 55″ E"),
    "NYLON_Holding_Area": ("1° 36′ 57″ N", "104° 6′ 24″ E"),
    "HOSBA_Holding_Area": ("1° 19′ 48″ N", "104° 24′ 18″ E"),
    "BATAM_Holding_Area": ("1° 8′ 13″ N", "104° 7′ 54″ E"),
    "BOBAG_Holding_Area": ("1° 2′ 33″ N", "103° 29′ 54″ E"),
    "VAMPO_Holding_Area": ("0° 58′ 33″ N", "103° 35′ 48″ E"),
    "ELALO_Holding_Area": ("4° 12′ 40″ N", "104° 33′ 35″ E"),
    "KILOT_Holding_Area": ("3° 2′ 17″ N", "104° 40′ 23″ E"),
    "KEXAS_Holding_Area": ("1° 10′ 19″ N", "104° 48′ 44″ E"),
    "REMES_Holding_Area": ("0° 28′ 26″ N", "105° 32′ 35″ E"),
    "MABAL_Holding_Area": ("0° 3′ 44″ N", "105° 32′ 16″ E"),
    "REPOV_Holding_Area": ("0° 16′ 23″ N", "104° 3′ 0″ E"),
    "UGEBO_Holding_Area": ("0° 35′ 24″ N", "104° 2′ 32″ E"),
    "PASPU_Holding_Area": ("0° 15′ 51″ N", "104° 6′ 18″ E"),
}

wsss_coord = ('012133N', '1035922E')
holding_fix_distances = compute_holding_fix_distances(wsss_coord, holding_fixes)

# Print results
for fix, (lat, lon, dist) in holding_fix_distances.items():
    print(f"{fix}: Lat={lat:.6f}, Lon={lon:.6f}, Distance={dist:.2f} NM")