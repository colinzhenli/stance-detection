from ast import IsNot
import os
import open3d as o3d
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import colors

def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1]/vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0]/vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)

def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return(mesh_frame)

def get_arrow(origin=[0, 0, 0], end=None, vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 0.1
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)

def get_vertices(box):
    points = np.asarray(box.get_box_points())
    reorder_index = [1, 0, 2, 7, 6, 3, 5, 4]
    reorder_points = np.copy(points)
    for i in range(8):
        reorder_points[i] = points[(reorder_index[i])]
    return reorder_points

def get_directions(box):
    trans_inv = box.R
    # obb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
    # obb_frame.translate(box.center)
    # obb_frame.rotate(trans_inv, center)
    up_direction_set = {}
    up_direction_set[0] = np.array((0.0, 1.0, 0.0))
    up_direction_set[1] = np.array((0.0, -1.0, 0.0))
    front_direction_set = {}
    front_direction_set[0] = np.matmul(trans_inv, np.array((1.0, 0.0, 0.0)))
    front_direction_set[1] = np.matmul(trans_inv, np.array((-1.0, 0.0, 0.0)))
    front_direction_set[2] = np.matmul(trans_inv, np.array((0.0, 0.0, 1.0)))
    front_direction_set[3] = np.matmul(trans_inv, np.array((0.0, 0.0, -1.0)))
    index_up = random.randint(0, 1)
    index_front = random.randint(0, 3)
    return [up_direction_set[index_up], front_direction_set[index_front]]
def custom_draw_geometry_with_camera_trajectory(pcd, camera_trajectory_path,
                                                render_option_path,
                                                output_path):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
        o3d.io.read_pinhole_camera_trajectory(camera_trajectory_path)
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    image_path = os.path.join(output_path, 'image')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    depth_path = os.path.join(output_path, 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    print("Saving color images in " + image_path)
    print("Saving depth images in " + depth_path)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            # Capture and save image using Open3D.
            vis.capture_depth_image(
                os.path.join(depth_path, "{:05d}.png".format(glb.index)), False)
            vis.capture_screen_image(
                os.path.join(image_path, "{:05d}.png".format(glb.index)), False)

            # Example to save image using matplotlib.
            # depth = vis.capture_depth_float_buffer()
            # image = vis.capture_screen_float_buffer()
            # image_array = np.asarray(image)
            # plt.imsave(os.path.join(depth_path, "{:05d}.png".format(glb.index)),
            #            np.asarray(depth),
            #            dpi=1)
            # plt.imsave(os.path.join(image_path, "{:05d}.png".format(glb.index)),
            #            np.asarray(image),
            #            dpi=1)

        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.destroy_window()

        # Return false as we don't need to call UpdateGeometry()
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    obb = pcd.get_oriented_bounding_box()
    aabb = pcd.get_axis_aligned_bounding_box()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json(render_option_path)
    vis.register_animation_callback(move_forward)
    vis.run()

def o3d_render(pcd_set, obb_set, output=None, win_width=640, win_height=480, with_diretions = False):
    center_a = (obb_set[0]["obb"]).get_center()
    R_a = obb_set[0]["obb"].R
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [1, 5],
        [0, 4],
        [2, 6],
        [3, 7],
    ]
    pcd_a = pcd_set[0]
    obb_points_a = get_vertices(obb_set[0]["obb"])
    line_set_a = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(obb_points_a),
        lines=o3d.utility.Vector2iVector(lines),
    )
    up_arrow_a = get_arrow(np.matmul(R_a, center_a), vec=obb_set[0]["directions"][0])
    front_arrow_a = get_arrow(np.matmul(R_a, center_a), vec=obb_set[0]["directions"][1])
    up_arrow_a.paint_uniform_color(colors.to_rgb('green'))
    front_arrow_a.paint_uniform_color(colors.to_rgb('black'))
    line_set_a.paint_uniform_color(colors.to_rgb('green'))
    pcd_a.paint_uniform_color(colors.to_rgb('blue'))
    if len(pcd_set) > 1:
        pcd_b = pcd_set[1]
        obb_points_b = get_vertices(obb_set[1]["obb"])
        line_set_b = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(obb_points_b),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set_b.paint_uniform_color(colors.to_rgb('yellow'))
        pcd_b.paint_uniform_color(colors.to_rgb('red'))
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Pointclouds', width=win_width, height=win_height, visible=True)
    vis.add_geometry(pcd_a)
    vis.add_geometry(line_set_a)
    if len(pcd_set) > 1:
        vis.add_geometry(pcd_b)
        vis.add_geometry(line_set_b)
    if with_diretions:
        vis.add_geometry(up_arrow_a)
        vis.add_geometry(front_arrow_a)
    if output:
        vis.capture_screen_image(filename=output, do_render=True)