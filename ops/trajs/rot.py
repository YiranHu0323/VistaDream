import torch
import numpy as np

def generate_rot_trajectory(num_frames, radius):
    t = np.linspace(0, 1, num_frames)
    r = radius * 1.5
    theta = 2 * np.pi * t
    x = r * np.cos(theta)
    y = np.full(num_frames, 5)
    z = r * np.sin(theta)
    return x, y, z 

def look_at(camera_position, target_position):
    # look at direction
    direction = target_position - camera_position
    direction /= np.linalg.norm(direction)
    # calculate rotation matrix
    up = np.array([0, 1, 0])
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    rotation_matrix = np.vstack([right, up, direction])
    rotation_matrix = np.linalg.inv(rotation_matrix)
    return rotation_matrix

def rot_camera_poses(num_frames, radius, forward_ratio = 0.2, backward_ratio = 0.8, rotation_times = 0.3, look_at_times = 0.5):
    x, y, z = generate_rot_trajectory(num_frames, radius)
    target_position = np.array([0,0,radius*look_at_times])
    camera_positions = np.vstack([x, y, z]).T
    camera_poses = []
    
    for pos in camera_positions:
        rotation_matrix = look_at(pos, target_position)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3,  3] = pos
        camera_poses.append(transform_matrix[None])
        
    camera_poses.reverse()
    camera_poses = np.concatenate(camera_poses,axis=0)
    
    return camera_poses