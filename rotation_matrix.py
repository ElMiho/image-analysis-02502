import numpy as np

def rotation_matrix(pitch, roll, yaw):
    rx = np.array([[1, 0, 0, 0], [0, np.cos(pitch), -np.sin(pitch), 0],
                   [0, np.sin(pitch), np.cos(pitch), 0], [0, 0, 0, 1]])
    
    ry = np.array([[np.cos(roll), 0, np.sin(roll), 0], [0, 1, 0, 0],
                   [-np.sin(roll), 0, np.cos(roll), 0], [0, 0, 0, 1]])
    
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], [np.sin(yaw), np.cos(yaw), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    
    return rx @ ry @ rz