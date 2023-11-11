import numpy as np
import cv2
import tensorflow as tf
import subprocess
from PIL import Image
from matplotlib import pyplot as plt
import cv2

def getTransformationMatrix():
    transformation_tensor = tf.io.read_file('transforms/tf_av_left_zivid.tf')
    print(transformation_tensor)
    transformation_string = transformation_tensor.numpy().decode('utf-8')  # Convert to a Python string
    # Split the string into individual lines and elements
    lines = transformation_string.strip().split('\n')[2:]
    matrix_elements = [list(map(float, line.split())) for line in lines]
    # Convert the matrix elements to a NumPy array
    transformation_matrix = np.array(matrix_elements)
    print(transformation_matrix.shape)
    print(transformation_matrix)
    return transformation_matrix
transformation_matrix = getTransformationMatrix()

# disparity_map from raft (for testing used google colab)
disp_path = "RAFT/suturing.npy"
disp = np.load(disp_path)

#calibaration
f = 1688.10117
cx = 657.660185
cy = 411.400296
Tx = -0.045530
cx_diff = 671.318549 - cx

# get depth map from disparity map
depth_image = (f * Tx) / abs(disp + cx_diff)
fx, fy, cx, cy = f, f, cx, cy
rows, cols = depth_image.shape
y, x = np.meshgrid(range(rows), range(cols), indexing="ij")
depth_image_wound = cv2.bitwise_and(depth_image, depth_image, mask=None) #todo: add mask here

#get point cloud

Z_wound = -depth_image_wound
X_wound = (x - cx) * Z_wound / fx
Y_wound = (y - cy) * Z_wound / fy
wound_points = np.column_stack((X_wound.flatten(), Y_wound.flatten(), Z_wound.flatten()))

#convert point cloud to overhead coordinates
R, t = transformation_matrix[1:], transformation_matrix[0]
overhead_wound_points = []
for pt in wound_points:
    overhead_wound_points.append(R @ pt + t)
overhead_wound_points = np.array(overhead_wound_points)
print(overhead_wound_points.shape)