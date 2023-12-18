import numpy as np
import cv2
import tensorflow as tf
import subprocess
from SAM import create_mask
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from utils import click_points_simple


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


def get_dilated_mask(img_path, dilation):
    # use same procedure as before to get the mask in 2D, then dilate
    img = Image.open(img_path)
    
    # asarray() class is used to convert
    # PIL images into NumPy arrays
    numpydata = np.asarray(img)
    left_coords, right_coords = click_points_simple(numpydata)

    num_left = len(left_coords)
    num_right = len(right_coords)

    fore_back = [1 for _ in range(num_left)] + [0 for _ in range(num_right)]

    mask, img = create_mask(img_path, np.array(left_coords + right_coords), np.array(fore_back), 'base')

    cv2.imwrite('original_mask.jpg', mask)


def get_transformed_points(image_path, disp_path, sam_mask):
    transformation_matrix = getTransformationMatrix()

    # disparity_map from raft (for testing used google colab)
    
    disp = np.load(disp_path)

    #mask from SAM
    sam_mask = sam_mask.astype('uint8')

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
    depth_image_wound = cv2.bitwise_and(depth_image, depth_image, mask=sam_mask)

    #get point cloud

    Z_wound = -depth_image_wound
    X_wound = (x - cx) * Z_wound / fx
    Y_wound = (y - cy) * Z_wound / fy
    wound_points = np.column_stack((X_wound.flatten(), Y_wound.flatten(), Z_wound.flatten()))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.scatter3D(X_wound.flatten(), Y_wound.flatten(), Z_wound.flatten())
    # plt.title("Allied Points")
    # plt.show()
    print("wound_points", wound_points.shape)

    # projecting onto left image
    left_image = cv2.imread(image_path)
    left_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.uint8)
    image_height, image_width, _ = left_image.shape
    left_camera_matrix = np.array(
        [[1688.10117, 0, 657.660185], [0, 1688.10117, 411.400296], [0, 0, 1]],
        dtype=np.float64,
    )
    left_dist_coeffs = np.array(
        [-0.13969738, 0.28183828, -0.00836148, -0.00180531, -1.65874481], dtype=np.float64
    )
    if not wound_points.shape[0] == 0:
        projected_points_wound, _ = cv2.projectPoints(
            wound_points,
            np.zeros(3),
            np.zeros(3),
            left_camera_matrix,
            distCoeffs=left_dist_coeffs,
        )
        image_points_wound = np.squeeze(projected_points_wound, axis=1).astype(int)
        for i in range(image_points_wound.shape[0]):
            x = int(image_points_wound[i, 0])
            y = int(image_points_wound[i, 1])
            if 0 <= x < image_width and 0 <= y < image_height:
                left_image[y, x] = 255
    # Visualizing the projection on the image
    cv2.namedWindow('Projected Points', cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.imshow('Projected Points', left_image)  # Show the modified image

    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()  # Close all OpenCV windows


    #convert point cloud to overhead coordinates
    R, t = transformation_matrix[1:], transformation_matrix[0]
    overhead_wound_points = []
    for pt in wound_points:
        overhead_wound_points.append(R @ pt + t)
    overhead_wound_points = np.array(overhead_wound_points)
    overhead_wound_points_transpose = overhead_wound_points.T
    ax.scatter3D(overhead_wound_points_transpose[0], overhead_wound_points_transpose[1], overhead_wound_points_transpose[2])
    plt.title("overhead points")
    plt.show()
    print(overhead_wound_points.shape)

disp_path = "RAFT/disp.npy"
img_path = "image_left_001.png"

# get the mask, save it
dilation = 5
get_dilated_mask(img_path, dilation)

sam_mask = cv2.imread("original_mask.jpg", cv2.IMREAD_GRAYSCALE)
get_transformed_points(img_path, disp_path, sam_mask)