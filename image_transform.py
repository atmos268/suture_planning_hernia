import numpy as np
import cv2
import tensorflow as tf
import subprocess
from SAM import create_mask
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from utils import click_points_simple
from largestCC import keep_largest_connected_component
from EdgeDetector import EdgeDetector
from fillHoles import fillHoles

def getTransformationMatrix():
    transformation_tensor = tf.io.read_file('transforms/tf_av_left_zivid.tf')
    # print(transformation_tensor)
    transformation_string = transformation_tensor.numpy().decode('utf-8')  # Convert to a Python string
    # Split the string into individual lines and elements
    lines = transformation_string.strip().split('\n')[2:]
    matrix_elements = [list(map(float, line.split())) for line in lines]
    # Convert the matrix elements to a NumPy array
    transformation_matrix = np.array(matrix_elements)
    # print(transformation_matrix.shape)
    # print(transformation_matrix)
    return transformation_matrix

transformation_matrix = getTransformationMatrix()


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

    # connected comp
    mask = keep_largest_connected_component('original_mask.jpg')
    cv2.imwrite('original_mask.jpg', mask)
    # cv2.imwrite('sam_img.jpg', img)
    
    new_edge_detector = EdgeDetector()
    mask = cv2.imread('original_mask.jpg')
    img_dilated = new_edge_detector.dilate_to_line(mask, 5)
    cv2.imwrite("original_mask.jpg", img_dilated)
    img_dilated = fillHoles('original_mask.jpg')
    cv2.imwrite("original_mask.jpg", img_dilated)

    # now we have our 'original'
    mask = cv2.imread('original_mask.jpg')
    img_dilated = new_edge_detector.dilate_to_line(mask, dilation)
    cv2.imwrite("dilated_mask.jpg", img_dilated)

def get_transformed_points(image_path, depth_image, sam_mask, viz=False, maintain_order=False, order_matrix=None):
    # disparity_map from raft (for testing used google colab)


    #mask from SAM
    sam_mask = sam_mask.astype('uint8')

    # ## for chicken
    # f = 2072.7670967549093
    # cx = 563.9893989562988
    # cy = 464.33528900146484
    # Tx = -0.04637584164697386


    ### for hernia
    # f = 1016.929931640625  # 1
    # f = 847.9979248046875  # 2
    # f = 1112.2060546875   # 3

    ### for youtubu
    # f = 821.169677734375   #1
    # f = 1140.134521484375  #2
    f = 878.539794921875   #3

    height, width = np.shape(depth_image)
    print("depth image shape", depth_image.shape)
    cx = width / 2
    cy = height / 2
    ###

    # get depth map from disparity map
    # depth_image = (f * Tx) / abs(disp + cx_diff)
    fx, fy, cx, cy = f, f, cx, cy
    rows, cols = depth_image.shape
    y, x = np.meshgrid(range(rows), range(cols), indexing="ij")
    depth_image_wound = cv2.bitwise_and(depth_image, depth_image, mask=sam_mask)

    if viz:
        # Normalize the depth image for visualization
        depth_image_normalized = cv2.normalize(depth_image_wound, None, 0, 255, cv2.NORM_MINMAX)

        # Apply a colormap to the depth image
        depth_colormap = cv2.applyColorMap(depth_image_normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)
        # Display the depth image
        cv2.imshow('Depth Image', depth_colormap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #get point cloud

    Z_wound = -depth_image_wound
    X_wound = (x - cx) * Z_wound / fx
    Y_wound = (y - cy) * Z_wound / fy

    # eliminiate extraneous Z values
    flat_X = X_wound.flatten()
    flat_Y = Y_wound.flatten()
    flat_Z = Z_wound.flatten()

    if not maintain_order:

        include_indices = []
        for i in range(len(flat_Z)):
            if flat_Z[i] != 0:
                include_indices.append(i)
        
        cleaned_X = [flat_X[include_idx] for include_idx in include_indices]
        cleaned_Y = [flat_Y[include_idx] for include_idx in include_indices]
        cleaned_Z = [flat_Z[include_idx] for include_idx in include_indices]

        wound_points = np.column_stack((cleaned_X, cleaned_Y, cleaned_Z))

    else:

        flat_order = order_matrix.flatten()

        len_line =int(np.max(flat_order)) + 1

        cleaned_X = [0 for i in range(len_line)]
        cleaned_Y = [0 for i in range(len_line)]
        cleaned_Z = [0 for i in range(len_line)]

        # add in order
        for i in range(len(flat_order)):
            if flat_order[i] != -1:
                cleaned_X[flat_order[i]] = flat_X[i]
                cleaned_Y[flat_order[i]] = flat_Y[i]
                cleaned_Z[flat_order[i]] = flat_Z[i]

        wound_points = np.column_stack((cleaned_X, cleaned_Y, cleaned_Z))
        


    # if viz:
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    # ax.scatter3D(X_wound.flatten(), Y_wound.flatten(), Z_wound.flatten())
    # plt.title("Allied Points")
    # plt.show()

    # Visualizing the projection onto the left image
    # if viz:
    #     left_image = cv2.imread(image_path)
    #     left_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.uint8)
    #     image_height, image_width, _ = left_image.shape
    #     left_camera_matrix = np.array(
    #         [[1688.10117, 0, 657.660185], [0, 1688.10117, 411.400296], [0, 0, 1]],
    #         dtype=np.float64,
    #     )
    #     left_dist_coeffs = np.array(
    #         [-0.13969738, 0.28183828, -0.00836148, -0.00180531, -1.65874481], dtype=np.float64
    #     )
    #     if not wound_points.shape[0] == 0:
    #         projected_points_wound, _ = cv2.projectPoints(
    #             wound_points,
    #             np.zeros(3),
    #             np.zeros(3),
    #             left_camera_matrix,
    #             distCoeffs=left_dist_coeffs,
    #         )
    #         image_points_wound = np.squeeze(projected_points_wound, axis=1).astype(int)
    #         for i in range(image_points_wound.shape[0]):
    #             x = int(image_points_wound[i, 0])
    #             y = int(image_points_wound[i, 1])
    #             if 0 <= x < image_width and 0 <= y < image_height:
    #                 left_image[y, x] = 255
    #     cv2.namedWindow('Projected Points', cv2.WINDOW_NORMAL)  # Create a resizable window
    #     cv2.imshow('Projected Points', left_image)  # Show the modified image

    #     cv2.waitKey(0)  # Wait for any key press
    #     cv2.destroyAllWindows()  # Close all OpenCV windows


    ### chicken
    # #convert point cloud to overhead coordinates
    # R, t = transformation_matrix[1:], transformation_matrix[0]
    # overhead_wound_points = []
    # for pt in wound_points:
    #     overhead_wound_points.append(R @ pt + t) 
    # overhead_wound_points = np.array(overhead_wound_points)
    # overhead_wound_points_transpose = overhead_wound_points.T

    overhead_wound_points = np.array(wound_points)


    
    # if viz:
    #     ax.scatter3D(overhead_wound_points_transpose[0], overhead_wound_points_transpose[1], overhead_wound_points_transpose[2])
    #     plt.title("overhead points")
    #     plt.show()
    #     print("selected points: ", overhead_wound_points.shape)

    return overhead_wound_points
    
    # R, t = transformation_matrix[1:], transformation_matrix[0]
    # left_wound_points = []
    # for pt in overhead_wound_points:
    #     left_wound_points.append(np.linalg.inv(R) @ (pt - t))
    # left_wound_points = np.array(left_wound_points)
    # left_wound_points_transpose = left_wound_points.T
    # # print("Overhead wound points shape", overhead_wound_points_transpose.shape)
    # # print("Left wound points shape", left_wound_points_transpose.shape)
    # # print(left_wound_points.shape == wound_points.shape)
    # # print(type(wound_points))
    # # print("left wound points", left_wound_points[0:10])
    # # print("wound points", wound_points[0:10])
    # #print(np.array_equal(np.array(wound_points), left_wound_points))
    # for i in range(wound_points.shape[0]):
    #     for j in range(wound_points.shape[1]):
    #         if wound_points[i][j] != left_wound_points[i][j]:
    #             print("left", left_wound_points[i])
    #             print("original", wound_points[i])

    # return left_wound_points


if __name__ == "__main__":
    disp_path = "RAFT/disp.npy"

    img_path = "chicken_images/image_left_001.png"

    # get the mask, save it
    dilation = 100
    get_dilated_mask(img_path, dilation)

    sam_mask = cv2.imread("original_mask.jpg", cv2.IMREAD_GRAYSCALE)
    mask_pts = get_transformed_points(img_path, disp_path, sam_mask)

    dilated_sam_mask = cv2.imread("dilated_mask.jpg", cv2.IMREAD_GRAYSCALE)
    surrounding_pts = get_transformed_points(img_path, disp_path, dilated_sam_mask)

    np.save('surrounding_pts.npy', surrounding_pts)
