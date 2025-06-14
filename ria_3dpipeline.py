from EdgeDetector import img_to_line, line_to_spline, line_to_spline_3d, click_points_simple
import tensorflow as tf
from main import suture_display_adj_pipeline
from SuturePlacer import SuturePlacer
from Optimizer3d import Optimizer3d
from MeshIngestor import MeshIngestor
from SutureDisplayAdjust import SutureDisplayAdjust
import math
import scipy.interpolate as inter
import cv2
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from enhance_image import adjust_contrast_saturation
from image_transform import get_transformed_points
from utils import get_mm_per_pixel
import subprocess
from SuturePlacement3d import SuturePlacement3d
import json
import os
import copy


######################
## HELPER FUNCTIONS ##
######################

def calculate_z(mesh, point, smallest_z, largest_z):
    """
    Given (x, y) for a point, find the depth (z).
    
    Args:
        mesh: 3D mesh object
        point: (x, y) coordinates for which to estimate depth
        smaallest_z: minimum depth bound to consider
        largest_z: maximum depth bound to consider
        
    Returns:
        Depth value where the point is closest to the mesh.
    """
    # sample 10,000 depth points within the specified range
    z_arr = np.linspace(smallest_z, largest_z, num=10000)
    min_dist = float('inf')
    min_z = 0
    
    # finds the z-value within in the above range with a point in the mesh that is closest to that depth value
    for z in z_arr:
        # finds the distance between the mesh to (x, y, z) for the sampled z-value
        dist, idx = mesh.get_nearest_point([point[0], point[1], z])
        if dist < min_dist:
            min_dist = dist
            min_z = z

    return min_z


def getTransformationMatrix():
    """
    Loads the transformation matrix from Tensorflow .tf file and converts to a NumPy array.
    
    Args:
        None
    
    Returns:
        np.ndarray: loaded transformation matrix.
    """
    # read the left zivid transform matrix & converts to string
    transformation_tensor = tf.io.read_file('transforms/tf_av_left_zivid.tf')
    transformation_string = transformation_tensor.numpy().decode('utf-8') # Convert to a Python string
    # split the string into individual lines and elements, stored in a matrix
    lines = transformation_string.strip().split('\n')[2:]
    matrix_elements = [list(map(float, line.split())) for line in lines]
    # convert the matrix elements to a NumPy array
    transformation_matrix = np.array(matrix_elements)
    return transformation_matrix

def project3d_to_2d(left_image, points):
    """
    Project 3D points into 2D pixel coordinates using the left camera image.
    
    Args:
        left_image: left camera image.
        points: array of 3D points in camera space.
        
    Returns:
    list: list of correspnding [x, y] pixel coordinates in the image range.
    """
    lst = []
    left_points = np.array(points)

    # projecting onto left image
    image_height, image_width, _ = left_image.shape
    # camera instrinsic matrix for the left camera
    left_camera_matrix = np.array(
        [[2072.7670967549093, 0, 563.9893989562988], [0, 2072.7670967549093, 464.33528900146484], [0, 0, 1]],
        dtype=np.float64,
    )
    # distortion coefficients ([k1, k2, p1, p2, k3])
    # helps estimate the amount of distortion in a camera image
    left_dist_coeffs = np.array(
        [-0.13969738, 0.28183828, -0.00836148, -0.00180531, -1.65874481], dtype=np.float64
    )
    
    # use the camera's instrinsic parameters to map 3D points into 2D image pixels
    if not left_points.shape[0] == 0:
        projected_points_wound, _ = cv2.projectPoints(
            left_points,
            np.zeros(3), # no rotation vector
            np.zeros(3), # no translation vector
            left_camera_matrix,
            distCoeffs=left_dist_coeffs,
            )
        
        # filter and keep points in the projection that fall within the image frame
        image_points_wound = np.squeeze(projected_points_wound, axis=1).astype(int)
        for i in range(image_points_wound.shape[0]):
            x = int(image_points_wound[i, 0])
            y = int(image_points_wound[i, 1])
            if 0 <= x < image_width and 0 <= y < image_height:
                lst.append([x, y])
    return lst

def sigmoid(x, L, k, x0):
    """
    Sigmoid function with parameters to control its shape.
    L: the curve's maximum value
    k: the logistic growth rate or steepness of the curve
    x0: the x-value of the sigmoid's midpoint
    """
    return L / (1 + np.exp(-k * (x - x0)))




###################
## MAIN FUNCTION ##
###################

if __name__ == "__main__":
    ##########################################
    ### VARIABLES TO UPDATE BEFORE RUNNING ###
    ##########################################
    # come back to ?? 
    box_method = True
    # come back to ?? 
    save_figs = True
    # which set of images to use
    chicken_number = 5
    # run synthetic vs physical experiments pipeline
    experiment_mode = "physical"
    # run 3d or 2d pipeline    come back to ?? 
    mode = '3d'
    # pick two random points to generate synthetic splines
    # can either use a random number generator or manually
    # num1, num2 = random.randrange(0, len(mesh.vertex_coordinates)), random.randrange(0, len(mesh.vertex_coordinates))
    # come back to ?? 
    num1, num2 = 21695, 8695
    
    
    # load images and define paths
    left_file = f'left_exp_00{chicken_number}.png'
    left_img_path = 'dan_chicken/' + left_file
    left_img_path_enhanced = 'chicken_images/enhanced/' + left_file
    right_file = f'right_exp_00{chicken_number}.png'
    right_img_path = 'chicken_images/' + right_file
    right_img_path_enhanced = 'chicken_images/enhanced/' + right_file

    # define file paths to store results
    image_pth = f"exp_00{chicken_number}"
    results_pth = "exp_temp"
    baseline_pth = "results/" + results_pth + "/baseline/"
    opt_pth = "results/" + results_pth + "/opt/"
    old_algo_pth = "results/" + results_pth + "/old_algo/"
    final_plan_pth = "final_plans/" + image_pth

    # create directories for results (if they don't already exist)
    if not os.path.isdir("results/"):
        os.mkdir("results/")
    if not os.path.isdir("results/" + results_pth):
        os.mkdir("results/" + results_pth)
    if not os.path.isdir(baseline_pth):
        os.mkdir(baseline_pth)
    if not os.path.isdir(opt_pth):
        os.mkdir(opt_pth)
    if not os.path.isdir(old_algo_pth):
        os.mkdir(old_algo_pth)
    if not os.path.isdir("final_plans/"):
        os.mkdir("final_plans/")
    if not os.path.isdir("final_plans/" + image_pth):
        os.mkdir("final_plans/" + image_pth)
    if not os.path.isdir("masks/"):
        os.mkdir("masks/")
    
    
    
    ##########################################################
    ## EXPERIMENT MODE 1: SYNTHETIC                         ##
    ## Use when running the pipeline on a synthentic spline ##
    ##########################################################
    if experiment_mode == "synthetic":
        # files with adjacency lists and coordinates for the synthetic spline
        # can represent a spline as a bunch of 3d points that are connected to each other (essentially a graph in 3d space)
        # adjacency list: tells which vertices are connected to other vertices
        # coordinates: contains the (x, y, z) position of coordinates on the spline
        adj_path = 'synth_adjacency.txt'
        loc_path = 'synth_coordinates.txt'
        
        # use to generate a mesh object representing the spline (by creating a graph)
        print("Initializing mesh")
        mesh = MeshIngestor(adj_path, loc_path)
        print("Using data to make mesh")
        mesh.generate_mesh()
        
        # come back to ?? why are these the points that are selected as endpoints
        print("calculating shortest path")
        # get the (x, y, z) coordinates of the 2 points in the mesh that are closest to the following set endpoints
        pt0 = mesh.get_point_location(mesh.get_nearest_point([0, -0.9, 1.5])[1])
        pt2 = mesh.get_point_location(mesh.get_nearest_point([1.4, -1, 1.3])[1])
        
        # then, find the (x, y, z) coordinates of the points along the shortest path between these 2 points
        shortest_path = mesh.get_a_star_path(pt0, pt2)
        shortest_path_xyz = np.array([mesh.get_point_location(pt_idx) for pt_idx in shortest_path])
        print('shortest path', shortest_path_xyz)
        
        # find the depth range of the shortest path
        smallest_z, largest_z = min(shortest_path_xyz[:, 2]), max(shortest_path_xyz[:, 2])
        
        # fit a spline to this line (shortest path) & generate a smoothed version
        spline3d = line_to_spline_3d(shortest_path_xyz, sample_ratio=30, viz=False)
        spline3d_smoothed = line_to_spline_3d(shortest_path_xyz, sample_ratio=30, viz=False, s_factor=0.0001)
        # number of points to sample
        granularity = 100
        
        # sample (x, y, z) points from the spline
        x_pts = [spline3d[0](t/granularity) for t in range(granularity)]
        y_pts = [spline3d[1](t/granularity) for t in range(granularity)]
        z_pts = [spline3d[2](t/granularity) for t in range(granularity)]

        # calculate the first and second derivatives of the spline components
        derivative_x, derivative_y, derivative_z = spline3d[0].derivative(), spline3d[1].derivative(), spline3d[2].derivative()
        derivative_x2, derivative_y2, derivative_z2 = spline3d[0].derivative(2), spline3d[1].derivative(2), spline3d[2].derivative(2)
        
        # calculate curvature along the spline
        curvature_arr = []
        for i in range(granularity):
            '''
            How to calculate curvature, K(t), for a curve, f(x):
            1. parameterize as r(t) = <x(t), y(t), z(t)> (3D vector)
            2. K(t) = ||r'(t) x r''(t)|| / ||r'(t)|| ^ 3
            '''
            t = i / granularity
            r_prime = np.array([derivative_x(t), derivative_y(t), derivative_z(t)])
            r_double_prime = np.array([derivative_x2(t), derivative_y2(t), derivative_z2(t)])
            curvature = np.linalg.norm(np.cross(r_prime, r_double_prime)) / np.linalg.norm(r_prime)**3
            curvature_arr.append(curvature)
        
        print('CURVATURE', curvature_arr)
        print("MIN", np.min(curvature_arr))
        print("MAX", np.max(curvature_arr))
        
        # plot curvature along the 3d spline
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title("Spline curvature")
        p = ax.scatter3D(x_pts, y_pts, z_pts, c=curvature_arr)
        fig.colorbar(p)
        plt.show()
        
        # scale curvature to be between 0 and 1
        curvature_arr = np.array(curvature_arr)
        scaled_curvature = curvature_arr / max(curvature_arr)
        L = 1 / 0.5 - 0.77  # The range of the spacing values
        k = 10  # The steepness of the curve
        x0 = 0.5  # The midpoint of the sigmoid

        # Calculate the spacing using the sigmoid function
        spacing = 0.77 + sigmoid(scaled_curvature, L, k, x0)
        print('SPACING', spacing)
        
        # Visualize eccentricity (spacing) over the 3d spline
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title("Sigmoid eccentricity")
        p = ax.scatter3D(x_pts, y_pts, z_pts, c=spacing)
        fig.colorbar(p)
        plt.show()
        
        # define hyperparameters
        suture_width = 0.15
        mm_per_pixel = 10
        c_ideal = 1000
        gamma = suture_width 
        c_var = 1000
        c_shear = 1
        c_closure = 1
        hyperparams = [c_ideal, gamma, c_var, c_shear, c_closure]
        force_model_parameters = {'ellipse_ecc': 1.0, 'force_decay': 0.5/0.15, 'verbose': 0, 'ideal_closure_force': None, 'imparted_force': None}
        
        # load image and optimizer
        left_image = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
        optim3d = Optimizer3d(mesh, spline3d, suture_width, hyperparams, force_model_parameters, spline3d_smoothed, spacing, left_image, synthetic=True)
        
        # use for 3d synthetic splines
        if mode == "3d":
            # display mesh as with spline
            mesh.plot_mesh(shortest_path_xyz)

            center_pts, insertion_pts, extraction_pts = optim3d.generate_inital_placement(mesh, spline3d, num_sutures=8)
            closure_loss, shear_loss, all_closure, per_insertion, per_extraction, insertion_forces, extraction_forces = optim3d.compute_closure_shear_loss(granularity=100)
            optim3d.plot_mesh_path_and_spline()


            fig = plt.figure()
            ax = plt.axes(projection='3d')
            plt.title("CLOSURE FORCES BEFORE")
            p = ax.scatter3D(x_pts, y_pts, z_pts, c=all_closure)
            fig.colorbar(p)
            plt.show()
            
            optim3d.optimize(eval=False)

            closure_loss, shear_loss, all_closure, per_insertion, per_extraction, insertion_forces, extraction_forces = optim3d.compute_closure_shear_loss(granularity=100)
            optim3d.plot_mesh_path_and_spline()

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            plt.title("CLOSURE FORCES AFTER")
            p = ax.scatter3D(x_pts, y_pts, z_pts, c=all_closure)
            fig.colorbar(p)
            plt.show()

            start_range = 4
            end_range = 12

            print("range:", start_range, end_range)

            equally_spaced_losses = {}
            post_algorithm_losses = {}

            best_baseline_loss = 1e8
            best_baseline_placement = None

            best_opt_loss = 1e8
            best_opt_insertion = None
            best_opt_extraction = None
            best_opt_center = None

            final_closure = None
            final_shear = None

            best_optim = None

            for num_sutures in range(start_range, end_range + 1):
                print("num sutures:", num_sutures)
                center_pts, insertion_pts, extraction_pts = optim3d.generate_inital_placement(mesh, spline3d, num_sutures=num_sutures)
                equally_spaced_losses[num_sutures] = optim3d.optimize(eval=True)
                print('Initial loss', equally_spaced_losses[num_sutures]["curr_loss"])
                optim3d.optimize(eval=False)
                
                post_algorithm_losses[num_sutures] = optim3d.optimize(eval=True)
                print('After loss', post_algorithm_losses[num_sutures]["curr_loss"])

                
                if post_algorithm_losses[num_sutures]["curr_loss"] < best_opt_loss:
                    print("num sutures", num_sutures, "best loss so far")
                    best_opt_loss = post_algorithm_losses[num_sutures]["curr_loss"]
                    print("BEST LOSS", best_opt_loss)
                    best_opt_insertion = optim3d.insertion_pts
                    best_opt_extraction = optim3d.extraction_pts
                    best_opt_center = optim3d.center_pts
                    baseline_insertion = insertion_pts
                    baseline_extraction = extraction_pts
                    baseline_center = center_pts
                    _, _, final_closure, _, _, _, _ = optim3d.compute_closure_shear_loss(granularity=100)
                    best_optim = copy.deepcopy(optim3d)

            best_optim.plot_mesh_path_and_spline()
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            plt.title("CLOSURE FORCES FINAL")
            p = ax.scatter3D(x_pts, y_pts, z_pts, c=final_closure)
            fig.colorbar(p)
            plt.show()
        
        # Previous example
        # start = mesh.get_nearest_point([0, -0.9, 0.8])[1]
        # end = mesh.get_nearest_point([0, -0.77, -1])[1]
        # Ken's example
        # end = mesh.get_nearest_point([1, -0.3, -1])[1]
        # start = mesh.get_nearest_point([0.6, -0.15, -2])[1]
        # middle = mesh.get_nearest_point([0.6, -0.15, -2])[1]
        # end = mesh.get_nearest_point([0.6, -0.15, -2])[1]
        # pt1 =  mesh.get_point_location(mesh.get_nearest_point([0.6, 0.2, -2])[1])
        # EXAMPLE 1
        # END EXAMPLE
        # pt1 = mesh.get_point_location(mesh.get_nearest_point([0.5, -0.8, 1.5])[1])
        # pt3 = mesh.get_point_location(mesh.get_nearest_point([-1, -0.4, -1])[1])
        # # pt3 = mesh.get_point_location(mesh.get_nearest_point([1, 0.1, -1])[1])
        # (0.6, -0.2, -2)
        # (0.6, -0.5, -1.9)
        # (0.75, -0.5, -1.7)
        # (0.8, -0.5, -1.5)
        # leg1 = mesh.get_point_location(mesh.get_nearest_point([1, -0.5, -0.25])[1])
        # leg2 = mesh.get_point_location(mesh.get_nearest_point([0.5, -1.5, -0.25])[1])
        
        # print("start and end indices", start, end)
        # start_pt = mesh.get_point_location(start)
        # wound_pt = mesh.get_point_location(end)
        # print("start and end locations", start_pt, wound_pt)
        # + mesh.get_a_star_path(pt2, pt3) + mesh.get_a_star_path(pt3, pt4)

        elif mode == "2d":
            suture_width = 0.005
            # project points onto xy plane
            avg_z = np.mean(shortest_path_xyz[:, 2])
            shortest_path_xy = [[shortest_path_xyz[i][0], shortest_path_xyz[i][1]]for i in range(len(shortest_path_xyz))]

            spline2d, tck = line_to_spline(shortest_path_xy, "", 1, viz=False)

            suture_placer = SuturePlacer(suture_width, mm_per_pixel)
            wound_parametric = lambda t, d: inter.splev(t, tck, der = d)

            # Put the wound into all the relevant objects
            newSuturePlacer = SuturePlacer(suture_width, mm_per_pixel)
            newSuturePlacer.tck = tck
            newSuturePlacer.DistanceCalculator.tck = tck
            newSuturePlacer.wound_parametric = wound_parametric
            newSuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
            newSuturePlacer.RewardFunction.wound_parametric = wound_parametric
            newSuturePlacer.image = left_img_path
            
            # Running 2d algorithm
            b_insert_pts, b_center_pts, b_extract_pts = newSuturePlacer.place_sutures(save_figs=False)

            # helper function to convert points back to 3d and find closest mesh points
            def twoD_to_3D(points):
                points = [pt + [calculate_z(mesh, pt, smallest_z, largest_z)] for pt in points]
                #print(points)
                #return np.array(points)
                points_mesh = np.array([mesh.get_point_location(mesh.get_nearest_point(pt)[1]) for pt in points])
                return points_mesh

            # converting points to 3d and plotting
            b_center_pts, b_insert_pts, b_extract_pts = twoD_to_3D(b_center_pts), twoD_to_3D(b_insert_pts), twoD_to_3D(b_extract_pts)
            center_pts_spline = line_to_spline_3d(b_center_pts, sample_ratio=30, viz=False)
            suturePlacement2dIn3d = SuturePlacement3d(center_pts_spline, b_center_pts, b_insert_pts, b_extract_pts, [])
            optim3d.plot_mesh_path_and_spline(mesh, center_pts_spline, suturePlacement2dIn3d, [], [])

            #print loss of the 2d placement
            loss2d = optim3d.loss_placement(suturePlacement2dIn3d)
            print("loss of 2d placement", str(loss2d))

            
    if mode == '2d' and experiment_mode == "physical":
        adj_path = 'adjacency_matrix.txt'
        loc_path = 'vertex_lookup.txt'
        disp_path = "RAFT/disparity_009.npy"

        # get the largest and smallest value in the mesh

        mesh = MeshIngestor(adj_path, loc_path)

        # Create the graph
        mesh.generate_mesh()

        max_z, min_z = mesh.get_point_location(0)[2], mesh.get_point_location(0)[2]
        for idx in range(mesh.graph.number_of_nodes()):
            max_z = max(max_z, mesh.get_point_location(idx)[2])
            min_z = min(min_z, mesh.get_point_location(idx)[2])

        # get reasonable upper and lower bounds on z
        # use it to convert back to 3d

        suture_width = 0.005 
        c_ideal = 1000
        gamma = suture_width 
        c_var = 1000
        c_shear = 1
        c_closure = 1

        hyperparams = [c_ideal, gamma, c_var, c_shear, c_closure]

        force_model_parameters = {'ellipse_ecc': 1.0, 'force_decay': 0.5/suture_width, 'verbose': 0, 'ideal_closure_force': None, 'imparted_force': None}

        img = Image.open(left_img_path)
    
        # asarray() class is used to convert
        # PIL images into NumPy arrays
        numpydata = np.asarray(img)

        # get scaling information
        mm_indicated = 10
        wound_width = 5
        left_pts, right_pts = click_points_simple(numpydata)

        if len(left_pts) < 2:
            print("not enough points clicked")
        
        mm_per_pixel = get_mm_per_pixel(left_pts[-2], left_pts[-1], mm_indicated)
        
        line, mask = img_to_line(left_img_path, box_method, viz=True, save_figs=save_figs)
        
        # build a line that is scaled to mm size
        scaled_line = []
        for i, elem in enumerate(line):
            scaled_line.append((line[i][0] * mm_per_pixel, line[i][1] * mm_per_pixel))
        
        # add contrast etc. to improve SAM results
        enhanced = adjust_contrast_saturation(img, 3, 1)
        enhanced.save(left_img_path_enhanced)

        scaled_spline, tck = line_to_spline(scaled_line, left_img_path_enhanced, mm_per_pixel, viz=True)

        # run suture placement pipeline
        suture_placer = SuturePlacer(wound_width, mm_per_pixel)
        wound_parametric = lambda t, d: inter.splev(t, tck, der = d)

        # Put the wound into all the relevant objects
        newSuturePlacer = SuturePlacer(wound_width, mm_per_pixel)
        newSuturePlacer.tck = tck
        newSuturePlacer.DistanceCalculator.tck = tck
        newSuturePlacer.wound_parametric = wound_parametric
        newSuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
        newSuturePlacer.RewardFunction.wound_parametric = wound_parametric
        newSuturePlacer.image = left_img_path

        newSuturePlacer.place_sutures(save_figs=save_figs)
        b_insert_pts, b_center_pts, b_extract_pts = newSuturePlacer.b_insert_pts, newSuturePlacer.b_center_pts, newSuturePlacer.b_extract_pts 

        smallest_z = min_z
        largest_z = max_z

        def twoD_to_3D(points):

            # rewrite - use raft disparity map strategy to transfer points to 3d

            # get back to pixels
            pts_pxl = [[int(float(pt[1]) / mm_per_pixel), int(float(pt[0]) / mm_per_pixel)] for pt in points]

            # make mask
            width, height = img.size

            point_mask = np.zeros((height, width))

            for col, row in pts_pxl:
                point_mask[row,col] = 1

            pts_3d = []
            
            for col, row in pts_pxl:
                # create mask with 1 point 
                img_width, img_height = Image.open(left_img_path).size
                line_mask = np.zeros((img_height, img_width))
                line_mask[row, col] = 1
                pt_3d = get_transformed_points(left_img_path, disp_path, line_mask, viz=False)
                pts_3d.append(pt_3d[0])
            pts_3d = np.array(pts_3d)
        
            # convert the spline to 3d using raft

            # return 3d points

            #print(points)
            #return np.array(points)
            points_mesh = np.array([mesh.get_point_location(mesh.get_nearest_point(pt)[1]) for pt in pts_3d])
            return points_mesh
                
        b_center_pts, b_insert_pts, b_extract_pts = twoD_to_3D(b_center_pts), twoD_to_3D(b_insert_pts), twoD_to_3D(b_extract_pts)
        center_pts_spline = line_to_spline_3d(b_center_pts, viz=False)
        suturePlacement2dIn3d = SuturePlacement3d(center_pts_spline, b_center_pts, b_insert_pts, b_extract_pts, [])

        optim3d = Optimizer3d(mesh, center_pts_spline, suture_width, hyperparams, force_model_parameters)
        optim3d.plot_mesh_path_and_spline(mesh, center_pts_spline, suturePlacement2dIn3d, [], [], viz=True)

        left_image_adjust = cv2.imread(left_img_path, cv2.IMREAD_COLOR)

        insertion_pts = project3d_to_2d(left_image_adjust, suturePlacement2dIn3d.insertion_pts)
        center_pts = project3d_to_2d(left_image_adjust, suturePlacement2dIn3d.center_pts)
        extraction_pts = project3d_to_2d(left_image_adjust, suturePlacement2dIn3d.extraction_pts)
        suture_display = SutureDisplayAdjust(insertion_pts, center_pts, extraction_pts, left_image_adjust)
        suture_display.user_display_pnts()

        # print loss of the 2d placement
        # loss2d = optim3d.loss_placement(suturePlacement2dIn3d)
        loss2d = optim3d.optimize(suturePlacement2dIn3d, eval=True)

        json_old = json.dumps(loss2d)

        old_losses_pth = old_algo_pth + "losses.json"
        f = open(old_losses_pth,"w")
        f.write(json_old)
        f.close()
        
        print("loss of 2d placement", str(loss2d))
        optim3d.plot_mesh_path_and_spline(mesh, center_pts_spline, suturePlacement2dIn3d, [], [], viz=False, results_pth=old_algo_pth)

        

        # suture_display_adj_pipeline(newSuturePlacer)

    elif mode == '3d' and experiment_mode == "physical":
        # COMMAND F
        start_time = time.time()
                
        viz = False
        use_prev = False
        suture_width = 0.005
        # suture_width = 0.05

        print("HELLO")
        
        # get the masks
        # save left and right masks

        left_mask_path = f'dan_masks/left_mask{chicken_number}.npy'
        # right_mask_path = f'masks/right_mask{chicken_number}.npy'
        left_line_path = f'dan_masks/left_line{chicken_number}.npy'
        border_pts_path = f'dan_masks/border_pts{chicken_number}.npy'
        # right_line_path = f'masks/right_line{chicken_number}.npy'
        left_dilated_mask_path = f'dan_masks/left_dilated_mask{chicken_number}.jpg'

        if use_prev:
            left_line = np.load(left_line_path)
            left_mask = np.load(left_mask_path)
            border_pts = np.load(border_pts_path)

        else:
            # Right click is not on wound
            print("NOW HERE")
            left_line, left_mask, border_pts = img_to_line(left_img_path, box_method=False, save_figs=save_figs)
            np.save(left_mask_path, left_mask)
            np.save(left_line_path, left_line)
            np.save(border_pts_path, border_pts)
        
        # do raft, no need to do rn, as we are using the existing RAFT output
        disp_path = f"dan_depth/depth_image_00{chicken_number}.npy"
        disp = np.load(disp_path)

        # dilate to get region
        dilation = 100

        kernel = np.ones((dilation, dilation), np.uint8)
        dilated_mask = cv2.dilate(left_mask, kernel, iterations=1) 
        cv2.imwrite(left_dilated_mask_path, dilated_mask) 

        # write out points to a file
        surrounding_pts = get_transformed_points(left_img_path, disp, dilated_mask)
        np.save('surrounding_pts.npy', surrounding_pts)

        left_img = Image.open(left_img_path)
        img_width, img_height = left_img.size
        all = np.ones((img_height, img_width))
        point_cloud_data = get_transformed_points(left_img_path, disp, all)
        np.save(f'dan_point_cloud_data/point_cloud{chicken_number}.npy', point_cloud_data)

        color_left = cv2.cvtColor(cv2.imread(left_img_path),cv2.COLOR_RGB2BGR)
        rgb_cloud_data = color_left.reshape(-1, 3)
        non_zero_indices = np.all(point_cloud_data != [0, 0, 0], axis=-1)
        rgb_cloud_data = rgb_cloud_data[non_zero_indices]
        point_cloud_data = point_cloud_data[non_zero_indices]
        np.save(f'dan_point_cloud_data/rgb_cloud{chicken_number}.npy', rgb_cloud_data)
    
        order_matrix = np.zeros((img_height, img_width), int) - 1
        line_mask = np.zeros((img_height, img_width))

        # assign index to location on image and add to 
        for i, coord in enumerate(left_line):
            line_mask[coord[0], coord[1]] = 1
            order_matrix[coord[0], coord[1]] = i

        line_pts_3d = get_transformed_points(left_img_path, disp, line_mask, viz=False, maintain_order=True, order_matrix=order_matrix)

        border_mask = np.zeros((img_height, img_width))

        # assign index to location on image and add to 
        for i, coord in enumerate(border_pts):
            border_mask[coord[1], coord[0]] = 1
        
        border_pts_3d = get_transformed_points(border_pts_path, disp, border_mask)
        # np.save('surrounding_pts.npy', border_pts_3d)

        # print("border pts", border_pts_3d)

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')

        # ax.grid(False)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        # ax.scatter3D([point[0] for point in border_pts_3d], [point[1] for point in border_pts_3d], [point[2] for point in border_pts_3d])
        # plt.show()

        order_matrix = np.zeros((img_height, img_width), int) - 1
        line_mask = np.zeros((img_height, img_width))

        # assign index to location on image and add to 
        for i, coord in enumerate(left_line):
            line_mask[coord[0], coord[1]] = 1
            order_matrix[coord[0], coord[1]] = i

        line_pts_3d = get_transformed_points(left_img_path, disp, line_mask, viz=False, maintain_order=True, order_matrix=order_matrix)

        np.save('line_pts_3d.npy', line_pts_3d)

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')

        # ax.scatter3D([point[0] for point in line_pts_3d], [point[1] for point in line_pts_3d], [point[2] for point in line_pts_3d])

        # get the spline from the left image
        # since we are not visualizing here, no need for scaling info
        # left_spline = line_to_spline(left_line, None, None, viz=False)
        # will actually need to use line_to_spline_3d (expect 3d points)
        left_spline = line_to_spline_3d(line_pts_3d, sample_ratio=30, viz=False, s_factor=0)
        left_spline_smoothed = line_to_spline_3d(line_pts_3d, sample_ratio=30, viz=False, s_factor=0.0001)
        granularity = 100

        x_pts = [left_spline[0](t/granularity) for t in range(granularity)]
        y_pts = [left_spline[1](t/granularity) for t in range(granularity)]
        z_pts = [left_spline[2](t/granularity) for t in range(granularity)]

        xs_pts = [left_spline_smoothed[0](t/granularity) for t in range(granularity)]
        ys_pts = [left_spline_smoothed[1](t/granularity) for t in range(granularity)]
        zs_pts = [left_spline_smoothed[2](t/granularity) for t in range(granularity)]

        derivative_x, derivative_y, derivative_z = left_spline_smoothed[0].derivative(), left_spline_smoothed[1].derivative(), left_spline_smoothed[2].derivative()
        derivative_x2, derivative_y2, derivative_z2 = left_spline_smoothed[0].derivative(2), left_spline_smoothed[1].derivative(2), left_spline_smoothed[2].derivative(2)

        curvature_arr = []
        for i in range(granularity):
            t = i / granularity
            r_prime = np.array([derivative_x(t), derivative_y(t), derivative_z(t)])
            r_double_prime = np.array([derivative_x2(t), derivative_y2(t), derivative_z2(t)])
            curvature = np.linalg.norm(np.cross(r_prime, r_double_prime)) / np.linalg.norm(r_prime)**3
            # print(f"AT MIDPOINT T {midpt_t} the CURVATURE IS", curvature)
            curvature_arr.append(curvature)

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # plt.title("Spline curvature")
        # print("max curve", max(curvature_arr))
        # p = ax.scatter3D(x_pts, y_pts, z_pts, c=curvature_arr)
        # fig.colorbar(p)
        # ax.grid(False)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])

        # plt.show()

        def sigmoid(x, L, k, x0):
            """
            Sigmoid function with parameters to control its shape.
            L: the curve's maximum value
            k: the logistic growth rate or steepness of the curve
            x0: the x-value of the sigmoid's midpoint
            """
            return L / (1 + np.exp(-k * (x - x0)))

        curvature_arr = np.array(curvature_arr)
        scaled_curvature = curvature_arr / 100
        L = (1/0.4) - (1/0.60)  # The range of the spacing values
        # more curve means 1/0.5 ellipse whereas less curve means greater
        k = 10  # The steepness of the curve
        x0 = 0.5  # The midpoint of the sigmoid

        # Calculate the spacing using the sigmoid function
        spacing = (1/0.60) + sigmoid(scaled_curvature, L, k, x0)
        # print('SPACING', spacing)
        # get mesh from the surrounding points

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # # plt.title("Sigmoid eccentricity")
        # p = ax.scatter3D(x_pts, y_pts, z_pts, c=spacing)
        # fig.colorbar(p)
        # ax.grid(False)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        # plt.show()

        with open("pipeline_xyz_pts.xyz", "w") as f:
            for point in surrounding_pts:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

        
        subprocess.run(["./generate_mesh"])

        # get the saved mesh data
        adj_path = 'adjacency_matrix.txt'
        loc_path = 'vertex_lookup.txt'

        mesh = MeshIngestor(adj_path, loc_path)

        # Create the graph
        mesh.generate_mesh()

        #hyperparameters
        gamma = suture_width # ideal distance between each suture
        c_ideal = 0 # variance from ideal
        c_var = 10 # variance between center points distances
        c_shear = 0 # shear loss
        c_closure = 0 # closure loss

        hyperparams = [c_ideal, gamma, c_var, c_shear, c_closure]

        force_model_parameters = {'ellipse_ecc': 1/0.5, 'force_decay': 0.5/0.005, 'verbose': 0, 'ideal_closure_force': None, 'imparted_force': None}

        left_image = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
        optim3d = Optimizer3d(mesh, left_spline, suture_width, hyperparams, force_model_parameters, left_spline_smoothed, spacing, left_image, border_pts_3d)
    
        spline_length = optim3d.calculate_spline_length(left_spline, mesh)
        # print("Spline length", spline_length)
        # num_sutures_initial = int(spline_length / (gamma)) #TODO: modify later 
        # print("Num sutures initial", num_sutures_initial)

        # TEST CLOSURE FORCE

        # center_pts, insertion_pts, extraction_pts = optim3d.generate_inital_placement(mesh, left_spline, num_sutures=8)
        # closure_loss, shear_loss, all_closure, per_insertion, per_extraction, insertion_forces, extraction_forces = optim3d.compute_closure_shear_loss(granularity=100)
        # optim3d.plot_mesh_path_and_spline()


        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # plt.title("CLOSURE FORCES BEFORE")
        # p = ax.scatter3D(x_pts, y_pts, z_pts, c=all_closure)
        # fig.colorbar(p)
        # plt.show()

        # for i in range(6):
        #     fig = plt.figure()
        #     ax = plt.axes(projection='3d')
        #     # print('per extraction', per_insertion[i] + per_extraction[i])
        #     plt.title(f"CLOSURE FORCES MAGNITUDE AT PT {i}")
        #     p = ax.scatter3D(x_pts, y_pts, z_pts, c=np.array(per_insertion[i]) + np.array(per_extraction[i]))
        #     ax.scatter3D(optim3d.insertion_pts[i][0], optim3d.insertion_pts[i][1], optim3d.insertion_pts[i][2], c='red')
        #     ax.scatter3D(optim3d.center_pts[i][0], optim3d.center_pts[i][1], optim3d.center_pts[i][2], c='green')
        #     ax.scatter3D(optim3d.extraction_pts[i][0], optim3d.extraction_pts[i][1], optim3d.extraction_pts[i][2], c='blue')
        #     fig.colorbar(p)
        #     plt.show()
        

        # for i in range(6):
        #     fig = plt.figure()
        #     ax = plt.axes(projection='3d')
        #     # print('per extraction', per_insertion[i] + per_extraction[i])
        #     plt.title(f"INSERTION FORCE MAGNITUDE AT {i}")
        #     p = ax.scatter3D(x_pts, y_pts, z_pts, c=[np.linalg.norm(force) for force in insertion_forces[i]])
        #     ax.scatter3D(optim3d.insertion_pts[i][0], optim3d.insertion_pts[i][1], optim3d.insertion_pts[i][2], c='red')
        #     ax.scatter3D(optim3d.center_pts[i][0], optim3d.center_pts[i][1], optim3d.center_pts[i][2], c='green')
        #     ax.scatter3D(optim3d.extraction_pts[i][0], optim3d.extraction_pts[i][1], optim3d.extraction_pts[i][2], c='blue')
        #     fig.colorbar(p)
        #     plt.show()
        
        # for i in range(6):
        #     fig = plt.figure()
        #     ax = plt.axes(projection='3d')
        #     # print('per extraction', per_insertion[i] + per_extraction[i])
        #     plt.title(f"CLOSURE MAGNITUDE AT {i}")
        #     p = ax.scatter3D(x_pts, y_pts, z_pts, c=[np.linalg.norm(iforce - eforce) for iforce, eforce in zip(insertion_forces[i], extraction_forces[i])])
        #     ax.scatter3D(optim3d.insertion_pts[i][0], optim3d.insertion_pts[i][1], optim3d.insertion_pts[i][2], c='red')
        #     ax.scatter3D(optim3d.center_pts[i][0], optim3d.center_pts[i][1], optim3d.center_pts[i][2], c='green')
        #     ax.scatter3D(optim3d.extraction_pts[i][0], optim3d.extraction_pts[i][1], optim3d.extraction_pts[i][2], c='blue')
        #     fig.colorbar(p)
        #     plt.show()

        # for i in range(6):
        #     fig = plt.figure()
        #     ax = plt.axes(projection='3d')
        #     print('per insertion', per_insertion[i])
        #     plt.title(f"CLOSURE FORCES FOR INSERTION {i}")
        #     p = ax.scatter3D(x_pts, y_pts, z_pts, c=per_insertion[i])
        #     ax.scatter3D(optim3d.insertion_pts[i][0], optim3d.insertion_pts[i][1], optim3d.insertion_pts[i][2], c='red')
        #     ax.scatter3D(optim3d.center_pts[i][0], optim3d.center_pts[i][1], optim3d.center_pts[i][2], c='green')
        #     ax.scatter3D(optim3d.extraction_pts[i][0], optim3d.extraction_pts[i][1], optim3d.extraction_pts[i][2], c='blue')
        #     fig.colorbar(p)
        #     plt.show()

        # for i in range(6):
        #     fig = plt.figure()
        #     ax = plt.axes(projection='3d')
        #     print('per extraction', per_extraction[i])
        #     plt.title(f"CLOSURE FORCES FOR EXTRACTION {i}")
        #     p = ax.scatter3D(x_pts, y_pts, z_pts, c=per_extraction[i])
        #     ax.scatter3D(optim3d.insertion_pts[i][0], optim3d.insertion_pts[i][1], optim3d.insertion_pts[i][2], c='red')
        #     ax.scatter3D(optim3d.center_pts[i][0], optim3d.center_pts[i][1], optim3d.center_pts[i][2], c='green')
        #     ax.scatter3D(optim3d.extraction_pts[i][0], optim3d.extraction_pts[i][1], optim3d.extraction_pts[i][2], c='blue')
        #     fig.colorbar(p)
        #     plt.show()
        
        # optim3d.optimize(eval=False)

        # closure_loss, shear_loss, all_closure, per_insertion, per_extraction, insertion_forces, extraction_forces = optim3d.compute_closure_shear_loss(granularity=100)
        # optim3d.plot_mesh_path_and_spline()

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # plt.title("CLOSURE FORCES AFTER")
        # p = ax.scatter3D(x_pts, y_pts, z_pts, c=all_closure)
        # fig.colorbar(p)
        # plt.show()

        # for i in range(6):
        #     fig = plt.figure()
        #     ax = plt.axes(projection='3d')
        #     # print('per extraction', per_insertion[i] + per_extraction[i])
        #     plt.title(f"CLOSURE MAGNITUDE AT {i}")
        #     p = ax.scatter3D(x_pts, y_pts, z_pts, c=[np.linalg.norm(iforce - eforce) for iforce, eforce in zip(insertion_forces[i], extraction_forces[i])])
        #     ax.scatter3D(optim3d.insertion_pts[i][0], optim3d.insertion_pts[i][1], optim3d.insertion_pts[i][2], c='red')
        #     ax.scatter3D(optim3d.center_pts[i][0], optim3d.center_pts[i][1], optim3d.center_pts[i][2], c='green')
        #     ax.scatter3D(optim3d.extraction_pts[i][0], optim3d.extraction_pts[i][1], optim3d.extraction_pts[i][2], c='blue')
        #     fig.colorbar(p)
        #     plt.show()

        start_range = 4
        end_range = 10

        # start_range = int(spline_length / 0.005)
        # end_range = int(spline_length / 0.003)

        print("range:", start_range, end_range)

        equally_spaced_losses = {}
        post_algorithm_losses = {}


        best_baseline_loss = 1e8
        best_baseline_placement = None

        best_opt_loss = 1e8
        best_opt_insertion = None
        best_opt_extraction = None
        best_opt_center = None

        final_closure = None
        final_shear = None

        for num_sutures in range(start_range, end_range + 1):
            print("Num sutures:", num_sutures)

            center_pts, insertion_pts, extraction_pts = optim3d.generate_inital_placement(mesh, left_spline, num_sutures=num_sutures)
            #print("Normal vector", normal_vectors)
            # optim3d.plot_mesh_path_and_spline(mesh, left_spline, viz=True, results_pth=baseline_pth)
            equally_spaced_losses[num_sutures] = optim3d.optimize(eval=True)
            print('Initial loss', equally_spaced_losses[num_sutures]["curr_loss"])
            optim3d.optimize(eval=False)
        
            # optim3d.plot_mesh_path_and_spline(mesh, left_spline, viz=True, results_pth=baseline_pth)
            # optim3d.plot_mesh_path_and_spline(mesh, left_spline, viz=viz, results_pth=opt_pth)

            post_algorithm_losses[num_sutures] = optim3d.optimize(eval=True)
            print('After loss', post_algorithm_losses[num_sutures]["curr_loss"])

            
            if post_algorithm_losses[num_sutures]["curr_loss"] < best_opt_loss:
                # print("Num sutures", num_sutures, "best loss so far")
                best_opt_loss = post_algorithm_losses[num_sutures]["curr_loss"]
                print("BEST LOSS", best_opt_loss)
                best_opt_insertion = optim3d.insertion_pts
                best_opt_extraction = optim3d.extraction_pts
                best_opt_center = optim3d.center_pts
                baseline_insertion = insertion_pts
                baseline_extraction = extraction_pts
                baseline_center = center_pts
                _, _, final_closure, _, _, _, _ = optim3d.compute_closure_shear_loss(granularity=100)

        end_time = time.time()
        print(f"execution time: {end_time - start_time}")

        # print("equally_spaced_losses", equally_spaced_losses)
        # print("post_algorithm_losses", post_algorithm_losses)
                
        # save data
                
        # json_equal = json.dumps(equally_spaced_losses)
        # json_post = json.dumps(post_algorithm_losses)

        # equal_losses_pth = baseline_pth + "losses.json"
        # opt_losses_pth = opt_pth + "losses.json"

        # f = open(equal_losses_pth,"w")
        # f.write(json_equal)
        # f.close()

        # f = open(opt_losses_pth,"w")
        # f.write(json_post)
        # f.close()

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title("CLOSURE FORCES FINAL")
        p = ax.scatter3D(x_pts, y_pts, z_pts, c=final_closure)
        fig.colorbar(p)
        plt.show()

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title("SHEAR FORCES FINAL")
        p = ax.scatter3D(x_pts, y_pts, z_pts, c=final_shear)
        fig.colorbar(p)
        plt.show()

        np.save(f'dan_insertion_extraction_pts/insertion_pts{chicken_number}.npy', np.array(best_opt_insertion))
        np.save(f'dan_insertion_extraction_pts/extraction_pts{chicken_number}.npy', np.array(best_opt_extraction))

        # visualize_pointcloud_node = VisualizePointcloudNode(np.array(best_opt_insertion), np.array(best_opt_extraction))
        
        left_image = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
        center_spline = project3d_to_2d(left_image, line_pts_3d)

        print("baseline")
        insertion_pts_base = project3d_to_2d(left_image, baseline_insertion)
        center_pts_base = project3d_to_2d(left_image, baseline_center)
        extraction_base = project3d_to_2d(left_image, baseline_extraction)
        suture_display_adjust = SutureDisplayAdjust(insertion_pts_base, center_pts_base, extraction_base, left_image, center_spline)
        suture_display_adjust.user_display_pnts(f"base{chicken_number}")

        print("optimized")
        left_image = cv2.imread(left_img_path, cv2.IMREAD_COLOR)

        insertion_pts = project3d_to_2d(left_image, best_opt_insertion)
        center_pts = project3d_to_2d(left_image, best_opt_center)
        extraction_pts = project3d_to_2d(left_image, best_opt_extraction)

        suture_display_adjust_optim = SutureDisplayAdjust(insertion_pts, center_pts, extraction_pts, left_image, center_spline)
        suture_display_adjust_optim.user_display_pnts(f"opt{chicken_number}")


