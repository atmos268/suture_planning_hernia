from EdgeDetector import img_to_line, line_to_spline, line_to_spline_3d, click_points_simple
import tensorflow as tf
from main import suture_display_adj_pipeline
from SuturePlacer import SuturePlacer
from Optimizer3d import Optimizer3d
from MeshIngestor import MeshIngestor
import math
import scipy.interpolate as inter
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from enhance_image import adjust_contrast_saturation
from image_transform import get_transformed_points
from utils import get_mm_per_pixel
import subprocess
import random
from SuturePlacement3d import SuturePlacement3d
import json
import os
import copy

def calculate_z(mesh, point, smallest_z, largest_z):
    z_arr = np.linspace(smallest_z, largest_z, num=10000)
    min_dist = float('inf')
    min_z = 0
    for z in z_arr:
        dist, idx = mesh.get_nearest_point([point[0], point[1], z])
        if dist < min_dist:
            min_dist = dist
            min_z = z

    return min_z


def getTransformationMatrix():
    transformation_tensor = tf.io.read_file('transforms/tf_av_left_zivid.tf')
    print(transformation_tensor)
    transformation_string = transformation_tensor.numpy().decode('utf-8') # Convert to a Python string
    # Split the string into individual lines and elements
    lines = transformation_string.strip().split('\n')[2:]
    matrix_elements = [list(map(float, line.split())) for line in lines]
    # Convert the matrix elements to a NumPy array
    transformation_matrix = np.array(matrix_elements)
    print(transformation_matrix.shape)
    print(transformation_matrix)
    return transformation_matrix

def dragging_helper(points, left_image):
    transformation_matrix = getTransformationMatrix()
    R, t = transformation_matrix[1:], transformation_matrix[0]
    left_center_points = []
    for pt in points:
        left_center_points.append(np.linalg.inv(R) @ (pt - t))
    left_center_points = np.array(left_center_points)
    # projecting onto left image
    image_height, image_width, _ = left_image.shape
    left_camera_matrix = np.array(
        [[1688.10117, 0, 657.660185], [0, 1688.10117, 411.400296], [0, 0, 1]],
        dtype=np.float64,
    )
    left_dist_coeffs = np.array(
        [-0.13969738, 0.28183828, -0.00836148, -0.00180531, -1.65874481], dtype=np.float64
    )
    if not left_center_points.shape[0] == 0:
        projected_points_wound, _ = cv2.projectPoints(
            left_center_points,
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
                for i in range(7):
                    for j in range(7):
                        left_image[y+i, x+j] = 0
    return left_image

if __name__ == "__main__":
    
    box_method = True
    save_figs = True
    left_file = 'left_test_001.png'
    left_img_path = 'chicken_images/' + left_file
    left_img_path_enhanced = 'chicken_images/enhanced/' + left_file

    right_file = 'right_test_001.png'
    right_img_path = 'chicken_images/' + right_file
    right_img_path_enhanced = 'chicken_images/enhanced/' + right_file

    experiment_mode = "physical" # Run synthetic vs physical experiments pipeline
    # pick two random points to generate synthetic splines
    #num1, num2 = random.randrange(0, len(mesh.vertex_coordinates)), random.randrange(0, len(mesh.vertex_coordinates))
    num1, num2 = 21695, 8695
    results_pth = "run_1"

    baseline_pth = "results/" + results_pth + "/baseline/"
    opt_pth = "results/" + results_pth + "/opt/"
    old_algo_pth = "results/" + results_pth + "/old_algo/"

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

    mode = '3d' # 3d

    if experiment_mode == "synthetic":
        adj_path = 'adjacency_matrix.txt'
        loc_path = 'vertex_lookup.txt'

        mesh = MeshIngestor(adj_path, loc_path)

        # Create the graph
        mesh.generate_mesh()
        start_pt = mesh.get_point_location(num1)
        wound_pt = mesh.get_point_location(num2)
        shortest_path = mesh.get_a_star_path(start_pt, wound_pt)
        shortest_path_xyz = np.array([mesh.get_point_location(pt_idx) for pt_idx in shortest_path])
        smallest_z, largest_z = min(shortest_path_xyz[:, 2]), max(shortest_path_xyz[:, 2])

        spline3d = line_to_spline_3d(shortest_path_xyz, sample_ratio=30, viz=False)
        suture_width = 0.005 
        mm_per_pixel = 10
        c_ideal = 1000
        gamma = suture_width 
        c_var = 1000
        c_shear = 1
        c_closure = 1

        hyperparams = [c_ideal, gamma, c_var, c_shear, c_closure]

        force_model_parameters = {'ellipse_ecc': 1.0, 'force_decay': 0.5/suture_width, 'verbose': 0, 'ideal_closure_force': None, 'imparted_force': None}

        optim3d = Optimizer3d(mesh, spline3d, suture_width, hyperparams, force_model_parameters)

        if mode == "3d":
            suturePlacement3d, normal_vectors, derivative_vectors = optim3d.generate_inital_placement(mesh, spline3d)

            optim3d.plot_mesh_path_and_spline(mesh, spline3d, suturePlacement3d, normal_vectors, derivative_vectors)

            optim3d.optimize(suturePlacement3d)

            optim3d.plot_mesh_path_and_spline(mesh, spline3d, suturePlacement3d, normal_vectors, derivative_vectors)

            loss3d = optim3d.loss_placement(suturePlacement3d)
            print("loss of 3d placement", str(loss3d))

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
        disp_path = "RAFT/disp_test_001.npy"

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
        
            # convert the spline to 3d using raft
            pts_3d = get_transformed_points(left_img_path, disp_path, point_mask)

            # return 3d points

            #print(points)
            #return np.array(points)
            points_mesh = np.array([mesh.get_point_location(mesh.get_nearest_point(pt)[1]) for pt in pts_3d])
            return points_mesh
                
        b_center_pts, b_insert_pts, b_extract_pts = twoD_to_3D(b_center_pts), twoD_to_3D(b_insert_pts), twoD_to_3D(b_extract_pts)
        center_pts_spline = line_to_spline_3d(b_center_pts, viz=False)
        suturePlacement2dIn3d = SuturePlacement3d(center_pts_spline, b_center_pts, b_insert_pts, b_extract_pts, [])

        optim3d = Optimizer3d(mesh, center_pts_spline, suture_width, hyperparams, force_model_parameters)
        optim3d.plot_mesh_path_and_spline(mesh, center_pts_spline, suturePlacement2dIn3d, [], [])

        # print loss of the 2d placement
        # loss2d = optim3d.loss_placement(suturePlacement2dIn3d)
        loss2d = optim3d.optimize(suturePlacement2dIn3d, eval=True)
        
        print("loss of 2d placement", str(loss2d))
        optim3d.plot_mesh_path_and_spline(mesh, center_pts_spline, suturePlacement2dIn3d, [], [], viz=False, results_pth=old_algo_pth)

        json_old = json.dumps(loss2d)

        old_losses_pth = old_algo_pth + "losses.json"
        f = open(old_losses_pth,"w")
        f.write(json_old)
        f.close()

        suture_display_adj_pipeline(newSuturePlacer)

    elif mode == '3d' and experiment_mode == "physical":
        
        viz = True
        use_prev = True
        suture_width = 0.005
        
        # get the masks
        # save left and right masks

        left_mask_path = 'temp_images/left_mask.npy'
        right_mask_path = 'temp_images/right_mask.npy'
        left_line_path = 'temp_images/left_line.npy'
        right_line_path = 'temp_images/right_line.npy'
        left_dilated_mask_path = 'temp_images/left_dilated_mask.jpg'

        if use_prev:
            left_line = np.load(left_line_path)
            left_mask = np.load(left_mask_path)

            right_line = np.load(right_line_path)
            right_mask = np.load(right_mask_path)

        else:

            left_line, left_mask = img_to_line(left_img_path, box_method, viz=True, save_figs=save_figs)
            np.save(left_mask_path, left_mask)
            np.save(left_line_path, left_line)
            right_line, right_mask = img_to_line(right_img_path, box_method, viz=True, save_figs=save_figs)
            np.save(right_mask_path, right_mask)
            np.save(right_line_path, right_line)
        
        # do raft, no need to do rn, as we are using the existing RAFT output
        disp_path = "RAFT/disp_test_001.npy"

        # dilate to get region
        dilation = 100

        kernel = np.ones((dilation, dilation), np.uint8)
        dilated_mask = cv2.dilate(left_mask, kernel, iterations=1) 
        cv2.imwrite(left_dilated_mask_path, dilated_mask) 

        # write out points to a file
        surrounding_pts = get_transformed_points(left_img_path, disp_path, dilated_mask)
        np.save('surrounding_pts.npy', surrounding_pts)

        # covert line into mask
        img_width, img_height = Image.open(left_img_path).size
        line_mask = np.zeros((img_height, img_width))

        for row, col in left_line:
            line_mask[row,col] = 1
        
        # convert the spline to 3d using raft
        line_pts_3d = get_transformed_points(left_img_path, disp_path, line_mask, viz=True)
        np.save('line_pts_3d.npy', line_pts_3d)

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter3D([point[0] for point in line_pts_3d], [point[1] for point in line_pts_3d], [point[2] for point in line_pts_3d])

        # get the spline from the left image
        # since we are not visualizing here, no need for scaling info
        # left_spline = line_to_spline(left_line, None, None, viz=False)
        # will actually need to use line_to_spline_3d (expect 3d points)
        left_spline = line_to_spline_3d(line_pts_3d, sample_ratio=30, viz=False)

        granularity = 100


        x_pts = [left_spline[0](t/granularity) for t in range(granularity)]
        y_pts = [left_spline[1](t/granularity) for t in range(granularity)]
        z_pts = [left_spline[2](t/granularity) for t in range(granularity)]

        ax.scatter3D(x_pts, y_pts, z_pts)

        plt.show()
        # get mesh from the surrounding points
                    
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

        gamma = suture_width * 2
        c_ideal = 1000
        c_var = 5000
        c_shear = 1
        c_closure = 0.5

        hyperparams = [c_ideal, gamma, c_var, c_shear, c_closure]

        force_model_parameters = {'ellipse_ecc': 1.0, 'force_decay': 0.5/suture_width, 'verbose': 0, 'ideal_closure_force': None, 'imparted_force': None}

        optim3d = Optimizer3d(mesh, left_spline, suture_width, hyperparams, force_model_parameters)
        
        spline_length = optim3d.calculate_spline_length(left_spline, mesh)
        num_sutures_initial = int(spline_length / (gamma)) #TODO: modify later 
        print("Num sutures initial", num_sutures_initial)
        start_range = max(int(num_sutures_initial * 0.5), 2)
        end_range = int(num_sutures_initial * 1.5)

        print("range:", start_range, end_range)

        equally_spaced_losses = {}
        post_algorithm_losses = {}


        best_baseline_loss = 1e8
        best_baseline_placement = None

        best_opt_loss = 1e8
        best_opt_placement = None

        for num_sutures in range(start_range, end_range + 1):

            print("num sutures:", num_sutures)

            suturePlacement3d, normal_vectors, derivative_vectors = optim3d.generate_inital_placement(mesh, left_spline, num_sutures=num_sutures)
            #print("Normal vector", normal_vectors)

            optim3d.plot_mesh_path_and_spline(mesh, left_spline, suturePlacement3d, normal_vectors, derivative_vectors, viz=viz, results_pth=baseline_pth)
            equally_spaced_losses[num_sutures] = optim3d.optimize(suturePlacement3d, eval=True)
            if equally_spaced_losses[num_sutures]["curr_loss"] < best_baseline_loss:
                best_baseline_loss = equally_spaced_losses[num_sutures]["curr_loss"]
                best_baseline_placement = copy.deepcopy(suturePlacement3d)

            optim3d.optimize(suturePlacement3d)

            optim3d.plot_mesh_path_and_spline(mesh, left_spline, suturePlacement3d, normal_vectors, derivative_vectors, viz=viz, results_pth=opt_pth)
            post_algorithm_losses[num_sutures] = optim3d.optimize(suturePlacement3d, eval=True)

            if post_algorithm_losses[num_sutures]["curr_loss"] < best_opt_loss:
                best_opt_loss = post_algorithm_losses[num_sutures]["curr_loss"]
                best_opt_placement = copy.deepcopy(suturePlacement3d)

        # print("equally_spaced_losses", equally_spaced_losses)
        # print("post_algorithm_losses", post_algorithm_losses)
        print("baseline")
        left_image = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
        print(left_image)
        print(len(best_baseline_placement.center_pts))
        left_image = dragging_helper(best_baseline_placement.center_pts, left_image)
        left_image = dragging_helper(best_baseline_placement.insertion_pts, left_image)
        left_image = dragging_helper(best_baseline_placement.extraction_pts, left_image)

        # # Visualizing the projection on the image
        cv2.namedWindow('Projected Points', cv2.WINDOW_NORMAL) # Create a resizable window
        cv2.imshow('Projected Points', left_image) # Show the modified image
        cv2.waitKey()


        print("optimized")
        left_image = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
        print(left_image)
        print(len(best_opt_placement.center_pts))
        left_image = dragging_helper(best_opt_placement.center_pts, left_image)
        left_image = dragging_helper(best_opt_placement.insertion_pts, left_image)
        left_image = dragging_helper(best_opt_placement.extraction_pts, left_image)

        # # Visualizing the projection on the image
        cv2.namedWindow('Projected Points', cv2.WINDOW_NORMAL) # Create a resizable window
        cv2.imshow('Projected Points', left_image) # Show the modified image
        cv2.waitKey()

        json_equal = json.dumps(equally_spaced_losses)
        json_post = json.dumps(post_algorithm_losses)

        equal_losses_pth = baseline_pth + "losses.json"
        opt_losses_pth = opt_pth + "losses.json"

        f = open(equal_losses_pth,"w")
        f.write(json_equal)
        f.close()

        f = open(opt_losses_pth,"w")
        f.write(json_post)
        f.close()

        # dragging codeeee
        # print(“Overhead center points”, np.array(suturePlacement3d.center_pts.shape))
        # print(“left center points”, left_center_points.shape)
        # projecting onto left image
        

    # else:
    #     print("invalid mode")






