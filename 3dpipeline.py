from EdgeDetector import img_to_line, line_to_spline, line_to_spline_3d, click_points_simple
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


    
    

if __name__ == "__main__":
    
    box_method = True
    save_figs = True
    left_file = 'image_left_001.png'
    left_img_path = 'chicken_images/' + left_file
    left_img_path_enhanced = 'chicken_images/enhanced/' + left_file

    right_file = 'image_right_001.png'
    right_img_path = 'chicken_images/' + right_file
    right_img_path_enhanced = 'chicken_images/enhanced/' + right_file

    mode = '2d' # Run 2d vs 3d
    experiment_mode = "synthetic" # Run synthetic vs physical experiments pipeline
    # pick two random points to generate synthetic splines
    #num1, num2 = random.randrange(0, len(mesh.vertex_coordinates)), random.randrange(0, len(mesh.vertex_coordinates))
    num1, num2 = 21695, 8695

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
            suturePlacement3dNew = SuturePlacement3d(center_pts_spline, b_center_pts, b_insert_pts, b_extract_pts, [])
            optim3d.plot_mesh_path_and_spline(mesh, center_pts_spline, suturePlacement3dNew, [], [])

            #TODO: check loss of the 2d placement vs 3d placement
            
    if mode == '2d' and experiment_mode == "physical":

        img = Image.open(left_img_path)
    
        # asarray() class is used to convert
        # PIL images into NumPy arrays
        numpydata = np.asarray(img)

        # get scaling information
        mm_indicated = 50
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

        plt.show()

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


        suture_display_adj_pipeline(newSuturePlacer)

    elif mode == '3d' and experiment_mode == "physical":
        use_prev = True
        
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
        disp_path = "RAFT/disp1.npy"

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
        line_pts_3d = get_transformed_points(left_img_path, disp_path, line_mask)
        np.save('line_pts_3d.npy', line_pts_3d)

        # get the spline from the left image
        # since we are not visualizing here, no need for scaling info
        # left_spline = line_to_spline(left_line, None, None, viz=False)
        # will actually need to use line_to_spline_3d (expect 3d points)
        left_spline = line_to_spline_3d(line_pts_3d, sample_ratio=30, viz=False)

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
        
        suture_width = 0.005
        c_ideal = 1000
        gamma = suture_width # TODO: Change once scaling is sorted
        c_var = 1000
        c_shear = 1
        c_closure = 1

        hyperparams = [c_ideal, gamma, c_var, c_shear, c_closure]

        force_model_parameters = {'ellipse_ecc': 1.0, 'force_decay': 0.5/suture_width, 'verbose': 0, 'ideal_closure_force': None, 'imparted_force': None}

        optim3d = Optimizer3d(mesh, left_spline, suture_width, hyperparams, force_model_parameters)
        suturePlacement3d, normal_vectors, derivative_vectors = optim3d.generate_inital_placement(mesh, left_spline)
        #print("Normal vector", normal_vectors)
        optim3d.plot_mesh_path_and_spline(mesh, left_spline, suturePlacement3d, normal_vectors, derivative_vectors)

        optim3d.optimize(suturePlacement3d)

        optim3d.plot_mesh_path_and_spline(mesh, left_spline, suturePlacement3d, normal_vectors, derivative_vectors)

    # else:
    #     print("invalid mode")






