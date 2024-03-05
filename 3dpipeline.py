from EdgeDetector import img_to_line, line_to_spline, click_points_simple
from main import suture_display_adj_pipeline
from SuturePlacer import SuturePlacer
import math
import scipy.interpolate as inter
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from enhance_image import adjust_contrast_saturation
from image_transform import get_transformed_points
from utils import get_mm_per_pixel

if __name__ == "__main__":
    
    box_method = True
    save_figs = True
    left_file = 'image_left_001.png'
    left_img_path = 'chicken_images/' + left_file
    left_img_path_enhanced = 'chicken_images/enhanced/' + left_file


    right_file = 'image_right_001.png'
    right_img_path = 'chicken_images/' + right_file
    right_img_path_enhanced = 'chicken_images/enhanced/' + right_file

    mode = '2d' # 3d

    if mode == '2d':

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

        scaled_spline, tck = line_to_spline(scaled_line, left_img_path_enhanced, mm_per_pixel)

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

    elif mode == '3d':

        # get the masks
        left_line, left_mask = img_to_line(left_img_path, box_method, viz=True, save_figs=save_figs)
        right_line, right_mask = img_to_line(right_img_path, box_method, viz=True, save_figs=save_figs)

        # do raft, no need to do rn, as we are using the existing RAFT output
        disp_path = "RAFT/disp.npy"

        # dilate to get region
        dilation = 100

        kernel = np.ones((dilation, dilation), np.uint8)
        dilated_mask = cv2.dilate(left_mask, kernel, iterations=1) 

        # write out points to a file
        surrounding_pts = get_transformed_points(left_mask, disp_path, dilated_mask)
        np.save('surrounding_pts.npy', surrounding_pts)

        # get the spline from the left image


        # convert the spline to 3d using raft

        # get mesh from the surrounding points

        # run optimizer
        pass

    else:
        print("invalid mode")






