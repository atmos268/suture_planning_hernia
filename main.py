import scipy_generate_sample_spline
import scipy.interpolate as inter
from SuturePlacer import SuturePlacer
from InsertionPointGenerator import InsertionPointGenerator
from ScaleGenerator import ScaleGenerator
from SutureDisplayAdjust import SutureDisplayAdjust
import numpy as np
import cv2
import math
import tkinter as tk
from tkinter import simpledialog
import sys

NUM_EXAMPLES = 5

def suture_placing_pipeline(sample_spline=None, image=None):
    if sample_spline is None:

        # make a new scale object to get the scale
        newScale = ScaleGenerator()
        space_between_sutures = 0.010  # 1 cm
        desired_compute_time = 1
        IPG = InsertionPointGenerator(cut_width=.0075, desired_compute_time=desired_compute_time,
                                      space_between_sutures=space_between_sutures)

        img_color = cv2.imread(image)
        img_point = np.load("record/img_point_inclined.npy")

        # get the scale measurement from surgeon
        scale_pts = newScale.get_scale_pts(img_color, img_point)

        # request the surgeon for a distance

        # make into GUI, and also request wound width, space between sutures
        # real_dist = input('Please enter the distance in mm that you measured: ')
        real_dist = simpledialog.askfloat(title="dist prompt",
                                          prompt="Please enter the distance in mm that you measured")

        wound_width = simpledialog.askfloat(title="width prompt", prompt="Please enter the width of suture in mm")

        cv2.destroyAllWindows()

        pixel_dist = math.sqrt((scale_pts[0][0] - scale_pts[1][0]) ** 2 + (scale_pts[0][1] - scale_pts[1][1]) ** 2)
        mm_per_pixel = real_dist / pixel_dist

        pnts = IPG.get_insertion_points_from_selection(img_color, img_point)
        x = [a[0] for a in pnts]
        y = [a[1] for a in pnts]

        # now, use our conversion factor to scale points appropriately
        x = [float(elem) * mm_per_pixel for elem in x]
        y = [float(elem) * -mm_per_pixel for elem in y]
        deg = 5
    else:
        mm_per_pixel=1
        point_data = (
            [[[0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 0]], 
             [[0, 3, 10, 15, 21, 25, 30], [0, 1, 3, 5, 3, 1, 0]],  
             [[0, 7, 10, 15, 21, 25, 30], [0, -5, 5, 35, 18, 7, 13]],
             [[0, 3, 10, 15, 10, 3, 1], [0, 1, 3, 5, 8, 10, 5]],
             [[0, 3, 10, 18, 29, 35, 40], [0, -10, 5, -20, 18, 7, 13]]
             ])
        
        width_data = [1.5, 1.5, 1.5, 1.5, 1.25]
        degree_data = [5, 2, 5, 5, 5]

        if 0 <= sample_spline < NUM_EXAMPLES:
            pnts = point_data[sample_spline]
            wound_width = width_data[sample_spline]
            deg = degree_data[sample_spline]

        else:
            raise Exception("not a pre-saved wound")

    tck, u = inter.splprep([x, y], k=deg)
    wound_parametric = lambda t, d: inter.splev(t, tck, der = d)

    def wound(x):
        pnts = inter.splev(x, tck)
        return pnts

    # Put the wound into all the relevant objects
    newSuturePlacer = SuturePlacer(wound_width, mm_per_pixel)
    newSuturePlacer.tck = tck
    newSuturePlacer.DistanceCalculator.tck = tck

    
    newSuturePlacer.wound_parametric = wound_parametric
    newSuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
    newSuturePlacer.RewardFunction.wound_parametric = wound_parametric

    newSuturePlacer.image = image
    # The main algorithm
    newSuturePlacer.place_sutures(sample_spline)
    return newSuturePlacer

def suture_display_adj_pipeline(newSuturePlacer):
    
    insert_pts = newSuturePlacer.b_insert_pts
    center_pts = newSuturePlacer.b_center_pts
    extract_pts = newSuturePlacer.b_extract_pts
    mm_per_pixel = newSuturePlacer.mm_per_pixel

    newSutureDisAdj = SutureDisplayAdjust(insert_pts, center_pts, extract_pts, mm_per_pixel)
    
    # display
    img_color = cv2.imread(newSuturePlacer.image)
    img_point = np.load("record/img_point_inclined.npy")

    # allow for edit
    newSutureDisAdj.adjust_points(img_color, img_point)
    return

if __name__ == "__main__":
    args = sys.argv
    ROOT = tk.Tk()
    ROOT.withdraw()

    if len(args) != 3:
        raise Exception("incorrect format: format is python main.py -i [image.jpg] || -s [spline_number]")
    
    if args[1] == "-s" :
        suturePlacerTest = suture_placing_pipeline(sample_spline=int(args[2]))
    elif args[1] == "-i":
        suturePlacerTest = suture_placing_pipeline(sample_spline=None, image=args[2])
        suture_display_adj_pipeline(suturePlacerTest)
    
    cv2.destroyAllWindows()
