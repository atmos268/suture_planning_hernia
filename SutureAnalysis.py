import scipy_generate_sample_spline
import scipy.interpolate as inter
import SuturePlacer
from InsertionPointGenerator import InsertionPointGenerator
from ScaleGenerator import ScaleGenerator
import numpy as np
import cv2
import math
from tkinter import simpledialog

# Viraj: this is the framework for out analysis of existing surgeon suturing jobs

# Process:
# Use clicking interface to get scale
# use clicking interface to get all other points, and use pixel to mm ratio to calculate the actual distances

# NB: most of the logic here is adapted directly from the main function, but we are using this for understanding existing sutures
# rather than optimizing new ones

def suture_analysis_pipeline():
    #  points along the wound. That'll return a spline.

    # make a new scale object to get the scale
    newScale = ScaleGenerator()

    space_between_sutures = 0.010  # 1 cm
    desired_compute_time = 1
    IPG = InsertionPointGenerator(cut_width=.0075, desired_compute_time=desired_compute_time,
                                  space_between_sutures=space_between_sutures)

    img_color = cv2.imread('short_vertical_suture.jpeg')
    img_point = np.load("record/img_point_inclined.npy")

    # get the scale measurement from surgeon
    scale_pts = newScale.get_scale_pts(img_color, img_point)
    
    print("scale pts: " + str(scale_pts))

    # request the surgeon for a distance
    float_real_dist = simpledialog.askfloat(title="dist prompt", prompt="Please enter the distance in mm that you measured")
    cv2.destroyAllWindows()

    print(float_real_dist)

    pixel_dist = math.sqrt((scale_pts[0][0] - scale_pts[1][0])**2 + (scale_pts[0][1] - scale_pts[1][1])**2)
    mm_per_pixel = float_real_dist/pixel_dist

    # now, query the center, insertion and extraction points
    print("select center points:")
    # center
    center_pts = IPG.get_insertion_points_from_selection(img_color, img_point)
    print("select insertion points:")
    # insertion
    insertion_pts = IPG.get_insertion_points_from_selection(img_color, img_point)
    print("select extraction points:")
    # extraction
    extraction_pts = IPG.get_insertion_points_from_selection(img_color, img_point)

    if (len(center_pts) != len(insertion_pts) or len(center_pts) != len(extraction_pts)):
        print("Error: must enter the same number of center, extraction and insertion points")
        return
    print(center_pts)
    print(insertion_pts)
    print(extraction_pts)

    real_center_pts = [[float(pt[0]) * mm_per_pixel, float(pt[1]) * mm_per_pixel] for pt in center_pts]
    real_insertion_pts = [[float(pt[0]) * mm_per_pixel, float(pt[1]) * mm_per_pixel] for pt in insertion_pts]
    real_extraction_pts = [[float(pt[0]) * mm_per_pixel, float(pt[1]) * mm_per_pixel] for pt in extraction_pts]

    print("After rescaling: ")

    print(real_center_pts)
    print(real_insertion_pts)
    print(real_extraction_pts)

    # to do this, start an instance of a SuturePlacer
    # testSuturePlacer = SuturePlacer(5, mm_per_pixel)

    # no need to worry about suture-width, as they have already been made

    # we then manually set the insertion pts and other pertinent varibles.


    # testSuturePlacer.RewardFunction.insert_dists = insert_dists
    # testSuturePlacer.RewardFunction.center_dists = center_dists
    # testSuturePlacer.RewardFunction.extract_dists = extract_dists

    # then, we feed pts into reward fn and output loss

if __name__ == "__main__":
    suturePlacer = suture_analysis_pipeline()
    cv2.destroyAllWindows()