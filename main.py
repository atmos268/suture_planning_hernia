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

def suture_placing_pipeline():
    # TODO Varun: will rope in Sam's code that has the interface for the surgeon to click
    #  points along the wound. That'll return a spline.

    # make a new scale object to get the scale
    newScale = ScaleGenerator()

    space_between_sutures = 0.010  # 1 cm
    desired_compute_time = 1
    IPG = InsertionPointGenerator(cut_width=.0075, desired_compute_time=desired_compute_time,
                                  space_between_sutures=space_between_sutures)

    img_color = cv2.imread('hand_image.png')
    img_point = np.load("record/img_point_inclined.npy")

    # get the scale measurement from surgeon
    scale_pts = newScale.get_scale_pts(img_color, img_point)

    # request the surgeon for a distance

    # make into GUI, and also request wound width, space between sutures 
    # real_dist = input('Please enter the distance in mm that you measured: ')
    real_dist = simpledialog.askfloat(title="dist prompt", prompt="Please enter the distance in mm that you measured")

    wound_width = simpledialog.askfloat(title="width prompt", prompt="Please enter the width of suture in mm")
    
    cv2.destroyAllWindows()

    pixel_dist = math.sqrt((scale_pts[0][0] - scale_pts[1][0])**2 + (scale_pts[0][1] - scale_pts[1][1])**2)
    mm_per_pixel = real_dist/pixel_dist

    sample_spline = False
    if not sample_spline:
        pnts = IPG.get_insertion_points_from_selection(img_color, img_point)
    else:
        pnts = [[46, 233], [50, 213], [57, 195], [67, 175], [77, 160], [91, 136], [107, 114], [121, 111], [137, 111],
         [144, 120], [158, 136], [166, 166], [175, 208], [193, 233], [227, 218], [251, 183], [275, 128]]

    # But for now, just use this sample spline. It's a Bezier spline

    # Varun/Viraj: For now this is OK, but maybe we will need to incorporate wounds that can't be represented as y(x) later using multiple B-spline curves or something else.
    """ Notes on old version of Bezier: So this bezier library can make arbitrary parametric [t -> x(t), y(t)] bezier curves which allows for wounds where y is not a function of x or vice versa,
    #  but I don't think it has a function to fit points to a bezier curve. SciPy's bezier module can fit points to a curve, but it is in the format [x -> y] which is more limiting
    #  for the types of curves we can handle. Goal is to fit points to a parametric bezier curve.
    """
    x = [a[0] for a in pnts]
    y = [a[1] for a in pnts]

    # now, use our conversion factor to scale points appropriately
    x = [float(elem) * mm_per_pixel for elem in x]
    y = [float(elem) * mm_per_pixel for elem in y]

    # x = [0.0, 0.7, 1.0, 1.5, 2.1, 2.5, 3.0] # OLD manually-chosen example
    # y = [0.0, -0.5, 0.5, 3.5, 1.8, 0.7, 1.3] # OLD manually-chosen example
    deg = 3

    # couldn't find reference to this in the codebase? I'm using make_interp_spline for now
    # wound = scipy_generate_sample_spline.generate_sample_spline()
    tck, u = inter.splprep([x, y], k=deg)
    wound_parametric = lambda t, d: inter.splev(t, tck, der = d)

    def wound(x):
        pnts = inter.splev(x, tck)
        # pnts[1] = pnts[1] * -1
        return pnts

    wound(3)
    # Put the wound into all the relevant objects
    newSuturePlacer = SuturePlacer(wound_width, mm_per_pixel)
    newSuturePlacer.wound = wound
    newSuturePlacer.tck = tck
    newSuturePlacer.DistanceCalculator.wound = wound
    newSuturePlacer.DistanceCalculator.tck = tck
    newSuturePlacer.Optimizer.wound = wound
    newSuturePlacer.Optimizer.tck = tck
    
    newSuturePlacer.wound_parametric = wound_parametric
    newSuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
    newSuturePlacer.Optimizer.wound_parametric = wound_parametric

    # The main algorithm
    newSuturePlacer.place_sutures()
    return newSuturePlacer

def suture_display_adj_pipeline(newSuturePlacer):
    
    insert_pts = newSuturePlacer.insert_pts
    center_pts = newSuturePlacer.center_pts
    extract_pts = newSuturePlacer.extract_pts
    mm_per_pixel = newSuturePlacer.mm_per_pixel

    newSutureDisAdj = SutureDisplayAdjust(insert_pts, center_pts, extract_pts, mm_per_pixel)

    # convert back to pixel values
    
    # display and allow for edit
    # (cv2)
    img_color = cv2.imread('hand_image.png')
    img_point = np.load("record/img_point_inclined.npy")

    newSutureDisAdj.adjust_points(img_color, img_point)
    
    # pull up image using cv2

    # plot points from optimization (need to source from somewhere)
    
    return

if __name__ == "__main__":
    ROOT = tk.Tk()
    ROOT.withdraw()

    suturePlacer = suture_placing_pipeline()

    suture_display_adj_pipeline(suturePlacer)

    cv2.destroyAllWindows()
