from EdgeDetector import img_to_line, line_to_spline, click_points_simple
from main import suture_display_adj_pipeline
from SuturePlacer import SuturePlacer
import math
import scipy.interpolate as inter
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    box_method = True
    img_path = 'image_left_001.png'

    img = Image.open(img_path)
    
    # asarray() class is used to convert
    # PIL images into NumPy arrays
    numpydata = np.asarray(img)


    # get scaling information
    mm_indicated = 50
    wound_width = 5
    left_pts, right_pts = click_points_simple(numpydata)

    if len(left_pts) < 2:
        print("not enough points clicked")
        
    start_pt = left_pts[-2]
    end_pt = left_pts[-1]

    pix_dist = math.sqrt((start_pt[0] - end_pt[0])**2 + (start_pt[1] - end_pt[1])**2)

    mm_per_pixel = mm_indicated / pix_dist

    line = img_to_line(img_path, box_method, viz=True)
    scaled_line = []

    # now we need to scale 
    for i, elem in enumerate(line):
        scaled_line.append((line[i][0] * mm_per_pixel, line[i][1] * mm_per_pixel))
    
    plt.plot([pt[1] for pt in line], [pt[0] for pt in line], color='b')
    plt.plot([pt[1] for pt in scaled_line], [pt[0] for pt in scaled_line], color='b')

<<<<<<< HEAD
    spline, tck = line_to_spline(line, img_path, mm_per_pixel)
=======
    spline, tck = line_to_spline(line, img_path)
>>>>>>> 528ce37 (add back files)
    scaled_spline, tck = line_to_spline(scaled_line, img_path, mm_per_pixel)

    plt.show()

    # now run original pipeline
    suture_placer = SuturePlacer(wound_width, mm_per_pixel)

    wound_parametric = lambda t, d: inter.splev(t, tck, der = d)

    def wound(x):
        pnts = inter.splev(x, tck)
        return pnts

    # Put the wound into all the relevant objects
    newSuturePlacer = SuturePlacer(wound_width, mm_per_pixel)
    newSuturePlacer.wound = wound
    newSuturePlacer.tck = tck
    newSuturePlacer.DistanceCalculator.wound = wound
    newSuturePlacer.DistanceCalculator.tck = tck
    
    newSuturePlacer.wound_parametric = wound_parametric
    newSuturePlacer.DistanceCalculator.wound_parametric = wound_parametric
    newSuturePlacer.RewardFunction.wound_parametric = wound_parametric

    newSuturePlacer.image = img_path

    img_color = cv2.imread(img_path)

    newSuturePlacer.place_sutures()
    suture_display_adj_pipeline(newSuturePlacer)





