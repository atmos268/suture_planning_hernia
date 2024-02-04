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
from utils import get_mm_per_pixel

if __name__ == "__main__":
    
    box_method = True
    save_figs = True
    file_name = 'image_left_001.png'
    img_path = 'chicken_images/' + file_name
    img_path_enhanced = 'chicken_images/enhanced/' + file_name

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
    
    mm_per_pixel = get_mm_per_pixel(left_pts[-2], left_pts[-1], mm_indicated)

    line = img_to_line(img_path, box_method, viz=True, save_figs=save_figs)
    
    # build a line that is scaled to mm size
    scaled_line = []
    for i, elem in enumerate(line):
        scaled_line.append((line[i][0] * mm_per_pixel, line[i][1] * mm_per_pixel))
    
    # add contrast etc. to improve SAM results
    enhanced = adjust_contrast_saturation(img, 3, 1)
    enhanced.save(img_path_enhanced)

    scaled_spline, tck = line_to_spline(scaled_line, img_path_enhanced, mm_per_pixel)

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
    newSuturePlacer.image = img_path

    newSuturePlacer.place_sutures(save_figs=save_figs)
    suture_display_adj_pipeline(newSuturePlacer)





