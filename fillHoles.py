import cv2
import numpy as np
def fillHoles(imgPath):
    im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    im_in = cv2.bitwise_not(im_in)
    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out