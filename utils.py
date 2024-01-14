# utils
import math
# mallika's code for point clicking
import matplotlib.pyplot as plt
import math

def click_points_simple(img):
    fig = plt.figure()
    plt.imshow(img)
    left_coords,right_coords = [], []
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        nonlocal left_coords,right_coords
        if(event.button==1):
            left_coords.append(coords)
        elif(event.button==3):
            right_coords.append(coords)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return left_coords,right_coords

def get_mm_per_pixel(start_pt, end_pt, mm_indicated):
    
    pix_dist = math.sqrt((start_pt[0] - end_pt[0])**2 + (start_pt[1] - end_pt[1])**2)
    return mm_indicated / pix_dist
    