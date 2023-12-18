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

# gets the euclidian distance from any two points in any dimension
def euclidean_dist(pt1, pt2):
    if len(pt1) != len(pt2):
        raise IndexError("Mismatched size of points to compare")
    
    total = 0

    for i in range(len(pt1)):
        total += (pt1[i] - pt2[i]) ** 2
    
    return math.sqrt(total)

def get_mm_per_pixel(start_pt, end_pt, mm_indicated):
    
    pix_dist = math.sqrt((start_pt[0] - end_pt[0])**2 + (start_pt[1] - end_pt[1])**2)
    return mm_indicated / pix_dist
    
