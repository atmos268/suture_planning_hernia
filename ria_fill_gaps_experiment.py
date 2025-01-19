from PIL import Image
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.interpolate as inter
from point_ordering import get_pt_ordering
# import plantcv
from SAM import create_mask
from largestCC import keep_largest_connected_component
from fillHoles import fillHoles
from matplotlib import colormaps
from utils import click_points_simple
import os

matplotlib.use('TkAgg')


border_pts = np.load("./EdgeDetector_img2line_experiment_borderpts.npy")
img = cv2.imread('chicken_images/image_left_005.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

'''
# show before image
plt.imshow(img)
colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'black', 'white']
step = len(border_pts) // 8
for i in range(8):
    plt.scatter(border_pts[step*i:min(len(border_pts),step*(i+1)), 0], border_pts[step*i:min(len(border_pts),step*(i+1)), 1], s=1, color=colors[i])   
#plt.scatter(border_pts[:, 0], border_pts[:, 1], s=1, color='red')
plt.show()

def euc_dist(x, y):
    return np.sqrt(abs(x[0]-y[0])**2 + abs(x[1]-y[1])**2)

def closest_point_right(index):
    new_arr = border_pts[np.where(border_pts[0] > border_pts[index][0])]
    if len(new_arr) > 0:
        pt_index = np.argmin(np.array([euc_dist(border_pts[index], b) for b in new_arr]))
        return pt_index, new_arr[pt_index]
    return None, None

def closest_point_left(index):
    new_arr = border_pts[np.where(border_pts[0] < border_pts[index][0])]
    if len(new_arr) > 0:
        pt_index = np.argmin(np.array([euc_dist(border_pts[index], b) for b in new_arr]))
        return pt_index, new_arr[pt_index]
    return None, None

distances = []
# linearly interpolate gaps
new_border_pts = np.copy(border_pts)
for i in range(len(border_pts)):
    pt_ind, pt = closest_point_right(i)
    if pt_ind and euc_dist(pt, border_pts[i]) >= 2:
        x1, x2, y1, y2 = pt[0], border_pts[i][0], pt[1], border_pts[i][1]
        # linearly interpolate along the direction with more sparsity
        if abs(x1-x2) > abs(y1-y2):
            for new_x in range(min(x1, x2)+1, max(x1, x2)):
                np.append(new_border_pts, [new_x, int(linear_int_y(x1, y1, x2, y2, new_x))])
        else:
            for new_y in range(min(y1, y2)+1, max(y1, y2)):
                np.append(new_border_pts, [int(linear_int_x(x1, y1, x2, y2, new_y)), new_y])
    
    pt_ind, pt = closest_point_left(i)
    if pt_ind and euc_dist(pt, border_pts[i]) >= 2:
        x1, x2, y1, y2 = pt[0], border_pts[i][0], pt[1], border_pts[i][1]
        # linearly interpolate along the direction with more sparsity
        if abs(x1-x2) > abs(y1-y2):
            for new_x in range(min(x1, x2)+1, max(x1, x2)):
                np.append(new_border_pts, [new_x, int(linear_int_y(x1, y1, x2, y2, new_x))])
        else:
            for new_y in range(min(y1, y2)+1, max(y1, y2)):
                np.append(new_border_pts, [int(linear_int_x(x1, y1, x2, y2, new_y)), new_y])
 
    distances = [euc_dist(border_pts[i], border_pts[i+1]) for i in range(len(border_pts)-1)]
    distances = np.array(distances)
print(np.min(distances), np.max(distances))
i = np.argmax(distances)
print(i, len(border_pts), border_pts[i], border_pts[i+1])


border_pts = border_pts[::5]

# after results
plt.imshow(img)
plt.scatter(border_pts[:, 0], border_pts[:, 1], s=1, color='red')
plt.show()

plt.imshow(img)
plt.scatter(new_border_pts[:, 0], new_border_pts[:, 1], s=1, color='red')
plt.show()

# sorted ??
sorted_border_pts = sort_coordinates(border_pts)
plt.imshow(img)
j = 0
colors = ['red', 'yellow', 'green', 'blue']
for b in border_pts:
    plt.scatter([b[0]], [b[1]], s=1, color=colors[j%4])
    j += 1
plt.show()

#plt.scatter([border_pts[0][0]], [border_pts[0][1]], s=5, color='yellow')
#plt.scatter([border_pts[-1][0]], [border_pts[-1][1]], s=5, color='white')
#plt.scatter([border_pts[i][0], border_pts[i+1][0]], [border_pts[i][1], border_pts[i+1][1]], s=3, color='green')
plt.show()
'''

##################
### FINAL CODE ###
##################
def linear_int_x(x1, y1, x2, y2, y):
    return x1 + (y - y1) *  (x2 - x1) / (y2 - y1)

def linear_int_y(x1, y1, x2, y2, x):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

def euc_dist(x, y):
    return np.sqrt(abs(x[0]-y[0])**2 + abs(x[1]-y[1])**2)

border_pts = np.append(border_pts, [border_pts[0]], axis=0)
print(border_pts)
new_border_pts = np.copy(border_pts)
for i in range(len(border_pts)-1):
    if euc_dist(border_pts[i], border_pts[i+1]) > 2:
        x1, x2, y1, y2 = border_pts[i][0], border_pts[i+1][0], border_pts[i][1], border_pts[i+1][1]
        # linearly interpolate along the direction with more sparsity
        if abs(x1-x2) > abs(y1-y2):
            for new_x in range(min(x1, x2)+1, max(x1, x2)):
                new_border_pts = np.append(new_border_pts, [[new_x, int(linear_int_y(x1, y1, x2, y2, new_x))]], axis=0)
        else:
            for new_y in range(min(y1, y2)+1, max(y1, y2)):
                new_border_pts = np.append(new_border_pts, [[int(linear_int_x(x1, y1, x2, y2, new_y)), new_y]], axis=0)

plt.imshow(img)
plt.scatter(border_pts[:, 0], border_pts[:, 1], s=1, color='red')
plt.show()

plt.imshow(img)
plt.scatter(new_border_pts[:, 0], new_border_pts[:, 1], s=1, color='red')
plt.show()
        


