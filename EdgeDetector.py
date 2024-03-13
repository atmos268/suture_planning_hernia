from PIL import Image
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.interpolate as inter
from point_ordering import get_pt_ordering
# import plantcv
from SAM import create_mask
from segment_anything import SamPredictor
from largestCC import keep_largest_connected_component
from fillHoles import fillHoles
from matplotlib import colormaps
from utils import click_points_simple
import os

'''This class will process an image, and produce a spline of where the wound is based on the image'''
class EdgeDetector:
    
    def find_edges(self, img):
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("grayscale_image.jpg", grayscale_image)
        # grayscale_image = np.clip(grayscale_image, 0, 170)
        cv2.imwrite("grayscale_image_clip.jpg", grayscale_image)
        # blur = cv2.bilateralFilter(grayscale_image, 5, 100, 150)
        cv2.imwrite("blur_clip.jpg", grayscale_image)
        return cv2.Canny(grayscale_image, 100, 600)

    def dilate_to_line(self, edge_mask, kernel_dim):
        kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
        return cv2.dilate(edge_mask, kernel, iterations=1) 
 
    def generate_spline(self, pixels):
        pass


def img_to_line(img_path, box_method, viz=False, save_figs=False):

    if not os.path.isdir("temp_images"):
        os.mkdir('temp_images')
    
    # load the image and convert into
    # numpy array
    img = Image.open(img_path)
    
    # asarray() class is used to convert
    # PIL images into NumPy arrays
    numpydata = np.asarray(img)

    if box_method:
        left_coords, right_coords = click_points_simple(numpydata)
        if len(left_coords) != 2:
            raise ValueError("Please select 2 points (top left, bottom right)")
        
        # display box and image
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(numpydata)

        # Create a Rectangle patch
        w = left_coords[1][0] - left_coords[0][0]
        h = left_coords[1][1] - left_coords[0][1]
        rect = patches.Rectangle(tuple(left_coords[0]), w, h, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.show()

        box = left_coords[0] + left_coords[1]
        mask = create_mask(img_path, box, [], 'base', box_method=True)

        # make the mask into an image and save
        im = Image.fromarray(mask)
        im.save('temp_images/sam_mask.jpg')

    else:

        left_coords, right_coords = click_points_simple(numpydata)

        num_left = len(left_coords)
        num_right = len(right_coords)

        fore_back = [1 for _ in range(num_left)] + [0 for _ in range(num_right)]

        mask, img = create_mask(img_path, np.array(left_coords + right_coords), np.array(fore_back), 'base')
    
        cv2.imwrite('temp_images/sam_mask.jpg', mask)
    mask = keep_largest_connected_component('temp_images/sam_mask.jpg')
    cv2.imwrite('temp_images/sam_mask.jpg', mask)

    # mask post-processing    
    new_edge_detector = EdgeDetector()
    mask = cv2.imread('temp_images/sam_mask.jpg')
    img_dilated = new_edge_detector.dilate_to_line(mask, 5)
    cv2.imwrite("temp_images/dilated_sam.jpg", img_dilated)
    img_dilated = fillHoles('temp_images/dilated_sam.jpg')
    cv2.imwrite("temp_images/filledHoles.jpg", img_dilated)
    
    # threshold to feed into skeletonize
    binary_image = np.where(img_dilated > 0, 1, 0)
    skeleton = skeletonize(binary_image)

    np.save('temp_images/binary_skeleton.npy', skeleton)

    plt.imshow(skeleton)
    if viz:
        plt.show()

    plt.imsave('temp_images/skeleton_sam.jpg', skeleton)

    # order points 
    ordered_points = get_pt_ordering(skeleton)

    # display results
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(numpydata)
    ax1.title.set_text("Original")

    filled_holes = Image.open("temp_images/sam_mask.jpg")
    numpydata = np.asarray(filled_holes)
    ax2.imshow(numpydata)
    ax2.title.set_text("SAM Mask")
    fig.tight_layout()

    # plot the image, dilation, skeleton
    if save_figs:
        plt.savefig('experimentation/point_results/chicken_result_left1.jpg', dpi=1200)

    # now, order the points

    # overlay ordered points over the original image
    fig_overlay =  plt.figure()

    img_np = np.asarray(img)
    plt.imshow(img_np)
    plt.plot([pt[1] for pt in ordered_points], [pt[0] for pt in ordered_points])

    if viz:
        plt.show()
    
    return ordered_points, numpydata

def line_to_spline(line, img_path, mm_per_pixel, viz=False):

    # fit spline to points
    exact_tck, u = inter.splprep([[pt[0] for pt in line], [pt[1] for pt in line]], k=3, s=0)
    exact_wound_parametric = lambda t, d: inter.splev(t, exact_tck, der = d)

    # from our ordered set of points, what fraction we pick: we will pick 1 in every sample_ratio points
    sample_ratio = 30

    sampled_pts = [line[i * sample_ratio] for i in range(len(line) // sample_ratio)] + [line[-1]]

    sampled_tck, u = inter.splprep([[pt[0] for pt in sampled_pts], [pt[1] for pt in sampled_pts]], k=2, s=0)
    sampled_wound_parametric = lambda t, d: inter.splev(t, sampled_tck, der = d)

    smoothed_tck, u = inter.splprep([[pt[0] for pt in line], [pt[1] for pt in line]], k=2)
    smoothed_wound_parametric = lambda t, d: inter.splev(t, smoothed_tck, der = d)

    # plot spline
    exact_spline_pts = []
    sampled_spline_pts = []
    smoothed_spline_pts = []

    for t_step in np.linspace(0, 1, 500):
        exact_spline_pts.append(exact_wound_parametric(t_step, 0))
        sampled_spline_pts.append(sampled_wound_parametric(t_step, 0))
        smoothed_spline_pts.append(smoothed_wound_parametric(t_step, 0))
    
    if viz:
        img = Image.open(img_path)
        img_np = np.asarray(img)
        plt.imshow(img_np)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # plt.plot([pt[1] for pt in spline_pts], [pt[0] for pt in spline_pts], color='r')
        ax1.imshow(img_np)
        ax1.plot([pt[1]/mm_per_pixel for pt in exact_spline_pts], [pt[0]/mm_per_pixel for pt in exact_spline_pts])
        ax2.imshow(img_np)
        ax2.plot([pt[1]/mm_per_pixel for pt in sampled_spline_pts], [pt[0]/mm_per_pixel for pt in sampled_spline_pts])
        ax3.imshow(img_np)
        ax3.plot([pt[1]/mm_per_pixel for pt in smoothed_spline_pts], [pt[0]/mm_per_pixel for pt in smoothed_spline_pts])
        
        # plot side by side
        plt.savefig("spline.png")

    return sampled_spline_pts, sampled_tck

def line_to_spline_3d(line, sample_ratio=30, viz=False):


    x = line[:, 0] # x-coordinates of the shortest path
    y = line[:, 1]
    z = line[:, 2]


    # define t based on cumulative dists
    distances = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))

    # Calculate cumulative distance
    cumulative_distance = np.insert(np.cumsum(distances), 0, 0)

    # Normalize t to range from 0 to 1
    t = cumulative_distance / cumulative_distance[-1]

    s_factor = len(x)/5
    # get spline in each dimension
    x_spline = inter.UnivariateSpline(t, x, s=s_factor)
    y_spline = inter.UnivariateSpline(t, y, s=s_factor)
    z_spline = inter.UnivariateSpline(t, z, s=s_factor)

    return [x_spline, y_spline, z_spline]

if __name__ == "__main__":
    
    box_method = True
    img_path = 'chicken_images/image_left_001.png'

    line, mask = img_to_line(img_path, box_method, viz=True)

    spline, tck = line_to_spline(line, img_path, viz=True)

    # now run original pipeline




    
    
    


    