from PIL import Image
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import plantcv
from SAM import create_mask
from segment_anything import SamPredictor
from largestCC import keep_largest_connected_component
from fillHoles import fillHoles
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

# def keep_largest_connected_component(imgPath):
#     # Ensure the input image is in binary format (0 and 255).
#     input_image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

#     print(len(np.unique(input_image)))
#     # Ensure the image is binary (threshold if needed).
#     _, binary_image = cv2.threshold(input_image, 128, 255, cv2.THRESH_BINARY)

#     print(len(np.unique(binary_image)))
#     # if len(np.unique(binary_image)) != 2:
#     #     raise ValueError("Input image must be binary (0 and 255).")

#     # Find connected components in the binary image.
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

#     # Find the index of the largest connected component (excluding the background).
#     largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

#     # Create a binary mask for the largest component.
#     largest_component_mask = (labels == largest_component_index).astype(np.uint8) * 255

#     return largest_component_mask


# mallika's code for point clicking

def click_points_simple(img):
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(img)
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


# Karim's DFS code
def find_length_and_endpoints(skeleton_img):
    #### IDEA: do DFS but have a left and right DFS with distances for one being negative and the other being positive 
    nonzero_pts = cv2.findNonZero(np.float32(skeleton_img))
    if nonzero_pts is None:
        nonzero_pts = [[[0,0]]]
    total_length = len(nonzero_pts)
    start_pt = (nonzero_pts[0][0][1], nonzero_pts[0][0][0])
    # run dfs from this start_pt, when we encounter a point with no more non-visited neighbors that is an endpoint
    endpoints = []
    NEIGHS = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1,-1), (-1,1), (1,-1),(1,1)]
    visited = set()
    q = [start_pt]
    dist_q = [0]
    # tells us if the first thing we look at is actually an endpoint
    initial_endpoint = False
    # carry out floodfill
    q = [start_pt]
    # carry out floodfill
    IS_LOOP = False
    ENTIRE_VISITED = [False] * int(np.nonzero(skeleton_img).sum())
    def dfs(q, dist_q, visited, start_pixel, increment_amt):
        '''
        q: queue with next point on skeleton for one direction
        dist_q: queue with distance from start point to next point for one direction
        visited: queue with visited points for only one direction
        increment_amt: counter that indicates direction +/- 1
        '''

        is_loop = ENTIRE_VISITED[start_pixel + increment_amt*len(visited)]
        if is_loop:
            return is_loop


        while len(q) > 0:
            next_loc = q.pop()
            distance = dist_q.pop()
            visited.add(next_loc)
            counter = 0
            for n in NEIGHS:
                test_loc = (next_loc[0]+n[0], next_loc[1]+n[1])
                if (test_loc in visited):
                    continue
                if test_loc[0] >= len(skeleton_img[0]) or test_loc[0] < 0 \
                        or test_loc[1] >= len(skeleton_img[0]) or test_loc[1] < 0:
                    continue
                if skeleton_img[test_loc[0]][test_loc[1]] == True:
                    counter += 1
                    #length_checker += 1
                    q.append(test_loc)
                    dist_q.append(distance+increment_amt)
            # this means we haven't added anyone else to the q so we "should" be at an endpoint
            if counter == 0:
                endpoints.append([next_loc, distance])
            # if next_loc == start_pt and counter == 1:
            #     endpoints.append([next_loc, distance])
            #     initial_endpoint = True
    counter = 0
    length_checker = 0
    increment_amt = 1
    visited = set([start_pt])
    for n in NEIGHS:
        test_loc = (start_pt[0]+n[0], start_pt[1]+n[1])
        # one of the neighbors is valued at one so we can dfs across it
        if skeleton_img[test_loc[0]][test_loc[1]] == True:
            counter += 1
            q = [test_loc]
            dist_q = [0]
            dfs(q, dist_q, visited, increment_amt)
            # the first time our distance will be incrementing but the second time
            # , i.e. when dfs'ing the opposite direction our distance will be negative to differentiate both paths
            increment_amt = -1
    # we only have one neighbor therefore we must be an endpoint
    if counter == 1:
        distance = 0
        endpoints.append([start_pt, distance])
        initial_endpoint = True

    final_endpoints = []
    
    largest_pos = None
    largest_neg = None

    for pt, distance in endpoints:
        if largest_pos is None or distance > endpoints[largest_pos][1]:
            largest_pos = endpoints.index([pt, distance])
        elif largest_neg is None or distance < endpoints[largest_neg][1]:
            largest_neg = endpoints.index([pt, distance])
    if initial_endpoint:
        final_endpoints = [endpoints[0][0], endpoints[largest_pos][0]]
    else:
        final_endpoints = [endpoints[largest_neg][0], endpoints[largest_pos][0]]
    
    #display results 
    plt.scatter(x = [j[0][1] for j in endpoints], y=[i[0][0] for i in endpoints],c='w')
    plt.scatter(x = [final_endpoints[1][1]], y=[final_endpoints[1][0]],c='r')
    plt.scatter(x = [final_endpoints[0][1]], y=[final_endpoints[0][0]],c='r')
    plt.title("final endpoints")
    plt.scatter(x=start_pt[1], y=start_pt[0], c='g')
    plt.imshow(skeleton_img, interpolation="nearest")
    plt.show() 

    print("the total length is ", total_length)
    return total_length, final_endpoints

if __name__ == "__main__":

    box_method = True

    img_path = 'image_left_001.png'

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
        mask = create_mask(img_path, box, [], 'huge', box_method=True)

        # make the mask into an image and save
        im = Image.fromarray(mask)
        im.save('sam_mask.jpg')

    else:

        left_coords, right_coords = click_points_simple(numpydata)
        print(left_coords, right_coords)

        cv2_img = cv2.imread(img_path)

        num_left = len(left_coords)
        num_right = len(right_coords)

        fore_back = [1 for _ in range(num_left)] + [0 for _ in range(num_right)]

        mask, img = create_mask(img_path, np.array(left_coords + right_coords), np.array(fore_back), 'base')
    
        cv2.imwrite('sam_mask.jpg', mask)
    mask = keep_largest_connected_component('sam_mask.jpg')
    cv2.imwrite('sam_mask.jpg', mask)
    # cv2.imwrite('sam_img.jpg', img)
    
    new_edge_detector = EdgeDetector()
    mask = cv2.imread('sam_mask.jpg')
    img_dilated = new_edge_detector.dilate_to_line(mask, 5)
    cv2.imwrite("dilated_sam.jpg", img_dilated)
    img_dilated = fillHoles('dilated_sam.jpg')
    cv2.imwrite("filledHoles.jpg", img_dilated)

    # check for binary?
    
    # threshold
    binary_image = np.where(img_dilated > 0, 1, 0)


    skeleton = skeletonize(binary_image)
    plt.imsave('skeleton_sam.jpg', skeleton)

    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(numpydata)
    ax1.title.set_text("Original")

    img = Image.open("filledHoles.jpg")
    numpydata = np.asarray(img)
    ax2.imshow(numpydata)
    ax2.title.set_text("New Dilation (filled holes)")

    img = Image.open('skeleton_sam.jpg')
    numpydata = np.asarray(img)
    ax3.imshow(numpydata)
    ax3.title.set_text("Skeletonization")

    fig.tight_layout()

    # plot the image, dilation, skeleton
    plt.savefig('box_results/chicken_result1.jpg', dpi=1200)

