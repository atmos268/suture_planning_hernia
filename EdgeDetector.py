from PIL import Image
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
# import plantcv
from SAM import create_mask

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
    img = cv2.imread('chicken_wound.jpg')
    cropped_img = img[1200:2200, 1700:2300]
    cv2.imwrite('cropped_wound.jpg', cropped_img)
    #img = cv2.imread('cropped_wound.jpg')

    
    mask, img = create_mask('Real Wound.jpeg', np.array([[40, 40], [136,130]]), np.array([0,1]), 'base')
    cv2.imwrite('sam_mask.jpg', mask)
    cv2.imwrite('sam_img.jpg', img)
    new_edge_detector = EdgeDetector()
    # edges = new_edge_detector.find_edges(mask)
    # cv2.imwrite('edges_sam.jpg', edges)
    mask = cv2.imread('sam_mask.jpg')
    img_dilated = new_edge_detector.dilate_to_line(mask, 20)
    #plt.subplot(121)
    #np.array([[349, 600]])
    #plt.imshow(img_dilated)
    #plt.show()
    cv2.imwrite("dilated_sam.jpg", img_dilated)

    skeleton = skeletonize(img_dilated)

    # prune 
    # pruned_image = plantcv.morphology.prune(skeleton)
    # print(pruned_image)

    plt.imsave('skeleton_sam.png', skeleton)

    # now, run the DFS function