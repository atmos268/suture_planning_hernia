import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_neighbors(pt, seen, binary_skeleton):

    neighs = []

    r, c = pt[0], pt[1]
    for n_r in range(max(0, r - 1), min(len(binary_skeleton), r + 2)):
        for n_c in range(max(0, c - 1), min(len(binary_skeleton[0]), c + 2)):
            if r == n_r and c == n_c:
                continue

            # otherwise we add to neighbor
            if binary_skeleton[n_r][n_c] != 0 and (n_r, n_c) not in seen:
                neighs.append((n_r, n_c))

    return neighs

def get_pt_ordering(mask_path):

    skeleton_mask = np.load(mask_path, allow_pickle=True)
    
    # asarray() class is used to convert
    # PIL images into NumPy arrays

    print(skeleton_mask)
    binary_skeleton = np.float32(skeleton_mask)

    seen = set()
    curr_layer = []
    prev_layer = []    

    nonzero_pts = cv2.findNonZero(np.float32(skeleton_mask))

    start_pt = (nonzero_pts[0][0][1], nonzero_pts[0][0][0])

    seen.add(start_pt)

    curr_layer = [start_pt]

    # check that actually 1
    print("check:", start_pt, binary_skeleton[start_pt[0]][start_pt[1]])

    # now, do a BFS from the point outwards

    while curr_layer:

        next_layer = []

        for fringe_pt in curr_layer:
            neighbors = get_neighbors(fringe_pt, seen, binary_skeleton)

            for neighbor in neighbors:
                seen.add(neighbor)

            # N.B these neighbors are unvisited
            next_layer.extend(neighbors)

        prev_layer = curr_layer
        curr_layer = next_layer
    
    print(prev_layer)

    plt.scatter([nonzero_pt[0][1] for nonzero_pt in nonzero_pts], [nonzero_pt[0][0] for nonzero_pt in nonzero_pts])
    plt.scatter([prev_layer[0][0]], [prev_layer[0][1]])
    plt.show()

    # now walk it back the other way, keeping a track of point seen (accomplish this with a prev or next dictionary)
    # Then, we can traverse this list and assign the points their ordering!

# Karim's DFS code
# FOR REFERENCE
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