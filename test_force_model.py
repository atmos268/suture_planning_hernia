import math
import scipy.interpolate as inter
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from heapq import heappop, heappush
from itertools import count

from utils import euclidean_dist

model_size = 40 # even number plz

def get_neighbors(x, y, size):

    neighbors = []
    candidates = [(x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1)]

    for candidate in candidates:
        if 0 <= candidate[0] < size and 0 <= candidate[1] < size:
            neighbors.append(candidate)

    return neighbors

# imagniary surface
def surface(x, y):
    return 0.05 * x ** 2 + 3 * math.sin(0.1 * x + 0.2 * y) + 2
     

x_vals = []
y_vals = []
z_vals = []

points = [[0 for i in range(model_size)] for j in range(model_size)]

for x_idx, x in enumerate(range(-model_size//2, model_size//2)):
    for y_idx, y in enumerate(range(-model_size//2, model_size//2)):
        x_ep = (random.randrange(10) - 5) / 10
        y_ep = (random.randrange(10) - 5) / 10
        z_ep = (random.randrange(10) - 5) / 40
        x_vals.append(x + x_ep)
        y_vals.append(y + y_ep)
        z_vals.append(surface(x, y) + z_ep)

        points[x_idx][y_idx] = (x + x_ep, y + y_ep, surface(x, y) + z_ep)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x_vals, y_vals, z_vals)
plt.title("test surface")
# plt.show()

def get_plane_estimation(indices, ep):
    # get grid of ep x ep around area
    x_lims = [max(indices[0] - ep//2, 0), min(indices[0] + ep//2)]
    y_lims = [max(indices[1] - ep//2, 0), min(indices[1] + ep//2)]

    local_area = points[x_lims[0]: x_lims[1]][y_lims[0]: y_lims[1]]

    # now, make the plane approximation

    data = np.ones((len(x_lims) * len(y_lims), 4))
    for i in x_lims:
        for j in y_lims:
            data[i * len(x_lims) + j][:-1] = local_area[i][j]

    coeffs = np.linalg.lstsq(data, np.zeros(len(x_lims) * len(y_lims)))

    # we worked out ax + bx + cx + d = 0
    return coeffs

def get_position(indices):
    return points[indices[0]][indices[1]]

# implement an a* search to get an accurate measure of path distance

# pick two points to do the search from one to the other

pt1 = (2, 5)
pt2 = (21, 34)

# now, do a*
c = count()
queue = [(0, next(c), pt1, 0, None)]


# credit: astar.py in Networks package
enqueued = {}
explored = {}

final_path = []
final_len = 0

while queue:
    priority, _,  popped_pt, curr_dist, parent = heappop(queue)

    if popped_pt == pt2:
        path = [popped_pt]
        node = parent
        while node is not None:
            path.append(node)
            node = explored[node]
        path.reverse()
        final_path = path
        final_len = priority
        break

    if popped_pt in explored:
        # Do not override the parent of starting node
        if explored[popped_pt] is None:
            continue

        # Skip bad paths that were enqueued before finding a better one
        qcost, heuristic = enqueued[popped_pt]
        if qcost < curr_dist:
            continue

    explored[popped_pt] = parent

    # enqueue neighbors unexplored and unqueue with relevant priority
    neighbors = get_neighbors(popped_pt[0], popped_pt[1], model_size)

    for neighbor in neighbors:
        step_dist = euclidean_dist(get_position(popped_pt), get_position(neighbor))

        new_cost = curr_dist + step_dist

        if neighbor in enqueued:
            queue_cost, heuristic = enqueued[neighbor]

            if queue_cost <= new_cost:
                continue

        else:
            heuristic = euclidean_dist(get_position(pt2), get_position(neighbor))

        enqueued[neighbor] = new_cost, heuristic
        heappush(queue, (new_cost + heuristic, next(c), neighbor, new_cost, popped_pt))
    

# print final list
print(final_path)
print("path length: ", final_len)
start_pt = final_path[0]
end_pt = final_path[-1]
euc_dist = math.sqrt((start_pt[0] - end_pt[0])** 2 + (start_pt[1] - end_pt[1]) ** 2)
print("euclidean distance: ", euc_dist)

start_plane = get_plane_estimation(final_path[0])
end_plane = get_plane_estimation(final_path[-1])

# project the start vector onto the start plane

# locally work out the direction of the wound (use spline fitting + smoothing)

# project the wound direction vecton onto the start and end planes

# for the 'project' method, take the original vector and directly project onto the end plane

# for the 'angle from path' method, take the wound line vector and the force vector, project and find angle
# then, project the wound line vector onto the end plane, sweep the same angle out. 

coords_list = [get_position(idx) for idx in final_path]
coords_x = [coord[0] for coord in coords_list]
coords_y = [coord[1] for coord in coords_list]
coords_z = [coord[2] for coord in coords_list]

plt.plot(coords_x, coords_y, coords_z, color='red')
plt.show()
