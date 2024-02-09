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

    # we worked out ax + bx + cx + d = 0 ### TODO: DON'T YOU MEAN ax + by + cz + d = 0 ?
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
    

def project_vector_onto_plane(vector, plane_normal):

    plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
    projection_onto_normal = np.dot(vector, plane_normal_normalized) * plane_normal_normalized
    projection_onto_plane = vector - projection_onto_normal
    return projection_onto_plane



def compute_felt_force(mesh, shortest_path, insertion_pt, wound_pt, insertion_force_vec, ellipse_ecc, points_to_sample, ep, force_decay=1):
    insertion_plane = get_plane_estimation(insertion_pt, ep)
    wound_plane = get_plane_estimation(wound_pt, ep)

    # Get the normal vectors from the coefficients (i.e. drop the constant term)
    insertion_plane_normal = insertion_plane[0:3]
    wound_plane_normal = wound_plane[0:3]

    # Normalize the normal vectors of the plane
    insertion_plane_normal = insertion_plane_normal / np.linalg.norm(insertion_plane_normal)
    wound_plane_normal = wound_plane_normal / np.linalg.norm(wound_plane_normal)

    insertion_vec_proj = project_vector_onto_plane(insertion_force_vec, insertion_plane_normal)
    insertion_vertex = insertion_pt #TODO: GET THE VERTEX FROM THE MESH
    wound_vertex = wound_pt #TODO: GET THE VERTEX FROM THE MESH
    #TODO: GET SHORTEST PATH FROM MESH
    
    # Calculate distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(shortest_path, axis=0)**2, axis=1))

    # Calculate cumulative distance
    cumulative_distance = np.insert(np.cumsum(distances), 0, 0)

    # Normalize t to range from 0 to 1
    t = cumulative_distance / cumulative_distance[-1]


    # FIT SPLINE TO SHORTEST PATH
    x = shortest_path[:][0]
    y = shortest_path[:][1]
    z = shortest_path[:][2]
    s_factor = len(x)  # A starting point for the smoothing factor; adjust based on noise level
    x_smooth = UnivariateSpline(t, x, s=s_factor)
    y_smooth = UnivariateSpline(t, y, s=s_factor)
    z_smooth = UnivariateSpline(t, z, s=s_factor)
    # GET DERIVATIVES OF SPLINE AT insertion_vertex AND wound_vertex
    x_smooth_deriv = x_smooth.derivative()
    y_smooth_deriv = y_smooth.derivative()
    z_smooth_deriv = z_smooth.derivative()
    dx_start = x_smooth_deriv(0)
    dy_start = y_smooth_deriv(0)
    dz_start = z_smooth_deriv(0)
    dx_end = x_smooth_deriv(1)
    dy_end = y_smooth_deriv(1)
    dz_end = z_smooth_deriv(1)

    # Calculate the length of the spline
    spline_length = np.trapz(np.sqrt(x_smooth_deriv(t)**2 + y_smooth_deriv(t)**2 + z_smooth_deriv(t)**2), t)

    # put together the path vectors at the insertion and wound points
    insertion_path_vec = np.array([dx_start, dy_start, dz_start])
    wound_path_vec = np.array([dx_end, dy_end, dz_end])    

    insertion_path_vec_proj = project_vector_onto_plane(insertion_path_vec, insertion_plane_normal)
    wound_path_vec_proj = project_vector_onto_plane(wound_path_vec, wound_plane_normal)
    insertion_path_vec_proj_normalized = insertion_path_vec_proj / np.linalg.norm(insertion_path_vec_proj)
    wound_path_vec_proj_normalized = wound_path_vec_proj / np.linalg.norm(wound_path_vec_proj)
    insertion_force =  np.linalg.norm(insertion_force_vec)
    force_angle = np.arccos(np.dot(insertion_path_vec_proj_normalized, insertion_force_vec) / (np.linalg.norm(insertion_path_vec_proj_normalized) * insertion_force))
    wound_force = insertion_force - force_decay * spline_length * np.sqrt((np.sin(force_angle) / ellipse_ecc) ** 2 + (np.cos(force_angle)) ** 2)
    wound_force = max(0, wound_force)

    # wound_direction is the direction of the wound on the plane wound_plane from wound_path_vec_proj_normalized by angle force_angle
    wound_cross_prod = np.cross(wound_plane_normal, wound_path_vec_proj_normalized)
    wound_direction = np.cos(force_angle) * wound_path_vec_proj_normalized + np.sin(force_angle) * wound_cross_prod

    wound_force_vec = wound_force * wound_direction

    return wound_force_vec









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
