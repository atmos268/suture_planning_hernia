import math
import scipy.interpolate as inter
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from heapq import heappop, heappush
from itertools import count
from MeshIngestor import MeshIngestor

from utils import euclidean_dist

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
     
def get_plane_estimation_chatgpt(indices, points, ep=20, verbose=1):
    x_lims = [max(indices[0] - ep//2, 0), min(indices[0] + ep//2, points.shape[0])]
    y_lims = [max(indices[1] - ep//2, 0), min(indices[1] + ep//2, points.shape[1])]

    local_area = points[x_lims[0]:x_lims[1], y_lims[0]:y_lims[1]].reshape(-1, 3)

    A = np.column_stack((local_area[:, 0], local_area[:, 1], np.ones_like(local_area[:, 0])))
    b = local_area[:, 2]

    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    normal = [coeffs[0], coeffs[1], -1]
    d = coeffs[2]  # Distance from origin

    return normal

def get_plane_estimation(mesh, pt, num_nearest=20):
    _, local_area = mesh.get_nearest_k_points(pt, num_nearest)

    #convert to xyz
    local_xyz = [mesh.get_point_location(pt_idx) for pt_idx in local_area]

    sep_xyz = [[pt[0] for pt in local_xyz], [pt[1] for pt in local_xyz], [pt[2] for pt in local_xyz]]


    A = np.column_stack((sep_xyz[0], sep_xyz[1], np.ones_like(sep_xyz[0])))
    b = sep_xyz[2]

    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    normal = [coeffs[0], coeffs[1], -1]
    d = coeffs[2]  # Distance from origin

    return normal


def get_position(indices):
    return points[indices[0]][indices[1]]


model_size = 40 # even number plz

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

def test_synthetic_mesh(vis=False):

    model_size = 40 # even number plz

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

    if vis:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x_vals, y_vals, z_vals)
        plt.title("test surface")
        plt.show()

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

    start_pt_xyz = get_position(start_pt)
    end_pt_xyz = get_position(end_pt)

    print("start pt: ", start_pt)
    print("end pt: ", end_pt)
    euc_dist = math.sqrt((start_pt_xyz[0] - end_pt_xyz[0])** 2 + (start_pt_xyz[1] - end_pt_xyz[1]) ** 2)
    print("euclidean distance: ", euc_dist)

def project_vector_onto_plane(vector, plane_normal):

    plane_normal_normalized = plane_normal / np.linalg.norm(plane_normal)
    projection_onto_normal = np.dot(vector, plane_normal_normalized) * plane_normal_normalized
    projection_onto_plane = vector - projection_onto_normal
    return projection_onto_plane

def get_path_xyz(mesh,path):
    path_xyz = []
    for i in range(len(path)):
        path_xyz.append(get_position(path[i]))

    path_xyz = np.array(path_xyz)
    return path_xyz

def compute_felt_force(mesh, insertion_pt, wound_pt, insertion_force_vec, ellipse_ecc, points_to_sample, ep, force_decay=1, verbose=10):
    
    # insertion_plane = get_plane_estimation_chatgpt(insertion_pt, points_array)
    # wound_plane = get_plane_estimation_chatgpt(wound_pt, points_array)

    insertion_plane = get_plane_estimation(mesh, insertion_pt)
    wound_plane = get_plane_estimation(mesh, wound_pt)

    if verbose > 0:
        print("insertion plane: ", insertion_plane)
        #print("wound plane: ", wound_plane)

    # Get the normal vectors from the coefficients (i.e. drop the constant term)
    insertion_plane_normal = insertion_plane
    wound_plane_normal = wound_plane

    insertion_plane_normal = insertion_plane_normal / np.linalg.norm(insertion_plane_normal)
    wound_plane_normal = wound_plane_normal / np.linalg.norm(wound_plane_normal)

    if verbose > 0:
        print("insertion plane normal: ", insertion_plane_normal)
        print("norm of insertion plane normal:", np.linalg.norm(insertion_plane_normal))

    # Normalize the normal vectors of the plane
    # insertion_plane_normal = insertion_plane_normal / np.linalg.norm(insertion_plane_normal)
    # wound_plane_normal = wound_plane_normal / np.linalg.norm(wound_plane_normal)

    # insertion_vec_proj = project_vector_onto_plane(insertion_force_vec, insertion_plane_normal)
    # insertion_vertex = insertion_pt #TODO: GET THE VERTEX FROM THE MESH
    # wound_vertex = wound_pt #TODO: GET THE VERTEX FROM THE MESH
        
    shortest_path = mesh.get_a_star_path(insertion_pt, wound_pt)
    shortest_path_xyz = np.array([mesh.get_point_location(pt_idx) for pt_idx in shortest_path])
    
    # Calculate distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(shortest_path_xyz, axis=0)**2, axis=1))

    # Calculate cumulative distance
    cumulative_distance = np.insert(np.cumsum(distances), 0, 0)

    # Normalize t to range from 0 to 1
    t = cumulative_distance / cumulative_distance[-1]

    # FIT SPLINE TO SHORTEST PATH
    x = shortest_path_xyz[:, 0] # x-coordinates of the shortest path
    y = shortest_path_xyz[:, 1]
    z = shortest_path_xyz[:, 2]

    print(t,x)
    print(len(t), len(x))

    s_factor = len(x)/5.0 # A starting point for the smoothing factor; adjust based on noise level
    #s_factor = 0.1
    x_smooth = inter.UnivariateSpline(t, x, s=s_factor)
    y_smooth = inter.UnivariateSpline(t, y, s=s_factor)
    z_smooth = inter.UnivariateSpline(t, z, s=s_factor)
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

    if verbose > 1:
        print('x_smooth: ', x_smooth)

    # Calculate the length of the spline
    spline_length = np.trapz(np.sqrt(x_smooth_deriv(t)**2 + y_smooth_deriv(t)**2 + z_smooth_deriv(t)**2), t)

    # put together the path vectors at the insertion and wound points
    insertion_path_vec = np.array([dx_start, dy_start, dz_start])
    insertion_path_vec = insertion_path_vec / np.linalg.norm(insertion_path_vec)

    wound_path_vec = np.array([dx_end, dy_end, dz_end])    
    wound_path_vec = wound_path_vec / np.linalg.norm(wound_path_vec)

    if verbose > 0:
        print("insertion path vec magnitude: ", np.linalg.norm(insertion_path_vec))
        print("wound path vec magnitude: ", np.linalg.norm(wound_path_vec))

    insertion_path_vec_proj = project_vector_onto_plane(insertion_path_vec, insertion_plane_normal)
    insertion_path_vec_proj_normalized = insertion_path_vec_proj / np.linalg.norm(insertion_path_vec_proj)
    
    wound_path_vec_proj = project_vector_onto_plane(wound_path_vec, wound_plane_normal)
    wound_path_vec_proj_normalized = wound_path_vec_proj / np.linalg.norm(wound_path_vec_proj)

    if verbose > 0:
        print('sanity check: dot product of projected vector and normal vector: ', np.dot(insertion_path_vec_proj, insertion_plane_normal))
        print('sanity check: dot product of projected vector and normal vector: ', np.dot(wound_path_vec_proj, wound_plane_normal))
        print('sanity check: projected insertion path vector difference ', np.linalg.norm(insertion_path_vec - insertion_path_vec_proj))
        print('sanity check: projected wound path vector difference ', np.linalg.norm(wound_path_vec - wound_path_vec_proj))
        print('sanity check: dot product of insertion path vector and normal vector: ', np.dot(insertion_path_vec, insertion_plane_normal))
        print('sanity check: dot product of wound path vector and normal vector: ', np.dot(wound_path_vec, wound_plane_normal))

    normals = [insertion_plane_normal, wound_plane_normal]

    insertion_force =  np.linalg.norm(insertion_force_vec)
    force_angle = np.arccos(np.dot(insertion_path_vec_proj_normalized, insertion_force_vec) / (np.linalg.norm(insertion_path_vec_proj_normalized) * insertion_force))
    wound_force = insertion_force - force_decay * spline_length * np.sqrt((np.sin(force_angle) / ellipse_ecc) ** 2 + (np.cos(force_angle)) ** 2)
    wound_force = max(0, wound_force)

    # wound_direction is the direction of the wound on the plane wound_plane from wound_path_vec_proj_normalized by angle force_angle
    wound_cross_prod = np.cross(wound_plane_normal, wound_path_vec_proj_normalized)
    wound_direction = np.cos(force_angle) * wound_path_vec_proj_normalized + np.sin(force_angle) * wound_cross_prod

    wound_force_vec = wound_force * wound_direction


    if verbose > 1:
        # TODO: Fix plotting
        spline = [x_smooth, y_smooth, z_smooth]
        plot_mesh_path_and_spline(mesh, shortest_path, spline, normals, insertion_force_vec, wound_force_vec)

    return wound_force_vec


def compute_total_force(mesh, suture_pts, suture_force_vecs, wound_pt, ellipse_ecc, points_to_sample, ep, force_decay=1, verbose=0):
    total_force = np.array([0, 0, 0])
    for i in range(len(suture_pts)):
        insertion_pt = suture_pts[i]
        insertion_force_vec = suture_force_vecs[i]
        if np.linalg.norm(insertion_pt, wound_pt) <= 1/force_decay:   #TODO: double check the distance
            felt_force = compute_felt_force(mesh, insertion_pt, wound_pt, insertion_force_vec, ellipse_ecc, points_to_sample, ep, force_decay, verbose)
            total_force += felt_force

    return total_force

def compute_closure_shear_force(mesh, insertion_pts, extraction_pts, wound_pt, wound_derivative, ellipse_ecc, points_to_sample, ep, force_decay=1, verbose=0):

    force_vecs = extraction_pts - insertion_pts

    wound_plane = get_plane_estimation(mesh, wound_pt)
    wound_derivative_proj = project_vector_onto_plane(wound_derivative, wound_plane)
    wound_derivative_proj = wound_derivative_proj / np.linalg.norm(wound_derivative_proj)

    wound_line_normal = np.cross(wound_derivative_proj, wound_plane)
    if verbose > 0:
        print("sanity check: magnitude of wound line normal (should be equal to 1): ", np.linalg.norm(wound_line_normal))
    wound_line_normal = wound_line_normal / np.linalg.norm(wound_line_normal)

    total_insertion_force = compute_total_force(mesh, insertion_pts, force_vecs, wound_pt, ellipse_ecc, points_to_sample, ep, force_decay, verbose)
    total_extraction_force = compute_total_force(mesh, extraction_pts, -force_vecs, wound_pt, ellipse_ecc, points_to_sample, ep, force_decay, verbose)

    insertion_closure_force = np.dot(total_insertion_force, wound_line_normal)
    extraction_closure_force = np.dot(total_extraction_force, wound_line_normal)

    closure_force = np.abs(insertion_closure_force - extraction_closure_force)

    insertion_shear_force = np.dot(total_insertion_force, wound_derivative_proj)
    extraction_shear_force = np.dot(total_extraction_force, wound_derivative_proj)

    shear_force = np.abs(insertion_shear_force - extraction_shear_force)

    return closure_force, shear_force


def compute_closure_shear_loss(mesh, insertion_pts, extraction_pts, wound_spline, granularity, ellipse_ecc, points_to_sample, ep, force_decay=1, closure_force_ideal=1, verbose=0):
    # granularity is the number of points to sample on the wound spline
    # TODO: step-out function inside here (i.e. get wound_suture_pts, then compute insertion_pts and extraction_pts from that)

    x_spline = wound_spline[0]
    y_spline = wound_spline[1]
    z_spline = wound_spline[2]

    t = np.linspace(0, 1, granularity)

    dx_spline = x_spline.derivative()
    dy_spline = y_spline.derivative()
    dz_spline = z_spline.derivative()

    closure_loss = 0
    shear_loss = 0

    for i in range(granularity):
        x = x_spline(t[i])
        y = y_spline(t[i])
        z = z_spline(t[i])

        dx = dx_spline(t[i])
        dy = dy_spline(t[i])
        dz = dz_spline(t[i])

        wound_pt = np.array([x, y, z])
        wound_derivative = np.array([dx, dy, dz])

        closure_force, shear_force = compute_closure_shear_force(mesh, insertion_pts, extraction_pts, wound_pt, wound_derivative, ellipse_ecc, points_to_sample, ep, force_decay, verbose)

        closure_loss += (closure_force - closure_force_ideal)**2/granularity
        shear_loss += shear_force**2/granularity

    return closure_loss, shear_loss


def generate_random_force_vector():
    return np.array([random.random(), random.random(), random.random()])


def plot_mesh_path_and_spline(mesh, path, spline, normals, insertion_force_vec, wound_force_vec,spline_segments=100):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("Mesh and Spline")

    mesh_coords = mesh.vertex_coordinates
    ax.scatter3D(mesh_coords[::5, 0], mesh_coords[::5, 1], mesh_coords[::5, 2], color='red', alpha=0.1)


    spline_x = spline[0]
    spline_y = spline[1]
    spline_z = spline[2]

    spline_coords_x = []
    spline_coords_y = []
    spline_coords_z = []

    for t in np.linspace(0, 1, spline_segments):
        spline_coords_x.append(spline_x(t))
        spline_coords_y.append(spline_y(t))
        spline_coords_z.append(spline_z(t))

    plt.plot(spline_coords_x, spline_coords_y, spline_coords_z, color='green')


    spline_direc_x = spline_x.derivative()
    spline_direc_y = spline_y.derivative()
    spline_direc_z = spline_z.derivative()

    spline_start_direc_x = spline_direc_x(0)
    spline_start_direc_y = spline_direc_y(0)
    spline_start_direc_z = spline_direc_z(0)

    spline_start_direc_magnitude = np.linalg.norm([spline_start_direc_x, spline_start_direc_y, spline_start_direc_z])

    spline_end_direc_x = spline_direc_x(1)
    spline_end_direc_y = spline_direc_y(1)
    spline_end_direc_z = spline_direc_z(1)

    spline_end_direc_magnitude = np.linalg.norm([spline_end_direc_x, spline_end_direc_y, spline_end_direc_z])

    spline_start_direc_x = 15 * spline_start_direc_x / spline_start_direc_magnitude
    spline_start_direc_y = 15 * spline_start_direc_y / spline_start_direc_magnitude
    spline_start_direc_z = 15 * spline_start_direc_z / spline_start_direc_magnitude

    spline_end_direc_x = 15 * spline_end_direc_x / spline_end_direc_magnitude
    spline_end_direc_y = 15 * spline_end_direc_y / spline_end_direc_magnitude
    spline_end_direc_z = 15 * spline_end_direc_z / spline_end_direc_magnitude

    plt.quiver(spline_coords_x[0], spline_coords_y[0], spline_coords_z[0], spline_start_direc_x, spline_start_direc_y, spline_start_direc_z, color='purple', normalize=True, length=0.01)
    plt.quiver(spline_coords_x[-1], spline_coords_y[-1], spline_coords_z[-1], spline_end_direc_x, spline_end_direc_y, spline_end_direc_z, color='purple', normalize=True, length=0.01)

    start_plane_normal = normals[0]
    end_plane_normal = normals[1]

    start_plane_normal = 15 * start_plane_normal
    end_plane_normal = 15 * end_plane_normal

    # plt.quiver(spline_coords_x[0], spline_coords_y[0], spline_coords_z[0], start_plane_normal[0], start_plane_normal[1], start_plane_normal[2], color='orange', length=0.01)
    # plt.quiver(spline_coords_x[-1], spline_coords_y[-1], spline_coords_z[-1], end_plane_normal[0], end_plane_normal[1], end_plane_normal[2], color='orange', length=0.01)
    
    # plot the insertion and wound forces
    plt.quiver(spline_coords_x[0], spline_coords_y[0], spline_coords_z[0], insertion_force_vec[0], insertion_force_vec[1], insertion_force_vec[2], color='orange', normalize=True, length=0.01)
    plt.quiver(spline_coords_x[-1], spline_coords_y[-1], spline_coords_z[-1], wound_force_vec[0], wound_force_vec[1], wound_force_vec[2], color='orange', normalize=True, length=0.01)

    plt.show()

def test_real_mesh(vis = False):

    # Specify the path to your text file
    adj_path = 'adjacency_matrix.txt'
    loc_path = 'vertex_lookup.txt'

    mesh = MeshIngestor(adj_path, loc_path)

    # Create the graph
    mesh.generate_mesh()

    # pick two random points for testing purposes
    rand_start_pt = mesh.get_point_location(random.randrange(0, len(mesh.vertex_coordinates)))
    rand_wound_pt = mesh.get_point_location(random.randrange(0, len(mesh.vertex_coordinates)))

    force_vec = generate_random_force_vector()
    print("force vector: ", force_vec)

    # now, compute the felt force
    # compute_felt_force(mesh, shortest_path, insertion_pt, wound_pt, insertion_force_vec, ellipse_ecc, points_to_sample, ep, force_decay=1, verbose=1)
    
    felt_force = compute_felt_force(mesh, rand_start_pt, rand_wound_pt, force_vec, 2, 10, 10, verbose=10)
    print("felt force: ", felt_force)

if __name__ == '__main__':
    test_real_mesh()