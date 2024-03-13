import numpy as np
from SuturePlacement3d import SuturePlacement3d
from MeshIngestor import MeshIngestor
from test_force_model import get_plane_estimation, project_vector_onto_plane
import scipy.interpolate as inter
import random
import scipy.optimize as optim
import matplotlib.pyplot as plt
from utils import euclidean_dist
class Optimizer3d:
    """
    This class takes in a mesh object, and a spline on the surface of the wound, and optimizes the position of
    sutures along the spline
    mesh: A MeshIngestor object, representing the surface of the wound.
    spline: The spline of the wound
    suture_width: how far, in mm, the insertion and extraction points should be from the wound line
    hyperparameters: hyperparameters for our optimization
    """
    def __init__(self, mesh, spline, suture_width, hyperparameters, force_model_parameters):
        self.mesh = mesh
        self.spline = spline
        self.suture_width = suture_width 
        self.suture_placement = None
        self.c_ideal = hyperparameters[0]
        self.gamma = hyperparameters[1]
        self.c_var = hyperparameters[2]
        self.c_shear = hyperparameters[3]
        self.c_closure = hyperparameters[4]
        self.force_model = force_model_parameters #Force model parameters are a dictionary 
        self.num_points_for_plane = 1000

        # Ideal closure force calculated according to properties of the original diamond force model
        # If you want, can specify ideal_closure_force yourself, otherwise set to None and this calculation will be done
        if force_model_parameters['ideal_closure_force'] is None:
            self.force_model['ideal_closure_force'] = np.max(2.0 - self.force_model['force_decay'] * self.suture_width, 0)
        
        if self.force_model['verbose'] > 0:
            print("Ideal closure force: ", self.force_model['ideal_closure_force'])

        if self.force_model['imparted_force'] is None:
            self.force_model['imparted_force'] = 1.0


    def calculate_spline_length(self, spline, mesh):
        spline_x, spline_y, spline_z = spline[0], spline[1], spline[2]
        start = [spline_x(0), spline_y(0), spline_z(0)]
        end = [spline_x(1), spline_y(1), spline_z(1)]
        shortest_path = mesh.get_a_star_path(start, end)
        shortest_path_xyz = np.array([mesh.get_point_location(pt_idx) for pt_idx in shortest_path])
        
        # Calculate distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(shortest_path_xyz, axis=0)**2, axis=1))

        # Calculate cumulative distance
        cumulative_distance = np.insert(np.cumsum(distances), 0, 0)
        print("Spline length", cumulative_distance[-1])
        return cumulative_distance[-1]
    
    def generate_inital_placement(self, mesh, spline):
        """
        This function should take in a mesh and a spline and output an initial placement of center, 
        insertion and extraction points (equally spaced works)

        The challenge here will be to work out how to place the points on the surface of the skin

        returns a SuturePlacement3d object with spline and points
        """

        new_placement = SuturePlacement3d(spline, None, None, None, None)
        spline_length = self.calculate_spline_length(spline, mesh)
        num_sutures_initial = int(spline_length / (self.gamma)) #TODO: modify later 
        print("Num sutures initial", num_sutures_initial)

        points_t_initial = np.linspace(0, 1, int(num_sutures_initial))

        # the below is for generating deliberately bad warm starts, to see whether loss function is functioning properly

        """
        
        for i in range(1,len(points_t_initial) - 1):

            cand_rand = np.random.normal(0, 0.1)
            while points_t_initial[i] + cand_rand >= 1 and points_t_initial[i]  + cand_rand <= 0:
                cand_rand = np.random.normal(0, 0.05)
            points_t_initial[i] += cand_rand
            points_t_initial.sort()

        """

        _, normals, derivs = self.update_placement(new_placement, mesh, spline, points_t_initial)

        return new_placement, normals, derivs
        
    def update_placement(self, placement, mesh, spline, points_t):
        """
        This function should take in a mesh and a spline and output center, 
        insertion and extraction points (step up function)

        returns a SuturePlacement3d object with spline and points
        """
        num_points = len(points_t)

        placement.t = points_t

        spline_x, spline_y, spline_z = spline[0], spline[1], spline[2]
        derivative_x, derivative_y, derivative_z = spline_x.derivative(), spline_y.derivative(), spline_z.derivative()

        # get center points
        center_points = [[spline_x(t), spline_y(t), spline_z(t)] for t in points_t]
        # print("magnitude center points", [np.linalg.norm(center_points[i]) for i in range(num_points)])


        # get derivative points
        derivative_points = [[derivative_x(t), derivative_y(t), derivative_z(t)] for t in points_t]

        #get tangent plane normal vectors
        normal_vectors = [get_plane_estimation(mesh, center_points[i], self.num_points_for_plane) for i in range(num_points)]

        # project derivatives onto the tangent plane
        derivative_vectors = [project_vector_onto_plane(derivative_points[i], normal_vectors[i]) for i in range(num_points)]

        # normalize normal vectors and derivative vectors
        normal_vectors = [self.normalize_vector(normal_vectors[i]) for i in range(num_points)]
        derivative_vectors = [self.normalize_vector(derivative_vectors[i]) for i in range(num_points)]

        # Insertion points = cross product 
        insertion_points = [mesh.get_point_location(mesh.get_nearest_point(center_points[i] + self.suture_width * np.cross(normal_vectors[i], derivative_vectors[i]))[1]) for i in range(num_points)]

        # get the closest point on the mesh to that point
        insertion_points 
        # print("magnitude insertion points", [np.linalg.norm(insertion_points[i]) for i in range(num_points)])

        # Extraction points = - cross product
        extraction_points = [mesh.get_point_location(mesh.get_nearest_point(center_points[i] + self.suture_width * (-np.cross(normal_vectors[i], derivative_vectors[i])))[1]) for i in range(num_points)]

        # update suture placement 3d object
        placement.center_pts = center_points
        placement.insertion_pts = insertion_points
        placement.extraction_pts = extraction_points

        # suturePlacement3d = SuturePlacement3d(spline, center_points, insertion_points, extraction_points, points_t)
        # print("Center points", center_points)
        # print("Insertion points", insertion_points)
        # print("Extraction points", extraction_points)

        self.suture_placement = placement
        return placement, normal_vectors, derivative_vectors
    
    def normalize_vector(self, vector):
        """
        Normalize a vector to have a unit length.
        
        Parameters:
            vector (list or tuple): The vector to be normalized.
            
        Returns:
            list: The normalized vector.
        """
        magnitude = np.sqrt(sum(component**2 for component in vector))
        normalized_vector = [component / magnitude for component in vector]
        return normalized_vector
    
    def plot_mesh_path_and_spline(self, mesh, spline, suturePlacement3d, normal_vectors, derivative_vectors, spline_segments=100):
        num_pts = len(suturePlacement3d.insertion_pts)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title("Mesh and Spline")

        mesh_coords = mesh.vertex_coordinates
        ax.scatter3D(mesh_coords[::5, 0], mesh_coords[::5, 1], mesh_coords[::5, 2], color='red', alpha=0.01)


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

        ax.plot(spline_coords_x, spline_coords_y, spline_coords_z, color='green')
        ax.scatter([suturePlacement3d.insertion_pts[i][0] for i in range(num_pts)], [suturePlacement3d.insertion_pts[i][1] for i in range(num_pts)], [suturePlacement3d.insertion_pts[i][2] for i in range(num_pts)], c="black", s=5)
        ax.scatter([suturePlacement3d.center_pts[i][0] for i in range(num_pts)], [suturePlacement3d.center_pts[i][1] for i in range(num_pts)], [suturePlacement3d.center_pts[i][2] for i in range(num_pts)], c="blue", s=5)
        ax.scatter([suturePlacement3d.extraction_pts[i][0] for i in range(num_pts)], [suturePlacement3d.extraction_pts[i][1] for i in range(num_pts)], [suturePlacement3d.extraction_pts[i][2] for i in range(num_pts)], c="purple", s=5)
       
        # ax.quiver([normal_vectors[i][0] + suturePlacement3d.center_pts[i][0] for i in range(num_pts)], [normal_vectors[i][1] + suturePlacement3d.center_pts[i][1] for i in range(num_pts)], [normal_vectors[i][2] + suturePlacement3d.center_pts[i][2] for i in range(num_pts)], [normal_vectors[i][0] for i in range(num_pts)], [normal_vectors[i][1] for i in range(num_pts)], [normal_vectors[i][2] for i in range(num_pts)])
        # ax.quiver([derivative_vectors[i][0] + suturePlacement3d.center_pts[i][0] for i in range(num_pts)], [derivative_vectors[i][1] + suturePlacement3d.center_pts[i][1] for i in range(num_pts)], [derivative_vectors[i][2] + suturePlacement3d.center_pts[i][2] for i in range(num_pts)], [derivative_vectors[i][0] for i in range(num_pts)], [derivative_vectors[i][1] for i in range(num_pts)], [derivative_vectors[i][2] for i in range(num_pts)], color="red")
        
        plt.show()


    def plot_mesh_path_spline_and_forces(self, mesh, spline, suturePlacement3d, wound_pt, tot_insertion_force, tot_extraction_force, spline_segments=100):
        num_pts = len(suturePlacement3d.insertion_pts)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title("Mesh and Spline")

        mesh_coords = mesh.vertex_coordinates
        ax.scatter3D(mesh_coords[::5, 0], mesh_coords[::5, 1], mesh_coords[::5, 2], color='red', alpha=0.01)


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

        ax.plot(spline_coords_x, spline_coords_y, spline_coords_z, color='green')
        ax.scatter([suturePlacement3d.insertion_pts[i][0] for i in range(num_pts)], [suturePlacement3d.insertion_pts[i][1] for i in range(num_pts)], [suturePlacement3d.insertion_pts[i][2] for i in range(num_pts)], c="green", s=10)
        ax.scatter([suturePlacement3d.center_pts[i][0] for i in range(num_pts)], [suturePlacement3d.center_pts[i][1] for i in range(num_pts)], [suturePlacement3d.center_pts[i][2] for i in range(num_pts)], c="blue", s=5)
        ax.scatter([suturePlacement3d.extraction_pts[i][0] for i in range(num_pts)], [suturePlacement3d.extraction_pts[i][1] for i in range(num_pts)], [suturePlacement3d.extraction_pts[i][2] for i in range(num_pts)], c="purple", s=10)
       
        ax.scatter(wound_pt[0], wound_pt[1], wound_pt[2], c="orange", s=5)

        arrow_rescaling = 0.03

        tot_insertion_force_scaled = arrow_rescaling * tot_insertion_force
        tot_extraction_force_scaled = arrow_rescaling * tot_extraction_force

        ax.quiver(wound_pt[0], wound_pt[1], wound_pt[2], tot_insertion_force_scaled[0], tot_insertion_force_scaled[1], tot_insertion_force_scaled[2], color="green")
        ax.quiver(wound_pt[0], wound_pt[1], wound_pt[2], tot_extraction_force_scaled[0], tot_extraction_force_scaled[1], tot_extraction_force_scaled[2], color="purple")

        # ax.quiver([normal_vectors[i][0] + suturePlacement3d.center_pts[i][0] for i in range(num_pts)], [normal_vectors[i][1] + suturePlacement3d.center_pts[i][1] for i in range(num_pts)], [normal_vectors[i][2] + suturePlacement3d.center_pts[i][2] for i in range(num_pts)], [normal_vectors[i][0] for i in range(num_pts)], [normal_vectors[i][1] for i in range(num_pts)], [normal_vectors[i][2] for i in range(num_pts)])
        # ax.quiver([derivative_vectors[i][0] + suturePlacement3d.center_pts[i][0] for i in range(num_pts)], [derivative_vectors[i][1] + suturePlacement3d.center_pts[i][1] for i in range(num_pts)], [derivative_vectors[i][2] + suturePlacement3d.center_pts[i][2] for i in range(num_pts)], [derivative_vectors[i][0] for i in range(num_pts)], [derivative_vectors[i][1] for i in range(num_pts)], [derivative_vectors[i][2] for i in range(num_pts)], color="red")
        
        plt.show()



    def optimize(self, placement: SuturePlacement3d):

        wound_points = placement.t

        # set up all of the constraints

        def min_suture_dist(t):

            # get points
            insert_pts, center_pts, extract_pts = placement.insertion_pts, placement.center_pts, placement.extraction_pts
            insert_dists, center_dists, extract_dists = self.get_dists(insert_pts), self.get_dists(center_pts), self.get_dists(extract_pts)

            # get distances
            h = self.gamma * (1/5)
            return min([i - h for i in insert_dists] + [i - h for i in center_dists] + [i - h for i in extract_dists])
    
        def max_suture_dist(t): # max distance b/w 2 sutures
            insert_pts, center_pts, extract_pts = placement.insertion_pts, placement.center_pts, placement.extraction_pts
            insert_dists, center_dists, extract_dists = self.get_dists(insert_pts), self.get_dists(center_pts), self.get_dists(extract_pts)
            
            h = self.gamma * 4
            return max([h - i for i in insert_dists] + [h - i for i in center_dists] + [h - i for i in extract_dists])
        
        insert_pts, center_pts, extract_pts = placement.insertion_pts, placement.center_pts, placement.extraction_pts
        insert_dists, center_dists, extract_dists = self.get_dists(insert_pts), self.get_dists(center_pts), self.get_dists(extract_pts)

        # make min/max constraints for every suture gap
        def get_ith_max_constraint(dist_list, i):
            return (self.gamma * 4) - dist_list[i]
        
        def get_ith_min_constraint(dist_list, i):
            return dist_list[i] -  (self.gamma * 0.2)
        

        def is_ordered(t):
            for i in range(len(t) - 1):
                if t[i + 1] - t[i] <= 0:
                    return - 1
            return 1
        
        constraints = [{'type': 'eq', 'fun': lambda t: 1 - t[-1]}, 
                       {'type': 'eq', 'fun': lambda t: t[0]},
                       {'type': 'ineq', 'fun': lambda t: min_suture_dist(t)},
                       {'type': 'ineq', 'fun': lambda t: max_suture_dist(t)},
                       ]
        
        # for i in range(len(t) - 1):
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_max_constraint(insert_dists, i)})
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_max_constraint(center_dists, i)})
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_max_constraint(extract_dists, i)})
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_min_constraint(insert_dists, i)})
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_min_constraint(center_dists, i)})
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_min_constraint(extract_dists, i)})


        def loss(wound_points):
            """
            This function should calculate the loss of a particular placement. As before, the 
            loss is entirely dependent on how far along the curve we are. Let t range from 
            0 to 1, and indicates how far along the wound we are. placement.t is an array of 
            t values for each point. This is what we are optimizing over (we want to find the 
            best values of t).
            """

            self.update_placement(placement, self.mesh, placement.spline, wound_points)
            
            # recalculate all point locations
            
            # closure_loss, shear_loss = compute_closure_shear_loss()
            #print("Closure loss", closure_loss)
            #print("Shear loss", shear_loss)

            # print("Updated placement")
            var_loss = self.get_point_dist_var_loss(placement)
            ideal_loss = self.get_ideal_loss(placement)

            # curr_loss = shear_loss * self.c_shear + closure_loss * self.c_closure + var_loss * self.c_var + ideal_loss * self.c_ideal
            curr_loss = var_loss * self.c_var + ideal_loss * self.c_ideal
            print("current_loss:" + str(curr_loss))

            return curr_loss
        
        def jac(t):
            return optim.approx_fprime(t, loss)

        def get_force_angle(normal_plane, force_vector, path_vector):
            """
            This function should calculate the angle between the force vector and the path vector in the plane normal_plane
            IT IS IMPORTANT THAT THIS ANGLE HAVE THE RIGHT SIGN
            """
            
            force_vector_proj = project_vector_onto_plane(force_vector, normal_plane)
            path_vector_proj = project_vector_onto_plane(path_vector, normal_plane)

            force_vector_proj = force_vector_proj / np.linalg.norm(force_vector_proj)
            path_vector_proj = path_vector_proj / np.linalg.norm(path_vector_proj)
            normal_plane_norm = normal_plane / np.linalg.norm(normal_plane)

            angle_magnitude = np.arccos(np.dot(force_vector_proj, path_vector_proj))
            
            force_cross = np.cross(normal_plane_norm, force_vector_proj)
            angle_sign = np.sign(np.dot(force_cross, path_vector_proj))

            force_angle = angle_sign * angle_magnitude

            return force_angle

        def get_force_direction(normal_plane, force_angle, path_vector):
            '''
            normal_plane = tangent plane at the wound point (expressed via normal vector)
            force_angle = angle between the force_vector and the path_vector in the plane normal_plane
            path_vector = the path vector at the wound point
            '''

            path_vector_proj = project_vector_onto_plane(path_vector, normal_plane)
            path_vector_proj = path_vector_proj / np.linalg.norm(path_vector_proj)

            force_direction = np.cos(force_angle) * path_vector_proj + np.sin(force_angle) * np.cross(normal_plane, path_vector_proj)

            return force_direction


        def compute_felt_force(in_ex_pt, in_ex_force_vec, point, num_nearest = 20):
    
            # Original function had points_to_sample, ep as parameters

            # in_ex_pt is the insertion or extraction point
            # point is the point on the wound we are measuring the force at
            # in_ex_vec is the force being applied at the insertion or extraction point

            mesh = self.mesh

            ellipse_ecc = self.force_model['ellipse_ecc']
            force_decay = self.force_model['force_decay']
            verbose = self.force_model['verbose']

            in_ex_plane = get_plane_estimation(mesh, in_ex_pt, num_nearest)
            wound_plane = get_plane_estimation(mesh, point, num_nearest)

            if verbose > 10:
                print("in_ex plane: ", in_ex_plane)
                #print("wound plane: ", wound_plane)

            # Get the normal vectors from the coefficients (i.e. drop the constant term)
            in_ex_plane_normal = in_ex_plane
            wound_plane_normal = wound_plane

            in_ex_plane_normal = in_ex_plane_normal / np.linalg.norm(in_ex_plane_normal)
            wound_plane_normal = wound_plane_normal / np.linalg.norm(wound_plane_normal)

            if verbose > 10:
                print("in_ex plane normal: ", in_ex_plane_normal)
                print("norm of in_ex plane normal:", np.linalg.norm(in_ex_plane_normal))

            # Normalize the normal vectors of the plane
            # in_ex_plane_normal = in_ex_plane_normal / np.linalg.norm(in_ex_plane_normal)
            # wound_plane_normal = wound_plane_normal / np.linalg.norm(wound_plane_normal)

            # in_ex_vec_proj = project_vector_onto_plane(in_ex_force_vec, in_ex_plane_normal)
            # in_ex_vertex = in_ex_pt #TODO: GET THE VERTEX FROM THE MESH
            # wound_vertex = point #TODO: GET THE VERTEX FROM THE MESH
                
            shortest_path = mesh.get_a_star_path(in_ex_pt, point)
            shortest_path_xyz = np.array([mesh.get_point_location(pt_idx) for pt_idx in shortest_path])
            
            # Calculate distances between consecutive points
            distances = np.sqrt(np.sum(np.diff(shortest_path_xyz, axis=0)**2, axis=1))

            # Calculate cumulative distance
            cumulative_distance = np.insert(np.cumsum(distances), 0, 0)

            # Normalize t to range from 0 to 1
            t = cumulative_distance / cumulative_distance[-1]

            a_star_distance = cumulative_distance[-1]

            # FIT SPLINE TO SHORTEST PATH
            x = shortest_path_xyz[:, 0] # x-coordinates of the shortest path
            y = shortest_path_xyz[:, 1]
            z = shortest_path_xyz[:, 2]

            #print(t,x)
            #print(len(t), len(x))

            s_factor = len(x)/5.0 # A starting point for the smoothing factor; adjust based on noise level
            #s_factor = 0.1
            x_smooth = inter.UnivariateSpline(t, x, s=s_factor)
            y_smooth = inter.UnivariateSpline(t, y, s=s_factor)
            z_smooth = inter.UnivariateSpline(t, z, s=s_factor)
            # GET DERIVATIVES OF SPLINE AT in_ex_vertex AND wound_vertex
            x_smooth_deriv = x_smooth.derivative()
            y_smooth_deriv = y_smooth.derivative()
            z_smooth_deriv = z_smooth.derivative()
            dx_start = x_smooth_deriv(0)
            dy_start = y_smooth_deriv(0)
            dz_start = z_smooth_deriv(0)
            dx_end = x_smooth_deriv(1)
            dy_end = y_smooth_deriv(1)
            dz_end = z_smooth_deriv(1)

            if verbose > 10:
                print('x_smooth: ', x_smooth)

            # Calculate the length of the spline
            ## TEST : SUBSTITUTE A* LENGTH FOR SPLINE LENGTH
            
            #spline_length = a_star_distance

            spline_length = np.trapz(np.sqrt(x_smooth_deriv(t)**2 + y_smooth_deriv(t)**2 + z_smooth_deriv(t)**2), t)
            #if verbose > 0:
            #    print("spline length: ", spline_length)

            # put together the path vectors at the in_ex and wound points
            in_ex_path_vec = np.array([dx_start, dy_start, dz_start])
            in_ex_path_vec = in_ex_path_vec / np.linalg.norm(in_ex_path_vec)

            wound_path_vec = np.array([dx_end, dy_end, dz_end])    
            wound_path_vec = wound_path_vec / np.linalg.norm(wound_path_vec)

            if verbose > 10:
                print("in_ex path vec magnitude: ", np.linalg.norm(in_ex_path_vec))
                print("wound path vec magnitude: ", np.linalg.norm(wound_path_vec))

            in_ex_path_vec_proj = project_vector_onto_plane(in_ex_path_vec, in_ex_plane_normal)
            in_ex_path_vec_proj_normalized = in_ex_path_vec_proj / np.linalg.norm(in_ex_path_vec_proj)
            
            wound_path_vec_proj = project_vector_onto_plane(wound_path_vec, wound_plane_normal)
            wound_path_vec_proj_normalized = wound_path_vec_proj / np.linalg.norm(wound_path_vec_proj)

            if verbose > 10:
                print('sanity check: dot product of projected vector and normal vector: ', np.dot(in_ex_path_vec_proj, in_ex_plane_normal))
                print('sanity check: dot product of projected vector and normal vector: ', np.dot(wound_path_vec_proj, wound_plane_normal))
                print('sanity check: projected in_ex path vector difference ', np.linalg.norm(in_ex_path_vec - in_ex_path_vec_proj))
                print('sanity check: projected wound path vector difference ', np.linalg.norm(wound_path_vec - wound_path_vec_proj))
                print('sanity check: dot product of in_ex path vector and normal vector: ', np.dot(in_ex_path_vec, in_ex_plane_normal))
                print('sanity check: dot product of wound path vector and normal vector: ', np.dot(wound_path_vec, wound_plane_normal))

            normals = [in_ex_plane_normal, wound_plane_normal]

            in_ex_force_vec_proj = project_vector_onto_plane(in_ex_force_vec, in_ex_plane_normal)
            in_ex_force_vec_proj_normalized = in_ex_force_vec_proj / np.linalg.norm(in_ex_force_vec_proj)

            force_angle = get_force_angle(in_ex_plane_normal, in_ex_force_vec, in_ex_path_vec)

            ellipse_dist_factor = np.sqrt((np.sin(force_angle) * ellipse_ecc) ** 2 + (np.cos(force_angle)) ** 2)

            wound_force = self.force_model['imparted_force'] - force_decay * spline_length * ellipse_dist_factor

            if verbose > 10:
                print('in_ex force: ', self.force_model['imparted_force'])
                print('force decay: ', force_decay)
                print('force angle: ', force_angle)
                print('suture width: ', self.suture_width)
                print('spline length: ', spline_length)
                print('euclidean distance: ', np.linalg.norm(in_ex_pt - point))
                print('distance on mesh path: ', a_star_distance)
                print('ellipse eccentricity: ', ellipse_ecc)
                print('elliptical distance compensation factor: ', ellipse_dist_factor)
                print('wound force: ', wound_force)

            wound_force = max(0, wound_force)

            # wound_direction is the direction of the wound on the plane wound_plane from wound_path_vec_proj_normalized by angle force_angle
            #wound_cross_prod = np.cross(wound_plane_normal, wound_path_vec_proj_normalized)
            #wound_direction = np.cos(force_angle) * wound_path_vec_proj_normalized - np.sin(force_angle) * wound_cross_prod

            wound_direction = get_force_direction(wound_plane_normal, force_angle, wound_path_vec_proj_normalized)

            wound_force_vec = wound_force * wound_direction

            #if verbose > 1:
                # TODO: Fix plotting
                #spline = [x_smooth, y_smooth, z_smooth]
                #plot_mesh_path_and_spline(mesh, shortest_path, spline, normals, in_ex_force_vec, wound_force_vec)

            if self.force_model['verbose'] > 0:
                print('wound force vec: ', wound_force_vec)

            return wound_force_vec
        
        def compute_total_force(in_ex_pts, in_ex_force_vecs, point):

            #mesh = self.mesh
            #compute_felt_force(in_ex_pt, point, in_ex_force_vec, num_nearest = 20)


            total_force = np.array([0.0, 0.0, 0.0])
            for i in range(len(in_ex_pts)):
                in_ex_pt = in_ex_pts[i]
                in_ex_force_vec = in_ex_force_vecs[i]
                if np.linalg.norm(in_ex_pt - point) <= 1/self.force_model['force_decay']:   #TODO: double check the distance
                    felt_force = compute_felt_force(in_ex_pt, in_ex_force_vec, point)
                    if self.force_model['verbose'] > 10:
                        print('felt force: ', felt_force)
                    total_force = felt_force + total_force

            if self.force_model['verbose'] > 10:
                print('total force: ', total_force)

            return total_force
        
        def compute_closure_shear_force(point, wound_derivative):

            # point is a new name for wound_pt

            mesh = self.mesh

            insertion_pts = np.array(placement.insertion_pts)
            extraction_pts = np.array(placement.extraction_pts)

            # forces are exerted by the suture from extraction to insertion points (and vice versa)
            force_vecs = extraction_pts - insertion_pts
            for i in range(len(force_vecs)):
                force_vecs[i] = force_vecs[i] / np.linalg.norm(force_vecs[i])

            dist_to_nearest_insertion_point = np.min(np.linalg.norm(insertion_pts - point, axis=1))
            dist_to_nearest_extraction_point = np.min(np.linalg.norm(extraction_pts - point, axis=1))

            

            mesh_dist_to_nearest_insertion_point = 1e8
            mesh_dist_to_nearest_extraction_point = 1e8
            for i in range(len(insertion_pts)):
                path_to_insertion_i = mesh.get_a_star_path(insertion_pts[i], point)
                path_to_extraction_i = mesh.get_a_star_path(extraction_pts[i], point)

                shortest_path_xyz_to_insertion_i = np.array([mesh.get_point_location(pt_idx) for pt_idx in path_to_insertion_i])
                shortest_path_xyz_to_extraction_i = np.array([mesh.get_point_location(pt_idx) for pt_idx in path_to_extraction_i])
    
                # Calculate distances between consecutive points
                distances_to_insertion_i = np.sqrt(np.sum(np.diff(shortest_path_xyz_to_insertion_i, axis=0)**2, axis=1))
                distances_to_extraction_i = np.sqrt(np.sum(np.diff(shortest_path_xyz_to_extraction_i, axis=0)**2, axis=1))

                # Calculate cumulative distance
                cumulative_distance_to_insertion_i = np.sum(distances_to_insertion_i)
                cumulative_distance_to_extraction_i = np.sum(distances_to_extraction_i)

                mesh_dist_to_nearest_insertion_point = min(mesh_dist_to_nearest_insertion_point, cumulative_distance_to_insertion_i)
                mesh_dist_to_nearest_extraction_point = min(mesh_dist_to_nearest_extraction_point, cumulative_distance_to_extraction_i)

            if self.force_model['verbose'] > 0:
                print("dist to nearest insertion point: ", dist_to_nearest_insertion_point)
                print("a star dist to nearest insertion point: ", mesh_dist_to_nearest_insertion_point)
                print("dist to nearest extraction point: ", dist_to_nearest_extraction_point)
                print("a star dist to nearest extraction point: ", mesh_dist_to_nearest_extraction_point)

            wound_plane = get_plane_estimation(mesh, point)
            wound_plane = wound_plane / np.linalg.norm(wound_plane)
            wound_derivative_proj = project_vector_onto_plane(wound_derivative, wound_plane)
            wound_derivative_proj = wound_derivative_proj / np.linalg.norm(wound_derivative_proj)

            wound_line_normal = np.cross(wound_derivative_proj, wound_plane)
            if self.force_model['verbose'] > 10:
                print("sanity check: magnitude of wound line normal (should be equal to 1): ", np.linalg.norm(wound_line_normal))
            wound_line_normal = wound_line_normal / np.linalg.norm(wound_line_normal)

            total_insertion_force = compute_total_force(insertion_pts, force_vecs, point)
            total_extraction_force = compute_total_force(extraction_pts, -force_vecs, point)

            insertion_closure_force = np.dot(total_insertion_force, wound_line_normal)
            extraction_closure_force = np.dot(total_extraction_force, wound_line_normal)

            closure_force = np.abs(insertion_closure_force - extraction_closure_force)

            insertion_shear_force = np.dot(total_insertion_force, wound_derivative_proj)
            extraction_shear_force = np.dot(total_extraction_force, wound_derivative_proj)

            shear_force = np.abs(insertion_shear_force - extraction_shear_force)

            if self.force_model['verbose'] > 0:
                print("Total insertion force: ", total_insertion_force)
                print("Total extraction force: ", total_extraction_force)
                print("Insertion closure force: ", insertion_closure_force)
                print("Extraction closure force: ", extraction_closure_force)
                print("Insertion shear force: ", insertion_shear_force)
                print("Extraction shear force: ", extraction_shear_force)
                print("Closure force: ", closure_force)
                print("Shear force: ", shear_force)

            return closure_force, shear_force

        def plot_closure_shear_force(t, wound_derivative):

            # t is location of the point on the wound (in spline parameterization)

            mesh = self.mesh
            point = np.array([self.spline[0](t), self.spline[1](t), self.spline[2](t)])

            insertion_pts = np.array(placement.insertion_pts)
            extraction_pts = np.array(placement.extraction_pts)

            # forces are exerted by the suture from extraction to insertion points (and vice versa)
            force_vecs = extraction_pts - insertion_pts
            for i in range(len(force_vecs)):
                force_vecs[i] = force_vecs[i] / np.linalg.norm(force_vecs[i])

            #dist_to_nearest_insertion_point = np.min(np.linalg.norm(insertion_pts - point, axis=1))
            #dist_to_nearest_extraction_point = np.min(np.linalg.norm(extraction_pts - point, axis=1))

            tot_insertion_force = compute_total_force(insertion_pts, force_vecs, point)
            tot_extraction_force = compute_total_force(extraction_pts, -force_vecs, point)

            self.plot_mesh_path_spline_and_forces(mesh, spline, suturePlacement3d, point, tot_insertion_force, tot_extraction_force, spline_segments=100)

        def compute_closure_shear_loss(granularity=5):
            # granularity is the number of points to sample on the wound spline

            wound_spline = self.spline

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

                wound_derivative = wound_derivative / np.linalg.norm(wound_derivative)

                closure_force, shear_force = compute_closure_shear_force(wound_pt, wound_derivative)

                if self.force_model['verbose'] > 0:
                    plot_closure_shear_force(t[i], wound_derivative)

                closure_loss += ((closure_force - self.force_model['ideal_closure_force'])**2)/granularity
                shear_loss += (shear_force**2)/granularity

            if self.force_model['verbose'] > 0:
                print("Closure loss", closure_loss)
                print("Shear loss", shear_loss)

            return closure_loss, shear_loss
        
        # run optimizer
        result = optim.minimize(loss, wound_points, constraints = constraints, options={"maxiter":10, "disp":True}, method = 'SLSQP', tol=1e-3, jac = jac)
        
        return result 

    
    '''
    def calculate_shear_force(point, placement):
        """
        Calculate forces acting along the wound at a point, due to the placement as a whole
        """
        return 0

    def calculate_closure_force(point, placement):
        """
        Calculate forces acting to close the wound at a point, due to the placement as a whole
        """
        return 0
    '''
    
    def get_point_dist_var_loss(self, placement):

        insertion_var = self.get_diff_var(placement.insertion_pts)
        center_var = self.get_diff_var(placement.center_pts)
        extraction_var = self.get_diff_var(placement.extraction_pts)

        return insertion_var + center_var + extraction_var
    
    def get_ideal_loss(self, placement):

        insertion_loss = self.get_me_opt(placement.insertion_pts, self.gamma)
        center_loss = self.get_me_opt(placement.center_pts, self.gamma)
        extraction_loss = self.get_me_opt(placement.extraction_pts, self.gamma)

        return insertion_loss + center_loss + extraction_loss

    def get_diff_var(self, points): 
        dists = np.zeros(len(points) - 1)

        for i in range(len(points) - 1):
            dists[i] = euclidean_dist(points[i], points[i+1])

        return np.var(dists)
    
    def get_me_opt(self, points, gamma):
        error = 0

        for i in range(len(points) - 1):
            dist = euclidean_dist(points[i], points[i+1])
            error += (dist - gamma)**2

        return error

    def get_dists(self, points):
        dists = [0 for i in range(len(points) -1)]
        for i in range(len(points) - 1):
            dists[i] = euclidean_dist(points[i], points[i+1])

        return dists
    
    def loss_placement(self, placement):
        """
        This function should calculate the loss of a particular placement. Used for synthetic splines
        """
        var_loss = self.get_point_dist_var_loss(placement)
        ideal_loss = self.get_ideal_loss(placement)

        # curr_loss = shear_loss * self.c_shear + closure_loss * self.c_closure + var_loss * self.c_var + ideal_loss * self.c_ideal
        curr_loss = var_loss * self.c_var + ideal_loss * self.c_ideal
        # print("current_loss:" + str(curr_loss))

        return curr_loss

if __name__ == '__main__':

    # Specify the path to your text file
    adj_path = 'adjacency_matrix.txt'
    loc_path = 'vertex_lookup.txt'


    mesh = MeshIngestor(adj_path, loc_path)

    # Create the graph
    mesh.generate_mesh()

    # pick two random points for testing purposes
    num1, num2 = random.randrange(0, len(mesh.vertex_coordinates)), random.randrange(0, len(mesh.vertex_coordinates))
    #num1, num2 = 21695, 8695
    #num1, num2 = 20000, 18000
    rand_start_pt = mesh.get_point_location(num1)
    rand_wound_pt = mesh.get_point_location(num2)
    shortest_path = mesh.get_a_star_path(rand_start_pt, rand_wound_pt)
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

    s_factor = len(x)/5.0 # A starting point for the smoothing factor; adjust based on noise level
    #s_factor = 0.1
    x_smooth = inter.UnivariateSpline(t, x, s=s_factor)
    y_smooth = inter.UnivariateSpline(t, y, s=s_factor)
    z_smooth = inter.UnivariateSpline(t, z, s=s_factor)
    spline = [x_smooth, y_smooth, z_smooth]
    suture_width = 0.002 # 0.002 TODO: Change once scaling is sorted

    c_ideal = 1
    gamma = suture_width # TODO: Change once scaling is sorted
    c_var = 1
    c_shear = 1
    c_closure = 1

    hyperparams = [c_ideal, gamma, c_var, c_shear, c_closure]

    force_model_parameters = {'ellipse_ecc': 1.0, 'force_decay': 0.5/suture_width, 'verbose': 0, 'ideal_closure_force': None, 'imparted_force': None}

    optim3d = Optimizer3d(mesh, spline, suture_width, hyperparams, force_model_parameters)
    suturePlacement3d, normal_vectors, derivative_vectors = optim3d.generate_inital_placement(mesh, spline)
    #print("Normal vector", normal_vectors)
    optim3d.plot_mesh_path_and_spline(mesh, spline, suturePlacement3d, normal_vectors, derivative_vectors)

    optim3d.optimize(suturePlacement3d)

    optim3d.plot_mesh_path_and_spline(mesh, spline, suturePlacement3d, normal_vectors, derivative_vectors)

