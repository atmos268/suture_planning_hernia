import numpy as np
from SuturePlacement3d import SuturePlacement3d
from MeshIngestor import MeshIngestor
from test_force_model import get_plane_estimation, project_vector_onto_plane
import scipy.interpolate as inter
import random
import copy

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
    def __init__(self, mesh: MeshIngestor, spline, suture_width, hyperparameters, force_model_parameters, smoothed_spline, spacing):
        self.mesh = mesh
        self.spline = spline
        self.smoothed_spline = smoothed_spline
        self.suture_width = suture_width 
        self.points_t = None
        self.center_pts = None
        self.insertion_pts = None
        self.extraction_pts = None
        self.length = None
        self.c_ideal = hyperparameters[0]
        self.gamma = hyperparameters[1]
        self.c_var = hyperparameters[2]
        self.c_shear = hyperparameters[3]
        self.c_closure = hyperparameters[4]
        self.force_model = force_model_parameters #Force model parameters are a dictionary 
        self.num_points_for_plane = 1000
        self.spacing = spacing

        # Ideal closure force calculated according to properties of the original diamond force model
        # If you want, can specify ideal_closure_force yourself, otherwise set to None and this calculation will be done
        if force_model_parameters['ideal_closure_force'] is None:
            self.force_model['ideal_closure_force'] = np.max(1, 0)
        
        if self.force_model['verbose'] > 0:
            print("Ideal closure force: ", self.force_model['ideal_closure_force'])

        if self.force_model['imparted_force'] is None:
            self.force_model['imparted_force'] = 1.0
    
    def compute_closure_shear_loss(self, granularity=5):
            # granularity is the number of points to sample on the wound spline

            x_spline = self.spline[0]
            y_spline = self.spline[1]
            z_spline = self.spline[2]

            xs_spline = self.smoothed_spline[0]
            ys_spline = self.smoothed_spline[1]
            zs_spline = self.smoothed_spline[2]

            t = np.linspace(0, 1, granularity)

            dx_spline = xs_spline.derivative()
            dy_spline = ys_spline.derivative()
            dz_spline = zs_spline.derivative()

            closure_loss = 0
            shear_loss = 0

            all_closure = []

            per_insertion = [[] for _ in range(len(self.insertion_pts))]
            per_extraction = [[] for _ in range(len(self.extraction_pts))]

            per_insertion_force = [[] for _ in range(len(self.insertion_pts))]
            per_extraction_force = [[] for _ in range(len(self.extraction_pts))]

            for i in range(granularity):
                x = x_spline(t[i])
                y = y_spline(t[i])
                z = z_spline(t[i])

                dx = dx_spline(t[i])
                dy = dy_spline(t[i])
                dz = dz_spline(t[i])

                sampled_pt = np.array([x, y, z])
                sampled_derivative = np.array([dx, dy, dz])

                # print(sampled_derivative)

                if np.linalg.norm(sampled_derivative) == 0:
                    sampled_derivative = np.array([0, 0, 0])
                else:
                    sampled_derivative = sampled_derivative / np.linalg.norm(sampled_derivative)

                closure_force, shear_force, peri, pere, insertion, extraction = self.compute_closure_shear_force(sampled_pt, sampled_derivative)
                # [1, 2, 3, ... 6]

                for i in range(len(self.insertion_pts)):
                    per_insertion[i].append(peri[i])
                    per_extraction[i].append(pere[i])

                for i in range(len(self.insertion_pts)):
                    per_insertion_force[i].append(insertion[i])
                    per_extraction_force[i].append(extraction[i])

                # if self.force_model['verbose'] > 0:
                # self.plot_insertion_extraction_force(t[i], sampled_derivative)
                
                all_closure.append(closure_force)

                closure_loss += ((closure_force - self.force_model['ideal_closure_force'])**2)/granularity
                shear_loss += (shear_force**2)/granularity
                # print("Closure loss", closure_loss)
                # print("Shear loss", shear_loss)
            

            # if self.force_model['verbose'] > 0:
            # print("Closure loss", closure_loss)
            # print("Shear loss", shear_loss)

            return closure_loss, shear_loss, all_closure, per_insertion, per_extraction, per_insertion_force, per_extraction_force

            # return closure_loss, shear_loss
    
    def compute_closure_shear_force(self, point, wound_derivative):

            # point is a new name for wound_pt

            mesh = self.mesh

            insertion_pts = np.array(self.insertion_pts)
            extraction_pts = np.array(self.extraction_pts)
            # print("INSERTION PTS", insertion_pts)
            # print("EXTRACTION PTS", extraction_pts)

            # forces are exerted by the suture from extraction to insertion points (and vice versa)
            force_vecs = extraction_pts - insertion_pts
            # print("force vecs", force_vecs)
            for i in range(len(force_vecs)):
                force_vecs[i] = force_vecs[i] / np.linalg.norm(force_vecs[i])

            wound_plane = get_plane_estimation(self.mesh, point)
            wound_plane = wound_plane / np.linalg.norm(wound_plane)
            wound_derivative_proj = project_vector_onto_plane(wound_derivative, wound_plane)
            wound_derivative_proj = wound_derivative_proj / np.linalg.norm(wound_derivative_proj)

            wound_line_normal = np.cross(wound_derivative_proj, wound_plane)
            if self.force_model['verbose'] > 10:
                print("sanity check: magnitude of wound line normal (should be equal to 1): ", np.linalg.norm(wound_line_normal))
            wound_line_normal = wound_line_normal / np.linalg.norm(wound_line_normal)

            # per insertion is for that point, index i is the force at insertion point i
            total_insertion_force, per_insertion, force_per_insertion = self.compute_total_force(insertion_pts, force_vecs, point)
            total_extraction_force, per_extraction, force_per_extraction = self.compute_total_force(extraction_pts, -force_vecs, point)

            insertion_closure_force = np.dot(total_insertion_force, wound_line_normal)
            extraction_closure_force = np.dot(total_extraction_force, wound_line_normal)

            insertion_force = []
            for force in force_per_insertion:
                insertion_force.append(np.dot(force, wound_line_normal))
            
            extraction_force = []
            for force in force_per_extraction:
                extraction_force.append(np.dot(force, wound_line_normal))

            insertion_closure_force = np.dot(total_insertion_force, wound_line_normal)
            extraction_closure_force = np.dot(total_extraction_force, wound_line_normal)

            # print('POINT', point)
            # print('insertion', insertion_closure_force)
            # print('extraction', extraction_closure_force)

            closure_force = np.abs(insertion_closure_force - extraction_closure_force)

            # print('closure', closure_force)

            insertion_shear_force = np.dot(total_insertion_force, wound_derivative_proj)
            extraction_shear_force = np.dot(total_extraction_force, wound_derivative_proj)

            shear_force = np.abs(insertion_shear_force - extraction_shear_force)

            # if self.force_model['verbose'] > 0:
            # print("Total insertion force: ", total_insertion_force)
            # print("Total extraction force: ", total_extraction_force)
            # print("Insertion closure force: ", insertion_closure_force)
            # print("Extraction closure force: ", extraction_closure_force)
            # print("Insertion shear force: ", insertion_shear_force)
            # print("Extraction shear force: ", extraction_shear_force)
            # print("Closure force: ", closure_force)
            # print("Shear force: ", shear_force)

            return closure_force, shear_force, per_insertion, per_extraction, insertion_force, extraction_force

    def compute_felt_force(self, in_ex_pt, in_ex_force_vec, point, i, num_nearest = 20):
    
            # Original function had points_to_sample, ep as parameters

            # in_ex_pt is the insertion or extraction point
            # point is the point on the wound we are measuring the force at
            # in_ex_vec is the force being applied at the insertion or extraction point

            mesh = self.mesh

            # ellipse_ecc = self.force_model['ellipse_ecc']
            ellipse_ecc =self.spacing[int(self.points_t[i] * 99)]
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
            
            # measure angle between (vec from in_ex to wound pt) and (in_ex_force_vec)
            force_angle = get_force_angle(in_ex_plane_normal, in_ex_force_vec, point - in_ex_pt)

            ellipse_dist_factor = np.sqrt((np.sin(force_angle) * ellipse_ecc) ** 2 + (np.cos(force_angle)) ** 2)
            # 1/2 / width
            wound_force = self.force_model['imparted_force'] - force_decay * (np.linalg.norm(point - in_ex_pt) - 0.002) * ellipse_dist_factor

            # if verbose > 10:
            #     print('in_ex force: ', self.force_model['imparted_force'])
            #     print('force decay: ', force_decay)
            #     print('force angle: ', force_angle)
            #     print('suture width: ', self.suture_width)
            #     print('spline length: ', spline_length)
            #     print('euclidean distance: ', np.linalg.norm(in_ex_pt - point))
            #     print('distance on mesh path: ', a_star_distance)
            #     print('ellipse eccentricity: ', ellipse_ecc)
            #     print('elliptical distance compensation factor: ', ellipse_dist_factor)
            #     print('wound force: ', wound_force)

            wound_force = max(0, wound_force)

            # wound_direction is the direction of the wound on the plane wound_plane from wound_path_vec_proj_normalized by angle force_angle
            #wound_cross_prod = np.cross(wound_plane_normal, wound_path_vec_proj_normalized)
            #wound_direction = np.cos(force_angle) * wound_path_vec_proj_normalized - np.sin(force_angle) * wound_cross_prod

            # wound_direction = get_force_direction(wound_plane_normal, force_angle, wound_path_vec_proj_normalized)

            # change the in_ex vector to be the same as the arg
            # project
            # return

            in_ex_force_vec = project_vector_onto_plane(in_ex_force_vec, wound_plane_normal)
            in_ex_force_vec = in_ex_force_vec / np.linalg.norm(in_ex_force_vec)

            wound_force_vec = wound_force * in_ex_force_vec

            #if verbose > 1:
                # TODO: Fix plotting
                #spline = [x_smooth, y_smooth, z_smooth]
                #plot_mesh_path_and_spline(mesh, shortest_path, spline, normals, in_ex_force_vec, wound_force_vec)

            if self.force_model['verbose'] > 0:
                print('wound force vec: ', wound_force_vec)

            
            # plot the wound force from each suture to each point
            # mesh = self.mesh

            # forces are exerted by the suture from extraction to insertion points (and vice versa)


            #dist_to_nearest_insertion_point = np.min(np.linalg.norm(insertion_pts - point, axis=1))
            #dist_to_nearest_extraction_point = np.min(np.linalg.norm(extraction_pts - point, axis=1))

            # self.plot_mesh_path_spline_and_forces(mesh, spline, self, point, wound_force_vec, wound_force_vec, spline_segments=100, in_ex_pt=in_ex_pt, grad_start=grad_start, grad_end=grad_end, shortest_path=[x_smooth, y_smooth, z_smooth])

            return wound_force_vec

    def compute_total_force(self, in_ex_pts, in_ex_force_vecs, point):

            #mesh = self.mesh
            #compute_felt_force(in_ex_pt, point, in_ex_force_vec, num_nearest = 20)

            total_force = np.array([0.0, 0.0, 0.0])
            per_insertion_pt = []
            force_per_insertion_pt = []
            for i in range(len(in_ex_pts)):
                in_ex_pt = in_ex_pts[i]
                in_ex_force_vec = in_ex_force_vecs[i]
                mag = 0
                force = np.array([0, 0, 0])
                if np.linalg.norm(in_ex_pt - point) <= 1/self.force_model['force_decay']:   #TODO: double check the distance
                    felt_force = self.compute_felt_force(in_ex_pt, in_ex_force_vec, point, i)
                    # self.plot_mesh_path_spline_and_forces(point, in_ex_pt, in_ex_force_vec)
                    if self.force_model['verbose'] > 10:
                        print('felt force: ', felt_force)
                    force = felt_force
                    total_force = felt_force + total_force
                    mag = np.linalg.norm(felt_force)
                force_per_insertion_pt.append(force)
                per_insertion_pt.append(mag)

            if self.force_model['verbose'] > 10:
                print('total force: ', total_force)

            return total_force, per_insertion_pt, force_per_insertion_pt
        

    def plot_insertion_extraction_force(self, t, wound_derivative):

            # t is location of the point on the wound (in spline parameterization)

            mesh = self.mesh
            point = np.array([self.spline[0](t), self.spline[1](t), self.spline[2](t)])

            insertion_pts = np.array(self.insertion_pts)
            extraction_pts = np.array(self.extraction_pts)

            # forces are exerted by the suture from extraction to insertion points (and vice versa)
            force_vecs = extraction_pts - insertion_pts
            for i in range(len(force_vecs)):
                force_vecs[i] = force_vecs[i] / np.linalg.norm(force_vecs[i])

            #dist_to_nearest_insertion_point = np.min(np.linalg.norm(insertion_pts - point, axis=1))
            #dist_to_nearest_extraction_point = np.min(np.linalg.norm(extraction_pts - point, axis=1))

            tot_insertion_force = self.compute_total_force(insertion_pts, force_vecs, point)
            tot_extraction_force = self.compute_total_force(extraction_pts, -force_vecs, point)
            # self.plot_mesh_path_spline_and_forces(point, tot_insertion_force, tot_extraction_force, spline_segments=100)



    def calculate_spline_length(self, spline, mesh):
        if self.length:
            return self.length

        spline_x, spline_y, spline_z = spline[0], spline[1], spline[2]

        # get an interpolated version of path

        granularity = 100
        x_pts = [spline_x(i / granularity) for i in range(100)]
        y_pts = [spline_y(i / granularity) for i in range(100)]
        z_pts = [spline_z(i / granularity) for i in range(100)]
        start = [spline_x(0), spline_y(0), spline_z(0)]
        end = [spline_x(1), spline_y(1), spline_z(1)]
        path = np.array([x_pts, y_pts, z_pts])
        # print(path)
        # Calculate distances between consecutive points
        distances = np.sqrt(np.sum(np.diff(path, axis=1)**2, axis=0))
        # Calculate cumulative distance
        
        length = np.sum(distances)
        self.length = length
        # print("Spline length", np.sum(distances))
        return length
    
    def generate_inital_placement(self, mesh, spline, num_sutures):
        """
        This function should take in a mesh and a spline and output an initial placement of center, 
        insertion and extraction points (equally spaced works)

        The challenge here will be to work out how to place the points on the surface of the skin

        returns a self object with spline and points
        """
        points_t_initial = np.linspace(0, 1, int(num_sutures))

        # the below is for generating deliberately bad warm starts, to see whether loss function is functioning properly
        
        # for i in range(1,len(points_t_initial) - 1):

        #     cand_rand = np.random.normal(0, 0.1)
        #     while points_t_initial[i] + cand_rand >= 1 and points_t_initial[i]  + cand_rand <= 0:
        #         cand_rand = np.random.normal(0, 0.05)
        #     points_t_initial[i] += cand_rand
        #     points_t_initial.sort()

        normals, derivs = self.update_placement(mesh, points_t_initial)

        return self.center_pts, self.insertion_pts, self.extraction_pts
        
    def update_placement(self, mesh, points_t):
        """
        This function should take in a mesh and a spline and output center, 
        insertion and extraction points (step up function)

        returns a self object with spline and points
        """
        num_points = len(points_t)

        spline_x, spline_y, spline_z = self.spline[0], self.spline[1], self.spline[2]
        spline_xs, spline_ys, spline_zs = self.smoothed_spline[0], self.smoothed_spline[1], self.smoothed_spline[2]
        derivative_x, derivative_y, derivative_z = spline_xs.derivative(), spline_ys.derivative(), spline_zs.derivative()

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
        # print("magnitude insertion points", [np.linalg.norm(insertion_points[i]) for i in range(num_points)])

        # Extraction points = - cross product
        extraction_points = [mesh.get_point_location(mesh.get_nearest_point(center_points[i] + self.suture_width * (-np.cross(normal_vectors[i], derivative_vectors[i])))[1]) for i in range(num_points)]

        # update suture placement 3d object
        self.center_pts = center_points
        self.insertion_pts = insertion_points
        self.extraction_pts = extraction_points
        self.points_t = points_t

        # print("Center points", center_points)
        # print("Insertion points", insertion_points)
        # print("Extraction points", extraction_points)

        return normal_vectors, derivative_vectors
            
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
    
    def plot_mesh_path_and_spline(self, spline_segments=100, viz=True, results_pth=None):
        
        # visaulize by default, save otherwise)
        
        num_pts = len(self.insertion_pts)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.view_init(elev=90, azim=0)
        plt.title(f"Surface and Spline for {num_pts} points")

        mesh_coords = self.mesh.vertex_coordinates
        ax.scatter3D(mesh_coords[::5, 0], mesh_coords[::5, 1], mesh_coords[::5, 2], color='red', alpha=0.01)


        spline_x = self.spline[0]
        spline_y = self.spline[1]
        spline_z = self.spline[2]

        spline_coords_x = []
        spline_coords_y = []
        spline_coords_z = []

        for t in np.linspace(0, 1, spline_segments):
            spline_coords_x.append(spline_x(t))
            spline_coords_y.append(spline_y(t))
            spline_coords_z.append(spline_z(t))

        ax.plot(spline_coords_x, spline_coords_y, spline_coords_z, color='green')
        ax.scatter([self.insertion_pts[i][0] for i in range(num_pts)], [self.insertion_pts[i][1] for i in range(num_pts)], [self.insertion_pts[i][2] for i in range(num_pts)], c="black", s=5)
        ax.scatter([self.center_pts[i][0] for i in range(num_pts)], [self.center_pts[i][1] for i in range(num_pts)], [self.center_pts[i][2] for i in range(num_pts)], c="blue", s=5)
        ax.scatter([self.extraction_pts[i][0] for i in range(num_pts)], [self.extraction_pts[i][1] for i in range(num_pts)], [self.extraction_pts[i][2] for i in range(num_pts)], c="purple", s=5)
       
        # ax.quiver([normal_vectors[i][0] + self.center_pts[i][0] for i in range(num_pts)], [normal_vectors[i][1] + self.center_pts[i][1] for i in range(num_pts)], [normal_vectors[i][2] + self.center_pts[i][2] for i in range(num_pts)], [normal_vectors[i][0] for i in range(num_pts)], [normal_vectors[i][1] for i in range(num_pts)], [normal_vectors[i][2] for i in range(num_pts)])
        # ax.quiver([derivative_vectors[i][0] + self.center_pts[i][0] for i in range(num_pts)], [derivative_vectors[i][1] + self.center_pts[i][1] for i in range(num_pts)], [derivative_vectors[i][2] + self.center_pts[i][2] for i in range(num_pts)], [derivative_vectors[i][0] for i in range(num_pts)], [derivative_vectors[i][1] for i in range(num_pts)], [derivative_vectors[i][2] for i in range(num_pts)], color="red")
        if viz:
            plt.show()
        else:
            # save images and losses
            save_str = results_pth + str(num_pts) + ".png"
            plt.savefig(save_str)
            plt.close()

    def plot_mesh_path_spline_and_forces(self, wound_pt, tot_insertion_force, tot_extraction_force, spline_segments=100):
        num_pts = len(self.insertion_pts)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.title("Mesh and Spline")

        mesh_coords = self.mesh.vertex_coordinates
        ax.scatter3D(mesh_coords[::5, 0], mesh_coords[::5, 1], mesh_coords[::5, 2], color='red', alpha=0.01)


        spline_x = self.spline[0]
        spline_y = self.spline[1]
        spline_z = self.spline[2]

        spline_coords_x = []
        spline_coords_y = []
        spline_coords_z = []

        for t in np.linspace(0, 1, spline_segments):
            spline_coords_x.append(spline_x(t))
            spline_coords_y.append(spline_y(t))
            spline_coords_z.append(spline_z(t))

        ax.plot(spline_coords_x, spline_coords_y, spline_coords_z, color='green')
        ax.scatter([self.insertion_pts[i][0] for i in range(num_pts)], [self.insertion_pts[i][1] for i in range(num_pts)], [self.insertion_pts[i][2] for i in range(num_pts)], c="black", s=5)
        ax.scatter([self.center_pts[i][0] for i in range(num_pts)], [self.center_pts[i][1] for i in range(num_pts)], [self.center_pts[i][2] for i in range(num_pts)], c="blue", s=5)
        ax.scatter([self.extraction_pts[i][0] for i in range(num_pts)], [self.extraction_pts[i][1] for i in range(num_pts)], [self.extraction_pts[i][2] for i in range(num_pts)], c="purple", s=5)
    
        ax.scatter(wound_pt[0], wound_pt[1], wound_pt[2], c="orange", s=5)
        # ax.scatter(in_ex_pt[0], in_ex_pt[1], in_ex_pt[2], c="orange", s=5)

        arrow_rescaling = 0.03

        tot_insertion_force_scaled = arrow_rescaling * tot_insertion_force
        tot_extraction_force_scaled = arrow_rescaling * tot_extraction_force

        # ax.quiver(in_ex_pt[0], in_ex_pt[1], in_ex_pt[2], grad_start[0], grad_start[1], grad_start[2], color="red")
        # ax.quiver(wound_pt[0], wound_pt[1], wound_pt[2], grad_end[0], grad_end[1], grad_end[2], color="red")

        ax.quiver(wound_pt[0], wound_pt[1], wound_pt[2], tot_insertion_force_scaled[0], tot_insertion_force_scaled[1], tot_insertion_force_scaled[2], color="green")
        ax.quiver(wound_pt[0], wound_pt[1], wound_pt[2], tot_extraction_force_scaled[0], tot_extraction_force_scaled[1], tot_extraction_force_scaled[2], color="purple")

        # ax.quiver([normal_vectors[i][0] + self.center_pts[i][0] for i in range(num_pts)], [normal_vectors[i][1] + self.center_pts[i][1] for i in range(num_pts)], [normal_vectors[i][2] + self.center_pts[i][2] for i in range(num_pts)], [normal_vectors[i][0] for i in range(num_pts)], [normal_vectors[i][1] for i in range(num_pts)], [normal_vectors[i][2] for i in range(num_pts)])
        # ax.quiver([derivative_vectors[i][0] + self.center_pts[i][0] for i in range(num_pts)], [derivative_vectors[i][1] + self.center_pts[i][1] for i in range(num_pts)], [derivative_vectors[i][2] + self.center_pts[i][2] for i in range(num_pts)], [derivative_vectors[i][0] for i in range(num_pts)], [derivative_vectors[i][1] for i in range(num_pts)], [derivative_vectors[i][2] for i in range(num_pts)], color="red")
        
        plt.show()



    def optimize(self, eval=False):

        wound_points = self.points_t

        def min_all_suture_dist(t):
            self.update_placement(self.mesh, t)  # Dynamically update placement
            insert_pts = self.insertion_pts
            # center_pts = self.center_pts
            extract_pts = self.extraction_pts

            insert_dists = self.get_all_dists(insert_pts)
            # center_dists = self.get_all_dists(center_pts)
            extract_dists = self.get_all_dists(extract_pts)

            # h = self.gamma * (1 / 2)
            h = 0.0025
            return min([i - h for i in insert_dists] + 
                    # [i - h for i in center_dists] + 
                    [i - h for i in extract_dists])

        def max_suture_dist(t):
            self.update_placement(self.mesh, t)  # Dynamically update placement
            insert_pts = self.insertion_pts
            center_pts = self.center_pts
            extract_pts = self.extraction_pts

            insert_dists = self.get_dists(insert_pts)
            center_dists = self.get_dists(center_pts)
            extract_dists = self.get_dists(extract_pts)

            # h = self.gamma * 4
            h = 0.008
            return min([h - i for i in insert_dists] + 
                    [h - i for i in center_dists] + 
                    [h - i for i in extract_dists])


        constraints = [
                        {'type': 'ineq', 'fun': lambda t: min(t)},
                        {'type': 'ineq', 'fun': lambda t: 1 - max(t)},
                    #    {'type': 'ineq', 'fun': lambda t: min_all_suture_dist(t)},
                    #    {'type': 'ineq', 'fun': lambda t: max_suture_dist(t)},
                       ]
        
        # for i in range(len(t) - 1):
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_max_constraint(insert_dists, i)})
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_max_constraint(center_dists, i)})
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_max_constraint(extract_dists, i)})
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_min_constraint(insert_dists, i)})
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_min_constraint(center_dists, i)})
        #     constraints.append({'type': 'ineq', 'fun': lambda t: get_ith_min_constraint(extract_dists, i)})


        def loss(wound_points, eval=False):
            """
            This function should calculate the loss of a particular placement. As before, the 
            loss is entirely dependent on how far along the curve we are. Let t range from 
            0 to 1, and indicates how far along the wound we are. placement.t is an array of 
            t values for each point. This is what we are optimizing over (we want to find the 
            best values of t).
            """
            # print("WOUND POINTS", wound_points)
            if not eval:
                self.update_placement(self.mesh, wound_points)
            
            # recalculate all point locations

            # print("Updated placement")
            # var_loss = self.get_point_dist_var_loss()
            # ideal_loss = self.get_ideal_loss()

            # curvature_loss = self.get_curvature_loss()
            closure_loss, shear_loss, all_closure, per_insertion, per_extraction, _, _ = self.compute_closure_shear_loss(granularity=100)
            # print("closure loss", closure_loss)
            # print("shear loss", shear_loss)
            # print("var_loss", var_loss)
            # print("curvature_loss",  np.sqrt(curvature_loss))
            # print("closure loss", np.sqrt(0.01 * closure_loss))

            # print("SHEAR COEF", self.c_shear)
            # print("CLOSURE COEF", self.c_closure)

            curr_loss = closure_loss + shear_loss
            # curr_loss = var_loss * self.c_var + ideal_loss * self.c_ideal
            # print("current_loss:" + str(curr_loss))

            if not eval:
                return curr_loss
            else:
                return {"closure_loss": closure_loss, "shear_loss": shear_loss, "curr_loss": curr_loss}
        
        def jac(t):
            return optim.approx_fprime(t, loss)


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
        
        if eval:
            return loss(wound_points, eval=True)
        
        initial_guess = wound_points
        bounds = [(0, 1) for _ in initial_guess]
        # run optimizer
        # result = optim.minimize(loss, initial_guess, bounds=bounds, options={"maxiter":50, "disp":True}, method = 'SLSQP', jac = jac)
        # print("RESULT")
        # print(result)
        cons = []
        # for i in range(len(initial_guess)):
        #     l = {'type': 'ineq',
        #         'fun': lambda t, i=i: t[i] - 0}
        #     u = {'type': 'ineq',
        #         'fun': lambda t, i=i: 1 - t[i]}
        #     cons.append(l)
        #     cons.append(u)
        
        cons.append({'type': 'ineq', 'fun': lambda t: min_all_suture_dist(t)})
        cons.append({'type': 'ineq', 'fun': lambda t: max_suture_dist(t)})
                    
        result = optim.minimize(
            fun=loss,
            x0=initial_guess,
            method='Nelder-Mead',
            bounds=bounds,
            options={'disp':True}
        )

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
    
    def get_point_dist_var_loss(self):

        insertion_var = self.get_diff_var(self.insertion_pts)
        center_var = self.get_diff_var(self.center_pts)
        extraction_var = self.get_diff_var(self.extraction_pts)

        return insertion_var + center_var + extraction_var
    
    def get_ideal_loss(self):

        insertion_loss = self.get_me_opt(self.insertion_pts, self.gamma)
        center_loss = self.get_me_opt(self.center_pts, self.gamma)
        extraction_loss = self.get_me_opt(self.extraction_pts, self.gamma)

        return insertion_loss + center_loss + extraction_loss
    
    def sigmoid(self, x, L, k, x0):
        """
        Sigmoid function with parameters to control its shape.
        L: the curve's maximum value
        k: the logistic growth rate or steepness of the curve
        x0: the x-value of the sigmoid's midpoint
        """
        return L / (1 + np.exp(-k * (x - x0)))

    def get_curvature_loss(self):
        # high loss when a low curvature
        num_points = len(self.points_t)
        
        error = 0
        for i in range(1, num_points-1):
            t = self.points_t[i]
            # print('At t', t)
            desired_spacing = self.spacing[int(t * 99)]
            dist1 = euclidean_dist(self.insertion_pts[i], self.insertion_pts[i+1])
            dist2 = euclidean_dist(self.insertion_pts[i], self.insertion_pts[i-1])
            dist3 = euclidean_dist(self.extraction_pts[i], self.extraction_pts[i+1])
            dist4 = euclidean_dist(self.extraction_pts[i], self.extraction_pts[i-1])

            # print(f"desired spacing {desired_spacing} vs actual {dist1} {dist2} {dist3} {dist4}")
            error += (dist1 - desired_spacing)**2
            error += (dist2 - desired_spacing)**2
            error += (dist3 - desired_spacing)**2
            error += (dist4 - desired_spacing)**2
        # print('ERROR', error)

        # print('spacing', spacing)
        # vs with thresholding + quadratic
        # print(spacing)

        # our prev method was to create spline , spline is not the best way to represent, we lose detail
        # keep detail if at sample time, we step out

        # the force model doesn't fully take this into account, imply that the surgical intuition is a more direct way
        # to what extent is the force model necessary  

        # add another , proximity of stitch to nearest maximum curvature (stick a loss on how far is a stich from local maximum curvature)

        # USE EXPONENTIAL / SIGMOID
        
        return np.sqrt(error)

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
    
    def get_all_dists(self, points):

        counter = 0
        dists = [0 for i in range((len(points) * (len(points) -1)) // 2)]

        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                dists[counter] = euclidean_dist(points[i], points[j])
                counter += 1

        return dists
    
    def loss_placement(self):
        """
        This function should calculate the loss of a particular placement. Used for synthetic splines
        """
        var_loss = self.get_point_dist_var_loss()
        ideal_loss = self.get_ideal_loss()

        # curr_loss = shear_loss * self.c_shear + closure_loss * self.c_closure + var_loss * self.c_var + ideal_loss * self.c_ideal
        curr_loss = var_loss * self.c_var + ideal_loss * self.c_ideal
        # print("current_loss:" + str(curr_loss))

        return curr_loss