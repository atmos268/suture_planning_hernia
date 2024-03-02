import numpy as np
from SuturePlacement3d import SuturePlacement3d
from MeshIngestor import MeshIngestor
from test_force_model import get_plane_estimation, project_vector_onto_plane
import scipy.interpolate as inter
import random
import matplotlib.pyplot as plt
class Optimizer3d:
    """
    This class takes in a mesh object, and a spline on the surface of the wound, and optimizes the position of
    sutures along the spline
    mesh: A MeshIngestor object, representing the surface of the wound.
    spline: The spline of the wound
    suture_width: how far, in mm, the insertion and extraction points should be from the wound line
    """
    def __init__(self, mesh, spline, suture_width):
        self.mesh = mesh
        self.spline = spline
        self.suture_width = suture_width 
        self.suture_placement = None

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
        spline_length = self.calculate_spline_length(spline, mesh)
        num_sutures_initial = int(spline_length / (self.suture_width * 3)) #TODO: modify later 
        print("Num sutures initial", num_sutures_initial)
        points_t_initial = np.linspace(0, 1, int(num_sutures_initial))
        return self.generate_placement(mesh, spline, points_t_initial)
        
    def generate_placement(self, mesh, spline, points_t):
        """
        This function should take in a mesh and a spline and output center, 
        insertion and extraction points (step up function)

        returns a SuturePlacement3d object with spline and points
        """
        num_points = len(points_t)

        spline_x, spline_y, spline_z = spline[0], spline[1], spline[2]
        derivative_x, derivative_y, derivative_z = spline_x.derivative(), spline_y.derivative(), spline_z.derivative()

        # get center points
        center_points = [[spline_x(t), spline_y(t), spline_z(t)] for t in points_t]
        print("magnitude center points", [np.linalg.norm(center_points[i]) for i in range(num_points)])


        # get derivative points
        derivative_points = [[derivative_x(t), derivative_y(t), derivative_z(t)] for t in points_t]

        #get tangent plane normal vectors
        normal_vectors = [get_plane_estimation(mesh, center_points[i]) for i in range(num_points)]

        # project derivatives onto the tangent plane
        derivative_vectors = [project_vector_onto_plane(derivative_points[i], normal_vectors[i]) for i in range(num_points)]

        # normalize normal vectors and derivative vectors
        normal_vectors = [self.normalize_vector(normal_vectors[i]) for i in range(num_points)]
        derivative_vectors = [self.normalize_vector(derivative_vectors[i]) for i in range(num_points)]

        # Insertion points = cross product 
        insertion_points = [center_points[i] + self.suture_width * np.cross(normal_vectors[i], derivative_vectors[i]) for i in range(num_points)]
        print("magnitude insertion points", [np.linalg.norm(insertion_points[i]) for i in range(num_points)])

        # Extraction points = - cross product
        extraction_points = [center_points[i] + self.suture_width * (-np.cross(normal_vectors[i], derivative_vectors[i])) for i in range(num_points)]

        # create suture placement 3d object

        suturePlacement3d = SuturePlacement3d(spline, center_points, insertion_points, extraction_points)
        print("Center points", center_points)
        print("Insertion points", insertion_points)
        print("Extraction points", extraction_points)
        return suturePlacement3d, normal_vectors, derivative_vectors
    
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







    # FORCE MODEL FUNCTIONS






    def loss(mesh, placement):
        """
        This function should calculate the loss of a particular placement. As before, the 
        loss is entirely dependent on how far along the curve we are. Let t range from 
        0 to 1, and indicates how far along the wound we are. placement.points_t is an array of 
        t values for each point. This is what we are optimizing over (we want to find the 
        best values of t).
        """
        pass

    def calculate_shear_force(point, placement):
        """
        Calculate forces acting along the wound at a point, due to the placement as a whole
        """
        pass

    def calculate_closure_force(point, placement):
        """
        Calculate forces acting to close the wound at a point, due to the placement as a whole
        """
        pass


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
    print("hello")
    print(num1)
    print(num2)
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
    suture_width = 0.002# 0.002 
    optim3d = Optimizer3d(mesh, spline, suture_width)
    suturePlacement3d, normal_vectors, derivative_vectors = optim3d.generate_inital_placement(mesh, spline)
    #print("Normal vector", normal_vectors)
    optim3d.plot_mesh_path_and_spline(mesh, spline, suturePlacement3d, normal_vectors, derivative_vectors)
