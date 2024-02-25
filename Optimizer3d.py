import numpy as np
from SuturePlacement3d import SuturePlacement3d
from MeshIngestor import MeshIngestor
from test_force_model import get_plane_estimation, project_vector_onto_plane
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
    
    def generate_inital_placement(self, mesh, spline):
        """
        This function should take in a mesh and a spline and output an initial placement of center, 
        insertion and extraction points (equally spaced works)

        The challenge here will be to work out how to place the points on the surface of the skin

        returns a SuturePlacement3d object with spline and points
        """
        num_sutures_initial = 5 #TODO: modify later based on spline length
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
        center_points = [[spline_x[t], spline_y[t], spline_z[t]] for t in points_t]

        # get derivative points
        derivative_points = [[derivative_x[t], derivative_y[t], derivative_z[t]] for t in points_t]

        #get tangent plane normal vectors
        normal_vectors = [get_plane_estimation(center_points[i], mesh) for i in range(num_points)]

        # project derivatives onto the tangent plane
        derivative_vectors = [project_vector_onto_plane(derivative_points[i], normal_vectors[i]) for i in range(num_points)]

        # normalize normal vectors and derivative vectors
        normal_vectors = [self.normalize_vector(normal_vectors[i]) for i in range(num_points)]
        derivative_vectors = [self.normalize_vector(derivative_vectors[i]) for i in range(num_points)]

        # Insertion points = cross product 
        insertion_points = [self.suture_width * np.cross(normal_vectors[i], derivative_vectors[i]) for i in range(num_points)]

        # Extraction points = - cross product
        extraction_points = [self.suture_width * -np.cross(normal_vectors[i], derivative_vectors[i]) for i in range(num_points)]

        # create suture placement 3d object

        suturePlacement3d = SuturePlacement3d(spline, center_points, insertion_points, extraction_points)
        return suturePlacement3d
    
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


