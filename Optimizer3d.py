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
        self.suture_placement = None

    def generate_inital_placement(mesh, spline):
        """
        This function should take in a mesh and a spline and output an initial placement of center, 
        insertion and extraction points (equally spaced works)

        The challenge here will be to work out how to place the points on the surface of the skin

        returns a SuturePlacement3d object with spline and points
        """
        pass

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

    def calculate_closure_force(point, placement):
        """
        Calculate forces acting to close the wound at a point, due to the placement as a whole
        """


