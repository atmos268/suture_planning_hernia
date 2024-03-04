class SuturePlacement3d:
    def __init__(self, spline, center_points, insertion_pts, extraction_pts, t):
        self.spline = spline
        self.t = t
        self.center_pts = center_points
        self.insertion_pts = insertion_pts
        self.extraction_pts = extraction_pts
    
