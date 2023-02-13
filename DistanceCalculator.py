class DistanceCalculator():

    def __init__(self, wound_width):
        self.wound_width = wound_width
        pass

    def calculate_distances(self, wound_points, curve):
        # wound points is the set of time-steps along the wound that correspond to wound points

        # Step 1: Calculate insertion and extraction points. Use the wound's
        # evaluate_hodograph function (see bezier_tutorial.py:34) to calculate the tangent/
        # normal, and then wound_width to see where these points are in terms of each wound_point

        # Step 2: Between each adjacent pair of wound_points, with i
        # as wound point index, must calculate dist(i, i + 1) for
        # insert, center extract: refer to slide 13 for details:
        
        insert_points = []
        center_points = []
        extract_points = []
        
        def get_derivative(x):
            d1 = curve.derivative()
            return d1(x)
        
        def get_normal(x):


        def euclidean_distance(point1, point2):
            x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
            return sqrt((x2-x1)**2+(y2-y1)**2)

        def dist_insert(i):
            return euclidean_distance(insert_points[i], insert_points[i+1])

        def dist_center(i):
            return euclidean_distance(center_points[i], center_points[i+1])

        def dist_extract(i):
            return euclidean_distance(extract_points[i], extract_points[i+1])
        
        insert_distances = []
        center_distances = []
        extract_distances = []
        for i in range(num_points-1):
            insert_distances.append(dist_insert(i))
            center_distances.append(dist_center(i))
            extract_distances.append(dist_extract(i))

        return insert_distances, center_distances, extract_distances
