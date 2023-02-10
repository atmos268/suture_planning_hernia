class DistanceCalculator():

    def __init__(self, wound_width):
        self.wound_width = wound_width
        pass

    def calculate_distances(self, wound_points):
        # wound points is the set of time-steps along the wound that correspond to wound points

        # Step 1: Calculate insertion and extraction points. Use the wound's
        # evaluate_hodograph function (see bezier_tutorial.py:34) to calculate the tangent/
        # normal, and then wound_width to see where these points are in terms of each wound_point

        # Step 2: Between each adjacent pair of wound_points, with i
        # as wound point index, must calculate dist(i, i + 1) for
        # insert, center extract: refer to slide 13 for details:

        def dist_insert(i):
            pass

        def dist_center(i):
            pass

        def dist_extract(i):
            pass

        for all i:
            calculate insert, center, extract distances

        return 3 lists: insert dists, center dists, extract dists
