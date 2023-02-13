import math
import matplotlib.pyplot as plt


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

        # gets the norm of the vector (1, gradient)

        num_pts = len(wound_points)

        def get_norm(gradient):
            return math.sqrt(1 + gradient**2)

        # might be worth vecotrizing in the future
            
        # get the curve and gradient for each point (the second argument allows you to on the fly take the derivative)
        wound_curve = self.wound(wound_points, 0)
        wound_derivatives = self.wound(wound_points, 1)

        # extract the norms of the vectors
        norms = [get_norm(wound_derivative) for wound_derivative in wound_derivatives]

        # get the normal vectors as norm = 1
        normal_vecs = [[wound_derivatives[i]/norms[i], -1/norms[i]] for i in range(num_pts)]
        
        # make norm width wound_width
        normal_vecs = [[normal_vec[0] * self.wound_width, normal_vec[1] * self.wound_width] for normal_vec in normal_vecs]

        # add and subtract for insertion and exit
        insert_pts = [[wound_points[i] + normal_vecs[i][0], wound_curve[i] + normal_vecs[i][1]] for i in range(num_pts)]

        extract_pts = [[wound_points[i] - normal_vecs[i][0], wound_curve[i] - normal_vecs[i][1]] for i in range(num_pts)]

        center_pts = [[wound_points[i], wound_curve[i]] for i in range(num_pts)]

        """
        # verify works by plotting
        plt.scatter([insert_pts[i][0] for i in range(num_pts)], [insert_pts[i][1] for i in range(num_pts)], c="red")
        plt.scatter([extract_pts[i][0] for i in range(num_pts)], [extract_pts[i][1] for i in range(num_pts)], c="blue")
        plt.scatter([center_pts[i][0] for i in range(num_pts)], [center_pts[i][1] for i in range(num_pts)], c="green")

        plt.show()
        """

        def dist_insert(i):
            pass

        def dist_center(i):
            pass

        def dist_extract(i):
            pass

        
        #for all i:
            # calculate insert, center, extract distances

        # return 3 lists: insert dists, center dists, extract dists

        return [], [], []

