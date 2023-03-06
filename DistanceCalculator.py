import math
import matplotlib.pyplot as plt
import torch as torch
import numpy as np
from scipy.interpolate import make_interp_spline

class DistanceCalculator():

    def __init__(self, wound_width):
        self.wound_width = wound_width
        self.gradients = {}
        pass

    def calculate_distances(self, wound_point_t):
        # wound points is the set of time-steps along the wound that correspond to wound points

        # Step 1: Calculate insertion and extraction points. Use the wound's
        # evaluate_hodograph function (see bezier_tutorial.py:34) to calculate the tangent/
        # normal, and then wound_width to see where these points are in terms of each wound_point

        # Step 2: Between each adjacent pair of wound_points, with i
        # as wound point index, must calculate dist(i, i + 1) for
        # insert, center extract: refer to slide 13 for details:

        # gets the norm of the vector (1, gradient)
        num_pts = len(wound_point_t)

        def get_norm(gradient):
            return math.sqrt(1 + gradient**2)

        # might be worth vecotrizing in the future
            
        # get the curve and gradient for each point (the second argument allows you to on the fly take the derivative)
        wound_points, wound_curve = self.wound_parametric(wound_point_t, 0)
        wound_derivatives_x, wound_derivatives_y = self.wound_parametric(wound_point_t, 1)
        wound_derivatives = np.divide(wound_derivatives_y, wound_derivatives_x)
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

        # verify works by plotting
        #plt.scatter([insert_pts[i][0] for i in range(num_pts)], [insert_pts[i][1] for i in range(num_pts)], c="red")
        #plt.scatter([extract_pts[i][0] for i in range(num_pts)], [extract_pts[i][1] for i in range(num_pts)], c="blue")
        #plt.scatter([center_pts[i][0] for i in range(num_pts)], [center_pts[i][1] for i in range(num_pts)], c="green")

        def euclidean_distance(point1, point2):
            x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
            return math.sqrt((x2-x1)**2+(y2-y1)**2)

        def dist_insert(i):
            return euclidean_distance(insert_pts[i], insert_pts[i+1])

        def dist_center(i):
            return euclidean_distance(center_pts[i], center_pts[i+1])

        def dist_extract(i):
            return euclidean_distance(extract_pts[i], extract_pts[i+1])

        
        #for all i:
            # calculate insert, center, extract distances
        # return 3 lists: insert dists, center dists, extract dists

        
        insert_distances = []
        center_distances = []
        extract_distances = []
        

        for i in range(num_pts - 1):
            insert_distances.append(dist_insert(i))
            center_distances.append(dist_center(i))
            extract_distances.append(dist_extract(i))
        #making insert_dist negative if there are crossings
        insert_distances = [self.intersect(insert_pts[i], extract_pts[i], insert_pts[i+1], extract_pts[i+1])*insert_distances[i] for i in range(len(insert_pts)-1)] 
        
        return insert_distances, center_distances, extract_distances, insert_pts, center_pts, extract_pts
    
    #check for intersections
    def on_segment(self, p1, p2, p):
        return min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1])
    
    def cross_product(self, p1, p2):
        return p1[0] * p2[1] - p2[0] * p1[1]
    
    def direction(self, p1, p2, p3):
        return self.cross_product((p3[0]-p1[0], p3[1] - p1[1]), (p2[0]-p1[0], p2[1] - p1[1]))
    
    # checks if line segment p1p2 and p3p4 intersect and returns -1 if True and 1 when False
    def intersect(self, p1, p2, p3, p4):
        d1 = self.direction(p3, p4, p1)
        d2 = self.direction(p3, p4, p2)
        d3 = self.direction(p1, p2, p3)
        d4 = self.direction(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return -1
        elif d1 == 0 and self.on_segment(p3, p4, p1):
            return -1
        elif d2 == 0 and self.on_segment(p3, p4, p2):
            return -1
        elif d3 == 0 and self.on_segment(p1, p2, p3):
            return -1
        elif d4 == 0 and self.on_segment(p1, p2, p4):
            return -1
        else:
            return 1
    

    def plot(self, wound_point_t):
        # wound points is the set of time-steps along the wound that correspond to wound points

        # Step 1: Calculate insertion and extraction points. Use the wound's
        # evaluate_hodograph function (see bezier_tutorial.py:34) to calculate the tangent/
        # normal, and then wound_width to see where these points are in terms of each wound_point

        # Step 2: Between each adjacent pair of wound_points, with i
        # as wound point index, must calculate dist(i, i + 1) for
        # insert, center extract: refer to slide 13 for details:

        # gets the norm of the vector (1, gradient)
        num_pts = len(wound_point_t)

        def get_norm(gradient):
            return math.sqrt(1 + gradient**2)

        # might be worth vecotrizing in the future
            
        # get the curve and gradient for each point (the second argument allows you to on the fly take the derivative)
        wound_points, wound_curve = self.wound_parametric(wound_point_t, 0)
        wound_derivatives_x, wound_derivatives_y = self.wound_parametric(wound_point_t, 1)
        wound_derivatives = np.divide(wound_derivatives_y, wound_derivatives_x)
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

        print(wound_point_t)
    
        # Returns evenly spaced numbers
        # over a specified interval.
        X_, Y_ = [], []
        for i in range(500):
            t = min(wound_point_t) + (max(wound_point_t) - min(wound_point_t))*i/500
            temp = self.wound_parametric(t, 0)
            X_.append(temp[0])
            Y_.append(temp[1])
        # X_ = np.linspace(wound_points.min(), wound_points.max(), 500)
        # Y_ = spline(X_)
        
        # Plotting the Graph
        plt.plot(X_, Y_)
        plt.scatter([insert_pts[i][0] for i in range(num_pts)], [insert_pts[i][1] for i in range(num_pts)], c="red")
        plt.scatter([extract_pts[i][0] for i in range(num_pts)], [extract_pts[i][1] for i in range(num_pts)], c="blue")
        plt.scatter([center_pts[i][0] for i in range(num_pts)], [center_pts[i][1] for i in range(num_pts)], c="green")
        plt.show()




