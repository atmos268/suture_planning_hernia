import torch as torch
import math
import numpy as np

class RewardFunction():
    def __init__(self, wound_width):
        self.insert_dists = []
        self.center_dists = []
        self.extract_dists = []
        self.wound_parametric = None
        self.wound_points = None

        # Closure Force
        self.influence_region = wound_width
        self.suture_force = 1 # The maximum force a suture exerts. So, in this code, we essentially scale to the force of 1 suture.
        self.ideal_suture_force = 1 # I don't know what this is exactly! 1 makes sense though, if a single suture is locally optimal
        # ...if you have too many sutures nearby this will amount to more than 1, and its also consistent with the wound_width
        # being double the distance, because with a straight wound, the closure force would be 1 everywhere if you space at the
        # ideal distance!

    # distance lists added to this object by SuturePlacer.
    # variance
    def final_loss(self, a = 1, b = 1, c=1):
        return a * self.lossVar() + b * self.lossIdeal() + c * self.lossClosureForce()

    def lossVar(self):
        mean_insert = sum(self.insert_dists) / len(self.insert_dists)
        var_insert = sum([(i - mean_insert)**2 for i in self.insert_dists])
        
        mean_center = sum(self.center_dists) / len(self.center_dists)
        var_center = sum([(i - mean_center)**2 for i in self.center_dists])
        
        mean_extract = sum(self.extract_dists) / len(self.extract_dists)
        var_extract = sum([(i - mean_extract)**2 for i in self.extract_dists])
        
        return var_insert + var_center + var_extract
    
    def lossIdeal(self):
        ideal = 0.4
        power = 2
        extra_pen = 100
        insertion = []
        extraction = []
        center = []
        for i in range(len(self.insert_dists)):
            ins = self.insert_dists[i]
            if ins < ideal:
                insertion.append((ins-ideal) ** power * extra_pen)
            else:
                insertion.append((ins-ideal) ** power)
            ext = self.extract_dists[i]
            if ext < ideal:
                extraction.append((ext-ideal) ** power * extra_pen)
            else:
                extraction.append((ext-ideal) ** power)
            cen = self.insert_dists[i]
            if cen < ideal:
                center.append((cen-ideal) ** power * extra_pen)
            else:
                center.append((cen-ideal) ** power)
        return sum(insertion + center + extraction)

    def distance_along(self, wound, a, b):
        'Note: Signed distance, and a can be greater than b'
        derivative = wound.derivative

        def get_norm(gradient):
            return math.sqrt(1 + gradient ** 2)

        # might be worth vecotrizing in the future

        # get the curve and gradient for each point (the second argument allows you to on the fly take the derivative)
        wound_points, wound_curve = self.wound_parametric(wound_point_t, 0)
        wound_derivatives_x, wound_derivatives_y = self.wound_parametric(wound_point_t, 1)
        wound_derivatives = np.divide(wound_derivatives_y, wound_derivatives_x)
        # extract the norms of the vectors
        norms = [get_norm(wound_derivative) for wound_derivative in wound_derivatives]

        # get the normal vectors as norm = 1
        normal_vecs = [[wound_derivatives[i] / norms[i], -1 / norms[i]] for i in range(num_pts)]

        # make norm width wound_width
        normal_vecs = [[normal_vec[0] * self.wound_width, normal_vec[1] * self.wound_width] for normal_vec in
                       normal_vecs]

        # add and subtract for insertion and exit
        insert_pts = [[wound_points[i] + normal_vecs[i][0], wound_curve[i] + normal_vecs[i][1]] for i in range(num_pts)]

        extract_pts = [[wound_points[i] - normal_vecs[i][0], wound_curve[i] - normal_vecs[i][1]] for i in
                       range(num_pts)]

        center_pts = [[wound_points[i], wound_curve[i]] for i in range(num_pts)]

    def lossClosureForce(self):

        def all_wounds_closure_force(t):
            suture_forces_running_sum = 0
            for c in self.wound_points:
                suture_forces_running_sum += single_wound_closure_force(c, t)
            return suture_forces_running_sum

        def single_wound_closure_force(c, t):
            dist_along = abs(self.distance_along(self.wound_parametric, c, t))
            if dist_along > self.influence_region:
                return 0
            else:
                distance_discount = dist_along / self.influence_region

                c_dx, c_dy = self.wound_parametric(c, 1)
                t_dx, t_dy = self.wound_parametric(t, 1)
                suture_vec_norm = math.dist([0,c_dx],[c_dy, 0])
                t_vec_norm = math.dist([0,t_dx],[t_dy, 0])
                suture_vec = np.array([c_dx, c_dy]) / suture_vec_norm
                t_vec = np.array([t_dx, t_dy]) / suture_vec_norm
                force_discount = np.dot(t_vec_norm, suture_vec)

                return self.suture_force * distance_discount * force_discount

        sample_points = np.linspace(0, 1, 100)
        closure_forces = [all_wounds_closure_force(i) for i in sample_points]
        return (closure_forces - 1) ** 2 # Squared error



    #min - max
    def lossMinMax(self):
        return (max(self.insert_dists) + max(self.center_dists) + max(self.extract_dists))
    


    # ... and so forth: refer to slide 14 from the presentation for details on how to design this.
    # It may be influenced by the optimizer as well.