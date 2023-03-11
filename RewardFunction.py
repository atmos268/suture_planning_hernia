import torch as torch
import math
import numpy as np
import numpy.linalg as LA

class RewardFunction():
    def __init__(self, wound_width, SuturePlacer):
        self.insert_dists = []
        self.center_dists = []
        self.extract_dists = []
        self.wound_parametric = None
        self.suture_points = None
        self.wound_points = None
        self.SuturePlacer = SuturePlacer

        # Closure Force
        self.closure_forces = None
        self.influence_region = wound_width
        self.suture_force = 1 # The maximum force a suture exerts. So, in this code, we essentially scale to the force of 1 suture.
        self.ideal_suture_force = 1 # I don't know what this is exactly! 1 makes sense though, if a single suture is locally optimal
        # ...if you have too many sutures nearby this will amount to more than 1, and its also consistent with the wound_width
        # being double the distance, because with a straight wound, the closure force would be 1 everywhere if you space at the
        # ideal distance!
        self.wcp_xs = None
        self.wcp_ys = None

    # distance lists added to this object by SuturePlacer.
    # variance
    def final_loss(self, c_lossVar = 1, c_lossIdeal = 1, c_lossClosure=1, c_lossShear=1):
        lossVar = self.lossVar()
        lossIdeal = self.lossIdeal()
        weighted_lossClosure = self.lossClosureForce(c_lossClosure, c_lossShear)
        return c_lossVar * lossVar + c_lossIdeal * lossIdeal + weighted_lossClosure

    def lossVar(self):
        mean_insert = sum(self.insert_dists) / len(self.insert_dists)
        var_insert = sum([(i - mean_insert)**2 for i in self.insert_dists])
        
        mean_center = sum(self.center_dists) / len(self.center_dists)
        var_center = sum([(i - mean_center)**2 for i in self.center_dists])
        
        mean_extract = sum(self.extract_dists) / len(self.extract_dists)
        var_extract = sum([(i - mean_extract)**2 for i in self.extract_dists])
        
        return var_insert + var_center + var_extract
    
    def lossIdeal(self):
        ideal = self.SuturePlacer.wound_width
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

    def distance_along(self, wound, a, b, num_samples_per_suture):
        'Note: Signed distance, and a can be greater than b'
        # return b - a # placeholder simple thing
        # return b - a
        sign = 1
        if a > b:
            old_a = a
            a = b
            b = old_a
            sign = -1
        steps = np.arange(a, b, self.influence_region / num_samples_per_suture)
        wound_points = [np.array(wound(s, 0)) for s in steps]
        cuml_dist = 0
        wound_vectors = [wound_points[i+1] - wound_points[i] for i in range(len(wound_points) - 1)]
        for v in wound_vectors:
            cuml_dist += math.sqrt(np.sum(v ** 2))
        return sign * cuml_dist
        # derivative = wound.derivative
        #
        # def get_norm(gradient):
        #     return math.sqrt(1 + gradient ** 2)
        #
        # # might be worth vecotrizing in the future
        #
        # # get the curve and gradient for each point (the second argument allows you to on the fly take the derivative)
        # wound_points, wound_curve = self.wound_parametric(wound_point_t, 0)
        # wound_derivatives_x, wound_derivatives_y = self.wound_parametric(wound_point_t, 1)
        # wound_derivatives = np.divide(wound_derivatives_y, wound_derivatives_x)
        # # extract the norms of the vectors
        # norms = [get_norm(wound_derivative) for wound_derivative in wound_derivatives]
        #
        # # get the normal vectors as norm = 1
        # normal_vecs = [[wound_derivatives[i] / norms[i], -1 / norms[i]] for i in range(num_pts)]
        #
        # # make norm width wound_width
        # normal_vecs = [[normal_vec[0] * self.wound_width, normal_vec[1] * self.wound_width] for normal_vec in
        #                normal_vecs]
        #
        # # add and subtract for insertion and exit
        # insert_pts = [[wound_points[i] + normal_vecs[i][0], wound_curve[i] + normal_vecs[i][1]] for i in range(num_pts)]
        #
        # extract_pts = [[wound_points[i] - normal_vecs[i][0], wound_curve[i] - normal_vecs[i][1]] for i in
        #                range(num_pts)]
        #
        # center_pts = [[wound_points[i], wound_curve[i]] for i in range(num_pts)]

    def lossClosureForce(self, c_lossClosure, c_lossShear, num_samples_per_suture=10):

        def all_wounds_closure_and_shear_force(t):
            suture_closure_forces_running_sum = 0
            suture_shear_forces_running_sum = 0
            for ice, w in zip(self.suture_points, self.wound_points):
                closure_force, shear = single_wound_closure_and_shear_force(ice, w, t)
                suture_closure_forces_running_sum += closure_force
                suture_shear_forces_running_sum += shear
            return suture_closure_forces_running_sum, suture_shear_forces_running_sum

        def single_wound_closure_and_shear_force(ice, w, t):
            i, c, e = ice
            if abs(w - t) > 4 * self.influence_region:
                return 0, 0
            w_dx, w_dy = self.wound_parametric(w, 1)
            t_dx, t_dy = self.wound_parametric(t, 1)
            xt, yt = self.wound_parametric(t, 0)
            suture_insert_vec = np.array([-w_dy, w_dx])
            suture_insert_vec = suture_insert_vec / LA.norm(suture_insert_vec)
            alpha = math.atan(w_dy / w_dx)

            ortho_to_wound_t_vec = np.array([-t_dy, t_dx])
            ortho_to_wound_t_vec = ortho_to_wound_t_vec / LA.norm(ortho_to_wound_t_vec)
            tangent_to_wound_t_vec = np.array([-ortho_to_wound_t_vec[1], ortho_to_wound_t_vec[0]])

            xi, yi = i
            pi = xt - xi
            qi = yt - yi

            xe, ye = e
            pe = xt - xe
            qe = yt - ye

            # Ellipse Axes: Found by Tuning so that the reward function is as constant
            # as possible for consecutive sutures on a line.
            a = 0.77 # Minor axis
            b = 1 # Major axis
            closure_r = 1 # Probably always 1, but can experiment with higher values
            shear_r = 1.3
            # Increase this!

            def ellipse_distance(p, q, alpha):
                return math.sqrt(
                    ((p * math.cos(alpha) + q * math.sin(alpha)) ** 2) /
                    ((self.SuturePlacer.wound_width * a) ** 2) +
                    ((-p * math.sin(alpha) + q * math.cos(alpha)) ** 2) /
                    ((self.SuturePlacer.wound_width * b) ** 2)
                )

            insert_ellipse_distance = ellipse_distance(pi, qi, alpha)
            extract_ellipse_distance = ellipse_distance(pe, qe, alpha)


            insert_closure_distance_discount = max(0, 1 - (1/closure_r) * (insert_ellipse_distance - 1))
            insert_shear_distance_discount = max(0, 1 - (1 / shear_r) * (insert_ellipse_distance - 1))

            extract_closure_distance_discount = max(0, 1 - (1 / closure_r) * (extract_ellipse_distance - 1))
            extract_shear_distance_discount = max(0, 1 - (1 / shear_r) * (extract_ellipse_distance - 1))

            # Find extract_closure_distance_discount and extract_shear_distance_discount too! Make these line up! Plus think about shear! and that you've combined everything right!

            closure_direction_discount = np.dot(suture_insert_vec, ortho_to_wound_t_vec)

            shear_direction_discount = abs(np.dot(suture_insert_vec, tangent_to_wound_t_vec))

            closure_force = self.suture_force * (insert_closure_distance_discount + extract_closure_distance_discount)/2 * closure_direction_discount
            shear_amount = self.suture_force * (insert_shear_distance_discount + extract_shear_distance_discount) * shear_direction_discount
            return closure_force, shear_amount

        sample_points = np.linspace(0, 1, 2 * int(num_samples_per_suture / self.influence_region))
        closure_forces = []
        shear_forces = []
        for p in sample_points:
            closure_force, shear = all_wounds_closure_and_shear_force(p)
            closure_forces.append(closure_force)
            shear_forces.append(shear)
        closure_forces = np.array(closure_forces)
        shear_forces = np.array(shear_forces)
        self.closure_forces = closure_forces
        self.shear_forces = shear_forces
        # print('closure forces!', closure_forces)
        wound_closure_points = [self.wound_parametric(t, 0) for t in sample_points]
        xs = [a[0] for a in wound_closure_points]
        ys = [a[1] for a in wound_closure_points]
        self.wcp_xs = xs
        self.wcp_ys = ys
        # return 0 # right now, I don't want to change the final result

        closure_loss = sum((closure_forces - 1) ** 2)
        shear_loss =  sum(shear_forces ** 2)

        print('closure loss: ', closure_loss)
        print('shear loss: ', shear_loss)

        return c_lossClosure * closure_loss + c_lossShear * shear_loss



    #min - max
    def lossMinMax(self):
        return (max(self.insert_dists) + max(self.center_dists) + max(self.extract_dists))
    


    # ... and so forth: refer to slide 14 from the presentation for details on how to design this.
    # It may be influenced by the optimizer as well.