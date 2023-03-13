import random

import DistanceCalculator
import RewardFunction
import Optimizer
import Constraints
import scipy.optimize as optim
import numpy as np
import matplotlib.pyplot as plt

class SuturePlacer:
    def __init__(self, wound_width, mm_per_pixel):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = wound_width
        self.mm_per_pixel = mm_per_pixel
        self.DistanceCalculator = DistanceCalculator.DistanceCalculator(self, self.wound_width, self.mm_per_pixel)
        self.RewardFunction = RewardFunction.RewardFunction(wound_width, self)
        self.Constraints = Constraints.Constraints(wound_width)
        self.Constraints.DistanceCalculator = self.DistanceCalculator

        # NB: Viraj: added a class in order to allow code to run
        self.Optimizer = Optimizer.Optimizer() # cvxpy? Feel free to make your own file with a class for Optimizer if you want one.

    def optimize(self, wound_points):
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(wound_points)
        self.RewardFunction.insert_dists = insert_dists
        self.RewardFunction.center_dists = center_dists
        self.RewardFunction.extract_dists = extract_dists

        self.Constraints.wound_points = wound_points
        start = wound_points[0]
        end = wound_points[-1]
        
        def final_loss(t):
            self.RewardFunction.insert_dists, self.RewardFunction.center_dists, self.RewardFunction.extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)    
            self.RewardFunction.wound_points = t
            self.RewardFunction.suture_points = list(zip(insert_pts, center_pts, extract_pts))
            return self.RewardFunction.final_loss(c_lossIdeal = 30, c_lossVar = 1, c_lossClosure = 20, c_lossShear = 2)

        result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints(), options={"maxiter":200}, method = 'COBYLA', tol=1e-2)
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(result.x)
        
        self.insert_pts = insert_pts
        self.center_pts = center_pts
        self.extract_pts = extract_pts


        # varun's printing code for shear/closure
        print(final_loss(wound_points))

        result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints(), options={"maxiter":200}, method = 'COBYLA', tol = 1e-2)
        #print(self.DistanceCalculator.calculate_distances(result.x))
        print(final_loss(result.x))
        print('final wound points', result.x)
        plt.clf()
        save_intermittent_plots = False
        if save_intermittent_plots:
            self.DistanceCalculator.plot(result.x, "closure plot", plot_closure=True, save_fig='images/' + str(len(wound_points)) + '_closure_' + str(random.randint(0, 1000000)))
            self.DistanceCalculator.plot(result.x, "shear plot", plot_shear=True, save_fig='images/' + str(len(wound_points)) + '_shear_' + str(random.randint(0, 1000000)))

        # plt.clf()
        # plt.plot(np.arange(len(self.RewardFunction.closure_forces)), (self.RewardFunction.closure_forces - 1) ** 2, c='blue')
        # plt.plot(np.arange(len(self.RewardFunction.shear_forces)), (self.RewardFunction.shear_forces ** 2), c='orange')

        # plt.show()
        # closure = (closure_forces - 1) ** 2
        # shear = shear_forces ** 2

        result_x, result_y = self.DistanceCalculator.wound_parametric(result.x, 0)
        print('result_x', result_x)
        print('result_y',
              result_y)
        # return result_x, result_y

        return insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, result.x
    
    def place_sutures(self):
        # I want it to have an initial placement and then a forward pass thru to the reward so we can test our code.
        # Maybe this initial placement could be based on some smart heuristic to make optimization faster...
       
        # choosing 8 points along curve as placeholder
        # currently, we have chosen a set of unequal points to demostrate visually what the optimization is doing
        # in reality, we would likely warm-start with equally spaced points.
        num_sutures = int(self.DistanceCalculator.initial_number_of_sutures(0, 1))
        print('NUM SUTURES: ', num_sutures)
        # num_sutures = 12
        heuristic = num_sutures
        best_loss = float('inf')
        wound_points = np.linspace(0, 1, num_sutures)
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = self.optimize(wound_points=wound_points)
        self.RewardFunction.insert_dists = insert_dists
        self.RewardFunction.center_dists = center_dists
        self.RewardFunction.extract_dists = extract_dists
        best_loss = self.RewardFunction.hyperLoss()
        print('loss: ', best_loss)
        b_insert_pts, b_center_pts, b_extract_pts, b_ts = insert_pts, center_pts, extract_pts, ts
        losses = [best_loss]
        first_downward = True
        while True:
            num_sutures += 1
            print('NUM SUTURES: ', num_sutures)
            wound_points = np.linspace(0, 1, num_sutures)
            insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = self.optimize(wound_points=wound_points)
            self.RewardFunction.insert_dists = insert_dists
            self.RewardFunction.center_dists = center_dists
            self.RewardFunction.extract_dists = extract_dists
            curr_loss = self.RewardFunction.hyperLoss()
            if curr_loss < best_loss:
                best_loss = curr_loss
                b_insert_pts, b_center_pts, b_extract_pts, b_ts = insert_pts, center_pts, extract_pts, ts
            if len(losses) >= 2 and curr_loss > max(losses[-1], losses[-2]):
                break
            losses.append(curr_loss)
            print("loss", curr_loss)
        num_sutures = heuristic
        while True:
            num_sutures -= 1
            print('NUM SUTURES: ', num_sutures)
            if num_sutures <= 1:
                break
            wound_points = np.linspace(0, 1, num_sutures)
            insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = self.optimize(wound_points=wound_points)
            self.RewardFunction.insert_dists = insert_dists
            self.RewardFunction.center_dists = center_dists
            self.RewardFunction.extract_dists = extract_dists
            curr_loss = self.RewardFunction.hyperLoss()
            if curr_loss < best_loss:
                best_loss = curr_loss
                b_insert_pts, b_center_pts, b_extract_pts, b_ts = insert_pts, center_pts, extract_pts, ts
            if not first_downward and (len(losses) >= 2 and curr_loss > max(losses[0], losses[1])):
                break
            losses = [curr_loss] + losses
            first_downward = False
            print("loss", curr_loss)
        self.insert_pts = b_insert_pts
        self.center_pts = b_center_pts
        self.extract_pts = b_extract_pts
        print(losses)
        self.DistanceCalculator.plot(b_ts, "Plotting after optimization", plot_closure=True)
        print("plotting")
        return b_insert_pts, b_center_pts, b_extract_pts

        
        
        # Varun: Eventually,  we'll have an overall reward that is the linear combination. [these two lines merge-conflicted, don't know which is right for now]
        # self.initial_reward = self.RewardFunction.rewardA(self.RewardFunction) # TODO Julia/Yashish
        # self.initial_reward = self.RewardFunction.rewardX() # TODO Julia/Yashish

        # Then, we can use the initial placement to warm-start the optimization process.
        # self.Optimizer.optimize_placement() # TODO Viraj/Yashish: the variables to optimize
        # [TODO] are the wound_points. These are parametric values for locations on the wound.
        #  [TODO] Wound should already be passed in by main.py:place_sutures.

        # we will feed optimized values in after merging with optimize code