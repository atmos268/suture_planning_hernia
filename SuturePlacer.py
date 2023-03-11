import DistanceCalculator
import RewardFunction
import Optimizer
import Constraints
import scipy.optimize as optim
import numpy as np

class SuturePlacer:
    def __init__(self, wound_width, mm_per_pixel):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = wound_width
        self.mm_per_pixel = mm_per_pixel
        self.DistanceCalculator = DistanceCalculator.DistanceCalculator(self, self.wound_width, self.mm_per_pixel)
        self.RewardFunction = RewardFunction.RewardFunction()
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
            return self.RewardFunction.final_loss(a = 1, b = 1)

        result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints())
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(result.x)
        
        self.insert_pts = insert_pts
        self.center_pts = center_pts
        self.extract_pts = extract_pts

        return insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, result.x
    
    def place_sutures(self):
        # I want it to have an initial placement and then a forward pass thru to the reward so we can test our code.
        # Maybe this initial placement could be based on some smart heuristic to make optimization faster...
       
        # choosing 8 points along curve as placeholder
        # currently, we have chosen a set of unequal points to demostrate visually what the optimization is doing
        # in reality, we would likely warm-start with equally spaced points.
        num_sutures = int(self.DistanceCalculator.initial_number_of_sutures(0, 1))
        heuristic = num_sutures
        best_loss = float('inf')
        wound_points = np.linspace(0, 1, num_sutures)
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = self.optimize(wound_points=wound_points)
        self.RewardFunction.insert_dists = insert_dists
        self.RewardFunction.center_dists = center_dists
        self.RewardFunction.extract_dists = extract_dists
        best_loss = self.RewardFunction.hyperLoss()
        print(best_loss)
        b_insert_pts, b_center_pts, b_extract_pts, b_ts = insert_pts, center_pts, extract_pts, ts
        losses = [best_loss]
        while True:
            num_sutures += 1
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
        num_sutures = heuristic
        while True:
            num_sutures -= 1
            wound_points = np.linspace(0, 1, num_sutures)
            insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = self.optimize(wound_points=wound_points)
            self.RewardFunction.insert_dists = insert_dists
            self.RewardFunction.center_dists = center_dists
            self.RewardFunction.extract_dists = extract_dists
            curr_loss = self.RewardFunction.hyperLoss()
            if curr_loss < best_loss:
                best_loss = curr_loss
                b_insert_pts, b_center_pts, b_extract_pts, b_ts = insert_pts, center_pts, extract_pts, ts
            if len(losses) >= 2 and curr_loss > max(losses[0], losses[1]):
                break
            losses = [curr_loss] + losses
        self.insert_pts = b_insert_pts
        self.center_pts = b_center_pts
        self.extract_pts = b_extract_pts
        print(losses)
        insert_pts, center_pts, extract_pts = self.DistanceCalculator.plot(b_ts, "Plotting after optimization")
        return b_insert_pts, b_center_pts, b_extract_pts

        
        
        # Varun: Eventually,  we'll have an overall reward that is the linear combination. [these two lines merge-conflicted, don't know which is right for now]
        # self.initial_reward = self.RewardFunction.rewardA(self.RewardFunction) # TODO Julia/Yashish
        # self.initial_reward = self.RewardFunction.rewardX() # TODO Julia/Yashish

        # Then, we can use the initial placement to warm-start the optimization process.
        # self.Optimizer.optimize_placement() # TODO Viraj/Yashish: the variables to optimize
        # [TODO] are the wound_points. These are parametric values for locations on the wound.
        #  [TODO] Wound should already be passed in by main.py:place_sutures.

        # we will feed optimized values in after merging with optimize code