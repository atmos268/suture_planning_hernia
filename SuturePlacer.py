import DistanceCalculator
import RewardFunction
import Optimizer
import Constraints
import scipy.optimize as optim

class SuturePlacer:
    def __init__(self, wound_width, mm_per_pixel):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = wound_width
        self.mm_per_pixel = mm_per_pixel
        self.DistanceCalculator = DistanceCalculator.DistanceCalculator(self, self.wound_width, self.mm_per_pixel)
        self.RewardFunction = RewardFunction.RewardFunction()
        self.Constraints = Constraints.Constraints()
        self.Constraints.DistanceCalculator = self.DistanceCalculator

        # NB: Viraj: added a class in order to allow code to run
        self.Optimizer = Optimizer.Optimizer() # cvxpy? Feel free to make your own file with a class for Optimizer if you want one.

    def place_sutures(self):
        # I want it to have an initial placement and then a forward pass thru to the reward so we can test our code.
        # Maybe this initial placement could be based on some smart heuristic to make optimization faster...
       
        # choosing 11 equally spaced points along curve as placeholder
<<<<<<< HEAD
        wound_points = [0.1*i for i in range(11)] # TODO Harshika/Viraj: Initial Placement, can put some placeholder here

        insert_dists, center_dists, extract_dists = self.DistanceCalculator.calculate_distances(wound_points) # TODO Harshika/Viraj
        self.RewardFunction.insert_dists = insert_dists
        self.RewardFunction.center_dists = center_dists
        self.RewardFunction.extract_dists = extract_dists
=======
        wound_points = [0.0, 0.05, 0.25, 0.45, 0.65, 0.75, 1.05, 1.1] # TODO Harshika/Viraj: Initial Placement, can put some placeholder here
        self.Constraints.wound_points = wound_points
        self.DistanceCalculator.plot(wound_points)
        # print("Initial wound points", wound_points)
        start = wound_points[0]
        end = wound_points[-1]
        
        def final_loss(t):
            self.RewardFunction.insert_dists, self.RewardFunction.center_dists, self.RewardFunction.extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)    
            return self.RewardFunction.final_loss(a = 1, b = 1)

        print(final_loss(wound_points))
        result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints())
        #print(self.DistanceCalculator.calculate_distances(result.x))
        print(final_loss(result.x))
        self.DistanceCalculator.plot(result.x)
        result_x, result_y = self.DistanceCalculator.wound_parametric(result.x, 0)
        print(result_x)
        print(result_y)
        return result_x, result_y
        


>>>>>>> optimization

        # Varun: Eventually,  we'll have an overall reward that is the linear combination. [these two lines merge-conflicted, don't know which is right for now]
        self.initial_reward = self.RewardFunction.rewardA(self.RewardFunction) # TODO Julia/Yashish
        # self.initial_reward = self.RewardFunction.rewardX() # TODO Julia/Yashish

        # Then, we can use the initial placement to warm-start the optimization process.
        self.Optimizer.optimize_placement() # TODO Viraj/Yashish: the variables to optimize
        # [TODO] are the wound_points. These are parametric values for locations on the wound.
        #  [TODO] Wound should already be passed in by main.py:place_sutures.

        # we will feed optimized values in after merging with optimize code