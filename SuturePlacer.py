import DistanceCalculator
import RewardFunction
import Optimizer

class SuturePlacer:
    def __init__(self, wound_width):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = wound_width
        self.DistanceCalculator = DistanceCalculator.DistanceCalculator(self.wound_width)
        self.RewardFunction = RewardFunction.RewardFunction

        # NB: Viraj: added a class in order to allow code to run
        self.Optimizer = Optimizer.Optimizer() # cvxpy? Feel free to make your own file with a class for Optimizer if you want one.

    def place_sutures(self):
        # I want it to have an initial placement and then a forward pass thru to the reward so we can test our code.
        # Maybe this initial placement could be based on some smart heuristic to make optimization faster...
       
        # choosing 11 equally spaced points along curve as placeholder
        wound_points = [0.1*i for i in range(11)] # TODO Harshika/Viraj: Initial Placement, can put some placeholder here

        insert_dists, center_dists, extract_dists = self.DistanceCalculator.calculate_distances(wound_points) # TODO Harshika/Viraj
        self.RewardFunction.insert_dists = insert_dists
        self.RewardFunction.center_dists = center_dists
        self.RewardFunction.extract_dists = extract_dists

        # Varun: Eventually,  we'll have an overall reward that is the linear combination. [these two lines merge-conflicted, don't know which is right for now]
        self.initial_reward = self.RewardFunction.rewardA(self.RewardFunction) # TODO Julia/Yashish
        # self.initial_reward = self.RewardFunction.rewardX() # TODO Julia/Yashish

        # Then, we can use the initial placement to warm-start the optimization process.
        self.Optimizer.optimize_placement() # TODO Viraj/Yashish: the variables to optimize
        # [TODO] are the wound_points. These are parametric values for locations on the wound.
        #  [TODO] Wound should already be passed in by main.py:place_sutures.