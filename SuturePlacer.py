import DistanceCalculator
import RewardFunction

class SuturePlacer():
    def __init__(self):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = 0.2 # TODO Varun: this is a random #, lookup
        self.DistanceCalculator = DistanceCalculator(self.wound_width)
        self.RewardFunction = RewardFunction
        self.Optimizer = None # cvxpy? Feel free to make your own file with a class for Optimizer if you want one.

    def place_sutures(self):
        # I want it to have an initial placement and then a forward pass thru to the reward so we can test our code.
        # Maybe this initial placement could be based on some smart heuristic to make optimization faster...
        wound_points = None # TODO Harshika/Viraj: Initial Placement, can put some placeholder here
        insert_dists, center_dists, extract_dists = self.DistanceCalculator.calculate_distances(wound_points) # TODO Harshika/Viraj
        self.RewardFunction.insert_dists = insert_dists
        self.RewardFunction.center_dists = center_dists
        self.RewardFunction.extract_dists = extract_dists
        self.initial_reward = self.RewardFunction.rewardX() # TODO Julia/Yashish

        # Then, we can use the initial placement to warm-start the optimization process.
        self.Optimizer.optimize_placement() # TODO Viraj/Yashish: the variables to optimize
        # [TODO] are the wound_points. These are parametric values for locations on the wound.
        #  [TODO] Wound should already be passed in by main.py:place_sutures.