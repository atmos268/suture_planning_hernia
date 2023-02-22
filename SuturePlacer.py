import DistanceCalculator
import RewardFunction
import Optimizer

class SuturePlacer:
    def __init__(self):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = 0.2 # TODO Varun: this is a random #, lookup
        self.DistanceCalculator = DistanceCalculator.DistanceCalculator(self.wound_width)
        self.RewardFunction = RewardFunction.RewardFunction()

        # NB: Viraj: added a class in order to allow code to run
        self.Optimizer = Optimizer.Optimizer() # cvxpy? Feel free to make your own file with a class for Optimizer if you want one.

    def place_sutures(self):
        # I want it to have an initial placement and then a forward pass thru to the reward so we can test our code.
        # Maybe this initial placement could be based on some smart heuristic to make optimization faster...
       
        # choosing 11 equally spaced points along curve as placeholder
        wound_points = [0.1*i for i in range(11)] # TODO Harshika/Viraj: Initial Placement, can put some placeholder here
        print("Initial wound points", wound_points)

        num_iters = 100
        lr = 0.1

        for i in range(num_iters):
            insert_dists, center_dists, extract_dists = self.DistanceCalculator.calculate_distances(wound_points) # TODO Harshika/Viraj
            self.RewardFunction.insert_dists = insert_dists
            self.RewardFunction.center_dists = center_dists
            self.RewardFunction.extract_dists = extract_dists

            # Varun: Eventually,  we'll have an overall reward that is the linear combination. [these two lines merge-conflicted, don't know which is right for now]
            #self.initial_reward = self.RewardFunction.rewardA() # TODO Julia/Yashish
            self.initial_reward = self.RewardFunction.lossX_torch() # TODO Julia/Yashish
            grads1, grads2 = self.RewardFunction.loss_gradient(), self.DistanceCalculator.grads(wound_points)
            final_grad = (grads2["center"] @ grads1["center"]) #+ (grads1["insert"] * grads2["insert"]) + (grads1["extract"] * grads2["extract"])
            wound_points = [wound_points[i] - lr * final_grad[i] for i in range(len(wound_points))]
            wound_points = [float(wound_points[i]) for i in range(len(wound_points))]
        print("Final wound points", wound_points)
        self.DistanceCalculator.plot(wound_points)
            


        # Then, we can use the initial placement to warm-start the optimization process.
        # self.Optimizer.optimize_placement(grads1, grads2) # TODO Viraj/Yashish: the variables to optimize
        # [TODO] are the wound_points. These are parametric values for locations on the wound.
        #  [TODO] Wound should already be passed in by main.py:place_sutures.