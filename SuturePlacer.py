import DistanceCalculator
import RewardFunction
import Optimizer
import Constraints
import scipy.optimize as optim

class SuturePlacer:
    def __init__(self):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = 0.3 # TODO Varun: this is a random #, lookup
        self.DistanceCalculator = DistanceCalculator.DistanceCalculator(self.wound_width, self)
        self.RewardFunction = RewardFunction.RewardFunction(self.wound_width, self)
        self.Constraints = Constraints.Constraints(self)
        self.Constraints.DistanceCalculator = self.DistanceCalculator

        # NB: Viraj: added a class in order to allow code to run
        self.Optimizer = Optimizer.Optimizer() # cvxpy? Feel free to make your own file with a class for Optimizer if you want one.

    def place_sutures(self):
        # I want it to have an initial placement and then a forward pass thru to the reward so we can test our code.
        # Maybe this initial placement could be based on some smart heuristic to make optimization faster...
       
        # choosing 11 equally spaced points along curve as placeholder
        wound_points = [0.0, 0.05, 0.25, 0.45, 0.65, 0.75, 0.95, 1.] # TODO Harshika/Viraj: Initial Placement, can put some placeholder here
        self.Constraints.wound_points = wound_points
        self.DistanceCalculator.plot(wound_points)
        print("Initial wound points", wound_points)
        start = wound_points[0]
        end = wound_points[-1]
        
        def final_loss(t):
            self.RewardFunction.insert_dists, self.RewardFunction.center_dists, self.RewardFunction.extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)    
            self.RewardFunction.wound_points = t
            return self.RewardFunction.final_loss(a = 1, b = 1, c = 1)

        print(final_loss(wound_points))

        result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints())
        #print(self.DistanceCalculator.calculate_distances(result.x))
        print(final_loss(result.x))
        print('final wound points', result.x)
        self.DistanceCalculator.plot(result.x, plot_closure=True)
        self.DistanceCalculator.plot(result.x, plot_shear=True)


        result_x, result_y = self.DistanceCalculator.wound_parametric(result.x, 0)
        print(result_x)
        print(result_y)
        return result_x, result_y





        # Then, we can use the initial placement to warm-start the optimization process.
        # self.Optimizer.optimize_placement(grads1, grads2) # TODO Viraj/Yashish: the variables to optimize
        # [TODO] are the wound_points. These are parametric values for locations on the wound.
        #  [TODO] Wound should already be passed in by main.py:place_sutures.