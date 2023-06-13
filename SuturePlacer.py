import random

import DistanceCalculator
import RewardFunction
import Optimizer
import Constraints
import scipy.optimize as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

        self.c_lossMin = 0
        self.c_lossIdeal = 1
        self.c_lossVarCenter = 12
        self.c_lossVarInsExt = 6
        self.c_lossClosure = 15
        self.c_lossShear = 5

    def optimize(self, wound_points):
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(wound_points)
        self.RewardFunction.insert_dists = insert_dists
        self.RewardFunction.center_dists = center_dists
        self.RewardFunction.extract_dists = extract_dists

        self.Constraints.wound_points = wound_points
        start = wound_points[0]
        end = wound_points[-1]
        
        def jac(t):
            return optim.approx_fprime(t, final_loss)

        def final_loss(t):
            self.RewardFunction.insert_dists, self.RewardFunction.center_dists, self.RewardFunction.extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(t)    
            self.RewardFunction.wound_points = t
            self.RewardFunction.suture_points = list(zip(insert_pts, center_pts, extract_pts))
            return self.RewardFunction.final_loss(c_lossMin=self.c_lossMin, c_lossIdeal = self.c_lossIdeal, c_lossVarCenter = self.c_lossVarCenter, c_lossVarInsExt=self.c_lossVarInsExt, c_lossClosure = self.c_lossClosure, c_lossShear = self.c_lossShear)

        result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints(), options={"maxiter":200}, method = 'SLSQP', tol=1e-2, jac = jac)
        insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts = self.DistanceCalculator.calculate_distances(result.x)
        
        self.insert_pts = insert_pts
        self.center_pts = center_pts
        self.extract_pts = extract_pts


        # varun's printing code for shear/closure
        # print(final_loss(wound_points))

        result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints(), options={"maxiter":200}, method = 'SLSQP', tol = 1e-2, jac = jac)
        #print(self.DistanceCalculator.calculate_distances(result.x))
        # print(final_loss(result.x))
        # print('final wound points', result.x)
        plt.clf()
        save_intermittent_plots = False
        if save_intermittent_plots:
            self.DistanceCalculator.plot(result.x, "closure plot", plot_closure=True, save_fig='s1/' + str(len(wound_points)) + '_closure_' + str(random.randint(0, 1000000)))
            self.DistanceCalculator.plot(result.x, "shear plot", plot_shear=True, save_fig='s1/' + str(len(wound_points)) + '_shear_' + str(random.randint(0, 1000000)))

        # plt.clf()
        # plt.plot(np.arange(len(self.RewardFunction.closure_forces)), (self.RewardFunction.closure_forces - 1) ** 2, c='blue')
        # plt.plot(np.arange(len(self.RewardFunction.shear_forces)), (self.RewardFunction.shear_forces ** 2), c='orange')

        # plt.show()
        # closure = (closure_forces - 1) ** 2
        # shear = shear_forces ** 2

        result_x, result_y = self.DistanceCalculator.wound_parametric(result.x, 0)
        # print('result_x', result_x)
        # print('result_y', result_y)
        # return result_x, result_y

        return insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, result.x
    
    def place_sutures(self, sample_spline):
        # I want it to have an initial placement and then a forward pass thru to the reward so we can test our code.
        # Maybe this initial placement could be based on some smart heuristic to make optimization faster...
       
        # choosing 8 points along curve as placeholder
        # currently, we have chosen a set of unequal points to demostrate visually what the optimization is doing
        # in reality, we would likely warm-start with equally spaced points.
        num_sutures_initial = int(self.DistanceCalculator.initial_number_of_sutures(0, 1)) # heuristic
        print("NUM SUTURES INITIAL", num_sutures_initial)
        d = {}
        losses = {}
        points_dict = {}
        for num_sutures in range(max(2, int(0.5 * num_sutures_initial)), int(2 * num_sutures_initial)): # This should be (0.8 * heuristic to 1.4 * heuristic)
            print('NUM SUTURES: ', num_sutures)
            d[num_sutures] = {}
            heuristic = num_sutures
            best_loss = float('inf')
            wound_points = np.linspace(0, 1, num_sutures)
            insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = self.optimize(wound_points=wound_points)
            self.RewardFunction.insert_dists = insert_dists
            self.RewardFunction.center_dists = center_dists
            self.RewardFunction.extract_dists = extract_dists
            best_loss = self.RewardFunction.hyperLoss()
            print('loss: ', best_loss)
            print('closure loss', self.RewardFunction.lossClosureForce(1, 0))
            print('shear loss', self.RewardFunction.lossClosureForce(0, 1))
            print('center var loss', self.RewardFunction.lossVar(1, 0))
            print('InsExt var loss', self.RewardFunction.lossVar(0, 1))
            print('ideal loss', self.RewardFunction.lossIdeal())
            d[num_sutures]['loss'] = best_loss
            d[num_sutures]['closure loss'] = self.RewardFunction.lossClosureForce(1, 0)
            d[num_sutures]['shear loss'] = self.RewardFunction.lossClosureForce(0, 1)
            d[num_sutures]['var loss - center'] = self.RewardFunction.lossVar(1, 0)
            d[num_sutures]['var loss - ins/ext'] = self.RewardFunction.lossVar(0, 1)
            d[num_sutures]['ideal loss'] = self.RewardFunction.lossIdeal()
            b_insert_pts, b_center_pts, b_extract_pts, b_ts = insert_pts, center_pts, extract_pts, ts
            losses[best_loss] = num_sutures
            first_downward = True
            # while True:
            #     num_sutures += 1
            #     print('NUM SUTURES: ', num_sutures)
            #     wound_points = np.linspace(0, 1, num_sutures)
            #     insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = self.optimize(wound_points=wound_points)
            #     self.RewardFunction.insert_dists = insert_dists
            #     self.RewardFunction.center_dists = center_dists
            #     self.RewardFunction.extract_dists = extract_dists
            #     curr_loss = self.RewardFunction.hyperLoss()
            #     if curr_loss < best_loss:
            #         best_loss = curr_loss
            #         b_insert_pts, b_center_pts, b_extract_pts, b_ts = insert_pts, center_pts, extract_pts, ts
            #     if len(losses) >= 2 and curr_loss > max(losses[-1], losses[-2]):
            #         break
            #     losses.append(curr_loss)
            #     print("loss", curr_loss)
            # num_sutures = heuristic
            # while True:
            #     num_sutures -= 1
            #     print('NUM SUTURES: ', num_sutures)
            #     if num_sutures <= 1:
            #         break
            #     wound_points = np.linspace(0, 1, num_sutures)
            #     insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, ts = self.optimize(wound_points=wound_points)
            #     self.RewardFunction.insert_dists = insert_dists
            #     self.RewardFunction.center_dists = center_dists
            #     self.RewardFunction.extract_dists = extract_dists
            #     curr_loss = self.RewardFunction.hyperLoss()
            #     if curr_loss < best_loss:
            #         best_loss = curr_loss
            #         b_insert_pts, b_center_pts, b_extract_pts, b_ts = insert_pts, center_pts, extract_pts, ts
            #     if not first_downward and (len(losses) >= 2 and curr_loss > max(losses[0], losses[1])):
            #         break
            #     losses = [curr_loss] + losses
            #     first_downward = False
            #     print("loss", curr_loss)
            self.insert_pts = b_insert_pts
            self.center_pts = b_center_pts
            self.extract_pts = b_extract_pts
            print(losses)
            # self.DistanceCalculator.plot(b_ts, "Sutures placed for " + str(num_sutures) + " sutures. Total loss = " + str(best_loss), save_fig= str(sample_spline) + '/sutures_' + sample_spline + "_" + str(num_sutures) + "_" + str(random.randint(0, 10000000))) # put the spine name here
            # self.DistanceCalculator.plot(b_ts, "Closure force for " + str(num_sutures) + " sutures", save_fig= str(sample_spline) + '/closure_' + sample_spline +  "_" + str(num_sutures) + "_" + str(random.randint(0, 10000000)), plot_closure=True)
            # self.DistanceCalculator.plot(b_ts, "Shear force for " + str(num_sutures) + " sutures", save_fig= str(sample_spline) + '/shear_' + sample_spline +  "_" + str(num_sutures) + "_" + str(random.randint(0, 10000000)), plot_shear=True)

            rand_int = str(random.randint(0, 10000000))
            self.DistanceCalculator.plot(b_ts, "Sutures placed for " + str(num_sutures) + " sutures. Total loss = " + str(best_loss), save_fig= "clicking" + '/sutures_' + str(num_sutures) + "_" + rand_int) # put the spine name here
            self.DistanceCalculator.plot(b_ts, "Closure force for " + str(num_sutures) + " sutures", save_fig= "clicking" + '/closure_' + str(num_sutures) + "_" + rand_int, plot_closure=True)
            self.DistanceCalculator.plot(b_ts, "Shear force for " + str(num_sutures) + " sutures", save_fig= "clicking" + '/shear_' + str(num_sutures) + "_" + rand_int, plot_shear=True)

            print("plotting")
            points_dict[num_sutures] = b_ts
            
        print(d)
        #save losses dictionary as a csv file
        # dict_to_csv(d, sample_spline + "_losses")
        # #save points 
        # save_dict_to_file(points_dict, sample_spline +"_points.txt")

        dict_to_csv(d, "clicked_losses")
        #save points 
        save_dict_to_file(points_dict, "clicked_points.txt")
        return b_insert_pts, b_center_pts, b_extract_pts

def save_dict_to_file(dic, filename):
    f = open(filename,'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file():
    f = open('dict.txt','r')
    data=f.read()
    f.close()
    return eval(data)

def dict_to_csv(d, filename):
    df = pd.DataFrame(columns = ['num_sutures', 'loss', 'closure loss', 'shear loss', 'var loss'])
    d2 = {}
    for k,v in d.items():
        d2["num_sutures"] = k
        d2 = {**d2, **v}
        df = df.append(d2,
            ignore_index = True)
    df = df.sort_values(by=['loss'])
    df.to_csv(filename + ".csv", index=False) 

        
        
        # Varun: Eventually,  we'll have an overall reward that is the linear combination. [these two lines merge-conflicted, don't know which is right for now]
        # self.initial_reward = self.RewardFunction.rewardA(self.RewardFunction) # TODO Julia/Yashish
        # self.initial_reward = self.RewardFunction.rewardX() # TODO Julia/Yashish

        # Then, we can use the initial placement to warm-start the optimization process.
        # self.Optimizer.optimize_placement() # TODO Viraj/Yashish: the variables to optimize
        # [TODO] are the wound_points. These are parametric values for locations on the wound.
        #  [TODO] Wound should already be passed in by main.py:place_sutures.

        # we will feed optimized values in after merging with optimize code