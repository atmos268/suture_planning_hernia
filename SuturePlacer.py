import random

import DistanceCalculator
import RewardFunction
import Constraints
import scipy.optimize as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os


class SuturePlacer:
    def __init__(self, wound_width, mm_per_pixel):
        # This object should contain the optimizer, the spline curve, the image, etc., i.e. all of the relevant objects involved, as attributes.
        self.wound_width = wound_width
        self.mm_per_pixel = mm_per_pixel
        self.DistanceCalculator = DistanceCalculator.DistanceCalculator(self, self.wound_width, self.mm_per_pixel)
        self.RewardFunction = RewardFunction.RewardFunction(wound_width, self)
        self.Constraints = Constraints.Constraints(wound_width)
        self.Constraints.DistanceCalculator = self.DistanceCalculator

        self.b_insert_pts = []
        self.b_center_pts = []
        self.b_extract_pts = []
        self.b_loss = float('inf')

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

        result = optim.minimize(final_loss, wound_points, constraints = self.Constraints.constraints(), options={"maxiter":200}, method = 'SLSQP', tol = 1e-2, jac = jac)
        plt.clf()
        save_intermittent_plots = False
        if save_intermittent_plots:
            self.DistanceCalculator.plot(result.x, "closure plot", plot_closure=True, save_fig='s1/' + str(len(wound_points)) + '_closure_' + str(random.randint(0, 1000000)))
            self.DistanceCalculator.plot(result.x, "shear plot", plot_shear=True, save_fig='s1/' + str(len(wound_points)) + '_shear_' + str(random.randint(0, 1000000)))

        return insert_dists, center_dists, extract_dists, insert_pts, center_pts, extract_pts, result.x
    
    def place_sutures(self, sample_spline=None, save_figs=False):

        # make a folder to store info

        if save_figs:
            if not os.path.isdir("clicking"):
                os.mkdir('clicking')
            
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
            os.mkdir('clicking/' + dt_string)
            os.mkdir('clicking/' + dt_string + '/sutures')
            os.mkdir('clicking/' + dt_string + '/closure')
            os.mkdir('clicking/' + dt_string + '/shear')


        num_sutures_initial = int(self.DistanceCalculator.initial_number_of_sutures(0, 1)) # heuristic
        print("NUM SUTURES INITIAL", num_sutures_initial)
        d = {}
        losses = {}
        points_dict = {}
        for num_sutures in range(max(2, int(0.8 * num_sutures_initial)), int(2 * num_sutures_initial)): # This should be (0.8 * heuristic to 1.4 * heuristic)
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
     
            self.insert_pts = b_insert_pts
            self.center_pts = b_center_pts
            self.extract_pts = b_extract_pts

            if best_loss < self.b_loss:
                self.b_loss = best_loss
                self.b_insert_pts = b_insert_pts
                self.b_center_pts = b_center_pts
                self.b_extract_pts = b_extract_pts

            print(losses)

            if save_figs:
            
                self.DistanceCalculator.plot(b_ts, "Number of Sutures: " + str(num_sutures) + ". Total loss: " + str(best_loss), save_fig=str(num_sutures), plot_type='sutures',save_dir='clicking/'+dt_string)
                self.DistanceCalculator.plot(b_ts, "Closure force for " + str(num_sutures) + " sutures", save_fig= str(num_sutures), plot_type='closure', save_dir='clicking/'+dt_string)
                self.DistanceCalculator.plot(b_ts, "Shear force for " + str(num_sutures) + " sutures", save_fig=str(num_sutures), plot_type='shear', save_dir='clicking/'+dt_string)

            points_dict[num_sutures] = b_ts

        dict_to_csv(d, "clicked_losses")
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