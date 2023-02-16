import numpy as np
import pickle
import scipy.interpolate as inter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

plt.clf()

def generate_sample_spline():
    # Sample Points [Will come from surgeon]
    x = [0.0, 0.7, 1.0, 1.5, 2.1, 2.5, 3.0]
    y = [0.0, -0.5, 0.5, 3.5, 1.8, 0.7, 1.3]
    deg = 3

    # Make B-Spline [which is a piecewise Beizer Spline] Curve with SciPy
    curve = inter.make_interp_spline(x=x, y=y, k=deg, bc_type="clamped" if deg == 3 else None)

    return curve