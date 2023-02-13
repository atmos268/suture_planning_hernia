import numpy as np
import bezier
import scipy.interpolate as inter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

plt.clf()

# Sample Points [Will come from surgeon]
x = [0.0, 0.7, 1.0, 1.5, 2.1, 2.5, 3.0]
y = [0.0, -0.5, 0.5, 3.5, 1.8, 0.7, 1.3]
deg = 3

# Make B-Spline [which is a piecewise Beizer Spline] Curve with SciPy
curve = inter.make_interp_spline(x=x, y=y, k=deg, bc_type="clamped" if deg == 3 else None)

# Evaluate values on the Bezier Curves: input x value to get y value.
xi = 1.2
yi = curve(xi)

# First Derivative
d1 = curve.derivative()

# Second Derivative
d2 = d1.derivative()

# PLOTTING
import seaborn
seaborn.set()
# Bezier Curves
xis = np.linspace(0, 3, 100)
yis = [curve(xi) for xi in xis]
ax = plt.plot(xis, yis, color='blue')

# Evaluated Point (xi, yi)
l1 = plt.plot(
    xi, yi,
    marker="o", linestyle="None", color="red")

# Derivative Lines
slope_point1 = (xi + 1, yi + d1(xi))
slope_point2 = (xi - 1, yi - d1(xi))

x = [slope_point1[0], slope_point2[0]]
y = [slope_point1[1], slope_point2[1]]

plt.plot(x, y, color='pink')

plt.savefig('test_spline')