import numpy as np
import bezier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

plt.clf()

# Control Points
nodes1 = np.array([
    [0.0, 0.5, 1.0, 1.3],
    [0.0, 1.0, 0.0, 0.1],
])

nodes2 = np.array([
    [0.0, 0.25,  -0.55, 0.75, 1.0],
    [0.0, 2.0 , -2.0, 2.0 , 0.0],
])

# Make Bezier Curves
curve1 = bezier.Curve(nodes1, degree=3)
curve2 = bezier.Curve.from_nodes(nodes2)

# Intersections
intersections = curve1.intersect(curve2)
s_vals = np.asfortranarray(intersections[0, :])
points = curve1.evaluate_multi(s_vals)

# Evaluate values on the Bezier Curves: input from [0, 1]
c1e = curve1.evaluate(0.7)
c2e = curve2.evaluate(0.5)

# First Derivative
hodograph1 = curve1.evaluate_hodograph(0.55)
print(hodograph1)
# derivate =  np.array([hodograph[1], hodograph[0]]) / 10 # hodograph is normal for some reason
point_on_curve1 = curve1.evaluate(0.55)
slope_point_c1_1 = point_on_curve1 + hodograph1 / 3
slope_point_c1_2 = point_on_curve1 - hodograph1 / 3

hodograph2 = curve2.evaluate_hodograph(0.84)
# derivate =  np.array([hodograph[1], hodograph[0]]) / 10 # hodograph is normal for some reason
point_on_curve2 = curve2.evaluate(0.84)
slope_point_c2_1 = point_on_curve2 + hodograph2 / 3
slope_point_c2_2 = point_on_curve2 - hodograph2 / 3

# Parametric Polynomial representation [needs SymPy]
c1sym = curve1.to_symbolic()
c2sym = curve2.to_symbolic()

# PLOTTING
import seaborn
seaborn.set()
# Bezier Curves
ax = curve1.plot(num_pts=256, color='blue')
_ = curve2.plot(num_pts=256, ax=ax, color='orange')

# Intersections
lines = ax.plot(
    points[0, :], points[1, :],
    marker="o", linestyle="None", color="black")

# Evaluated Points
l1 = ax.plot(
    c1e[0], c1e[1],
    marker="o", linestyle="None", color="red")

# l2 = ax.plot(
#     slope_point[0], slope_point[1],
#     marker="o", linestyle="None", color="purple")
#
# l5 = ax.plot(
#     point_on_curve[0], point_on_curve[1],
#     marker="o", linestyle="None", color="yellow")

l3 = ax.plot(
    c2e[0], c2e[1],
    marker="o", linestyle="None", color="green")

l4 = ax.plot(
    c2e[0], c2e[1],
    marker="o", linestyle="None", color="green")

# Derivative Lines'
x = [slope_point_c1_1[0], slope_point_c1_2[0]]
y = [slope_point_c1_1[1], slope_point_c1_2[1]]

plt.plot(x, y, color='pink')

x = [slope_point_c2_1[0], slope_point_c2_2[0]]
y = [slope_point_c2_1[1], slope_point_c2_2[1]]

plt.plot(x, y, color='brown')

_ = ax.axis("scaled")
_ = ax.set_xlim(-0.125, 1.125)
_ = ax.set_ylim(-0.0625, 0.625)

plt.savefig('test')