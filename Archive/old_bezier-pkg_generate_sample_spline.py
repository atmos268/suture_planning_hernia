import numpy as np
import bezier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

plt.clf()

def generate_sample_spline(savefig):
    # Control Points
    nodes = np.asfortranarray([
        [0.0, 0.7, 1.0, 1.5, 2.1, 2.5, 3.0],
        [0.0, -0.5, 0.5, 3.5, 1.8, 0.7, 1.3],
    ])

    # Make Bezier Curve
    curve = bezier.Curve.from_nodes(nodes)

    curve_symbolic = curve.to_symbolic()

    # Plotting
    import seaborn
    seaborn.set()
    # Bezier Curves
    ax = curve.plot(num_pts=256, color='blue')

    # _ = ax.axis("scaled")
    # _ = ax.set_xlim(-0.125, 1.125)
    # _ = ax.set_ylim(-0.0625, 0.625)

    if savefig:
        plt.savefig('sample_spline')

    return curve, curve_symbolic