import numpy as np
import pickle
import scipy.interpolate as inter
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize
import sympy.functions.special.bsplines as sympy_bsplines
from sympy import interpolating_spline
from sympy.abc import x
matplotlib.use('TkAgg')
from root_finder import RootFinder

plt.clf()

def generate_sample_spline():
    # Sample Points [Will come from surgeon]
    xs = [0.0, 0.7, 1.0, 1.5, 2.1, 2.5, 3.0]
    ys = [0.0, -0.5, 0.5, 3.5, 1.8, 0.7, 1.3]
    deg = 3

    # Make B-Spline [which is a piecewise Beizer Spline] Curve with SciPy
    curve = inter.make_interp_spline(x=xs, y=ys, k=deg)
    x_eval = np.linspace(0, 3, 500)
    fig, ax = plt.subplots()
    ax.plot(x_eval, [curve(x_) for x_ in x_eval], color='orange')
    ax.scatter(xs, ys)

    # SciPy splprep curve
    x_eval = np.linspace(0, 1, 500)
    tck, u = inter.splprep([xs, ys], s=0)
    new_points = inter.splev(x_eval, tck)
    ax.plot(new_points[0], new_points[1], color='blue')

    # Sympy's fit
    # sympy_spline = interpolating_spline(deg, x, xs, ys)
    # ax.plot(x_eval, [sympy_spline.subs(x, x_) for x_ in x_eval], color='blue')
    # # y_s = sympy_spline(0.5)

    plt.show()

    return curve

def generate_parametric_bezier():
    import numpy as np
    from scipy import interpolate

    # x = np.array([-1, 0, 2])
    # y = np.array([0, 2, 0])

    x = [0.0, 0.7, 1.0, 1.5, 2.1, 2.5, 3.0, 1.5, 3.0, 2.7]
    y = [0.0, -0.5, 0.5, 3.5, 1.8, 0.7, 1.3, 5.0, 4.0, 3.5]

    old_x = x
    old_y = y

    tck, u = inter.splprep([x, y], s=0)

    t, c, k = tck

    new_points = inter.splev(np.linspace(0,1,500), tck, der=0)
    new_points_der = inter.splev(np.linspace(0,1,500), tck, der=1)

    # spline = inter.BSpline(t, c, k)

    # spl = inter.splrep(x, y, k=3)

    def spline_derivative_x(i, tck):
        return inter.splev(i, tck, der=1)[0]

    def spline_derivative_y(i, tck):
        return inter.splev(i, tck, der=1)[1]

    root_finder = RootFinder(0, 1)
    roots = root_finder.find(spline_derivative_x, tck)
    roots_on_curve = roots * 3

    root_points = inter.splev(roots, tck)

    new_x = [0.0, 0.7, 1.0, 1.5, 2.1, 2.5, 3.0, 3.07]
    new_y = [0.0, -0.5, 0.5, 3.5, 1.8, 0.7, 1.3, 1.73]

    new_spline = inter.make_interp_spline(x=new_x, y=new_y, k=3, bc_type=([(1, -0.5)], [(1, 1e1)]))


    # spl_der = inter.splder(spl)
    # spl_der_para_x = inter.splder((t, c[0], k))
    # spl_der_para_y = inter.splder((t, c[1], k))

    fig, ax = plt.subplots()
    ax.scatter(root_points[0], root_points[1], c='k')
    ax.plot(np.linspace(0, 3.07, 500), [new_spline(x) for x in np.linspace(0, 3.07, 500)], c='purple')
    ax.plot(x, y, 'ro')
    # ax.plot(np.linspace(0, 3, 500), [spline_derivative_x(i, tck) for i in np.linspace(0, 1, 500)], 'b')
    ax.plot(new_points[0], new_points[1], 'g')
    # ax.plot(new_points[0], new_points_der[1], 'r')
    plt.show()



    ts = np.linspace(0, 1, 500)

    a = u(0.5)
    # ax.plot()

    print('knots: ', list(tck[0]))
    print('coefficients x: ', list(c[0]))
    print('coefficients y: ', list(c[1]))
    print('degree: ', tck[2])
    print('parameter: ', list(u))

if __name__ == '__main__':
    # generate_parametric_bezier()
    generate_sample_spline()