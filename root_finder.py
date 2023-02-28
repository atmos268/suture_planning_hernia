"""
Simple zero/root finder for functions with one variable.

Finds all the zeroes within a range of values, using user-defined searchstep and tolerance.

Utilizes the scipy.optimize.fsolve: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html

LICENCE: MIT
"""

import numpy as np
import scipy
from scipy.optimize import fsolve
from matplotlib import pyplot as plt


class RootFinder:
    def __init__(self, start, stop, step=0.01, root_dtype="float64", xtol=1e-9):

        self.start = start
        self.stop = stop
        self.step = step
        self.xtol = xtol
        self.roots = np.array([], dtype=root_dtype)

    def add_to_roots(self, x):

        if (x < self.start) or (x > self.stop):
            return  # outside range
        if any(abs(self.roots - x) < self.xtol):
            return  # root already found.

        self.roots = np.append(self.roots, x)

    def find(self, f, *args):
        current = self.start

        for x0 in np.arange(self.start, self.stop + self.step, self.step):
            if x0 < current:
                continue
            x = self.find_root(f, x0, *args)
            if x is None:  # no root found.
                continue
            current = x
            self.add_to_roots(x)

        return self.roots

    def find_root(self, f, x0, *args):

        x, _, ier, _ = fsolve(f, x0=x0, args=args, full_output=True, xtol=self.xtol)
        if ier == 1:
            return x[0]
        return None


def f(u, V=90, ell=5):
    # Example from https://stackoverflow.com/q/14878110/3015186
    w = np.sqrt(V ** 2 - u ** 2)

    jl = scipy.special.jn(ell, u)
    jl1 = scipy.special.yn(ell - 1, u)
    kl = scipy.special.kn(ell, w)
    kl1 = scipy.special.kn(ell - 1, w)

    return jl / (u * jl1) + kl / (w * kl1)


if __name__ == "__main__":

    r = RootFinder(1, 20, 0.01)
    args = (90, 5)
    roots = r.find(f, *args)
    print("Roots: ", roots)

    # plot results
    u = np.linspace(1, 20, num=600)
    fig, ax = plt.subplots()
    ax.plot(u, f(u, *args))
    ax.scatter(roots, f(np.array(roots), *args), color="r", s=10)
    ax.grid(color="grey", ls="--", lw=0.5)
    plt.show()