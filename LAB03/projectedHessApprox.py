# Optimization for Engineers - Dr.Johannes Hild
# projected Hessian Approximation

# Purpose: Approximates Hessian times direction with central differences and subject to projection

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# delta: tolerance for termination. Default value: 1.0e-6
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# dH: Hessian times direction, column vector in R ** n

# Required files:
# < none >

# Test cases:
# a = np.array([[-2], [2]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])

# dH = projectedHessApprox(myObjective, myBox, x, d)
# should return dH = [[1.55491],[0]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


def projectedHessApprox(f, P, x: np.array, d: np.array, delta=1.0e-6, verbose=0):

    if verbose:
        print('Start projectedHessApprox...')

    xp = P.project(x)
    A = P.activeIndexSet(xp)
    dr = d.copy()
    dr[A, :] = 0
    norm_d = np.linalg.norm(dr)
    if norm_d > 0:
        dH = 0.5*norm_d/delta*(f.gradient(x+delta/norm_d*dr)-f.gradient(x-delta/norm_d*dr))
        dH[A, :] = d[A, :]
    else:
        dH = d

    if verbose:
        print('projectedHessApprox terminated with dH=', dH)

    return dH
