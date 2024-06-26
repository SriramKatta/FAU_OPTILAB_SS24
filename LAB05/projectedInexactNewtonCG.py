# Optimization for Engineers - Dr.Johannes Hild
# projected inexact Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = projectedHessApprox(f, P, x, d) from projectedHessApprox.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
import projectedBacktrackingSearch as PB
import projectedHessApprox as PHA

def matrnr():
    # set your matriculation number here
    matrnr =  23322375
    return matrnr


def projectedInexactNewtonCG(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedInexactNewtonCG...')

    # INCOMPLETE CODE STARTS
    countIter = 0
    x_k = P.project(x0)
    eps_check = np.linalg.norm(x_k - P.project(x_k - f.gradient(x_k)))
    diff = np.sqrt(np.linalg.norm(x_k - P.project(x_k - f.gradient(x_k))))
    eta_k = np.min((0.5, diff)) * eps_check
    curvaturefail = False
    while eps_check > eps:
        countIter = countIter + 1
        x_j = x_k
        r_j = f.gradient(x_k)
        d_j = -r_j
        while np.linalg.norm(r_j) > eta_k:
            d_a = PHA.projectedHessApprox(f, P, x_k, d_j)
            rho_j = d_j.T @ d_a
            if rho_j <= eps * np.square(np.linalg.norm(d_j)):
                curvaturefail = True
                break
            t_j = np.square(np.linalg.norm(r_j)) / rho_j
            x_j = x_j + t_j * d_j
            r_old = r_j
            r_j = r_old + t_j * d_a
            beta_j = np.square(np.linalg.norm(r_j)/np.linalg.norm(r_old))
            d_j = -r_j + beta_j * d_j
        if curvaturefail:
            d_k = -f.gradient(x_k)
        else :
            d_k = x_j - x_k
        tk = PB.projectedBacktrackingSearch(f, P, x_k, d_k)
        x_k = P.project(x_k+ tk*d_k)
        eps_check = np.linalg.norm(x_k - P.project(x_k - f.gradient(x_k)))
        diff = np.sqrt(np.linalg.norm(x_k - P.project(x_k - f.gradient(x_k))))
        eta_k = np.min((0.5, diff)) * eps_check
    # INCOMPLETE CODE ENDS
    if verbose:
        gradx = f.gradient(x_k)
        stationarity = np.linalg.norm(x_k - P.project(x_k - gradx))
        print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity))

    return x_k

