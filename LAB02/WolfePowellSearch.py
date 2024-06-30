# Optimization for Engineers - Dr.Johannes Hild
# Wolfe-Powell line search

# Purpose: Find t to satisfy f(x+t*d)<=f(x) + t*sigma*gradf(x).T@d and gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set, such that t satisfies both Wolfe - Powell conditions

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=1

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.2], [1]])
# d = np.array([[0.1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=16

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-0.2], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.25

import numpy as np

def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr

def WolfePowellSearch(f, xk: np.array, dk: np.array, sigma=0.25, rho=0.5, verbose=0):
    fx = f.objective(xk)
    gradx = f.gradient(xk)
    descent = gradx.T @ dk

    if descent >= 0:
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5:
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1:
        raise TypeError('range of rho is wrong!')

    if verbose:
        print('Start WolfePowellSearch...')

    def W1(t):
        xt = xk + t * dk
        return f.objective(xt) <= fx + t * sigma * descent

    def W2(t):
        xt = xk + t * dk
        return f.gradient(xt).T @ dk >= rho * descent

    t = 1.0

    # Backtracking
    if not W1(t):
        t *= 0.5
        while not W1(t):
            t *= 0.5
        t_minus = t
        t_plus = 2 * t
    # Fronttracking
    elif W2(t):
        return t
    else:
        t *= 2
        while W1(t):
            t *= 2
        t_minus = 0.5 * t
        t_plus = t

    t = t_minus

    # Refining
    while not W2(t):
        t = (t_minus + t_plus) / 2
        if W1(t):
            t_minus = t
        else:
            t_plus = t

    if verbose:
        xt = xk + t * dk
        fxt = f.objective(xt)
        gradxt = f.gradient(xt)
        print('WolfePowellSearch terminated with t=', t)
        print('Wolfe-Powell: ', fxt, '<=', fx + t * sigma * descent, ' and ', gradxt.T @ dk, '>=', rho * descent)

    return t

