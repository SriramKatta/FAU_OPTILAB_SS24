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
from copy import copy


def matrnr():
    # set your matriculation number here
    matrnr = 23322375
    return matrnr


def WolfePowellSearch(f, x: np.array, d: np.array, sigma=1.0e-3, rho=1.0e-2, verbose=0):
    fx = f.objective(x)
    gradx = f.gradient(x)
    descent = gradx.T @ d

    if descent >= 0:
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5:
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1:
        raise TypeError('range of rho is wrong!')

    if verbose:
        print('Start WolfePowellSearch...')

    def WP1(ft, s):
        isWP1 = ft <= fx + s*sigma*descent
        return isWP1

    def WP2(gradft: np.array):
        isWP2 = gradft.T @ d >= rho*descent
        return isWP2

    # INCOMPLETE CODE STARTS
    if gradx.T @ d >= 0:
        raise TypeError('descent direction check failed!')
    
    t = 1    
    if WP1(f.objective(x+t*d), t) == False:
        t = t/2
        while WP1(f.objective(x + t*d), t) == False:
            t = t/2
        t_minus = copy(t)
        t_plus = 2*t
    elif WP2(f.gradient(x + t*d)) == True:
        t_star = copy(t)
        return t_star
    else :
        t = 2*t
        while WP1(f.objective(x+t*d), t) == True:
            t = 2*t
        t_minus = t/2
        t_plus = copy(t)

    t = t_minus
    while WP2(f.gradient(x + t*d)) == False:
        t = (t_minus + t_plus)/2
        if WP1(f.objective(x + t*d),t) == True:
            t_minus = t
        else:
            t_plus = t
    t_star = copy(t_minus)
    # INCOMPLETE CODE ENDS

    if verbose:
        xt = x + t * d
        fxt = f.objective(xt)
        gradxt = f.gradient(xt)
        print('WolfePowellSearch terminated with t=', t)
        print('Wolfe-Powell: ', fxt, '<=', fx+t*sigma*descent, ' and ', gradxt.T @ d, '>=', rho*descent)

    return t_star
