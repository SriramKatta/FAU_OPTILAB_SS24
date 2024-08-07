# Optimization for Engineers - Dr.Johannes Hild
# augmented Lagrangian descent

# Purpose: Find xmin to satisfy the stationarity condition for the augmented Lagrangian: norm(xmin - P(xmin - gradA(xmin)))<=eps
# And xmin also satisfies the feasibility condition norm(h(xmin))<=delta. 

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# h: objective class with methods .objective() and .gradient(), equality constraint.
# x0: column vector in R ** n(domain point)
# alpha0: real value, starting guess for Lagrangian multiplier for h. Default value: 0.
# eps: tolerance for termination. Default value: 1.0e-3
# delta: positive value in (0,eps), tolerance for feasibility. Default value: 1.0e-6.
# verbose: bool, if set to true, verbose information is displayed.

# Output Definition:
# xmin: column vector in R ** n(domain point)
# alphamin: real value, approximates Lagrangian multiplier.

# Required files:
# xmin_sub = projectedInexactNewtonCG(f, P, x_sub, eps_sub)
# myAugmentedObjective = augmentedLagrangianObjective(f, h, alpha, gamma)

# Test cases:
# A = np.array([[4, 0], [0, 2]], dtype=float)
# B = np.array([[0], [0]], dtype=float)
# C = 1
# myObjective = quadraticObjective(A, B, C)
# a = np.array([[0], [0]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# D = np.array([[2, 0], [0, 2]], dtype=float)
# E = np.array([[0], [0]], dtype=float)
# F = -1
# myConstraint = quadraticObjective(D, E, F)
# x0 = np.array([[1], [1]], dtype=float)
# alpha0 = 0
# eps = 1.0e-3
# delta = 1.0e-6
# [xmin, alphamin] = augmentedLagrangianDescent(myObjective, myBox, myConstraint, x0, alpha0, eps, delta, 1)
# should return xmin close to [[0], [1]] and alphamin close to -1


import numpy as np
import projectedInexactNewtonCG as PCG
import augmentedLagrangianObjective as AO


def matrnr():
    # set your matriculation number here
    matrnr = 23322375
    return matrnr


def augmentedLagrangianDescent(f, P, h, x0: np.array, alpha0=0, eps=1.0e-3, delta=1.0e-6, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if delta <= 0 or delta >= eps:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start augmentedLagrangianDescent...')

    countIter = 0


    x_k = P.project(x0)
    alphak = alpha0
    gammak = 10
    epsk = 1 / gammak
    deltak = 1 / (gammak ** 0.1)
    A = AO.augmentedLagrangianObjective(f, h, alphak, gammak)

    # INCOMPLETE CODE STARTS
    while np.linalg.norm(x_k - P.project(x_k - A.gradient(x_k))) > eps \
            or np.linalg.norm(h.objective(x_k)) > delta:
        countIter += 1
        x_k = PCG.projectedInexactNewtonCG(A, P, x0,epsk)
        if np.linalg.norm(h.objective(x_k)) <= deltak:
            alphak += gammak*h.objective(x_k)
            epsk = np.max((epsk/gammak, eps))
            deltak = np.max((deltak/(gammak**0.9),delta))
        else:
            gammak = np.max((10, gammak**0.5))*gammak
            epsk = 1/gammak
            deltak = 1/(gammak**0.1)
        A = AO.augmentedLagrangianObjective(f, h, alphak, gammak)        

    # INCOMPLETE CODE ENDS
    if verbose:
        stationarity = np.linalg.norm(x_k - P.project(x_k - A.gradient(x_k)))
        feasibility = np.linalg.norm(h.objective(x_k))
        print('augmentedLagrangianDescent terminated after ', countIter, ' steps with feasibility =', feasibility, ' and stationarity = ', stationarity)

    return [x_k, alphak]
