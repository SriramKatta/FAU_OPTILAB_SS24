# Optimization for Engineers - Dr.Johannes Hild
# quadratic box objective that is not defined outside its box
# Do not change this file

# n-dimensional quadratic function mapping x -> 0.5*x'*A*x + b'*x +c

# Class parameters:
# A: real valued matrix nxn
# b: column vector in R^n
# c: real number
# aa: lower bounds of the box, column vector in R^n
# bb: upper bounds of the box, column vector in R^n

# Input Definition:
# x: vector in R**n (domain space)

# Output Definition:
# objective(): real number, evaluation at x
# gradient(): vector in R**n, evaluation of gradient wrt x
# hessian(): matrix in R**nxn, evaluation of hessian wrt x

# Required files:
# < none >

# Test cases:
# A = np.eye(2)
# b = np.ones((2,1))
# c = 1
# aa = -np.ones((2,1))
# bb =  np.ones((2,1))
# myObjective = quadraticObjective(A,b,c,aa,bb)
# y = myObjective.objective(b)
# should return y = 4

# grad = myObjective.gradient(b)
# should return grad = [[2],[2]]

# hess = myObjective.hessian(b)
# should return hess = [[1, 0],[0, 1]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


class boxObjective:

    def __init__(self, A: np.array, b: np.array, c: float, aa: np.array, bb: np.array):
        self.A = A
        self.b = b
        self.c = c
        self.aa = aa
        self.bb = bb

    def isFeasible(self, x: np.array):
        n = x.shape[0]
        for i in range(n):
            if x[i, 0] < self.aa[i, 0]:
                return False

            if x[i, 0] > self.bb[i, 0]:
                return False
        return True


    def objective(self, x: np.array):
        if self.isFeasible(x):
            f = 0.5 * (x.T @ (self.A @ x)) + self.b.T @ x + self.c
            return f
        else:
            raise TypeError('boxObjective is not defined outside the box!')


    def gradient(self, x: np.array):
        if self.isFeasible(x):
            g = self.A @ x + self.b
            return g
        else:
            raise TypeError('boxObjective is not defined outside the box!')


    def hessian(self, x: np.array):
        if self.isFeasible(x):
            h = self.A
            return h
        else:
            raise TypeError('boxObjective is not defined outside the box!')