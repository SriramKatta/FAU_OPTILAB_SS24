# Optimization for Engineers - Dr.Johannes Hild
# projection in box constraints 

# Purpose: if x[i, 0] is bigger or smaller than the bounds, x[i, 0] is set to the closest boundary

# Class parameters:
# a: column vector in R ** n, lower bounds for x, must be smaller than b by at least eps in each component.
# b: column vector in R ** n, upper bounds for x  
# eps: nonnegative value, tolerance for accepting being active. Default value: 1.0e-6.

# Input Definition:
# x: column vector in R ** n(domain space)

# Output Definition:
# projectedX: column vector in R ** n, satisfies box constraints
# activeIndexSet: list of indices, collected indices mark x[i, 0] components with projectedX[i, 0]-a[i, 0] <= eps
# or projectedX[i, 0] - b[i, 0] >= -eps

# Required files:
# < none >

# Test cases:
# a = np.array([[0], [2], [0.9], [0], [0]], dtype=float)
# b = np.array([[2], [3], [3], [0.5], [1.1]], dtype=float)
# eps = 0.2
# myBox = projectionInBox(a, b, eps)
# x = np.array([[1], [1], [1], [1], [1]], dtype=float)
# myBox.project(x) should return [[1], [2], [1], [0.5], [1]]
# myBox.activeIndexSet(x) should return [1, 2, 3, 4]

# a = np.array([[0], [0], [0.9], [-1]], dtype=float)
# b = np.array([[2], [3], [3], [-0.25]], dtype=float)
# eps = 0.2
# myBox = projectionInBox(a, b, eps)
# x = np.array([[1], [1], [1], [-0.51]], dtype=float)
# myBox.project(x) should return [[1], [1], [1], [-0.5]]
# myBox.activeIndexSet(x) should return [2]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


class projectionInBox:
  
    def __init__(self, a: np.array, b: np.array, eps=1.0e-6):
        self.a = a
        self.b = b
        self.eps = eps
        if np.min(b - a) < eps:
            raise TypeError('a and b forming box is degenerate.')
        
    def project(self, x: np.array):
        n = x.shape[0]
        projectedX = x.copy()
        for i in range(n):
            if x[i, 0] < self.a[i, 0]:
                projectedX[i, 0] = self.a[i, 0]
          
            if x[i, 0] > self.b[i, 0]:
                projectedX[i, 0] = self.b[i, 0]
                
        return projectedX
    
    def activeIndexSet(self, x: np.array):
        n = x.shape[0]
        myList = []
        for i in range(n):
            if x[i, 0] <= self.a[i, 0]+self.eps or x[i, 0] >= self.b[i, 0]-self.eps:
                myList.append(i)

        return myList
