# Optimization for Engineers - Dr.Johannes Hild
# Mock Homework to check setup
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your setup is not working correctly.')
print('First we check if the math package numpy is installed.')

import numpy as np

X = np.power(2, 3)
Y = 2**3
if X == Y:
    print('=> numpy seems to work.\n')

print('Next we check if the function definitions in modelObjective.py are available.')

import modelObjective as MO

p = np.array([[3], [2], [16]])
x = np.array([[0], [0], [-0.5]])
myObjective = MO.modelObjective(p)
f = myObjective.objective(x)
fg = myObjective.gradient(x)
fh = myObjective.hessian(x)
xdata = myObjective.getXData()
fdata = myObjective.getFData()

print('The modelObjective returns')
print(f)
print('at x and the gradient is')
print(fg)
print('and the hessian is')
print(fh,'\n')

print('The modelObjective measure points are')
print(xdata)
print('with measure results')
print(fdata,'\n')

import incompleteCholesky as IC
print('An important feature for solving linear systems of equations with spd matrices is the decomposition of A into L @ L.T')
print('We use imcompleteCholesky to decompose the Hessian of the modelProblem into L @ L.T, with the triangle matrix L.')
alpha = 0
delta = 0
L = IC.incompleteCholesky(fh, alpha, delta)
print('=> Result: The Hessian ')
print(fh)
print('is decomposed into L @ L.T with L = ')
print(L,'\n')

import LLTSolver as LLT

print('We can use this triangle matrix L to quickly solve the Newton step condition, i.e. Hessian @ d = - Gradient. ')
print('We do this by calling LLTSolver.')
print('=> Result: The Newton step is d = ')
d = LLT.LLTSolver(L, -fg)
print(d)

print('\nSo we already have a toolset to compute an exact Newton step for the modelProblem. We will expand and improve this toolset in the upcoming LABs.')