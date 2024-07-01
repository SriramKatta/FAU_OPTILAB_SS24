# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import boxObjective as BO
import noHessianObjective as NO
import multidimensionalObjective as MO
import projectionInBox as PB
import projectedBacktrackingSearch as PS
import projectedInexactNewtonCG as PCG

p = np.array([[0], [1]])
myObjective = SO.simpleValleyObjective(p)
a = np.array([[-1], [-1]])
b = np.array([[2], [2]])
myBox = PB.projectionInBox(a, b)
x = np.array([[-1.01], [1]])
d = np.array([[1], [1]])
sigma = 1.0e-3
t = PS.projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, 1)
te = 1
if t == te:
    print('Check 01 okay')
else:
    raise Exception('Your projected backtracking search is not recognizing t = 1 as valid starting point.')


p = np.array([[0], [1]])
myObjective = SO.simpleValleyObjective(p)
a = np.array([[-2], [1]])
b = np.array([[2], [2]])
myBox = PB.projectionInBox(a, b)
x = np.array([[1], [1]])
d = 4*np.array([[-1.99], [0]])
sigma = 0.5
t = PS.projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, 1)
te = 0.125
if t == te:
    print('Check 02 okay')
else:
    raise Exception('Your projected backtracking search is not backtracking correctly.')

A = -np.eye(3)
B = np.array([[-1.5],[-1.5],[-1.5]], dtype=float)
C = 1
aa = np.array([[1], [1], [1]])
bb = np.array([[2], [3], [4]])
myBox = PB.projectionInBox(aa, bb)
myObjective = BO.boxObjective(A, B, C, aa, bb)
x0 = np.array([[1], [1], [3]], dtype=float)
eps = 1.0e-3
xmin = PCG.projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
xe = np.array([[2], [3], [4]], dtype=float)
if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 03 is okay')
else:
    raise Exception('Your projectedInexactNewtonCG is not curvature failure at the very first iteration correctly.')

myObjective = NO.noHessianObjective()
x0 = np.array([[0.15], [2.0]], dtype=float)
eps = 1.0e-3
a = np.array([[-2], [1]])
b = np.array([[2], [2]])
myBox = PB.projectionInBox(aa, bb)
xmin = PCG.projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
xe = np.array([[1], [1]], dtype=float)
if np.linalg.norm(xmin-xe) < 1.0e-2:
    print('Check 04 is okay')
else:
    raise Exception('Your projectedInexactNewtonCG is not working for Hessian free objective class.')

myObjective = MO.multidimensionalObjective()
a = np.array([[1], [1], [1], [1], [-1], [-1], [-1], [-1]], dtype=float)
b = np.array([[2], [2], [2], [2], [2], [2], [2], [2]], dtype=float)
myBox = PB.projectionInBox(a, b)
x0 = np.array([[1], [1], [1], [1], [2], [2], [2], [2]], dtype=float)
eps = 1.0e-6
xmin = PCG.projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
xe = np.array([[1], [1], [1], [1], [-0.40749], [0.01116], [0.04147], [-0.01356]], dtype=float)
if np.linalg.norm(xmin-xe) < 1.0e-2:
    print('Check 05 is okay')
else:
    raise Exception('Your projectedInexactNewtonCG is not working for higher dimensions.')

if PS.matrnr() == 0:
    raise Exception('Please set your matriculation number in projectedBacktrackingSearch.py!')
elif PCG.matrnr() == 0:
    raise Exception('Please set your matriculation number in projectedInexactNewtonCG.py!')
else:
    print('Check completed.')

print('\nWe finished now projectedInexactNewtonCG, which is a q-superlinear descent algorithm for box constraints, that converges globally for unconstrained problems.')
print('It does not require Hessian information or a linear system solver like CG.')
print('We will implement an algorithm for data fitting in the next LAB.')
