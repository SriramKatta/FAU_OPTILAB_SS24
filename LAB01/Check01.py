# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import quadraticObjective as QO
import bananaValleyObjective as BO
import PrecCGSolver as PCG
import NewtonDescent as ND

print('Checking PrecCGSolver...')
A = np.array([[4, 1, 0], [1, 7, 0], [0, 0, 3]], dtype=float)
b = np.array([[10], [16], [6]], dtype=float)
delta = 1.0e-6
x = PCG.PrecCGSolver(A, b, delta, 1)
xe = np.array([[2], [2], [2]])

if np.linalg.norm(x - xe) < 1.0e-3:
    print('Check 01 okay')
else:
    raise Exception('Your PrecCGSolver is not working correctly for a simple example.')

A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6], [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
x = PCG.PrecCGSolver(A, b, delta, 1)
xe = np.array([[1], [0], [2], [0], [3]])

if np.linalg.norm(x - xe) < 1.0e-3:
    print('Check 02 okay')
else:
    raise Exception('Your PrecCGSolver is not working correctly for other dimensions.')

A = np.array([[2, 0], [0, -2]], dtype=float)
b = np.array([[3], [1]], dtype=float)
delta = 1.0e-6
x = PCG.PrecCGSolver(A, b, delta, 1)
xe = np.array([[1.5], [-0.5]])

if np.linalg.norm(x - xe) < 1.0e-3:
    print('Check 03 okay')
else:
    raise Exception('Your PrecCGSolver is not using the preconditioning correctly.')

A = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -2]], dtype=float)
b = np.array([[1], [1], [1]], dtype=float)
myObjective = QO.quadraticObjective(A, b, 1)
x0 = np.array([[0], [0], [0]])
xmin2 = ND.NewtonDescent(myObjective, x0, 1.0e-6, 1)
xe = np.array([[1], [1], [0.5]])

if np.linalg.norm(xmin2 - xe) < 1.0e-2:
    print('Check 04 okay')
else:
    raise Exception('Your NewtonDescent does not jump to the hill point of the quadratic objective. At some point you do not use a true Newton direction.')


myObjective = BO.bananaValleyObjective()
x0 = np.array([[0], [1]])
xmin = ND.NewtonDescent(myObjective, x0, 1.0e-6, 1)
xe = np.array([[1], [1]])
if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 05 okay')
else:
    raise Exception('Your Newton Descent does not work for the banana valley objective')

if PCG.matrnr() == 0:
    raise Exception('Please set your matriculation number in PrecCGSolver.py!')
elif ND.matrnr() == 0:
    raise Exception('Please set your matriculation number in NewtonDescent.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')

print('\nWe finished now NewtonDescent, which is a q-quadratic descent algorithm, but it has two big flaws:')
print('First, it only works reliably for convex objectives. Second, it requires expensive Hessian information.')
print('We will improve this in all aspects in the next LAB.')

