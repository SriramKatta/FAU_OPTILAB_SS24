# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import noHessianObjective as NO
import bananaValleyObjective as BO
import WolfePowellSearch as WP
import multidimensionalObjective as MO
import BFGSDescent as BD


p = np.array([[0], [1]])
myObjective = SO.simpleValleyObjective(p)
x = np.array([[-1.01], [1]])
d = np.array([[1], [1]])
sigma = 1.0e-3
rho = 1.0e-2
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 1
if t == te:
    print('Check 01 okay')
else:
    raise Exception('Your Wolfe-Powell search is not recognizing t = 1 as valid starting point.')


x = np.array([[-1.2], [1]])
d = np.array([[0.1], [1]])
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 16
if t == te:
    print('Check 02 okay')
else:
    raise Exception('Your Wolfe-Powell search is not front tracking correctly.')

x = np.array([[-0.2], [1]])
d = np.array([[1], [1]])
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 0.25
if t == te:
    print('Check 03 okay')
else:
    raise Exception('Your Wolfe-Powell search is not refining correctly.')

myObjective = MO.multidimensionalObjective()
x = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
d = -myObjective.gradient(x)
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 0.0625
if np.abs(t-te) < 1.0e-3:
    print('Check 04 okay')
else:
    raise Exception('Your Wolfe-Powell search is not working for multidimensional objective.')

myObjective = MO.multidimensionalObjective()
x0 = np.array([[1], [1], [1], [1], [1], [1], [1], [1]])
xmin = BD.BFGSDescent(myObjective, x0, 1.0e-6, 1)
xe = np.array([[1.02614], [0], [0], [0], [0], [0], [0], [0]])
if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 05 okay')
else:
    raise Exception('Your BFGS Descent does not work for the 8-dimensional test function')

myObjective = NO.noHessianObjective()
x0 = np.array([[-0.01], [0.01]])
xmin = BD.BFGSDescent(myObjective, x0, 1.0e-6, 1)
xe = np.array([[0.26], [-0.21]])
if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 06 okay')
else:
    raise Exception('Your BFGS Descent is not working correctly for the Hessian free test function.')

myObjective = BO.bananaValleyObjective()
x0 = np.array([[0], [0]])
xmin = BD.BFGSDescent(myObjective, x0, 1.0e-6, 1)
xe = np.array([[1], [1]])

if np.linalg.norm(xmin - xe) < 1.0e-2:
    print('Check 07 okay')
else:
    raise Exception('Your BFGS Descent does not work for the banana valley objective')

if WP.matrnr() == 0:
    raise Exception('Please set your matriculation number in WolfePowellSearch.py!')
elif BD.matrnr() == 0:
    raise Exception('Please set your matriculation number in BFGSDescent.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')

print('\nWe finished now BFGSDescent, which is a q-superlinear descent algorithm, that converges globally for unconstrained problems.')
print('It does not require Hessian information or a linear system solver like CG.')
print('We will implement an algorithm that can handle convex constraints with a projection in the next LAB.')