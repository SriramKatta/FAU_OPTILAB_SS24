# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import quadraticObjective as QO
import modelObjective as MO
import augmentedLagrangianObjective as AO
import augmentedLagrangianDescent as AD
import projectionInBox as PB

A = np.array([[2, 0], [0, 2]], dtype=float)
B = np.array([[0], [0]], dtype=float)
C = 1
myObjective = QO.quadraticObjective(A, B, C)
D = np.array([[2, 0], [0, 2]], dtype=float)
E = np.array([[0], [0]], dtype=float)
F = -1
myConstraint = QO.quadraticObjective(D, E, F)
x0 = np.array([[2], [2]])
alpha = -1
gamma = 10
myAugLag = AO.augmentedLagrangianObjective(myObjective, myConstraint, alpha, gamma)

y1 = myAugLag.objective(x0)
y1e = 247
if np.linalg.norm(y1-y1e) < 1.0e-1:
    print('Check 01 is okay')
else:
    raise Exception('Your augmentedLagrangianObjective returns a wrong objective')

y2 = myAugLag.gradient(x0)
y2e = np.array([[280], [280]])
if np.linalg.norm(y2-y2e) < 1.0e-1:
    print('Check 02 is okay')
else:
    raise Exception('Your augmentedLagrangianObjective returns a wrong gradient')

A = np.array([[4, 0], [0, 2]], dtype=float)
B = np.array([[0], [0]], dtype=float)
C = 1
myObjective = QO.quadraticObjective(A, B, C)
a = np.array([[0], [0]])
b = np.array([[2], [2]])
myBox = PB.projectionInBox(a, b)
D = np.array([[2, 0], [0, 2]], dtype=float)
E = np.array([[0], [0]], dtype=float)
F = -1
myConstraint = QO.quadraticObjective(D, E, F)
x0 = np.array([[1], [1]], dtype=float)
alpha0 = 0
eps = 1.0e-3
delta = 1.0e-6
[xmin, alphamin] = AD.augmentedLagrangianDescent(myObjective, myBox, myConstraint, x0, alpha0, eps, delta, 1)
xmine = np.array([[0], [1]])
if np.linalg.norm(xmin-xmine) < 1.0e-1:
    print('Check 03 is okay')
else:
    raise Exception('Your augmentedLagrangianDescent returns a wrong xmin')

alphamine = -1
if np.linalg.norm(alphamin-alphamine) < 1.0e-1:
    print('Check 04 is okay')
else:
    raise Exception('Your augmentedLagrangianDescent returns a wrong alphamin')

p = np.array([[ 2.9999039 ], [ 1.99851503], [16.05570494]], dtype=float)
myObjective = MO.modelObjective(p)
a = np.array([[0], [-4], [-1]])
b = np.array([[8], [4], [1]])
myBox = PB.projectionInBox(a, b)
D = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
E = np.array([[-8], [0], [0]], dtype=float)
F = 7
myConstraint = QO.quadraticObjective(D, E, F)
x0 = np.array([[4], [-2], [0]], dtype=float)
alpha0 = 0
eps = 1.0e-3
delta = 1.0e-6
[xmin, alphamin] = AD.augmentedLagrangianDescent(myObjective, myBox, myConstraint, x0, alpha0, eps, delta, 1)
xmine = np.array([[5.56], [-2.55], [-0.20]])
alphamine = 16.4
if np.linalg.norm(xmin-xmine) < 1.0e-1 and np.linalg.norm(alphamin-alphamine) < 1.0e-1:
    print('Check 05 is okay')
else:
    raise Exception('Your augmentedLagrangianDescent returns a wrong result for the model problem')

if AO.matrnr() == 0:
    raise Exception('Please set your matriculation number in augmentedLagrangianObjective.py!')
elif AD.matrnr() == 0:
    raise Exception('Please set your matriculation number in augmentedLagrangianDescent.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')

print('\nWe finished now augmentedLagrangianDescent, which is a q-superlinear descent algorithm for objectives subject to box constraints and equality constraints.')
print('We use it to solve the model problem.')
print('The (LMP) you found for the model problem is:\n', xmin)
print('Congratulations! You finished the LAB.')


