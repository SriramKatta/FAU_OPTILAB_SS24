# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import modelObjective as MO
import leastSquaresModel as LSM
import levenbergMarquardtDescent as LMD

p0 = np.array([[2], [3]])
myObjective = SO.simpleValleyObjective(p0)
xk = np.array([[0, 0, 1, 2], [1, 2, 3, 4]])
fk = np.array([[2, 3, 2.54, 4.76]])
myErrorVector = LSM.leastSquaresModel(myObjective, xk, fk)
res = myErrorVector.residual(p0)
rese = np.array([[2], [3], [10], [20]])
if np.linalg.norm(res-rese) < 1.0e-2:
    print('Check 01 is okay')
else:
    raise Exception('Your leastSquaresModel returns a wrong residual')

res = myErrorVector.jacobian(p0)
rese = np.array([[0, 1], [1, 1], [4, 1],  [9, 1]])
if np.linalg.norm(res - rese) < 1.0e-2:
    print('Check 02 is okay')
else:
    raise Exception('Your leastSquaresModel returns a wrong jacobian')

p0 = np.array([[2],[1],[3]])
myObjective = MO.modelObjective(p0)
xk = np.array([[3, 0, 0], [-1, -1, 0], [-1, -1, -1]])
fk = np.array([[3, 2, 1]])
myErrorVector = LSM.leastSquaresModel(myObjective, xk, fk)

res = myErrorVector.residual(p0)
rese = np.array([[4], [2], [2]])
if np.linalg.norm(res-rese) < 1.0e-2:
    print('Check 03 is okay')
else:
    raise Exception('Your leastSquaresModel returns a wrong residual for the model objective')

res = myErrorVector.jacobian(p0)
rese = np.array([[0, -1, 2], [0, -1, 1], [0, 0, 1]])
if np.linalg.norm(res-rese) < 1.0e-2:
    print('Check 04 is okay')
else:
    raise Exception('Your leastSquaresModel returns a wrong jacobian for the model objective')

p0 = np.array([[180],[0]])
myObjective = SO.simpleValleyObjective(p0)
xk = np.array([[0, 0], [1, 2]])
fk = np.array([[2, 3]])
myErrorVector = LSM.leastSquaresModel(myObjective, xk, fk)
eps = 1.0e-4
alpha0 = 1.0e-3
beta = 100
pmin = LMD.levenbergMarquardtDescent(myErrorVector, p0, eps, alpha0, beta, 1)
pe = np.array([[1], [1]])
if np.linalg.norm(pmin-pe) < 1.0e-2:
    print('Check 05 is okay')
else:
    raise Exception('Your levenbergMarquardtDescent returns a wrong result')

p0 = np.array([[0], [0], [0]])
myObjective = MO.modelObjective(p0)
xk = myObjective.getXData()
fk = myObjective.getFData()
myErrorVector = LSM.leastSquaresModel(myObjective, xk, fk)
eps = 1.0e-4
alpha0 = 1.0e-3
beta = 100
pmin = LMD.levenbergMarquardtDescent(myErrorVector, p0, eps, alpha0, beta, 1)
pe = np.array([[3], [2], [16]])
if np.linalg.norm(pmin-pe) < 1.0e-1:
    print('Check 06 is okay')
else:
    raise Exception('Your levenbergMarquardtDescent returns wrong parameters for the model problem')

if LSM.matrnr() == 0:
    raise Exception('Please set your matriculation number in leastSquaresModel.py!')
elif LMD.matrnr() == 0:
    raise Exception('Please set your matriculation number in levenbergMarquardtDescent.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')

print('\nWe finished now levenbergMarquardtDescent, which is a q-superlinear descent algorithm for least squares objectives.')
print('We use it to find the parameters of the model problem with data fitting.')
print('The parameters you computed for the model problem are:\n', pmin)
print('We will implement an descent algorithm to solve the model problem with its box and equality constraints in the next LAB.')

