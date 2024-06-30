# Optimization for Engineers - Dr.Johannes Hild
# global BFGS descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k is the BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the inverse BFGS matrix is reset.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = noHessianObjective()
# x0 = np.array([[-0.01], [0.01]])
# xmin = BFGSDescent(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[0.26],[-0.21]] with the inverse BFGS matrix being close to [[0.0078, 0.0005], [0.0005, 0.0080]]


import numpy as np
import WolfePowellSearch as WP

def matrnr():
    # set your matriculation number here
    matrnr = 23356687
    return matrnr

def BFGSDescent(f, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start BFGSDescent...')

    countIter = 0
    xk = x0
    n = x0.shape[0]
    Bk = np.eye(n)  # Initialize Bk as identity matrix
    
    while np.linalg.norm(f.gradient(xk)) > eps:
        gradxk = f.gradient(xk)
        dk = -Bk @ gradxk
        
        # Descent direction check
        if gradxk.T @ dk >= 0:
            dk = -gradxk
            Bk = np.eye(n)  # Reset Bk to identity matrix
        
        tk = WP.WolfePowellSearch(f, xk, dk)
        
        xk_new = xk + tk * dk
        gradxk_new = f.gradient(xk_new)
        
        deltaxk = xk_new - xk
        deltagk = gradxk_new - gradxk
        
        # BFGS update
        if deltagk.T @ deltaxk > 0:
            Bk = Bk + (deltaxk @ deltaxk.T) / (deltaxk.T @ deltagk) - (Bk @ deltagk @ deltagk.T @ Bk) / (deltagk.T @ Bk @ deltagk)
        
        xk = xk_new
        countIter += 1

        if countIter >= 30:
            print("Warning: Maximum iterations exceeded. Switching to steepest descent-like steps.")
            dk = -gradxk  # Switch to steepest descent
            Bk = np.eye(n)  # Reset Bk to identity matrix

    if verbose:
        print('BFGSDescent terminated after', countIter, 'steps with norm of gradient =', np.linalg.norm(f.gradient(xk)), 'and the inverse BFGS matrix is')
        print(Bk)

    return xk

