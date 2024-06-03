---
Optimization for Engineers - Dr.Johannes Hild
LAB02
---

---
Files
---
WolfePowellSearch: Highly effective line search method, needs to be completed.
BFGSDescent: Descent method with global q-superlinear convergence rate. Does not require Hessian information or a linear system solver. Needs to be completed.
simpleValleyObjective: Test problem for Wolfe-Powell.
noHessianObjective: Test problem without Hessian information.
multidimensionalObjective: Test problem in 8 dimensions.
Check02: Run this to check your files for correctness, requires files from previous LABs.

---
Tasks
---
Complete WolfePowellSearch.py and BFGSDescent.py and check them with Check02.py for correctness.
Everything is fine if you get "Process finished with exit code 0" AND the number of iterations of the last check is smaller than 30.
Upload WolfePowellSearch.py and BFGSDescent.py within the deadline.

---
Hints
---
For Wolfe-Powell see Algorithm 4.10 on script page 56
Define inner functions to check the Wolfe-Powell rules
Make sure that a change in t leads to a proper change in all variables depending on t
For BFGSDescent implement Algorithm 6.7 on script page 82
BFGSDescent has a similar structure like NewtonDescent
Use numpy.eye(n) to construct a identity matrix, use @ for matrix - vector or matrix - matrix product, use .T to transpose
Use x.shape[0] to get the vector dimension
Do a descent direction check right before calling Wolfe-Powell: If d_k is not a descent direction, switch to steepest descent AND reset B_k to the identity matrix
For the last check you should have fewer than 30 iterations, otherwise you use steepest-descent-like steps too often and end up with a poor q-linear algorithm.