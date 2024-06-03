---
Optimization for Engineers - Dr.Johannes Hild
LAB03
---

---
Files
---
projectedBacktrackingSearch: Line search method for projection methods, works similar to Wolfe-Powell backtracking, needs to be completed.
projectedInexactNewtonCG: Descent method for box constraints with global q-superlinear convergence rate. Does not require Hessian information or a linear system solver. Needs to be completed.
projectionInBox: Provides projection into boxes and active index sets.
projectedHessApprox: Provides projected directional Hessian approximations, i.e. Algorithm 11.6.
boxObjective: Test problem that is not defined outside a box. If you get an error from this file, you probably miss a projection.
Check03: Run this to check your files for correctness, requires files from previous LABs.

---
Tasks
---
Complete projectedBacktrackingSearch.py and projectedInexactNewtonCG.py and check them with Check03.py for correctness.
Everything is fine if you get "Process finished with exit code 0" AND the number of iterations of the last check is smaller than 30.
Upload projectedBacktrackingSearch.py and projectedInexactNewtonCG.py within the deadline.

---
Hints
---
For projectedBacktrackingSearch implement Algorithm 4.17 on script page 63. 
P is already provided, can project() and can generate the active index set, you can look in the file projectionInBox.py to see how it works.
For projectedInexactNewtonCG you have to implement Alg. 11.7 on script page 123. In projectedHessApprox Algorithm 11.6. is already implemented.
Use numpy.sqrt and numpy.min([,]) for eta_k.
Make sure that a change in x leads to a proper change in all variables depending on x. And do not forget to project.
be careful: numpy arrays are not copied for assignments like "x = x_j".
A statement like  "x_j += td" would then also change "x", even if not intended.
You can avoid this by using ".copy()" and not using "+=".
For the last check you should have fewer than 10 iterations, otherwise you use steepest-descent-like steps too often and end up with a poor q-linear algorithm.