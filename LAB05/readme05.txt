---
Optimization for Engineers - Dr.Johannes Hild
LAB05
---

---
Files
---
augmentedLagrangianObjective: Constructs an objective for augmentedLagrangianDescent out of a given objective and equality constraints, needs to be completed.
augmentedLagrangianDescent: Descent method with global q-superlinear convergence rate for equality constraints and box constraints. Needs to be completed.
Check05: Run this to check your files for correctness, requires files from previous LABs.

---
Tasks
---
Complete augmentedLagrangianObjective.py and augmentedLagrangianDescent.py and check them with Check05.py for correctness.
Everything is fine if you get "Process finished with exit code 0" AND the number of iterations of the last check is smaller than 30.
Upload augmentedLagrangianObjective.py and augmentedLagrangianDescent.py within the deadline.

---
Hints
---
For augmentedLagrangianObjective you need to write the code that generates the objective and gradient for the augmentedLagrangianObjective, compare Definition 7.3 in script page 94.
For augmentedLagrangianDecent implement Algorithm 7.4 on script page 95 and use projectedInexactNewtonCG as the projection method with tolerance eps_k!
Please take note that there is a difference between delta and delta_k, same for eps and eps_k.
Update all dependent variables, if either x changes or the tolerances change.

