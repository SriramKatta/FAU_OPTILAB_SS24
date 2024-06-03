---
Optimization for Engineers - Dr.Johannes Hild
LAB04
---

---
Files
---
leastSquaresModel: Constructs an objective for Levenberg-Marquardt out of a model objective, needs to be completed.
levenbergMarquardtDescent: Descent method for least squares objectives with global q-superlinear convergence rate. Needs to be completed.
Check04: Run this to check your files for correctness, requires files from previous LABs.

---
Tasks
---
Complete leastSquaresModel.py and levenbergMarquardtDescent.py and check them with Check04.py for correctness.
Everything is fine if you get "Process finished with exit code 0" AND the number of iterations of the last check is smaller than 30.
Upload leastSquaresModel.py and levenbergMarquardtDescent.py within the deadline.

---
Hints
---
For leastSquaresModel you need to complete the code that generates the residual and jacobian for the least squares objective of the following type:
f(p) = 0.5*sum_k (fmodel(xData_k,p)-fData_k)**2, whereas f(p) is the least squares objective, fmodel is a model depending on measure points xData_k and the parameters p, and fdata_k are the measure results.
Inspect the files simpleValleyObjective and multidimensionalObjective, both have class methods required for leastSquaresObjective.
Look at the input definitions of leastSquaresModel, there you can see how you can access the data correctly.
Remember that each row of the Jacobian is a transposed gradient of the model with respect to p, you will need .parameterGradient() to call this.
Use .reshape((n,1)) to bring extracted data in column vector form. use x.shape[0] to get the vector dimension.
You can overwrite entries in numpy.arrays with other numpy.arrays, here is some example:
myA[3, 0:3] = numpy.array([[1, 0, 0]]) will replace the entries  myA[3, 0], myA[3, 1] and myA[3, 2] with 1,0,0 respectively.
For levenbergMarquardtDescent implement Algorithm 6.14 on script page 89.
You will need PrecCGSolver from previous LABs.
Remember the connection f = 1/2 R'*R and grad_f = J'*R.
