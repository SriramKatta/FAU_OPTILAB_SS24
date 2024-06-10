import WolfePowellSearch as WPS
import numpy as np

class myobj:
  def objective(self, x):
    return x.T @ x
  
  def gradient(self, x):
    return 2*x
  
if __name__ == "__main__":
  obj = myobj()
  print(WPS.WolfePowellSearch(obj, np.array([[-0.001], [-0.001]]), np.array([[0.5], [0.5]]), 0.25, 0.7 ))
  