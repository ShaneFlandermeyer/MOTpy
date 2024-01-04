import numpy as np
from typing import List
from motpy.distributions.gaussian import GaussianState


# @profile
def mahalanobis(mean: np.ndarray, covar: np.ndarray, points: List[np.ndarray]):
  Si = np.linalg.inv(covar)

  x = np.array(points)
  y = x - mean
  return np.sqrt(np.einsum('ni, ii, in -> n', y, Si, y.swapaxes(-1, -2)))
