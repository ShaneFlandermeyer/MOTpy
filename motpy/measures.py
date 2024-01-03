import numpy as np
from typing import List
from motpy.distributions.gaussian import GaussianState


def mahalanobis(ref_dist: GaussianState, states: List[GaussianState]):
  mean = ref_dist.mean
  covar = ref_dist.covar
  Si = np.linalg.inv(covar)

  x = np.array([s.mean for s in states])
  y = x - mean
  return np.sqrt(np.einsum('ni, ii, in -> n', y, Si, y.swapaxes(-1, -2)))
