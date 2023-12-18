import datetime
from typing import List, Tuple, Union

import numpy as np
from scipy.stats import multivariate_normal


class GaussianState():
  def __init__(
    self,
    mean: np.ndarray,
    covar: np.ndarray,
    timestamp: Union[float, datetime.datetime] = {},
    metadata: dict = None,
    **kwargs
  ):
    self.mean = mean
    self.covar = covar
    self.timestamp = timestamp
    self.metadata = metadata
    # Set additional kwargs as attributes for flexibility
    for key, value in kwargs.items():
      setattr(self, key, value)
    
  def __repr__(self):
    return f'GaussianState(\n mean={self.mean}\n covar=\n{self.covar})'

def mix_gaussians(means: List[np.ndarray],
                  covars: List[np.ndarray],
                  weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Compute a Gaussian mixture as a weighted sum of N Gaussian distributions, each with dimension D.

  Parameters
  ----------
  means : np.ndarray
      Length-N list of D-dimensional arrays of mean values for each Gaussian components
  covars : np.ndarray
      Length-N list of D x D covariance matrices for each Gaussian component
  weights : np.ndarray
      Length-N array of weights for each component

  Returns
  -------
  Tuple[np.ndarray, np.ndarray]
    Mixture PDF mean and covariance

  """
  assert len(means) == len(covars) == len(weights)

  N = len(weights)
  x = np.array(means).reshape(N, -1)
  P = np.array(covars)
  w = weights / np.sum(weights)

  mix_mean = np.dot(weights, x)
  mix_covar = np.zeros((x.shape[1], x.shape[1]))
  for i in range(N):
    mix_covar += w[i] * (P[i] + np.outer(x[i], x[i]))
  mix_covar -= np.outer(mix_mean, mix_mean)
  return mix_mean, mix_covar


def likelihood(z: np.ndarray,
               z_pred: np.ndarray,
               P_pred: np.ndarray,
               H: np.ndarray,
               R: np.ndarray,
               ) -> float:
  z = np.atleast_2d(z)
  z_pred = z_pred.flatten()
  S = H @ P_pred @ H.T + R
  return multivariate_normal.pdf(z, z_pred, S)