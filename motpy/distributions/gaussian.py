import datetime
from typing import List, Tuple, Union

import numpy as np

class GaussianState():
  def __init__(
      self,
      mean: np.ndarray,
      covar: np.ndarray,
      timestamp: Union[float, datetime.datetime] = {},
      metadata: dict = {},
      **kwargs
  ):
    self.mean = mean
    self.covar = covar
    self.timestamp = timestamp
    self.metadata = metadata

  def __repr__(self):
    return f"""GaussianState(
      mean={self.mean}
      covar=\n{self.covar})
      meta={self.metadata})"""

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
  x = np.array(means)
  P = np.array(covars)
  w = weights / np.sum(weights)

  mix_mean = np.dot(weights, x)
  mix_covar = np.zeros((x.shape[1], x.shape[1]))
  mix_covar = np.einsum('i,ijk->jk', w, P) + np.einsum('i,ij,ik->jk', w, x, x)
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
  Si = np.linalg.inv(S)

  k = z_pred.size
  den = np.sqrt((2 * np.pi) ** k * np.linalg.det(S))
  x = z - z_pred
  l = np.squeeze(np.exp(-0.5 * x[..., None, :] @ Si @ x[..., None])) / den
  return l