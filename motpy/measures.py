import numpy as np
from typing import List


def mahalanobis(mean: np.ndarray,
                covar: np.ndarray,
                points: np.ndarray) -> np.ndarray:
  """
  Batched mahalanobis distance.

  Parameters
  ----------
  mean : np.ndarray
      Means of the reference Gaussian distributions. Shape (N, D).
  covar : np.ndarray
      Covars of the reference Gaussian distributions. Shape (N, D, D).
  points : np.ndarray
      Query points. Shape (M, D).

  Returns
  -------
  np.ndarray
      Mahalanobis distance for each reference/query pair. Shape (N, M).
  """
  mu = np.atleast_2d(mean)
  x = np.atleast_2d(points)
  n, d = mu.shape
  m, _ = x.shape
  Si = np.linalg.inv(covar).reshape(n, d, d)
  
  y = x.reshape(1, m, d) - mu.reshape(n, 1, d)
  dist = np.sqrt(np.einsum('nmi, nii, nim -> nm', y, Si, y.swapaxes(-1, -2)))

  if mean.ndim == 1:
    dist = dist.squeeze(0)
  return dist
