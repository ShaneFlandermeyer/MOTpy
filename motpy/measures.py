from functools import partial
import numpy as np


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
  y = mean[...,  None, :] - points[..., None, :, :]
  Si_y = np.linalg.inv(covar) @ y.swapaxes(-1, -2)
  dist = np.sqrt(np.einsum('...nmd, ...ndm ->...nm', y, Si_y))
  return dist