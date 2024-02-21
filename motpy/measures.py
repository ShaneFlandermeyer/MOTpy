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
  mu = np.atleast_2d(mean)
  x = np.atleast_2d(points)
  n, d = mu.shape
  m, _ = x.shape

  y = x.reshape(1, m, d) - mu.reshape(n, 1, d)
  Si_y = np.linalg.solve(covar, y.swapaxes(-1, -2))
  dist = np.sqrt(np.einsum('nmi, nim -> nm', y, Si_y))

  if mean.ndim == 1:
    dist = dist.squeeze(0)
  return dist
