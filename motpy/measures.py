from functools import partial
import jax
import numpy as np
import jax
import jax.numpy as jnp


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


def pairwise_euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """
  Pairwise Euclidean distance.

  Parameters
  ----------
  x : np.ndarray
      First set of points. Shape (N, D).
  y : np.ndarray
      Second set of points. Shape (M, D).

  Returns
  -------
  np.ndarray
      Pairwise Euclidean distance. Shape (N, M).
  """
  return jnp.sqrt(jnp.sum((x[:, None] - y[None])**2, axis=-1))


def pairwise_mahalanobis(means: np.ndarray, covars: np.ndarray):
  y = means[None, :, :] - means[:, None, :]
  Si = jnp.linalg.inv(covars)
  return jnp.sqrt(jnp.einsum('nmi, nij, nim -> nm', y, Si, y.swapaxes(-1, -2)))
