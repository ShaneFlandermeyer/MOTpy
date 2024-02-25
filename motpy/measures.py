import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from motpy.common import nextpow2


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


def mahalanobis(mean: np.ndarray,
                covar: np.ndarray,
                points: np.ndarray) -> jnp.ndarray:
  y = points - mean
  dist = jnp.sqrt(y @ jnp.linalg.inv(covar) @ y.T)
  return dist


_pairwise_mahalanobis = jit(
    vmap(vmap(mahalanobis, in_axes=(None, None, 0)), in_axes=(0, 0, None)))


def pairwise_mahalanobis(mean: np.ndarray,
                         covar: np.ndarray,
                         points: np.ndarray) -> np.ndarray:
  """
  Pairwise mahalanobis distance.

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
  # Pad each input to the next power of 2
  ndist = mean.shape[0]
  nquery = points.shape[0]
  mean = np.pad(mean, ((0, nextpow2(ndist) - ndist), (0, 0)))
  covar = np.pad(covar, ((0, nextpow2(ndist) - ndist), (0, 0), (0, 0)))
  points = np.pad(points, ((0, nextpow2(nquery) - nquery), (0, 0)))
  
  # Compute the pairwise Mahalanobis distance and remove padding
  dists = _pairwise_mahalanobis(mean, covar, points)
  
  return np.array(dists)[:ndist, :nquery]


if __name__ == '__main__':
  x = np.ones((100, 4))
  mu = np.ones((50, 4))
  cov = np.random.rand(50, 4, 4)
  pairwise_mahalanobis(mu, cov, x)
