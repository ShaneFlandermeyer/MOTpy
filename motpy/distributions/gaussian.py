from __future__ import annotations

import dataclasses
from typing import *

import numpy as np
from motpy.distributions.base import Distribution


class Gaussian(Distribution):
  """
  A general class to represent both Gaussian state and Gaussian mixture distributions.

  Representing these with one class simplifies the code and makes it easy to support batched operations.
  """

  def __init__(
      self,
      mean: Optional[np.ndarray],
      covar: Optional[np.ndarray],
      weight: Optional[np.ndarray] = None,
  ):
    shape, state_dim = mean.shape[:-1], mean.shape[-1]

    self.mean = np.reshape(mean, shape + (state_dim,))
    self.covar = np.reshape(covar, shape + (state_dim, state_dim))
    if weight is not None:
      self.weight = np.reshape(weight, shape)
    else:
      self.weight = None
    self.state_dim = state_dim

  def __repr__(self):
    return f"""Gaussian(
      shape={self.shape},
      state_dim={self.state_dim},
      means={self.mean}
      covars=\n{self.covar})
      weights={self.weight}
      """

  @property
  def shape(self) -> Tuple[int]:
    return self.mean.shape[:-1]

  @property
  def size(self) -> int:
    return np.prod(self.shape)

  def __getitem__(self, idx) -> Gaussian:
    return Gaussian(
        mean=self.mean[idx],
        covar=self.covar[idx],
        weight=self.weight[idx] if self.weight is not None else None
    )

  def __setitem__(self, idx: int, value: Gaussian) -> None:
    self.mean[idx] = value.mean
    self.covar[idx] = value.covar
    if self.weight is not None:
      self.weight[idx] = value.weight

  def append(self, state: Gaussian, axis: int = 0) -> Gaussian:
    means = np.append(self.mean, state.mean, axis=axis)
    covars = np.append(self.covar, state.covar, axis=axis)
    if self.weight is not None:
      weights = np.append(self.weight, state.weight, axis=axis)
    else:
      weights = None
    return Gaussian(mean=means, covar=covars, weight=weights)

  @staticmethod
  def empty(shape: Tuple[int], state_dim: int) -> Gaussian:
    return Gaussian(
        mean=np.zeros(shape + (state_dim,)),
        covar=np.zeros(shape + (state_dim, state_dim)),
        weight=np.zeros(shape)
    )

  def sample(self,
             num_points: int,
             dims: Optional[Sequence[int]] = None,
             max_distance: Optional[float] = None,
             rng: np.random.Generator = np.random.default_rng(),
             ) -> np.ndarray:
    """
    Sample points from each Gaussian component at the specified dimensions.
    """
    if dims is None:
      dims = np.arange(self.state_dim)

    covar_inds = np.ix_(dims, dims)
    P = self.covar[..., covar_inds[0], covar_inds[1]]

    mu = self.mean[..., None, dims]
    std_normal = rng.normal(size=(*self.shape, num_points, mu.shape[-1]))
    if max_distance is not None:
      max_val = max_distance / np.sqrt(len(dims))
      std_normal = std_normal.clip(-max_val, max_val)
    return mu + np.einsum('nij, nmj -> nmi', np.linalg.cholesky(P), std_normal)


def merge_gaussians(
        means: np.ndarray,
        covars: np.ndarray,
        weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  mu = means
  P = covars
  w = weights

  w_merged = np.sum(w, axis=-1)
  w /= w_merged[..., None] + 1e-15
  mu_merged = np.einsum('...i, ...ij -> ...j', w, mu)

  y = mu - mu_merged[..., None, :]
  y_outer = np.einsum('...i, ...j -> ...ij', y, y)
  P_merged = np.einsum('...i, ...ijk -> ...jk', w, P + y_outer)
  return w_merged, mu_merged, P_merged


def likelihood(
        x: np.ndarray,
        mean: np.ndarray,
        covar: np.ndarray,
        subtract_fn: Callable[
            [np.ndarray, np.ndarray], np.ndarray
        ] = np.subtract,
) -> np.ndarray:
  x = np.atleast_2d(x)
  y = subtract_fn(x[..., None, :, :], mean[..., None, :])

  Pi = np.linalg.inv(covar)
  det_P = np.linalg.det(covar)

  num = np.exp(-0.5 * np.einsum('...mi, ...ii, ...im -> ...m',
                                y, Pi, y.swapaxes(-1, -2))
               )
  den = np.sqrt((2 * np.pi) ** x.shape[-1] * det_P)
  likelihood = num / den[..., None]

  return likelihood


def mahalanobis(x: np.ndarray,
                mean: np.ndarray,
                covar: np.ndarray,
                subtract_fn: Callable[
                    [np.ndarray, np.ndarray], np.ndarray
                ] = np.subtract,
                ) -> np.ndarray:
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
  x = np.atleast_2d(x)
  y = subtract_fn(x[..., None, :, :], mean[..., None, :])

  dist = np.sqrt(
      np.einsum(
          '...nmi, ...nim ->...nm',
          y, np.linalg.inv(covar) @ y.swapaxes(-1, -2)
      )
  )
  return dist
