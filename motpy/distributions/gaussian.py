from __future__ import annotations

from typing import *

import numpy as np


class Gaussian():
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
    shape, ndim = mean.shape[:-1], mean.shape[-1]

    self.mean = np.reshape(mean, shape + (ndim,))
    self.covar = np.reshape(covar, shape + (ndim, ndim))
    if weight is not None:
      self.weight = np.reshape(weight, shape)
    else:
      self.weight = None
    self.ndim = ndim

  def __repr__(self):
    return f"""GaussianMixture(
      shape={self.shape},
      ndim={self.ndim},
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
      dims = np.arange(self.ndim)

    covar_inds = np.ix_(dims, dims)
    P = self.covar[..., covar_inds[0], covar_inds[1]]

    mu = self.mean[..., None, dims]
    std_normal = rng.normal(size=(*self.shape, num_points, mu.shape[-1]))
    if max_distance is not None:
      max_val = max_distance / np.sqrt(len(dims))
      std_normal = std_normal.clip(-max_val, max_val)
    return mu + np.einsum('nij, nmj -> nmi', np.linalg.cholesky(P), std_normal)


class SigmaPointGaussian():
  def __init__(self,
               distribution: Gaussian,
               sigma_points: np.ndarray,
               Wm: np.ndarray,
               Wc: np.ndarray,
               ):
    """
    Container class for Gaussian distributions and their sigma points

    # NOTE: Assuming the weights are the same for all sigma points

    Parameters
    ----------
    distribution : Gaussian
    sigma_points : np.ndarray
    Wm : np.ndarray
    Wc : np.ndarray
    """
    self.distribution = distribution
    self.sigma_points = sigma_points
    self.Wm = Wm
    self.Wc = Wc

  @property
  def shape(self) -> Tuple[int]:
    return self.distribution.shape

  @property
  def size(self) -> int:
    return self.distribution.size

  @property
  def mean(self) -> np.ndarray:
    return self.distribution.mean

  @property
  def covar(self) -> np.ndarray:
    return self.distribution.covar

  @property
  def weight(self) -> np.ndarray:
    return self.distribution.weight

  def __getitem__(self, idx: int) -> SigmaPointGaussian:
    return SigmaPointGaussian(
        distribution=self.distribution[idx],
        sigma_points=self.sigma_points[idx],
        Wm=self.Wm,
        Wc=self.Wc
    )

  def __setitem__(self, idx: int, value: SigmaPointGaussian) -> None:
    self.distribution[idx] = value.distribution
    self.sigma_points[idx] = value.sigma_points
    self.Wm = value.Wm
    self.Wc = value.Wc

  def append(self,
             value: SigmaPointGaussian,
             axis: int = 0
             ) -> SigmaPointGaussian:
    return SigmaPointGaussian(
        distribution=self.distribution.append(value.distribution),
        sigma_points=np.append(
            self.sigma_points, value.sigma_points, axis=axis
        ),
        Wm=self.Wm,
        Wc=self.Wc
    )


def merge_gaussians(
        means: np.ndarray,
        covars: np.ndarray,
        weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
  mu = means
  P = covars
  w = weights

  w_merged = np.sum(w, axis=-1)
  w /= w_merged[..., None] + 1e-15
  mu_merged = np.einsum('...i, ...ij -> ...j', w, mu)

  y = mu - mu_merged
  y_outer = np.einsum('...i, ...j -> ...ij', y, y)
  P_merged = np.einsum('...i, ...ijk -> ...jk', w, P + y_outer)
  return w_merged, mu_merged, P_merged


def likelihood(
        x: np.ndarray,
        mean: np.ndarray,
        covar: np.ndarray,
) -> np.ndarray:
  x = np.atleast_2d(x)
  y = x[..., None, :, :] - mean[..., None, :]

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
  y = x[..., None, :, :] - mean[..., None, :]

  dist = np.sqrt(
      np.einsum('...nmi, ...nim ->...nm',
                y, np.linalg.inv(covar) @ y.swapaxes(-1, -2))
  )
  return dist
