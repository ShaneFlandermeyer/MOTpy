from __future__ import annotations

import datetime
from typing import *

import jax
import jax.numpy as jnp
import numpy as np

from motpy.measures import pairwise_euclidean, pairwise_mahalanobis


class GaussianState():
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

  def __repr__(self):
    return f"""GaussianMixture(
      shape={self.shape},
      state_dim={self.state_dim},
      means={self.mean}
      covars=\n{self.covar})
      weights={self.weight}
      """

  @property
  def shape(self):
    shape = self.mean.shape[:-1]
    if self.covar.shape[:-2] != shape:
      raise ValueError("Mean and covariance shapes do not match")
    if self.weight is not None and self.weight.shape != shape:
      raise ValueError("Mean and weight shapes do not match")
    return shape

  @property
  def state_dim(self):
    state_dim = self.mean.shape[-1]
    if self.covar.shape[-2:] != (state_dim, state_dim):
      raise ValueError("Covariance matrix has incorrect state dimension")

    return self.mean.shape[-1]

  def __len__(self):
    return len(self.mean)

  def __getitem__(self, idx):
    return GaussianState(
        mean=self.mean[idx],
        covar=self.covar[idx],
        weight=self.weight[idx] if self.weight is not None else None
    )

  def append(self, state: GaussianState, axis: int = 0) -> None:
    means = np.append(self.mean, state.mean, axis=axis)
    covars = np.append(self.covar, state.covar, axis=axis)
    if self.weight is not None:
      weights = np.append(self.weight, state.weight, axis=axis)
    else:
      weights = None
    return GaussianState(mean=means, covar=covars, weight=weights)

  def stack(self, state: GaussianState, axis: int = 0) -> None:
    means = np.stack([self.mean, state.mean], axis=axis)
    covars = np.stack([self.covar, state.covar], axis=axis)
    if self.weight is not None:
      weights = np.stack([self.weight, state.weight], axis=axis)
    else:
      weights = None
    return GaussianState(mean=means, covar=covars, weight=weights)

  def concatenate(self, state: GaussianState, axis: int = 0) -> None:
    means = np.concatenate([self.mean, state.mean], axis=axis)
    covars = np.concatenate([self.covar, state.covar], axis=axis)
    if self.weight is not None:
      weights = np.concatenate([self.weight, state.weight], axis=axis)
    else:
      weights = None
    return GaussianState(mean=means, covar=covars, weight=weights)

  def sample(self,
             num_points: int,
             dims: Sequence[int],
             rng: np.random.Generator = np.random.default_rng()
             ) -> np.ndarray:
    """
    Sample points from each Gaussian component at the specified dimensions.
    """
    covar_inds = np.ix_(dims, dims)
    P = self.covar[..., covar_inds[0], covar_inds[1]]

    mu = self.mean[..., None, dims]
    std_normal = rng.normal(size=(len(self), num_points, len(dims)))
    return mu + np.einsum('nij, nmj -> nmi', np.linalg.cholesky(P), std_normal)


def match_moments(means: np.ndarray,
                  covars: np.ndarray,
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
  if len(weights) == 1:
    return means, covars
  x = means
  P = covars
  w = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-15)

  mix_mean = np.einsum('...i, ...ij -> ...j', w, x)
  mix_covar = np.einsum('...i, ...ijk->...jk', w, P)
  mix_covar += np.einsum('...i,...ij,...ik->...jk', w, x, x)
  mix_covar -= np.einsum('...i,...j->...ij', mix_mean, mix_mean)
  return mix_mean, mix_covar


def likelihood(z: np.ndarray,
               z_pred: np.ndarray,
               S: np.ndarray
               ) -> np.ndarray:
  """
  Compute the likelihood for a set of measurement/state pairs.

  Parameters>
  ----------
  z : np.ndarray
      Array of measurements. Shape: (M, nz)
  z_pred : np.ndarray
      Array of predicted measurements. Shape: (N, nz)
  S : np.ndarray
      Innovation covariance. Shape: (N, nz, nz)

  Returns
  -------
  np.ndarray
      Likelihood for each measurement/state pair. Shape: (M, N)
  """
  nz = z.shape[-1]

  z = z[np.newaxis, ...]
  z_pred = z_pred[..., np.newaxis, :]
  y = z - z_pred  # (N, M, nz)

  Si = np.linalg.inv(S)
  det_S = np.linalg.det(S)[:, np.newaxis]

  exponent = -0.5 * np.einsum('nmi, nii, nim -> nm', y, Si, y.swapaxes(-1, -2))
  likelihoods = np.exp(exponent) / np.sqrt((2 * np.pi) ** nz * det_S)

  return likelihoods


@ jax.jit
def merge_mixture(means: np.ndarray,
                  covars: np.ndarray,
                  weights: np.ndarray,
                  threshold: np.ndarray,
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Merge a Gaussian mixture by clustering components that are close to each other.

  Parameters
  ----------
  means : np.ndarray
      _description_
  covars : np.ndarray
      _description_
  weights : np.ndarray
      _description_
  threshold : np.ndarray
      _description_

  Returns
  -------
  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
      _description_
  """
  # Compute pairwise distances between all points
  # dists = pairwise_mahalanobis(means, covars)
  dists = pairwise_euclidean(means, means)
  mask = dists < threshold

  # Sort distance matrix rows by weight
  inds = jnp.argsort(weights, descending=True)
  dists = dists[inds]
  mask = mask[inds]

  # Determine cluster indices and normalized mixture weights
  masked_weights = jnp.empty((means.shape[0], weights.size))
  for i in range(means.shape[0]):
    # Compute (unnormalized) mixture weights for this cluster
    masked_weights = masked_weights.at[i].set(jnp.where(mask[i], weights, 0))

    # Mark points in this cluster as used
    mask = jnp.where(mask[i], False, mask)

  # Match moments for all clusters
  weight_sum = jnp.sum(masked_weights, axis=-1, keepdims=True)
  mix_weights = masked_weights / (weight_sum + 1e-15)
  mix_means = jnp.einsum('...i, ...ij -> ...j', mix_weights, means)
  mix_covars = jnp.einsum('...i, ...ijk->...jk', mix_weights, covars) + \
      jnp.einsum('...i,...ij,...ik->...jk', mix_weights, means, means) - \
      jnp.einsum('...i,...j->...ij', mix_means, mix_means)

  return mix_means, mix_covars, weight_sum.ravel()
