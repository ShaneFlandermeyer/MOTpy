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
      mean: np.ndarray,
      covar: np.ndarray,
      weight: Optional[np.ndarray] = None,
  ):
    self.mean = np.atleast_2d(mean)
    self.state_dim = self.mean.shape[-1]

    n_components = self.mean.shape[0]
    self.covar = covar.reshape(n_components, self.state_dim, self.state_dim)

    if weight is None:
      weight = np.zeros(n_components)
    else:
      weight = np.asarray(weight).reshape(n_components)
    self.weight = weight

  def __repr__(self):
    return f"""GaussianMixture(
      n_components={len(self)},
      means={self.mean}
      covars=\n{self.covar})
      weights={self.weight}
      """

  def __len__(self):
    return len(self.mean)

  def __getitem__(self, idx):
    return GaussianState(mean=self.mean[idx], covar=self.covar[idx], weight=self.weight[idx])

  def append(self, state: GaussianState) -> None:
    self.mean = np.concatenate((self.mean, state.mean), axis=0)
    self.covar = np.concatenate((self.covar, state.covar), axis=0)
    self.weight = np.concatenate((self.weight, state.weight), axis=0)


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

  Parameters
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


@jax.jit
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
