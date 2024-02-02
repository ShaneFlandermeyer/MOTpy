from __future__ import annotations
import datetime
from typing import *

import numpy as np


class GaussianState():
  def __init__(
      self,
      mean: np.ndarray,
      covar: np.ndarray,
      weight: Optional[np.ndarray] = None,
  ):
    self.state_dim = mean.shape[-1]
    self.mean = np.atleast_2d(mean)
    if covar.ndim == 2:
      covar = covar[np.newaxis, ...]
    self.covar = covar
    self.weight = np.atleast_1d(weight)

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


# class GaussianMixture():
#   def __init__(
#       self,
#       mean: np.ndarray,
#       covar: np.ndarray,
#       weight: np.ndarray,
#   ):
#     self.state_dim = mean.shape[-1]
#     self.mean = np.atleast_2d(mean)
#     if covar.ndim == 2:
#       covar = covar[np.newaxis, ...]
#     self.covar = covar
#     self.weight = np.atleast_1d(weight)

#   def __repr__(self):
#     return f"""GaussianMixture(
#       n_components={len(self)},
#       means={self.mean}
#       covars=\n{self.covar})
#       weights={self.weight}
#       """

#   def __len__(self):
#     return len(self.mean)

#   def __getitem__(self, idx):
#     return GaussianMixture(mean=self.mean[idx], covar=self.covar[idx], weight=self.weight[idx])

#   def append(self, state: GaussianMixture) -> None:
#     self.mean = np.concatenate((self.mean, state.mean), axis=0)
#     self.covar = np.concatenate((self.covar, state.covar), axis=0)
#     self.weight = np.concatenate((self.weight, state.weight), axis=0)


def mix_gaussians(means: np.ndarray,
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
  w = weights / (np.sum(weights) + 1e-15)

  mix_mean = np.dot(w, x)
  mix_covar = np.einsum('i, ijk->jk', w, P)
  mix_covar += np.einsum('i,ij,ik->jk', w, x, x)
  mix_covar -= np.outer(mix_mean, mix_mean)
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
  Si = np.linalg.inv(S)

  z_pred = np.atleast_2d(z_pred)
  z = np.atleast_2d(z)
  n, d = z_pred.shape
  m, _ = z.shape
  Si = Si.reshape(n, d, d)

  den = np.sqrt((2 * np.pi) ** d * np.linalg.det(S))[:, None]
  x = z.reshape(1, m, d) - z_pred.reshape(n, 1, d)
  l = np.exp(
      -0.5 * np.einsum('nmi, nii, nim -> nm', x, Si, x.swapaxes(-1, -2))) / den
  if z_pred.ndim == 1:
    l = l.squeeze(0)

  return l
