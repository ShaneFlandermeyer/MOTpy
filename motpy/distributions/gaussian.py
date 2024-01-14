from __future__ import annotations
import datetime
from typing import List, Tuple, Union, Dict

import numpy as np


class GaussianState():
  def __init__(
      self,
      mean: np.ndarray,
      covar: np.ndarray,
  ):
    self.mean = mean
    self.covar = covar

  def __repr__(self):
    return f"""GaussianState(
      mean={self.mean}
      covar=\n{self.covar})
      """


class GaussianMixture():
  def __init__(
      self,
      means: np.ndarray,
      covars: np.ndarray,
      weights: np.ndarray,
  ):
    self.state_dim = means.shape[-1]
    self.means = means.reshape(-1, self.state_dim)
    self.covars = covars.reshape(-1, self.state_dim, self.state_dim)
    self.weights = np.atleast_1d(weights)

  def __repr__(self):
    return f"""GaussianMixture(
      n_components={len(self)},
      means={self.means}
      covars=\n{self.covars})
      weights={self.weights}
      """

  def __len__(self):
    return len(self.means)

  def __getitem__(self, idx):
    return GaussianMixture(means=self.means[idx], covars=self.covars[idx], weights=self.weights[idx])

  def append(self, state: GaussianMixture) -> None:
    self.means = np.concatenate((self.means, state.means), axis=0)
    self.covars = np.concatenate((self.covars, state.covars), axis=0)
    self.weights = np.concatenate((self.weights, state.weights), axis=0)


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
               P_pred: np.ndarray,
               H: np.ndarray,
               R: np.ndarray,
               ) -> np.ndarray:
  """
  Compute the likelihood for a set of measurement/state pairs.

  TODO: Replace P_pred, H, and R with innovation covariance matrix S

  Parameters
  ----------
  z : np.ndarray
      Array of measurements. Shape: (M, nz)
  z_pred : np.ndarray
      Array of predicted measurements. Shape: (N, nz)
  P_pred : np.ndarray
      Predicted state covariance. Shape: (N, nx, nx)
  H : np.ndarray
      Measurement model matrix. Shape: (nz, nx)
  R : np.ndarray
      Measurement noise covariance. Shape: (nz, nz)

  Returns
  -------
  np.ndarray
      Likelihood for each measurement/state pair. Shape: (M, N)
  """
  S = H @ P_pred @ H.swapaxes(-1, -2) + R
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
