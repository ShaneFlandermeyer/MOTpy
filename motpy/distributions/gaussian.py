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
    self.state_dim = mean.shape[-1]
    self.mean = np.atleast_2d(mean)
    self.covar = covar.reshape(-1, self.state_dim, self.state_dim)

    

  def __repr__(self):
    return f"""GaussianState(
      mean={self.mean}
      covar=\n{self.covar})
      """

  def __len__(self):
    return len(self.mean)

  def __getitem__(self, idx):
    return GaussianState(mean=self.mean[idx], covar=self.covar[idx])

  def append(self, state: GaussianState) -> None:
    self.mean = np.concatenate((self.mean, state.mean), axis=0)
    self.covar = np.concatenate((self.covar, state.covar), axis=0)


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
  mix_covar = np.einsum('i,ijk->jk', w, P)
  mix_covar += np.einsum('i,ij,ik->jk', w, x, x)
  mix_covar -= np.outer(mix_mean, mix_mean)
  return mix_mean, mix_covar


def likelihood(z: np.ndarray,
               z_pred: np.ndarray,
               P_pred: np.ndarray,
               H: np.ndarray,
               R: np.ndarray,
               ) -> float:
  S = H @ P_pred @ H.T + R
  Si = np.linalg.inv(S)

  k = z_pred.shape[-1]
  den = np.sqrt((2 * np.pi) ** k * np.linalg.det(S))
  x = z - z_pred
  l = np.exp(-0.5 * np.einsum('...i, ...ij, j...', x, Si, x.T)) / den
  return l
