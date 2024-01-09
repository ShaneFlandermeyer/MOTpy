import datetime
from typing import List, Tuple, Union, Dict

import numpy as np
from tensordict import TensorDict
import torch


class GaussianState(TensorDict):
  def __init__(self,
               mean: torch.Tensor,
               covar: torch.Tensor):
    assert mean.shape[0] == covar.shape[0]
    batch_size = mean.shape[0]
    source = dict(mean=mean, covar=covar)
    super().__init__(source=source, batch_size=batch_size)


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
  z = np.atleast_2d(z)
  z_pred = z_pred.flatten()
  S = H @ P_pred @ H.T + R
  Si = np.linalg.inv(S)

  k = z_pred.size
  den = np.sqrt((2 * np.pi) ** k * np.linalg.det(S))
  x = z - z_pred
  l = np.squeeze(np.exp(-0.5 * x[..., None, :] @ Si @ x[..., None])) / den
  return l
