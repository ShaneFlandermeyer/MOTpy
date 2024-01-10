import datetime
from typing import List, Tuple, Union, Dict

import numpy as np
from tensordict import TensorDict
import torch


class GaussianState(TensorDict):
  def __init__(self,
               mean: torch.Tensor,
               covar: torch.Tensor):
    # Handle batch dimension
    mean = torch.atleast_2d(mean)
    if covar.ndim == 2:
      covar = covar.view(1, *covar.shape)

    assert mean.shape[0] == covar.shape[0]

    batch_size = mean.shape[0]
    source = dict(mean=mean, covar=covar)
    super().__init__(source=source, batch_size=batch_size)


def mix_gaussians(means: torch.Tensor,
                  covars: torch.Tensor,
                  weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
  x, P = means, covars
  w = weights / (torch.sum(weights) + 1e-15)

  mix_mean = torch.einsum('i, ij -> j', w, x)
  mix_covar = torch.einsum('i, ijk -> jk', w, P)
  mix_covar += torch.einsum('i, ij, ik -> jk', w, x, x)
  mix_covar -= torch.outer(mix_mean, mix_mean)
  return mix_mean, mix_covar


def likelihood(z: torch.Tensor,
               z_pred: torch.Tensor,
               P_pred: torch.Tensor,
               H: torch.Tensor,
               R: torch.Tensor,
               ) -> float:
  S = H @ P_pred @ H.T + R
  Si = torch.linalg.inv(S)

  k = z_pred.shape[-1]
  den = torch.sqrt((2 * torch.pi) ** k * torch.linalg.det(S))
  x = z - z_pred
  l = torch.exp(-0.5 * torch.einsum('...i, ...ij, j...', x, Si, x.T)) / den
  return l
