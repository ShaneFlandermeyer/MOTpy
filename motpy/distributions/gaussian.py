from __future__ import annotations

import datetime
from typing import List, Tuple, Union, Dict

import numpy as np
import torch
from tensordict import TensorDict


class GaussianState():
  def __init__(
      self,
      mean: torch.Tensor,
      covar: torch.Tensor,
      cache: TensorDict = None,
  ):
    self.state_dim = mean.shape[-1]
    self.mean = mean.view(-1, self.state_dim)
    self.covar = covar.view(-1, self.state_dim, self.state_dim)
    self.cache = cache if cache is not None else TensorDict({}, batch_size=[])

  def __len__(self):
    return len(self.mean)

  def __repr__(self):
    return f"""GaussianState(
      mean={self.mean}
      covar=\n{self.covar})
      cache={self.cache})"""

  def __getitem__(self, idx):
    cache = self.cache[idx] if len(self.cache) > 0 else self.cache
    return GaussianState(self.mean[idx], self.covar[idx], cache)

  def append(self, state: GaussianState):
    self.mean = torch.cat([self.mean, state.mean])
    self.covar = torch.cat([self.covar, state.covar])
    
    if len(state.cache) < 0:
      return 
    
    if len(self.cache) > 0:
      self.cache = torch.cat([self.cache, state.cache])
    else:
      self.cache = state.cache


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
  x = means
  P = covars
  w = weights / (torch.sum(weights) + 1e-15)

  mix_mean = w @ x
  mix_covar = torch.einsum('i,ijk->jk', w, P)
  mix_covar += torch.einsum('i,ij,ik->jk', w, x, x)
  mix_covar -= torch.outer(mix_mean, mix_mean)
  return mix_mean, mix_covar


def likelihood(z: torch.Tensor,
               z_pred: torch.Tensor,
               P_pred: torch.Tensor,
               H: torch.Tensor,
               R: torch.Tensor,
               ) -> float:
  # z = np.atleast_2d(z)
  # z_pred = z_pred.flatten()
  S = H @ P_pred @ H.T + R
  Si = torch.linalg.inv(S)

  k = z_pred.numel()
  den = np.sqrt((2 * torch.pi) ** k * torch.linalg.det(S))
  x = z - z_pred
  l = torch.exp(-0.5 * torch.einsum('...i, ...ij, j... -> ...', x, Si, x.T)) / den
  return l
