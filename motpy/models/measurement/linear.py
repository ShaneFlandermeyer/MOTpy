from typing import List, Union

import numpy as np

from motpy.models.measurement.base import MeasurementModel


class LinearMeasurementModel(MeasurementModel):
  def __init__(self,
               ndim_state: int,
               covar: np.ndarray,
               measured_dims: np.ndarray = None,
               seed: int = np.random.randint(0, 2**32-1),
               ):
    if measured_dims is None:
      measured_dims = np.arange(self.noise_covar.shape[0])
    
    self.ndim_state = ndim_state  
    self.noise_covar = np.array(covar)
    self.measured_dims = np.array(measured_dims, dtype=int)
    self.ndim = len(measured_dims)
    self.np_random = np.random.RandomState(seed)

  def __call__(self,
               x: List[np.ndarray],
               noise: bool = False) -> List[np.ndarray]:
    xarr = np.atleast_2d(x)
    n_measurements = xarr.shape[0]
    
    out = x if self.measured_dims is None else xarr[:, self.measured_dims]
    if noise:
      noise = self.sample_noise(size=n_measurements)
      out = out.astype(noise.dtype) + noise.reshape(out.shape)
    return list(out) if xarr.shape[0] > 1 else out[0]

  def matrix(self, **kwargs):
    H = np.zeros((self.ndim, self.ndim_state))
    H[np.arange(self.ndim), self.measured_dims] = 1
    return H

  def covar(self, **kwargs):
    return self.noise_covar

  def sample_noise(self, size: int = 1):
    noise = self.np_random.multivariate_normal(
        mean=np.zeros(self.ndim), cov=self.noise_covar, size=size)
    return noise
