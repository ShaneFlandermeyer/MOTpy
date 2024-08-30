import functools
from typing import List, Optional, Union

import numpy as np

from motpy.models.measurement.base import MeasurementModel


class LinearMeasurementModel(MeasurementModel):
  def __init__(self,
               ndim_state: int,
               covar: np.ndarray,
               measured_dims: Optional[np.ndarray] = None,
               seed: int = np.random.randint(0, 2**32-1),
               ):

    self.ndim_state = ndim_state
    self.noise_covar = np.array(covar, dtype=float)
    self.np_random = np.random.RandomState(seed)

    if measured_dims is None:
      measured_dims = np.arange(self.noise_covar.shape[0])
    self.measured_dims = np.array(measured_dims, dtype=int)
    self.ndim = len(measured_dims)

  def __call__(self,
               x: np.ndarray,
               noise: bool = False,
               **_
               ) -> List[np.ndarray]:
    out = x[..., self.measured_dims].astype(float)

    if noise:
      n_measurements = x.shape[0] if x.ndim > 1 else 1
      noise = self.sample_noise(size=n_measurements)
      out = out + noise.reshape(out.shape)

    return out

  @functools.lru_cache(maxsize=1)
  def matrix(self, **_):
    H = np.zeros((self.ndim, self.ndim_state))
    H[np.arange(self.ndim), self.measured_dims] = 1
    return H

  def covar(self, **_):
    return self.noise_covar

  def sample_noise(self, size: int = 1):
    noise = self.np_random.multivariate_normal(
        mean=np.zeros(self.ndim), cov=self.noise_covar, size=size)
    return noise
