import functools
from typing import List, Optional, Tuple, Union

import numpy as np

from motpy.models.measurement.base import MeasurementModel


class LinearMeasurementModel(MeasurementModel):
  def __init__(self,
               state_dim: int,
               covar: np.ndarray,
               measured_dims: Optional[np.ndarray] = None,
               seed: int = np.random.randint(0, 2**32-1),
               ):

    self.state_dim = state_dim
    self.noise_covar = np.array(covar, dtype=float)
    self.np_random = np.random.RandomState(seed)

    if measured_dims is None:
      measured_dims = np.arange(self.noise_covar.shape[0])
    self.measured_dims = np.array(measured_dims, dtype=int)
    self.measurement_dim = len(measured_dims)

  def __call__(self,
               x: np.ndarray,
               noise: bool = False,
               **kwargs,
               ) -> List[np.ndarray]:
    out = x[..., self.measured_dims].astype(float)

    if noise:
      out += self.sample_noise(size=out.shape[:-1])

    return out

  @functools.lru_cache(maxsize=1)
  def matrix(self):
    H = np.zeros((self.measurement_dim, self.state_dim))
    H[np.arange(self.measurement_dim), self.measured_dims] = 1
    return H

  def covar(self):
    return self.noise_covar

  def sample_noise(self, size: Tuple[int, ...]) -> np.ndarray:
    noise = self.np_random.multivariate_normal(
        mean=np.zeros(self.measurement_dim), cov=self.noise_covar, size=size)
    return noise
