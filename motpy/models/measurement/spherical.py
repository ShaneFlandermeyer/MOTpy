import functools
from typing import List, Optional, Tuple, Union

import numpy as np

from motpy.models.measurement import MeasurementModel


class RangeAzimuthVelocity2D(MeasurementModel):
  def __init__(self,
               covar: np.ndarray,
               pos_inds: List[int] = [0, 2],
               vel_inds: List[int] = [1, 3],
               seed: int = np.random.randint(0, 2**32-1),
               ):
    # TODO: Some noise covars depend on the sensor state
    self.noise_covar = np.array(covar, dtype=float)
    self.pos_inds = pos_inds
    self.vel_inds = vel_inds

    self.np_random = np.random.RandomState(seed)

  def __call__(self,
               x: np.ndarray,
               sensor_state: np.ndarray,
               noise: bool = False
               ) -> np.ndarray:
    sensor_pos = sensor_state[..., self.pos_inds]
    sensor_vel = sensor_state[..., self.vel_inds]
    
    rel_pos = x[..., self.pos_inds] - sensor_pos
    rel_vel = x[..., self.vel_inds] - sensor_vel

    r = np.linalg.norm(rel_pos, axis=-1)
    az = np.arctan2(rel_pos[..., 1], rel_pos[..., 0])
    # Negative by convention (positive velocity = moving towards radar)
    v = -np.einsum('...i, ...i->...', rel_pos, rel_vel) / (r + 1e-15)

    measurement = np.stack([r, az, v], axis=-1)

    if noise:
      measurement += self.sample_noise(size=measurement.shape[:-1])

    # print(measurement)
    return measurement

  def covar(self):
    return self.noise_covar

  def sample_noise(self, size: Tuple[int, ...]) -> np.ndarray:
    measurement_dim = 3
    noise = self.np_random.multivariate_normal(
        mean=np.zeros(measurement_dim), cov=self.noise_covar, size=size
    )
    return noise
