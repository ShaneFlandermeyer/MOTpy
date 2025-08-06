import functools
from typing import List, Optional, Tuple, Union

import numpy as np

from motpy.models.measurement import MeasurementModel


def normalize_angle(x: np.ndarray) -> np.ndarray:
  """Normalize angles to the range [-pi, pi]."""
  return np.mod(x + np.pi, 2 * np.pi) - np.pi


class Radar2D(MeasurementModel):
  def __init__(self,
               covar: np.ndarray,
               pos_inds: List[int] = [0, 2],
               vel_inds: List[int] = [1, 3],
               ):
    """
    2D radar measurement model.

    Converts a cartesian input state to [range, azimuth, radial velocity] measurements.

    Parameters
    ----------
    covar : np.ndarray
        Sensor noise covariance matrix.
    pos_inds : List[int], optional
        Position state vector indices, by default [0, 2]
    vel_inds : List[int], optional
        Velocity state vector indices, by default [1, 3]
    """
    self.noise_covar = np.array(covar, dtype=float)
    self.pos_inds = pos_inds
    self.vel_inds = vel_inds
    self.measurement_dim = 3  # range, azimuth, radial velocity

  def __call__(self,
               x: np.ndarray,
               sensor_pos: np.ndarray,
               sensor_vel: np.ndarray,
               noise: bool = False,
               rng: Optional[np.random.RandomState] = None,
               **kwargs,
               ) -> np.ndarray:
    rel_pos = x[..., self.pos_inds] - sensor_pos
    rel_vel = x[..., self.vel_inds] - sensor_vel

    r = np.linalg.norm(rel_pos, axis=-1)
    az = normalize_angle(np.arctan2(rel_pos[..., 1], rel_pos[..., 0]))
    v = -np.einsum('...i, ...i->...', rel_pos, rel_vel) / (r + 1e-15)
    measurement = np.stack([r, az, v], axis=-1)

    if noise:
      measurement += self.sample_noise(size=measurement.shape[:-1], rng=rng)

    return measurement

  def covar(self):
    return self.noise_covar

  def sample_noise(
      self,
      size: Tuple[int, ...],
      rng: Optional[np.random.RandomState] = None
  ) -> np.ndarray:
    if rng is None:
      rng = np.random.default_rng()

    noise = rng.multivariate_normal(
        mean=np.zeros(self.measurement_dim), cov=self.noise_covar, size=size
    )
    return noise

  @staticmethod
  def subtract_fn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = a - b
    y[..., 1] = normalize_angle(y[..., 1])

    return y

  @staticmethod
  def average_fn(
      z: np.ndarray,
      axis: Optional[int] = None,
      weights: Optional[np.ndarray] = None
  ) -> np.ndarray:
    if weights is None:
      w = np.full((*z.shape[:-1], 1), 1 / z.shape[axis], dtype=float)
    else:
      w = abs(weights)[..., None]
      w = w / (np.sum(w, axis=axis, keepdims=True) + 1e-15)
    
    az_mean = np.arctan2(
        np.sum(w * np.sin(z[..., 1][..., None]), axis=axis),
        np.sum(w * np.cos(z[..., 1][..., None]), axis=axis)
    )
    r_mean = np.sum(w * z[..., 0][..., None], axis=axis)
    v_mean = np.sum(w * z[..., 2][..., None], axis=axis)

    return np.concatenate([r_mean, az_mean, v_mean], axis=-1)


if __name__ == '__main__':
  radar2d = Radar2D(covar=np.eye(3))

  x = np.array([10, 1, 10, 1])
  z = radar2d(
      x=x,
      sensor_pos=np.array([0, 0]),
      sensor_vel=np.array([0, 0]),
      noise=False,
      rng=None,
  )
  zm = radar2d.average_fn(z=np.stack(
      [z, z], axis=0), weights=np.array([0.5, 0.5]))
  print(zm-z)
