import functools
from typing import Union, Literal
from scipy.linalg import block_diag
import numpy as np
from motpy.models.transition.base import TransitionModel


class ConstantVelocity(TransitionModel):
  """
  A nearly constant velocity (NCV) kinematic transition model with continuous or discrete white noise acceleration (C/DWNA).


  """

  def __init__(self,
               ndim: float,
               w: float,
               position_mapping: np.array = None,
               velocity_mapping: np.array = None,
               noise_type: Literal["continuous", "discrete"] = "continuous",
               seed: int = np.random.randint(0, 2**32-1),
               ):
    self.ndim = ndim
    self.ndim_state = 2*self.ndim
    self.w = w
    self.noise_type = noise_type.lower()

    self.position_mapping = position_mapping
    self.velocity_mapping = velocity_mapping
    if position_mapping is None:
      self.position_mapping = np.arange(0, self.ndim_state, 2)
    if velocity_mapping is None:
      self.velocity_mapping = np.arange(1, self.ndim_state, 2)

    self.np_random = np.random.RandomState(seed)

  def __call__(
      self,
      x: np.ndarray,
      dt: float = 0,
      noise: bool = False
  ) -> np.ndarray:
    next_state = x.astype(float)
    next_state[..., self.position_mapping] += x[..., self.velocity_mapping]*dt

    if noise:
      n = x.shape[0] if x.ndim > 1 else 1
      mean = np.zeros((self.ndim_state))
      covar = self.covar(dt)
      noise = self.np_random.multivariate_normal(mean=mean, cov=covar, size=n)
      next_state += noise

    return next_state

  @functools.lru_cache()
  def matrix(self, dt: float):
    F = np.zeros((self.ndim_state, self.ndim_state))

    pos, vel = self.position_mapping, self.velocity_mapping
    F[pos, pos] = 1
    F[pos, vel] = dt
    F[vel, vel] = 1
    return F

  @functools.lru_cache()
  def covar(self, dt: float):
    pos, vel = self.position_mapping, self.velocity_mapping
    Q = np.zeros((self.ndim_state, self.ndim_state))

    if self.noise_type == "continuous":
      Q[pos, pos] = dt**3 / 3
      Q[pos, vel] = dt**2 / 2
      Q[vel, pos] = dt**2 / 2
      Q[vel, vel] = dt
    elif self.noise_type == "discrete":
      Q[pos, pos] = dt**4 / 4
      Q[pos, vel] = dt**3 / 2
      Q[vel, pos] = dt**2 / 2
      Q[vel, vel] = dt**2
    Q *= self.w

    return Q
