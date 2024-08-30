import functools
from typing import Union, Literal, Optional
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
               position_inds: Optional[np.ndarray] = None,
               velocity_inds: Optional[np.ndarray] = None,
               noise_type: Literal["continuous", "discrete"] = "continuous",
               seed: int = np.random.randint(0, 2**32-1),
               ):
    self.ndim = ndim
    self.ndim_state = 2*self.ndim
    self.w = w
    self.noise_type = noise_type.lower()

    if position_inds is None:
      position_inds = np.arange(0, self.ndim_state, 2)
    if velocity_inds is None:
      velocity_inds = np.arange(1, self.ndim_state, 2)
    self.position_inds = position_inds
    self.velocity_inds = velocity_inds

    self.np_random = np.random.RandomState(seed)

  def __call__(
      self,
      x: np.ndarray,
      dt: float = 0,
      noise: bool = False,
      **_
  ) -> np.ndarray:
    next_state = x.astype(float)
    next_state[..., self.position_inds] += x[..., self.velocity_inds]*dt

    if noise:
      n_samples = x.shape[0] if x.ndim > 1 else 1
      mean = np.zeros((self.ndim_state))
      covar = self.covar(dt)
      noise = self.np_random.multivariate_normal(mean, covar, size=n_samples)
      next_state += noise.reshape(next_state.shape)

    return next_state

  @functools.lru_cache()
  def matrix(self, dt: float, **_):
    F = np.zeros((self.ndim_state, self.ndim_state))

    pos, vel = self.position_inds, self.velocity_inds
    F[pos, pos] = 1
    F[pos, vel] = dt
    F[vel, vel] = 1
    return F

  @functools.lru_cache()
  def covar(self, dt: float, **_):
    ipos, ivel = self.position_inds, self.velocity_inds
    Q = np.zeros((self.ndim_state, self.ndim_state))

    if self.noise_type == "continuous":
      Q[ipos, ipos] = dt**3 / 3
      Q[ipos, ivel] = dt**2 / 2
      Q[ivel, ipos] = dt**2 / 2
      Q[ivel, ivel] = dt
    elif self.noise_type == "discrete":
      Q[ipos, ipos] = dt**4 / 4
      Q[ipos, ivel] = dt**3 / 2
      Q[ivel, ipos] = dt**2 / 2
      Q[ivel, ivel] = dt**2
    Q *= self.w

    return Q
