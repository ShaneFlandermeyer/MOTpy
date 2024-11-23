import functools
from typing import Tuple, Union, Literal, Optional
from scipy.linalg import block_diag
import numpy as np
from motpy.models.transition.base import TransitionModel


class ConstantVelocity(TransitionModel):
  """
  A nearly constant velocity (NCV) kinematic transition model with continuous or discrete white noise acceleration (C/DWNA).
  """

  def __init__(self,
               ndim_state: float,
               w: float,
               position_inds: Optional[np.ndarray] = None,
               velocity_inds: Optional[np.ndarray] = None,
               noise_type: Literal["continuous", "discrete"] = "continuous",
               seed: int = np.random.randint(0, 2**32-1),
               ):
    self.ndim = ndim_state
    self.w = w
    self.noise_type = noise_type.lower()

    if position_inds is None:
      position_inds = np.arange(0, self.ndim, 2)
    if velocity_inds is None:
      velocity_inds = np.arange(1, self.ndim, 2)
    self.position_inds = position_inds
    self.velocity_inds = velocity_inds

    self.np_random = np.random.RandomState(seed)

  def __call__(
      self,
      x: np.ndarray,
      dt: float = 0,
      noise: bool = False,
  ) -> np.ndarray:
    next_x = np.array(x).astype(float)
    next_x[..., self.position_inds] += x[..., self.velocity_inds]*dt

    if noise:
      next_x += self.sample_noise(covar=self.covar(dt=dt), size=x.shape[:-1])

    return next_x

  @functools.lru_cache()
  def matrix(self, dt: float):
    F = np.zeros((self.ndim, self.ndim))

    pos, vel = self.position_inds, self.velocity_inds
    F[pos, pos] = 1
    F[pos, vel] = dt
    F[vel, vel] = 1
    return F

  @functools.lru_cache()
  def covar(self, dt: float):
    ipos, ivel = self.position_inds, self.velocity_inds
    Q = np.zeros((self.ndim, self.ndim))

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

  def sample_noise(self, covar, size: Tuple[int, ...]) -> np.ndarray:
    return self.np_random.multivariate_normal(
        mean=np.zeros(self.ndim), cov=covar, size=size)
