import functools
from typing import Tuple, Optional
import numpy as np
from scipy.linalg import block_diag


class CoordinatedTurn():
  """
  Coordinated turn model assuming piecewise constant white acceleration

  References:
    - Bostrom-rost2021 - Sensor Management for Search and Track Using the Poisson Multi-Bernoulli Mixture Filter
    - Li2003 - Survey of Maneuvering Target Tracking. Part I: Dynamic Models
  """

  def __init__(self,
               w_linear: float,
               w_turn: float,
               position_inds: Optional[np.ndarray] = None,
               velocity_inds: Optional[np.ndarray] = None,
               turn_rate_ind: Optional[int] = None,
               degrees: bool = False,
               ):
    self.state_dim = 5
    self.w_linear = w_linear
    self.w_turn = w_turn

    if position_inds is None:
      position_inds = np.array([0, 2])
    if velocity_inds is None:
      velocity_inds = np.array([1, 3])
    if turn_rate_ind is None:
      turn_rate_ind = -1
    self.position_inds = position_inds
    self.velocity_inds = velocity_inds
    self.turn_rate_ind = turn_rate_ind
    self.degrees = degrees

  def __call__(
      self,
      x: np.array,
      dt: float,
      noise: bool = False,
      rng: Optional[np.random.RandomState] = None,
  ) -> np.ndarray:

    # Extract position, velocity, and turn rate components from current state
    ipx, ipy = self.position_inds
    ivx, ivy = self.velocity_inds
    iw = self.turn_rate_ind
    px, py = x[..., ipx], x[..., ipy]
    vx, vy = x[..., ivx], x[..., ivy]
    w = x[..., iw] + 1e-10
    
    # Propagate next state - expressing in equation form for readability and state vector ordering flexibility
    if self.degrees:
      w = w * np.pi / 180
    next_state = np.zeros_like(x)
    SWT = np.sin(w * dt)
    CWT = np.cos(w * dt)
    next_state[..., ipx] = px + vx * SWT/w + vy * -(1-CWT)/w
    next_state[..., ivx] = vx * CWT + vy * -SWT
    next_state[..., ipy] = vx * (1-CWT)/w + py + vy * SWT/w
    next_state[..., ivy] = vx * SWT + vy * CWT
    if self.degrees:
      w = w * 180 / np.pi
    next_state[..., iw] = w

    if noise:
      next_state += self.sample_noise(
          covar=self.covar(dt=dt), size=x.shape[:-1], rng=rng
      )

    return next_state

  @functools.lru_cache()
  def covar(self, dt: float):
    ipos, ivel, iw = self.position_inds, self.velocity_inds, self.turn_rate_ind
    Q = np.zeros((self.state_dim, self.state_dim))
    Q[ipos, ipos] = self.w_linear * dt**4 / 4
    Q[ipos, ivel] = self.w_linear * dt**3 / 2
    Q[ivel, ipos] = self.w_linear * dt**3 / 2
    Q[ivel, ivel] = self.w_linear * dt**2
    Q[iw, iw] = self.w_turn
    return Q

  def sample_noise(
      self,
      covar: np.ndarray,
      size: Tuple[int, ...],
      rng: Optional[np.random.RandomState] = None,
  ) -> np.ndarray:
    if rng is None:
      rng = np.random.default_rng()
    return rng.multivariate_normal(
        mean=np.zeros(self.state_dim), cov=covar, size=size
    )
