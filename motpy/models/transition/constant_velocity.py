import functools
from typing import Union
from scipy.linalg import block_diag
import numpy as np
from motpy.models.transition.base import TransitionModel


class ConstantVelocity(TransitionModel):
  r"""This is a class implementation of a discrete, time-variant 1D
    Linear-Gaussian Constant Velocity Transition Model.

    The target is assumed to move with (nearly) constant velocity, where
    target acceleration is modelled as white noise.

    This is a faster than the :class:`ConstantVelocity` model in StoneSoup since it can be vectorized and avoids the for loops in the transition and covariance matrix computations. This also makes it less extensible to higher-order motion models.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel} d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & q\cdot dW_t,\ W_t \sim \mathcal{N}(0,q^2) & | \
                Speed on\ X-axis (m/s)
            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel}
                \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                        1 & dt\\
                        0 & 1
                \end{bmatrix}

        .. math::
            Q_t & = & \begin{bmatrix}
                        \frac{dt^3}{3} & \frac{dt^2}{2} \\
                        \frac{dt^2}{2} & dt
                \end{bmatrix} q
    """

  def __init__(self,
               ndim_pos: float,
               q: float,
               position_mapping: np.array = None,
               velocity_mapping: np.array = None,
               seed: int = np.random.randint(0, 2**32-1),
               ):
    self.ndim_pos = ndim_pos
    self.ndim = self.ndim_pos*2
    self.q = q

    self.position_mapping = position_mapping
    self.velocity_mapping = velocity_mapping
    if position_mapping is None:
      self.position_mapping = np.arange(0, self.ndim, 2)
    if velocity_mapping is None:
      self.velocity_mapping = np.arange(1, self.ndim, 2)
    self.np_random = np.random.RandomState(seed)

  def __call__(
      self,
      x: np.ndarray,
      dt: float = 0,
      noise: bool = False
  ) -> np.ndarray:
    next_state = x.copy().astype(float)
    next_state[self.position_mapping] += x[self.velocity_mapping]*dt
    if noise:
      next_state += self.sample_noise(dt).reshape(x.shape)
    return next_state

  @functools.lru_cache()
  def matrix(self, dt: float):
    # TODO: Account for state mapping
    F = np.array([[1, dt],
                  [0, 1]])
    F = block_diag(*[F]*self.ndim_pos)
    return F.astype(float)

  @functools.lru_cache()
  def covar(self, dt: float):
    # TODO: Account for state mapping
    Q = np.array([[1/3*dt**3, 1/2*dt**2],
                  [1/2*dt**2, dt]])
    Q = block_diag(*[Q for _ in range(self.ndim_pos)]) * self.q
    return Q.astype(float)

  def sample_noise(self,
                   dt: float = 0) -> np.array:
    covar = self.covar(dt)
    noise = self.np_random.multivariate_normal(
        mean=np.zeros((self.ndim)), cov=covar)
    return np.asarray(noise)
