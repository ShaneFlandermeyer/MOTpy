import functools
import numpy as np
from typing import List, Tuple
from scipy.stats import norm, chi2
import math


class EllipsoidalGate:
  def __init__(self, pg: float, ndim: int):
    self.pg = pg
    self.ndim = ndim

  def __call__(self,
               measurements: List[np.ndarray],
               predicted_measurement: np.ndarray,
               innovation_covar: np.ndarray
               ) -> Tuple[List[np.ndarray], np.ndarray]:
    z = np.array(measurements).reshape((-1, self.ndim, 1))
    z_pred = predicted_measurement.reshape((-1, 1))
    S = innovation_covar
    y = z - z_pred
    Si = np.linalg.inv(S)

    # Compute the distance for all measurements
    dist = np.einsum('nji, ii, nij -> n', y.swapaxes(-1, -2), Si, y)

    # Return measurements in the gate and their indices
    t = self.threshold(pg=self.pg, ndim=self.ndim)
    in_gate = dist < t
    valid_inds = np.where(in_gate)[0]
    valid_meas = [z[i].reshape(measurements[0].shape) for i in valid_inds]
    return valid_meas, valid_inds

  @staticmethod
  @functools.lru_cache
  def threshold(pg: int, ndim: int):
    return chi2.ppf(pg, ndim)

  def volume(self, innovation_covar: np.ndarray):
    gamma = self.threshold
    S = innovation_covar

    c = np.pi**(self.ndim/2) / math.gamma(self.ndim/2+1)
    return c*gamma**(self.ndim/2) * np.sqrt(np.linalg.det(S))


def gate_probability(threshold: float, ndim: int) -> float:
  """
  Compute the probability of a measurement falling within the ellipsoidal gate.

  Parameters
  ----------
  gate_dim : int
      _description_
  threshold : float
      _description_

  Returns
  -------
  float
      _description_
  """
  # Standard Gaussian probability integral
  def gc(G): return norm.cdf(G) - norm.cdf(0)
  G = threshold
  sqrt_G = np.sqrt(G)
  if ndim == 1:
    return 2*gc(sqrt_G)
  elif ndim == 2:
    return 1 - np.exp(-G/2)
  elif ndim == 3:
    return 2*gc(sqrt_G) - np.sqrt(2*G/np.pi)*np.exp(-G/2)
  elif ndim == 4:
    return 1 - (1+G/2)*np.exp(-G/2)
  elif ndim == 5:
    return 2*gc(sqrt_G) - (1+G/3)*np.sqrt(2*G/np.pi)*np.exp(-G/2)
  elif ndim == 6:
    return 1 - 0.5*(G**2/4+G+2)*np.exp(-G/2)


if __name__ == '__main__':
  measurements = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
  z_pred = np.array([1, 1])
  S = np.eye(2)
  pg = 0.99

  gate = EllipsoidalGate(pg=pg, ndim=2)

  print(gate(measurements=measurements,
        predicted_measurement=z_pred, innovation_covar=S))
  print(gate.volume(innovation_covar=S))