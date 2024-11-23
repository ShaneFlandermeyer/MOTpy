import functools
import numpy as np
from typing import List, Tuple
from scipy.stats import norm, chi2
import math

from motpy.distributions.gaussian import mahalanobis


class EllipsoidalGate:
  def __init__(self, pg: float, ndim: int):
    self.pg = pg
    self.ndim = ndim

  def __call__(self,
               x: np.ndarray,
               mean: np.ndarray,
               covar: np.ndarray,
               ) -> Tuple[np.ndarray, np.ndarray]:
    # Thresholding for all pairs
    dist = mahalanobis(
        x=x,
        mean=mean,
        covar=covar
    )
    t = self.threshold(pg=self.pg, ndim=self.ndim)
    in_gate = dist**2 < t
    return in_gate, dist

  @staticmethod
  @functools.lru_cache
  def threshold(pg: int, ndim: int) -> float:
    return chi2.ppf(pg, ndim)

  def volume(self, covar: np.ndarray):
    gamma = self.threshold(pg=self.pg, ndim=self.ndim)

    c = np.pi**(self.ndim/2) / math.gamma(self.ndim/2+1)
    return c*gamma**(self.ndim/2) * np.sqrt(np.linalg.det(covar))


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