import functools
import numpy as np
from typing import List, Tuple, Callable
from scipy.stats import norm, chi2
import math

from motpy.distributions.gaussian import mahalanobis


def ellipsoidal_gate(
    pg: float,
    ndim: int,
    x: np.ndarray,
    mean: np.ndarray,
    covar: np.ndarray,
    subtract_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.subtract,
) -> Tuple[np.ndarray, np.ndarray]:
  
  dist = mahalanobis(x=x, mean=mean, covar=covar, subtract_fn=subtract_fn)
  t = gate_threshold(pg=pg, ndim=ndim)
  in_gate = dist**2 < t
  return in_gate, dist


@functools.lru_cache
def gate_threshold(pg: float, ndim: int) -> float:
  return chi2.ppf(pg, ndim)


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


def gate_volume(pg: float, ndim: int, covar: np.ndarray) -> float:
  gamma = gate_threshold(pg=pg, ndim=ndim)

  c = np.pi**(ndim/2) / math.gamma(ndim/2+1)
  volume = c*gamma**(ndim/2) * np.sqrt(np.linalg.det(covar))
  return volume
