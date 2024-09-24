from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple

from motpy.kalman import KalmanFilter
from motpy.distributions.gaussian import Gaussian
import numpy as np


class Poisson:
  """
  Class to hold all poisson distributions. Methods include birth, prediction, merge, prune, recycle.
  """

  def __init__(
      self,
      birth_state: Gaussian,
      state: Optional[Gaussian] = None,
      static: Optional[bool] = False
  ):
    self.birth_state = birth_state
    self.state = state
    self.static = static

  def __repr__(self):
    return f"""Poisson(birth_distribution={self.birth_state},
  distribution={self.state})"""

  @property
  def shape(self) -> Tuple[int]:
    return self.state.shape

  @property
  def size(self) -> int:
    return self.state.size

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float) -> Poisson:
    predicted = copy.deepcopy(self)
    predicted.state.weight *= ps

    if not self.static:
      predicted.state, filter_state = state_estimator.predict(
          state=predicted.state, dt=dt)
      predicted.state = predicted.state.append(self.birth_state)

    return predicted

  def prune(
          self,
          threshold: float,
          meta: Optional[List[Dict[str, Any]]] = None
  ) -> Poisson:
    """
    Prune components below weight threshold
    """
    pruned = copy.deepcopy(self)
    valid = pruned.state.weight > threshold
    pruned.state = pruned.state[valid]

    if meta is None:
      new_meta = None
    else:
      new_meta = [meta[i] for i in range(len(meta)) if valid[i]]

    return pruned, new_meta

  def merge(self) -> Poisson:
    """
    Merge the birth distribution back into the main distribution.

    NOTE: This assumes that only the birth distribution contributes, such that we only need to change the weights and covariance matrices.
    """
    assert isinstance(self.birth_state, Gaussian)
    merged = copy.deepcopy(self)

    if self.static:
      merged_state = Gaussian(
          mean=self.state.mean,
          covar=self.state.covar,
          weight=self.state.weight + self.birth_state.weight,
      )
    else:
      nbirth = self.birth_state.size
      state, birth_state = self.state[:nbirth], self.birth_state
      wmix = np.stack((state.weight, birth_state.weight), axis=0)
      wmix /= np.sum(wmix + 1e-15, axis=0)
      xmix = np.stack((state.mean, birth_state.mean), axis=0)
      Pmix = np.stack((state.covar, birth_state.covar), axis=0)
      merged_state = Gaussian(
          mean=np.einsum('i..., i...j -> ...j', wmix, xmix),
          covar=np.einsum('i..., i...jk -> ...jk', wmix, Pmix),
          weight=state.weight + birth_state.weight,
      )
      
    
    
    merged.state = merged_state
    return merged
