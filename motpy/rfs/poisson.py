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
  ):
    self.birth_state = birth_state
    self.state = state

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