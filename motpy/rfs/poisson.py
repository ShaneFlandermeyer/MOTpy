from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple

from motpy.estimators import StateEstimator
from motpy.distributions import Distribution
import numpy as np


class Poisson:
  """
  Class to hold all poisson distributions. Methods include birth, prediction, merge, prune, recycle.
  """

  def __init__(
      self,
      birth_distribution: Distribution,
      state: Optional[Distribution] = None,
  ):
    self.birth_distribution = birth_distribution
    self.state = state

  def __repr__(self):
    return f"""Poisson(birth_distribution={self.birth_distribution},
  distribution={self.state})"""

  @property
  def shape(self) -> Tuple[int]:
    return self.state.shape

  @property
  def size(self) -> int:
    return self.state.size

  def predict(self,
              state_estimator: StateEstimator,
              ps: float,
              dt: float,
              **kwargs
              ) -> Tuple[Poisson, Dict[str, Any]]:
    pred_state = state_estimator.predict(
        state=self.state, dt=dt, **kwargs
    )
    pred_state.weight = ps * self.state.weight
    pred_state = pred_state.append(self.birth_distribution)
    pred_poisson = Poisson(
        birth_distribution=self.birth_distribution,
        state=pred_state
    )

    return pred_poisson

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
      pruned_meta = None
    else:
      pruned_meta = [meta[i] for i in range(len(meta)) if valid[i]]

    return pruned, pruned_meta
