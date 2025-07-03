from __future__ import annotations
from typing import *
from motpy.estimators import StateEstimator
from motpy.distributions import Distribution


class Poisson:
  """
  Class to hold all poisson distributions. Methods include birth, prediction, merge, prune, recycle.
  """

  def __init__(
      self, state: Optional[Distribution] = None,
  ):
    self.state = state

  def __repr__(self):
    return f"""Poisson(
  distribution={self.state}
  )"""

  @property
  def shape(self) -> Tuple[int]:
    return self.state.shape

  @property
  def size(self) -> int:
    return self.state.size

  def __len__(self) -> int:
    return self.state.shape[0] if self.state is not None else 0

  def predict(self,
              state_estimator: StateEstimator,
              birth_distribution: Poisson,
              ps: float,
              dt: float,
              **kwargs
              ) -> Tuple[Poisson, Dict[str, Any]]:
    pred_state = state_estimator.predict(
        state=self.state, dt=dt, **kwargs
    )
    pred_state.weight = ps * self.state.weight
    pred_state = pred_state.append(birth_distribution)
    pred_poisson = Poisson(state=pred_state)

    return pred_poisson

  def prune(
          self,
          threshold: float,
          meta: Optional[List[Dict[str, Any]]] = None
  ) -> Poisson:
    """
    Prune components below weight threshold
    """
    valid = self.state.weight > threshold
    new_poisson = Poisson(state=self.state[valid])

    if meta is None:
      new_meta = None
    else:
      new_meta = [meta[i] for i in range(len(meta)) if valid[i]]

    return new_poisson, new_meta
