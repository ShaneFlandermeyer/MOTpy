from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from motpy.distributions import Distribution
from motpy.estimators import StateEstimator


class MultiBernoulli():
  def __init__(self,
               state: Optional[Distribution] = None,
               r: Optional[np.ndarray] = None
               ) -> None:
    self.state = state
    self.r = r

  def __repr__(self) -> str:
    return f"""MultiBernoulli(
      r={self.r}
      state={self.state})"""

  @property
  def shape(self) -> Tuple[int]:
    return self.state.shape if self.state is not None else ()

  @property
  def size(self) -> int:
    return self.state.size if self.state is not None else 0

  def __getitem__(self, idx) -> MultiBernoulli:
    return MultiBernoulli(state=self.state[idx], r=self.r[idx])

  def __setitem__(self, idx, value: MultiBernoulli) -> None:
    self.r[idx] = value.r
    self.state[idx] = value.state

  def append(self, state: Distribution, r: np.ndarray) -> MultiBernoulli:
    if self.state is not None:
      state = self.state.append(state)

    if self.r is not None:
      r = np.append(self.r, r)

    return MultiBernoulli(state=state, r=r)

  def predict(self,
              state_estimator: StateEstimator,
              dt: float,
              ps: float,
              **kwargs
              ) -> MultiBernoulli:
    predicted_state = state_estimator.predict(
        state=self.state, dt=dt, **kwargs
    )

    predicted_mb = MultiBernoulli(r=self.r * ps, state=predicted_state)
    return predicted_mb

  def prune(self: MultiBernoulli,
            threshold: float = 1e-4,
            meta: Optional[Dict[str, Any]] = None
            ) -> Tuple[MultiBernoulli, Optional[Dict[str, Any]]]:
    pruned = copy.deepcopy(self)

    valid = self.r > threshold
    pruned = pruned[valid]

    if meta is None:
      new_meta = None
    else:
      new_meta = [meta[i] for i in range(len(meta)) if valid[i]]

    return pruned, new_meta
