from __future__ import annotations
import copy
import numpy as np
from motpy.kalman import KalmanFilter
from motpy.distributions.gaussian import GaussianState
from typing import Dict, Tuple, Optional, List, Union


class MultiBernoulli():
  def __init__(self,
               state: Optional[GaussianState] = None,
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

  def append(self, state: GaussianState, r: np.ndarray) -> MultiBernoulli:
    if self.state is not None:
      state = self.state.append(state)

    if self.r is not None:
      r = np.append(self.r, r)

    return MultiBernoulli(state=state, r=r)

  def predict(self,
              state_estimator: KalmanFilter,
              dt: float,
              ps: float,
              filter_state: Optional[Dict] = None) -> MultiBernoulli:
    predicted_state, filter_state = state_estimator.predict(
        state=self.state, dt=dt, filter_state=filter_state)

    predicted_mb = MultiBernoulli(r=self.r * ps, state=predicted_state)
    return predicted_mb, filter_state
