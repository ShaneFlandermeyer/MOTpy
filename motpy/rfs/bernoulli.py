from __future__ import annotations
import copy
import numpy as np
from motpy.kalman import KalmanFilter
from motpy.distributions.gaussian import GaussianState
from typing import Dict, Tuple, Optional, List, Union


class MultiBernoulli():
  def __init__(self,
               r: np.ndarray = None,
               state: GaussianState = None):
    self.r = r if r is not None else np.empty(0)
    self.state = state

  def __repr__(self) -> str:
    return f"""MultiBernoulli(
      r={self.r}
      state={self.state})"""

  def __len__(self) -> int:
    return len(self.r)

  def __getitem__(self, idx) -> MultiBernoulli:
    return MultiBernoulli(r=self.r[idx], state=self.state[idx])

  def append(self,
             r: np.ndarray,
             state: GaussianState) -> None:
    if self.state is None:
      self.state = state
    else:
      self.state.append(state)

    self.r = np.append(self.r, r)

  def predict(self,
              state_estimator: KalmanFilter,
              dt: float,
              ps: float,
              filter_state: Optional[Dict] = None) -> MultiBernoulli:
    if len(self) == 0:
      return copy.copy(self)

    pred_state, filter_state = state_estimator.predict(
        state=self.state, dt=dt, filter_state=filter_state)

    pred_mb = MultiBernoulli(r=self.r * ps, state=pred_state)
    return pred_mb, filter_state
