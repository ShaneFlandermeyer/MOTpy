from __future__ import annotations
import copy
import numpy as np
from motpy.kalman import KalmanFilter
from motpy.distributions.gaussian import GaussianState
from typing import Dict, Tuple, Optional, List, Union


class MultiBernoulli():
  def __init__(self,
               state: GaussianState,
               r: np.ndarray
               ) -> None:
    self.state = state
    self.r = r.reshape(state.shape + (1,))

  def __repr__(self) -> str:
    return f"""MultiBernoulli(
      r={self.r}
      state={self.state})"""

  def __len__(self) -> int:
    return len(self.r)

  def __getitem__(self, idx) -> MultiBernoulli:
    return MultiBernoulli(r=self.r[idx], state=self.state[idx])

  def append(self, state: GaussianState, r: np.ndarray) -> None:
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
    predicted_state, filter_state = state_estimator.predict(
        state=self.state, dt=dt, filter_state=filter_state)

    predicted_mb = MultiBernoulli(r=self.r * ps, state=predicted_state)
    return pred_mb, filter_state
