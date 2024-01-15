from __future__ import annotations
import copy
import numpy as np
from motpy.kalman import KalmanFilter
from motpy.distributions.gaussian import GaussianMixture, GaussianState
from typing import Tuple, Optional, List, Union


class MultiBernoulli():
  def __init__(self,
               r: np.ndarray = None,
               state: GaussianMixture = None):
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
             state: Union[GaussianState, GaussianMixture],
             weight: float = None) -> None:
    if isinstance(state, GaussianState):
      state = GaussianMixture(
          means=state.mean, covars=state.covar, weights=weight)
      
    if self.state is None:
      self.state = state
    else:
      self.state.append(state)
      
    self.r = np.append(self.r, r)

  def predict(self,
              state_estimator: KalmanFilter,
              dt: float,
              ps: float) -> MultiBernoulli:
    if len(self) == 0:
      return copy.copy(self)

    return MultiBernoulli(
        r=self.r * ps,
        state=state_estimator.predict(state=self.state, dt=dt),
    )


class Bernoulli():
  def __init__(self,
               r: float,
               state: GaussianState,
               ) -> None:
    self.r = r
    self.state = state

  def __repr__(self) -> str:
    return f"""Bernoulli(
      r={self.r}
      state={self.state})"""

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float,
              ) -> Bernoulli:
    """
    Performs prediction step for a Bernoulli component

    Parameters
    ----------
    state_estimator : KalmanFilter
        State prediction model
    ps : float
        Survival probability
    dt : float
        Prediction timestep

    Returns
    -------
    Tuple[State, float]
        - Predicted state
        - Predicted survival probability
    """
    pred = Bernoulli(
        r=self.r * ps,
        state=state_estimator.predict(state=self.state, dt=dt),
    )

    return pred

  def update(self,
             pd: float,
             measurement: np.ndarray = None,
             state_estimator: KalmanFilter = None,
             ) -> Bernoulli:
    """
    Update the state of the Bernoulli component with an associated measurement. If no measurement is associated to this component, the state is predicted 

    Parameters
    ----------
    pd : float
        Detection probability, assumed constant in state space
    measurement : np.ndarray, optional
        Measurement associated to bernoulli component, by default None
    state_estimator : KalmanFilter, optional
        Measurement update model, which only needs to be provided if a measurement is associated to the component, by default None

    Returns
    -------
    Tuple[State, float, float]
        - Updated state
        - Updated existence probability
    """
    if measurement is None:
      state_post = self.state
      r_post = self.r * (1 - pd) / (1 - self.r + self.r * (1 - pd))
    else:
      state_post = state_estimator.update(
          measurement=measurement, predicted_state=self.state)
      r_post = 1

    posterior = Bernoulli(r=r_post, state=state_post)
    return posterior
