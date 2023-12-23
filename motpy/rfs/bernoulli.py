from __future__ import annotations
import numpy as np
from motpy.kalman import KalmanFilter
from motpy.distributions.gaussian import GaussianState
from typing import Tuple, Optional, List


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