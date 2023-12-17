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

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float,
              ) -> Bernoulli:
    """
    Performs prediction step for a Bernoulli component

    # TODO: Handle death

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

  def likelihood(self,
                 pd: float,
                 measurements: List[np.ndarray] = None,
                 state_estimator: KalmanFilter = None,
                 ) -> float:
    """
    Compute the LOG likelihood of a measurement given the predicted state

    Parameters
    ----------
    pd : float
        Detection probability, assumed constant in state space
    measurements : List[np.ndarray], optional
        List of measurements. If no measurements are specified, this function computes the likelihood that no measurement is associated to this Bernoulli, by default None
    state_estimator : KalmanFilter, optional
        Object implementing the likelihood for the measurement model. Not required if there are no measurements, by default None

    Returns
    -------
    float
        _description_
    """
    if measurements is None:
      log_likelihood = np.log(1 - self.r + self.r * (1 - pd))
    else:
      zs = np.array(measurements).reshape(-1, len(measurements))
      log_likelihood = np.log(self.r * pd * state_estimator.likelihood(
          measurement=zs, predicted_state=self.state))

    return log_likelihood


class MultiBernoulli():
  def __init__(self,
               rs: np.ndarray,
               states: List[GaussianState],
               weight: float = None,
               ) -> None:
    self.r = np.array(rs)
    self.states = states
    self.weight = weight

  def __repr__(self) -> str:
    return f"""MultiBernoulli(
      weight={self.weight}
      rs={np.array(self.r).tolist()}
      states={self.states})"""

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float,
              ) -> MultiBernoulli:
    """
    Performs prediction step for each Bernoulli component

    Parameters
    ----------
    state_estimator : KalmanFilter
        State prediction model
    ps : float
        Survival probability
    dt : float
        Prediction timestep
    """
    pred_states = []
    pred_rs = np.empty_like(self.r)
    for i, (r, state) in enumerate(zip(self.r, self.states)):
      pred_bern = Bernoulli(r=r, state=state).predict(
          state_estimator=state_estimator, ps=ps, dt=dt)
      pred_states.append(pred_bern.state)
      pred_rs[i] = pred_bern.r

    pred = MultiBernoulli(rs=pred_rs, states=pred_states)
    return pred
