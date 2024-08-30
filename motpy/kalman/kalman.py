from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from motpy.models.measurement.base import MeasurementModel
from motpy.models.transition.base import TransitionModel
from motpy.distributions.gaussian import GaussianState
import motpy.distributions.gaussian as gaussian
from motpy.gate import EllipsoidalGate


class KalmanFilter():
  def __init__(self,
               transition_model: Optional[TransitionModel] = None,
               measurement_model: Optional[MeasurementModel] = None,
               ):
    self.transition_model = transition_model
    self.measurement_model = measurement_model

  def predict(self,
              state: GaussianState,
              dt: float,
              filter_state: Optional[Dict] = None,
              **kwargs
              ) -> Tuple[GaussianState, Dict]:
    assert self.transition_model is not None

    x, P = state.mean, state.covar

    F = self.transition_model.matrix(dt=dt)
    Q = self.transition_model.covar(dt=dt)

    x_pred = self.transition_model(x, dt=dt, **kwargs)
    P_pred = F @ P @ F.T + Q

    predicted_state = GaussianState(
        state_dim=state.state_dim,
        mean=x_pred,
        covar=P_pred,
        weight=state.weight)

    return predicted_state, filter_state

  def update(self,
             predicted_state: GaussianState,
             measurement: Optional[np.ndarray] = None,
             filter_state: Optional[Dict] = None,
             **kwargs
             ) -> Tuple[GaussianState, Dict]:
    assert self.measurement_model is not None

    x_pred, P_pred = predicted_state.mean, predicted_state.covar

    z = measurement
    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    P_post = P_pred - K @ S @ K.swapaxes(-1, -2)
    P_post = (P_post + P_post.swapaxes(-1, -2)) / 2

    if z is None:
      x_post = x_pred
    else:
      z_pred = self.measurement_model(x_pred, **kwargs)
      x_post = x_pred + np.einsum('...ij, ...j -> ...i', K, z - z_pred)

    post_state = GaussianState(
        state_dim=predicted_state.state_dim,
        mean=x_post,
        covar=P_post,
        weight=predicted_state.weight)

    return post_state, filter_state

  def gate(self,
           predicted_state: GaussianState,
           measurements: np.ndarray,
           pg: float = 0.999,
           ) -> np.ndarray:
    """
    Gate a set of measurements with respect to one or more Gaussian components using the squared mahalanobis distance.

    Parameters
    ----------
    measurements : np.ndarray
        Measurements to gate. Shape: (M, nz)
    predicted_state : Union[GaussianState, GaussianMixture]
        Predicted state distribution.
    pg : float, optional
        Gate probability, by default 0.999

    Returns
    -------
    Boolean array indicating whether each measurement is within the gate.
    """
    assert self.measurement_model is not None

    if pg == 1.0:
      return np.ones((len(predicted_state), len(measurements)), dtype=bool)

    x, P = predicted_state.mean, predicted_state.covar

    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()
    z_pred = x @ H.T
    S = H @ P @ H.T + R
    gate = EllipsoidalGate(pg=pg, ndim=measurements[0].size)
    return gate(measurements=measurements,
                predicted_measurement=z_pred,
                innovation_covar=S)[0]

  def likelihood(
      self,
      measurement: np.ndarray,
      predicted_state: GaussianState,
  ) -> float:
    """
    Compute the likelihood of a measurement given the predicted state

    Parameters
    ----------
    measurement : np.ndarray
        Measurement
    predicted_state : GaussianState
        Predicted state

    Returns
    -------
    float
        Likelihood
    """

    x, P = predicted_state.mean, predicted_state.covar

    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()
    S = H @ P @ H.swapaxes(-1, -2) + R
    return gaussian.likelihood(
        z=measurement,
        z_pred=x @ H.T,
        S=S
    )
