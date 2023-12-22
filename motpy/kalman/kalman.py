from typing import Tuple
import numpy as np

from motpy.models.measurement.base import MeasurementModel
from motpy.models.transition.base import TransitionModel
from motpy.distributions.gaussian import GaussianState
import motpy.distributions.gaussian as gaussian
from motpy.gate import EllipsoidalGate


class KalmanFilter():
  def __init__(self,
               transition_model: TransitionModel = None,
               measurement_model: MeasurementModel = None,
               ):
    self.transition_model = transition_model
    self.measurement_model = measurement_model

  def predict(self,
              state: GaussianState,
              dt: float,
              ) -> GaussianState:
    x, P = state.mean, state.covar
    F = self.transition_model.matrix(dt)
    Q = self.transition_model.covar(dt)
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return GaussianState(mean=x_pred, covar=P_pred)

  def update(self,
             measurement: np.ndarray,
             predicted_state: GaussianState) -> Tuple[np.ndarray, np.ndarray]:
    x_pred, P_pred = predicted_state.mean, predicted_state.covar
    z = measurement
    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()

    z_pred = H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    P_post = P_pred - K @ S @ K.T
    P_post = (P_post + P_post.T) / 2

    if z is None:
      x_post = None
    else:
      y = z - z_pred
      x_post = x_pred + K @ y

    return GaussianState(mean=x_post, covar=P_post,
                         metadata=dict(S=S, K=K, z_pred=z_pred))

  def gate(self,
           measurements: np.ndarray,
           predicted_state: GaussianState,
           pg: float = 0.999,
           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gate measurements using the predicted state

    Parameters
    ----------
    measurements : np.ndarray
        Measurements
    predicted_state : GaussianState
        Predicted state
    pg : float
        Gate probability

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Measurements in the gate and their indices
    """
    x, P = predicted_state.mean, predicted_state.covar
    gate = EllipsoidalGate(pg=pg, ndim=measurements[0].size)
    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()
    z_pred = self.measurement_model(x, noise=False)
    S = H @ P @ H.T + R
    return gate(measurements=measurements,
                predicted_measurement=z_pred,
                innovation_covar=S)

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
    return gaussian.likelihood(
        z=measurement,
        z_pred=self.measurement_model(x, noise=False),
        P_pred=P,
        H=self.measurement_model.matrix(),
        R=self.measurement_model.covar(),
    )
