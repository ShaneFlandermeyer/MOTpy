import datetime
from typing import Tuple, Union
import numpy as np

from motpy.models.measurement import MeasurementModel
from motpy.models.transition import TransitionModel
from motpy.distributions.gaussian import GaussianState
import motpy.distributions.gaussian as gaussian


class ExtendedKalmanFilter():
  def __init__(self,
               transition_model: TransitionModel = None,
               measurement_model: MeasurementModel = None,
               state_residual_fn: callable = np.subtract,
               measurement_residual_fn: callable = np.subtract,
               ):
    self.transition_model = transition_model
    self.measurement_model = measurement_model

    self.state_residual_fn = state_residual_fn
    self.measurement_residual_fn = measurement_residual_fn

  def predict(self,
              state: GaussianState,
              dt: float,
              ) -> GaussianState:
    mean_pred, covar_pred = self.ekf_predict(
        x=state.mean,
        P=state.covar,
        F=self.transition_model.matrix(x=state.mean, dt=dt),
        Q=self.transition_model.covar(dt=dt),
        f=self.transition_model,
        dt=dt,
    )
    return GaussianState(mean=mean_pred, covar=covar_pred)

  def update(self,
             measurement: np.ndarray,
             predicted_state: GaussianState) -> Tuple[np.ndarray, np.ndarray]:
    x_post, P_post, S, K, z_pred = self.ekf_update(
        x_pred=predicted_state.mean,
        P_pred=predicted_state.covar,
        z=measurement,
        H=self.measurement_model.matrix(x=predicted_state.mean),
        R=self.measurement_model.covar(),
        h=self.measurement_model,
        residual_fn=self.measurement_residual_fn,
    )
    return GaussianState(mean=x_post, covar=P_post,
                         metadata=dict(S=S, K=K, z_pred=z_pred))

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
        H=self.measurement_model.matrix(x),
        R=self.measurement_model.covar(),
    )

  @staticmethod
  def ekf_predict(x: np.ndarray,
                  P: np.ndarray,
                  F: np.ndarray,
                  Q: np.ndarray,
                  f: callable,
                  dt: float
                  ) -> Tuple[np.ndarray, np.ndarray]:
    # Propagate the state forward in time
    x_pred = f(x, dt=dt, noise=False)
    P_pred = F @ P @ F.T + Q

    return x_pred, P_pred

  @staticmethod
  def ekf_update(x_pred: np.ndarray,
                 P_pred: np.ndarray,
                 z: np.ndarray,
                 H: np.ndarray,
                 R: np.ndarray,
                 h: callable,
                 residual_fn: callable,
                 ) -> Tuple[np.ndarray, np.ndarray]:
    z_pred = h(x_pred, noise=False)
    y = residual_fn(z, z_pred)

    # Compute the Kalman gain
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    # Compute the updated state and covariance
    x_post = x_pred + K @ y
    P_post = P_pred - K @ S @ K.T

    return x_post, P_post, S, K, z_pred
