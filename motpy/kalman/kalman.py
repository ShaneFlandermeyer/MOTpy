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
    mean_pred, covar_pred = self.kf_predict(
        x=state.mean,
        P=state.covar,
        F=self.transition_model.matrix(dt),
        Q=self.transition_model.covar(dt),
    )
    return GaussianState(mean=mean_pred, covar=covar_pred)

  def update(self,
             measurement: np.ndarray,
             predicted_state: GaussianState) -> Tuple[np.ndarray, np.ndarray]:
    x_post, P_post, S, K, z_pred = self.kf_update(
        x_pred=predicted_state.mean,
        P_pred=predicted_state.covar,
        z=measurement,
        H=self.measurement_model.matrix(),
        R=self.measurement_model.covar(),
    )
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

  @staticmethod
  def kf_predict(
      x: np.ndarray,
      P: np.ndarray,
      F: np.ndarray,
      Q: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kalman predict step

    See: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/  blob/master/06-Multivariate-Kalman-Filters.ipynb

    Parameters
    ----------
    x : np.ndarray
        State vector
    P : np.ndarray
        Covariance
    F : np.ndarray
        State transition matrix
    Q : np.ndarray
        Transition model noise covariance
    Returns
    -------
    Tuple[np.ndarray]
        _description_
    """
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

  @staticmethod
  def kf_update(x_pred: np.ndarray,
                P_pred: np.ndarray,
                H: np.ndarray,
                R: np.ndarray,
                z: np.ndarray) -> Tuple[np.ndarray]:
    """
    Kalman filter update step

    See: 
    - https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb
    - https://stonesoup.readthedocs.io/en/v0.1b5/stonesoup.updater.html?highlight=kalman#module-stonesoup.updater.kalman

    Parameters
    ----------
    x_pred : np.ndarray
        State prediction
    P_pred : np.ndarray
        Covariance prediction
    z : np.ndarray
        Measurement
    H : np.ndarray
        Measurement model matrix
    R : np.ndarray
        Measurement noise covariance
    Returns
    -------
    Tuple[np.ndarray]
        Updated state and covariance
    """
    # Compute the Kalman gain and innovation covar
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    z_pred = H @ x_pred

    # Compute the updated state and covariance
    if z is None:
      x_post = None
    else:
      y = z - z_pred
      x_post = x_pred + K @ y
    P_post = P_pred - K @ S @ K.T
    P_post = (P_post + P_post.T) / 2

    return x_post, P_post, S, K, z_pred
