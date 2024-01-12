from typing import List, Optional, Tuple
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
              ) -> GaussianState:
    assert self.transition_model is not None

    x, P = state.mean, state.covar
    F = self.transition_model.matrix(dt=dt)
    Q = self.transition_model.covar(dt=dt)

    x_pred = x @ F.T
    P_pred = F @ P @ F.T + Q

    # Clear cache from previous update step
    # meta = state.metadata.copy()
    # if 'cache' in meta:
    #   for key in ['S', 'K', 'P_post']:
    #     meta['cache'].pop(key, None)

    return GaussianState(mean=x_pred, covar=P_pred)

  def update(self,
             measurement: np.ndarray,
             predicted_state: GaussianState) -> GaussianState:
    assert self.measurement_model is not None

    x_pred, P_pred = predicted_state.mean, predicted_state.covar
    z = measurement
    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    P_post = P_pred - K @ S @ K.swapaxes(-1, -2)
    P_post = (P_post + P_post.swapaxes(-1, -2)) / 2

    z_pred = x_pred @ H.T
    x_post = x_pred + np.einsum('nij, nj -> ni', K, z - z_pred)

    # Update cache
    post_state = GaussianState(mean=x_post, covar=P_post)

    return post_state

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
    assert self.measurement_model is not None

    if pg == 1.0:
      return measurements, np.ones((len(measurements),), dtype=bool)

    x, P = predicted_state.mean, predicted_state.covar
    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()
    z_pred = x @ H.T
    S = H @ P @ H.T + R
    gate = EllipsoidalGate(pg=pg, ndim=measurements[0].size)
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
    H = self.measurement_model.matrix()
    return gaussian.likelihood(
        z=measurement,
        z_pred=x @ H.T,
        P_pred=P,
        H=self.measurement_model.matrix(),
        R=self.measurement_model.covar(),
    )
