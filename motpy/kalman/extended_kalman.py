import datetime
from typing import Dict, Optional, Tuple, Union
import numpy as np
from motpy.distributions import gaussian
from motpy.gate import EllipsoidalGate

from motpy.models.measurement import MeasurementModel
from motpy.models.transition import TransitionModel
from motpy.distributions.gaussian import GaussianState


class ExtendedKalmanFilter():
  def __init__(self,
               transition_model: Optional[TransitionModel] = None,
               measurement_model: Optional[MeasurementModel] = None,
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
              metadata: Optional[Dict] = dict(),
              **kwargs
              ) -> Tuple[GaussianState, Dict]:
    x, P = state.mean, state.covar

    F = self.transition_model.matrix(dt=dt, **kwargs)
    Q = self.transition_model.covar(dt=dt, **kwargs)

    x_pred = self.transition_model(x, dt=dt, **kwargs)
    P_pred = F @ P @ F.T + Q

    pred_state = GaussianState(
        state_dim=state.state_dim,
        mean=x_pred,
        covar=P_pred,
        weight=state.weight)

    return pred_state, metadata

  def update(self,
             predicted_state: GaussianState,
             measurement: np.ndarray,
             metadata: Optional[Dict] = dict(),
             **kwargs
             ) -> Tuple[GaussianState, Dict]:
    assert self.measurement_model is not None

    x_pred, P_pred = predicted_state.mean, predicted_state.covar

    z = measurement
    H = self.measurement_model.matrix(**kwargs)
    R = self.measurement_model.covar(**kwargs)

    S = H @ P_pred @ H.swapaxes(-1, -2) + R
    K = P_pred @ H.swapaxes(-1, -2) @ np.linalg.inv(S)
    P_post = P_pred - K @ S @ K.swapaxes(-1, -2)
    P_post = (P_post + P_post.swapaxes(-1, -2)) / 2

    z_pred = self.measurement_model(x_pred, **kwargs)
    y = self.measurement_residual_fn(z, z_pred)
    x_post = x_pred + np.einsum('...ij, ...j -> ...i', K, y)

    post_state = GaussianState(
        state_dim=predicted_state.state_dim,
        mean=x_post,
        covar=P_post,
        weight=predicted_state.weight)

    return post_state, metadata

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
                innovation_covar=S)[0]

  def likelihood(
      self,
      measurement: np.ndarray,
      state: GaussianState,
  ) -> float:
    """
    Compute the likelihood of a measurement given the predicted state

    Parameters
    ----------
    measurement : np.ndarray
        Measurement
    state : GaussianState
        Predicted state

    Returns
    -------
    float
        Likelihood
    """

    x, P = state.mean, state.covar

    H = self.measurement_model.matrix(x=x)
    R = self.measurement_model.covar()
    S = H @ P @ H.swapaxes(-1, -2) + R
    return gaussian.likelihood(
        z=measurement,
        z_pred=self.measurement_model(x, noise=False),
        S=S
    )
