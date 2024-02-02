import datetime
from typing import Optional, Tuple, Union
import numpy as np
from motpy.distributions import gaussian
from motpy.gate import EllipsoidalGate

from motpy.models.measurement import MeasurementModel
from motpy.models.transition import TransitionModel
from motpy.distributions.gaussian import GaussianState, GaussianMixture


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
              state: Union[GaussianState, GaussianMixture],
              dt: float,
              ) -> Union[GaussianState, GaussianMixture]:
    if isinstance(state, GaussianMixture):
      x, P = state.mean, state.covar
    elif isinstance(state, GaussianState):
      x, P = state.mean, state.covar
    else:
      raise ValueError(f"Unknown state type {type(state)}")

    F = self.transition_model.matrix(x=x, dt=dt)
    Q = self.transition_model.covar(dt=dt)
    x_pred = self.transition_model(x, dt=dt, noise=False)

    P_pred = F @ P @ F.swapaxes(-1, -2) + Q

    if isinstance(state, GaussianMixture):
      return GaussianMixture(mean=x_pred, covar=P_pred, weight=state.weight)
    elif isinstance(state, GaussianState):
      return GaussianState(mean=x_pred, covar=P_pred)

  def update(self,
             measurement: np.ndarray,
             predicted_state: GaussianState) -> Tuple[np.ndarray, np.ndarray]:
    assert self.measurement_model is not None

    if isinstance(predicted_state, GaussianMixture):
      x_pred, P_pred = predicted_state.mean, predicted_state.covar
    elif isinstance(predicted_state, GaussianState):
      x_pred, P_pred = predicted_state.mean, predicted_state.covar

    z = measurement
    H = self.measurement_model.matrix(x=x_pred)
    R = self.measurement_model.covar()

    S = H @ P_pred @ H.swapaxes(-1, -2) + R
    K = P_pred @ H.swapaxes(-1, -2) @ np.linalg.inv(S)
    P_post = P_pred - K @ S @ K.swapaxes(-1, -2)
    P_post = (P_post + P_post.swapaxes(-1, -2)) / 2

    z_pred = self.measurement_model(x_pred, noise=False)
    y = self.measurement_residual_fn(z, z_pred)
    x_post = x_pred + np.einsum('...ij, ...j -> ...i', K, y)

    if isinstance(predicted_state, GaussianMixture):
      return GaussianMixture(
          mean=x_post, covar=P_post, weight=predicted_state.weight)
    elif isinstance(predicted_state, GaussianState):
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

    if isinstance(predicted_state, GaussianMixture):
      x, P = predicted_state.mean, predicted_state.covar
    elif isinstance(predicted_state, GaussianState):
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
      predicted_state: Union[GaussianState, GaussianMixture],
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

    if isinstance(predicted_state, GaussianMixture):
      x, P = predicted_state.mean, predicted_state.covar
    elif isinstance(predicted_state, GaussianState):
      x, P = predicted_state.mean, predicted_state.covar

    H = self.measurement_model.matrix(x=x)
    R = self.measurement_model.covar()
    return gaussian.likelihood(
        z=measurement,
        z_pred=self.measurement_model(x, noise=False),
        P_pred=P,
        H=H,
        R=R,
    )