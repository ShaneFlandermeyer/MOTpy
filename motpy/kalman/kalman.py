from typing import List, Optional, Tuple, Union
import numpy as np

from motpy.models.measurement.base import MeasurementModel
from motpy.models.transition.base import TransitionModel
from motpy.distributions.gaussian import GaussianMixture, GaussianState
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
              state: Union[GaussianState, GaussianMixture],
              dt: float,
              ) -> Union[GaussianState, GaussianMixture]:
    assert self.transition_model is not None

    if isinstance(state, GaussianMixture):
      x, P = state.means, state.covars
    elif isinstance(state, GaussianState):
      x, P = state.mean, state.covar

    F = self.transition_model.matrix(dt=dt)
    Q = self.transition_model.covar(dt=dt)

    x_pred = x @ F.T
    P_pred = F @ P @ F.T + Q

    if isinstance(state, GaussianMixture):
      return GaussianMixture(means=x_pred, covars=P_pred, weights=state.weights)
    elif isinstance(state, GaussianState):
      return GaussianState(mean=x_pred, covar=P_pred)

  def update(self,
             measurement: np.ndarray,
             predicted_state: Union[GaussianState, GaussianMixture]
             ) -> Union[GaussianState, GaussianMixture]:
    assert self.measurement_model is not None

    if isinstance(predicted_state, GaussianMixture):
      x_pred, P_pred = predicted_state.means, predicted_state.covars
    elif isinstance(predicted_state, GaussianState):
      x_pred, P_pred = predicted_state.mean, predicted_state.covar

    z = measurement
    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    P_post = P_pred - K @ S @ K.swapaxes(-1, -2)
    P_post = (P_post + P_post.swapaxes(-1, -2)) / 2

    z_pred = x_pred @ H.T
    x_post = x_pred + np.einsum('...ij, ...j -> ...i', K, z - z_pred)

    if isinstance(predicted_state, GaussianMixture):
      return GaussianMixture(
          means=x_post, covars=P_post, weights=predicted_state.weights)
    elif isinstance(predicted_state, GaussianState):
      post_state = GaussianState(mean=x_post, covar=P_post)

    return post_state

  def gate(self,
           measurements: np.ndarray,
           predicted_state: Union[GaussianState, GaussianMixture],
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
      if isinstance(predicted_state, GaussianMixture):
        return np.ones((len(predicted_state), len(measurements)), dtype=bool)
      elif isinstance(predicted_state, GaussianState):
        return np.ones((len(measurements)), dtype=bool)

    if isinstance(predicted_state, GaussianMixture):
      x, P = predicted_state.means, predicted_state.covars
    elif isinstance(predicted_state, GaussianState):
      x, P = predicted_state.mean, predicted_state.covar
    else:
      raise ValueError('Invalid predicted state type')

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
      x, P = predicted_state.means, predicted_state.covars
    elif isinstance(predicted_state, GaussianState):
      x, P = predicted_state.mean, predicted_state.covar
    else:
      raise ValueError('Invalid predicted state type')

    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()
    S = H @ P @ H.swapaxes(-1, -2) + R
    return gaussian.likelihood(
        z=measurement,
        z_pred=x @ H.T,
        S=S
    )
