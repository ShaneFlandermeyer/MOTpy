from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from motpy.models.measurement.base import MeasurementModel
from motpy.models.transition.base import TransitionModel
from motpy.distributions.gaussian import Gaussian
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
              state: Gaussian,
              dt: float,
              model_args: Optional[Dict] = dict(),
              ) -> Gaussian:
    assert self.transition_model is not None

    x, P = state.mean, state.covar

    F = self.transition_model.matrix(dt=dt)
    Q = self.transition_model.covar(dt=dt)

    x_pred = self.transition_model(x, dt=dt, **model_args)
    P_pred = F @ P @ F.T + Q

    predicted_state = Gaussian(mean=x_pred, covar=P_pred, weight=state.weight)

    return predicted_state

  def update(self,
             state: Gaussian,
             measurement: Optional[np.ndarray] = None,
             model_args: Optional[Dict] = dict(),
             ) -> Tuple[Gaussian, Dict]:
    assert self.measurement_model is not None

    x_pred, P_pred = state.mean, state.covar

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
      z_pred = self.measurement_model(x_pred, **model_args)
      x_post = x_pred + np.einsum('...ij, ...j -> ...i', K, z - z_pred)

    post_state = Gaussian(mean=x_post, covar=P_post, weight=state.weight)

    return post_state

  def likelihood(
      self,
      measurement: np.ndarray,
      state: Gaussian,
  ) -> np.ndarray:
    """
    Likelihood of measurement(s) conditioned on state estimate(s).

    Parameters
    ----------
    measurement : np.ndarray
        Array of measurements. Shape: (..., m, dz)
    predicted_state : GaussianState
        Predicted state distribution. Shape: (..., n, dx)

    Returns
    -------
    np.ndarray
        A matrix of likelihoods. Shape: (..., n, m)
    """

    x, P = state.mean, state.covar

    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()
    S = H @ P @ H.swapaxes(-1, -2) + R
    return gaussian.likelihood(
        mean=self.measurement_model(x),
        covar=S,
        x=measurement,
    )

  def gate(self,
           measurements: np.ndarray,
           state: Gaussian,
           pg: float,
           **kwargs,
           ) -> np.ndarray:
    """
    Gate a set of measurements with respect to one or more Gaussian components using the squared mahalanobis distance.

    Parameters
    ----------
    measurements : np.ndarray
        Measurements to gate. Shape: (M, nz)
    state : Union[GaussianState, GaussianMixture]
        Predicted state distribution.
    pg : float, optional
        Gate probability

    Returns
    -------
    Boolean array indicating whether each measurement is within the gate.
    """
    assert self.measurement_model is not None

    x, P = state.mean, state.covar

    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()
    S = H @ P @ H.T + R
    gate = EllipsoidalGate(pg=pg, ndim=measurements.shape[-1])
    gate_mask, _ = gate(
        x=measurements,
        mean=self.measurement_model(x, **kwargs),
        covar=S
    )

    return gate_mask
