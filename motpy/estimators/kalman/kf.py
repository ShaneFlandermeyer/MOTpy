import dataclasses
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from motpy.models.measurement.base import MeasurementModel
from motpy.models.transition.base import TransitionModel
from motpy.distributions.gaussian import Gaussian
import motpy.distributions.gaussian as gaussian
from motpy.gate import EllipsoidalGate
from motpy.estimators import StateEstimator


class KalmanFilter(StateEstimator):
  def __init__(self,
               transition_model: Optional[TransitionModel] = None,
               measurement_model: Optional[MeasurementModel] = None,
               ):
    self.transition_model = transition_model
    self.measurement_model = measurement_model

  def predict(self,
              state: Gaussian,
              dt: float,
              **kwargs,
              ) -> Gaussian:
    assert self.transition_model is not None

    x, P = state.mean, state.covar

    F = self.transition_model.matrix(dt=dt)
    Q = self.transition_model.covar(dt=dt)

    x_pred = self.transition_model(x, dt=dt, **kwargs)
    P_pred = F @ P @ F.T + Q

    predicted_state = Gaussian(
        mean=x_pred, covar=P_pred, weight=state.weight
    )

    return predicted_state

  def update(self,
             state: Gaussian,
             measurement: Optional[np.ndarray] = None,
             **kwargs,
             ) -> Gaussian:
    assert self.measurement_model is not None
    
    measurement = np.asarray(measurement)
    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()

    x_pred = np.expand_dims(state.mean, axis=-2)
    P_pred = state.covar
    z = np.atleast_2d(measurement)
    z_pred = self.measurement_model(x_pred, **kwargs)
    y = z - z_pred
    
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_post = x_pred + \
        np.einsum('...ij, ...j -> ...i', np.expand_dims(K, axis=-3), y)
    P_post = P_pred - K @ S @ K.swapaxes(-1, -2)
    P_post = 0.5 * (P_post + P_post.swapaxes(-1, -2))

    
    # Handle broadcasting for multiple measurements
    weight = state.weight
    if measurement.ndim == 1:
      x_post = x_post.squeeze(axis=-2)
    else:
      P_post = np.expand_dims(P_post, axis=-3).repeat(z.shape[-2], axis=-3)
      if weight is not None:
        weight = np.expand_dims(weight, axis=-1).repeat(z.shape[-2], axis=-1)

    post_state = Gaussian(
        mean=x_post, covar=P_post, weight=weight
    )

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
