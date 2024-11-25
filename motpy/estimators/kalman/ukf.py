from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

import motpy.distributions.gaussian as gaussian
from motpy.distributions import Gaussian
from motpy.estimators import StateEstimator
from motpy.estimators.kalman.sigma_points import (merwe_scaled_sigma_points,
                                                  merwe_sigma_weights)
from motpy.gate import ellipsoidal_gate
from motpy.models.measurement import MeasurementModel
from motpy.models.transition import TransitionModel


class UnscentedKalmanFilter(StateEstimator):
  def __init__(self,
               transition_model: TransitionModel,
               measurement_model: MeasurementModel,
               state_residual_fn: callable = np.subtract,
               measurement_residual_fn: callable = np.subtract,
               # Sigma point parameters
               sigma_params: Dict[str, Any] = dict(alpha=0.1, beta=2, kappa=0),
               ):
    self.transition_model = transition_model
    self.measurement_model = measurement_model
    self.state_residual_fn = state_residual_fn
    self.measurement_residual_fn = measurement_residual_fn

    self.sigma_params = sigma_params

  def predict(self,
              state: Gaussian,
              dt: float,
              **kwargs,
              ) -> Gaussian:
    assert self.transition_model is not None

    sigma_points = merwe_scaled_sigma_points(
        x=state.mean, P=state.covar, **self.sigma_params
    )
    Wm, Wc = merwe_sigma_weights(
        ndim_state=state.mean.shape[-1], **self.sigma_params
    )

    # Transform sigma points to prediction space
    predicted_sigmas = self.transition_model(sigma_points, dt=dt, **kwargs)

    x_pred, P_pred = unscented_transform(
        sigmas=predicted_sigmas,
        Wm=Wm,
        Wc=Wc,
        noise_covar=self.transition_model.covar(dt=dt),
        residual_fn=self.state_residual_fn,
    )

    predicted_state = Gaussian(mean=x_pred, covar=P_pred, weight=state.weight)

    return predicted_state

  def update(self,
             state: Gaussian,
             measurement: np.ndarray,
             **kwargs
             ) -> Gaussian:
    assert self.measurement_model is not None
    sigma_points = merwe_scaled_sigma_points(
        x=state.mean, P=state.covar, **self.sigma_params
    )
    Wm, Wc = merwe_sigma_weights(
        ndim_state=state.mean.shape[-1], **self.sigma_params
    )

    # Unscented transform in measurement space
    measured_sigmas = self.measurement_model(sigma_points, **kwargs)
    z_pred, S = unscented_transform(
        sigmas=measured_sigmas,
        Wm=Wm,
        Wc=Wc,
        noise_covar=self.measurement_model.covar(),
        residual_fn=self.measurement_residual_fn,
    )

    # Standard kalman update
    x_pred = np.expand_dims(state.mean, axis=-2)
    P_pred = state.covar
    z = np.atleast_2d(measurement)
    z_pred = np.expand_dims(z_pred, axis=-2)

    Pxz = np.einsum(
        '...n, ...ni, ...nj -> ...ij',
        Wc,
        self.state_residual_fn(sigma_points, x_pred),
        self.measurement_residual_fn(measured_sigmas, z_pred)
    )
    y = self.measurement_residual_fn(z, z_pred)
    K = Pxz @ np.linalg.inv(S)
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

    post_state = Gaussian(mean=x_post, covar=P_post, weight=weight)

    return post_state

  def likelihood(
      self,
      measurement: np.ndarray,
      state: Gaussian,
      **kwargs
  ) -> np.ndarray:
    """
    Likelihood of measurement(s) conditioned on state estimate(s).

    Parameters
    ----------
    measurement : np.ndarray
        Array of measurements. Shape: (..., m, dz)
    predicted_state : SigmaPointGaussian
        Predicted state distribution. Shape: (..., n, dx)

    Returns
    -------
    np.ndarray
        A matrix of likelihoods. Shape: (..., n, m)
    """
    sigma_points = merwe_scaled_sigma_points(
        x=state.mean, P=state.covar, **self.sigma_params
    )
    Wm, Wc = merwe_sigma_weights(
        ndim_state=state.mean.shape[-1], **self.sigma_params
    )
    measured_sigmas = self.measurement_model(sigma_points, **kwargs)
    z_pred, S = unscented_transform(
        sigmas=measured_sigmas,
        Wm=Wm,
        Wc=Wc,
        noise_covar=self.measurement_model.covar(),
        residual_fn=self.measurement_residual_fn,
    )
    return gaussian.likelihood(x=measurement, mean=z_pred, covar=S)

  def gate(self,
           measurements: np.ndarray,
           state: Gaussian,
           pg: float,
           **kwargs,
           ) -> np.ndarray:
    assert self.measurement_model is not None
    sigma_points = merwe_scaled_sigma_points(
        x=state.mean, P=state.covar, **self.sigma_params
    )
    Wm, Wc = merwe_sigma_weights(
        ndim_state=state.mean.shape[-1], **self.sigma_params
    )

    measured_sigmas = self.measurement_model(sigma_points, **kwargs)
    z_pred, S = unscented_transform(
        sigmas=measured_sigmas,
        Wm=Wm,
        Wc=Wc,
        noise_covar=self.measurement_model.covar(),
        residual_fn=self.measurement_residual_fn,
    )
    gate_mask, _ = ellipsoidal_gate(
        pg=pg,
        ndim=measurements.shape[-1],
        x=measurements,
        mean=z_pred,
        covar=S
    )

    return gate_mask


def unscented_transform(sigmas: np.ndarray,
                        Wm: np.ndarray,
                        Wc: np.ndarray,
                        noise_covar: np.ndarray,
                        residual_fn: callable = np.subtract,
                        ) -> Tuple[np.ndarray, np.ndarray]:
  Wm = np.expand_dims(Wm, axis=-1)
  Wc = np.expand_dims(Wc, axis=(-1, -2))

  # Mean
  x = np.sum(Wm * sigmas, axis=-2)

  # Covariance
  y = residual_fn(sigmas, np.expand_dims(x, axis=-2))
  y_outer = np.einsum('...i, ...j -> ...ij', y, y)
  P = np.sum(Wc * y_outer, axis=-3) + noise_covar

  return x, P
