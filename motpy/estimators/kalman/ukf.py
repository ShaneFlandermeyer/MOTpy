from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

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
  def __init__(
      self,
      transition_model: TransitionModel,
      measurement_model: MeasurementModel,
      state_subtract_fn: Callable[
          [np.ndarray, np.ndarray], np.ndarray
      ] = np.subtract,
      state_average_fn: Callable[
          [np.ndarray, np.ndarray], np.ndarray
      ] = np.average,
      measurement_subtract_fn: Callable[
          [np.ndarray, np.ndarray], np.ndarray
      ] = np.subtract,
      measurement_average_fn: Callable[
          [np.ndarray, np.ndarray], np.ndarray
      ] = np.average,
      # Sigma point parameters
      sigma_params: Dict[str, Any] = dict(alpha=0.1, beta=2, kappa=0),
  ):
    self.transition_model = transition_model
    self.measurement_model = measurement_model
    self.state_subtract_fn = state_subtract_fn
    self.state_average_fn = state_average_fn
    self.measurement_subtract_fn = measurement_subtract_fn
    self.measurement_average_fn = measurement_average_fn

    self.sigma_params = sigma_params

  def predict(self,
              state: Gaussian,
              dt: float,
              **kwargs,
              ) -> Gaussian:
    sigma_points = merwe_scaled_sigma_points(
        x=state.mean,
        P=state.covar,
        subtract_fn=self.state_subtract_fn,
        **self.sigma_params
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
        subtract_fn=self.state_subtract_fn,
        average_fn=self.state_average_fn,
    )

    predicted_state = Gaussian(mean=x_pred, covar=P_pred, weight=state.weight)

    return predicted_state

  def update(self,
             state: Gaussian,
             measurement: np.ndarray,
             **kwargs
             ) -> Gaussian:
    sigma_points = merwe_scaled_sigma_points(
        x=state.mean,
        P=state.covar,
        subtract_fn=self.state_subtract_fn,
        **self.sigma_params
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
        subtract_fn=self.measurement_subtract_fn,
        average_fn=self.measurement_average_fn,
    )

    # Standard kalman update
    x_pred = state.mean
    P_pred = state.covar
    z = np.asarray(measurement)

    Pxz = np.einsum(
        '...n, ...ni, ...nj -> ...ij',
        Wc,
        self.state_subtract_fn(sigma_points, x_pred[..., None, :]),
        self.measurement_subtract_fn(measured_sigmas, z_pred[..., None, :])
    )
    K = Pxz @ np.linalg.inv(S)
    if measurement is None:
      x_post = x_pred
    else:
      y = self.measurement_subtract_fn(z, z_pred)
      x_post = x_pred + np.einsum('...ij, ...j -> ...i', K, y)

    P_post = P_pred - K @ S @ K.swapaxes(-1, -2)
    P_post = 0.5 * (P_post + P_post.swapaxes(-1, -2))

    post_state = Gaussian(mean=x_post, covar=P_post, weight=state.weight)

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
        x=state.mean,
        P=state.covar,
        subtract_fn=self.state_subtract_fn,
        **self.sigma_params
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
        subtract_fn=self.measurement_subtract_fn,
        average_fn=self.measurement_average_fn,
    )
    return gaussian.likelihood(
        x=measurement,
        mean=z_pred,
        covar=S,
        subtract_fn=self.measurement_subtract_fn
    )

  def gate(self,
           measurements: np.ndarray,
           state: Gaussian,
           pg: float,
           **kwargs,
           ) -> np.ndarray:
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
        subtract_fn=self.measurement_subtract_fn,
        average_fn=self.measurement_average_fn,
    )
    gate_mask, _ = ellipsoidal_gate(
        pg=pg,
        ndim=measurements.shape[-1],
        x=measurements,
        mean=z_pred,
        covar=S,
        subtract_fn=self.measurement_subtract_fn,
    )

    return gate_mask


def unscented_transform(
    sigmas: np.ndarray,
    Wm: np.ndarray,
    Wc: np.ndarray,
    noise_covar: np.ndarray,
    subtract_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.subtract,
    average_fn: Optional[
        Callable[[np.ndarray], np.ndarray]
    ] = None,
) -> Tuple[np.ndarray, np.ndarray]:
  # Mean
  x = average_fn(sigmas, weights=Wm, axis=-2)

  # Covariance
  y = subtract_fn(sigmas, x[..., None, :])
  y_outer = np.einsum('...i, ...j -> ...ij', y, y)
  P = np.sum(Wc[..., None, None] * y_outer, axis=-3) + noise_covar

  return x, P
