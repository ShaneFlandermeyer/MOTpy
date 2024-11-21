from typing import Any, Dict, Optional, Tuple

import numpy as np

from motpy.kalman.sigma_points import merwe_scaled_sigma_points, merwe_sigma_weights
from motpy.models.measurement import MeasurementModel
from motpy.models.transition import TransitionModel
from motpy.distributions.gaussian import Gaussian


class UnscentedKalmanFilter():
  def __init__(self,
               transition_model: TransitionModel,
               measurement_model: MeasurementModel,
               state_residual_fn: callable = np.subtract,
               measurement_residual_fn: callable = np.subtract,
               ):
    self.transition_model = transition_model
    self.measurement_model = measurement_model
    self.state_residual_fn = state_residual_fn
    self.measurement_residual_fn = measurement_residual_fn

  def predict(self,
              state: Gaussian,
              dt: float,
              sigma_points: np.ndarray,
              Wm: np.ndarray,
              Wc: np.ndarray,
              model_args: Optional[Dict] = dict(),
              ) -> Tuple[Gaussian, Dict]:
    """
    UKF predict step

    Parameters
    ----------
    state : Gaussian
        Object states
    dt : float
        Prediction time step
    sigma_points : np.ndarray
        2n + 1 x n matrix of sigma points
    Wm : np.ndarray
        Mean weights for the unscented transform
    Wc : np.ndarray
        Covariance weights for the unscented transform

    Returns
    -------
    Tuple[Gaussian, Dict]
        A tuple containing the:
        - Predicted state object
        - Filter state dict with the following updated keys:
            - predicted_sigmas: Predicted sigma points
            - Wm: Mean weights for the unscented transform
            - Wc: Covariance weights for the unscented transform
    """
    # Transform sigma points to the prediction space
    predicted_sigmas = self.transition_model(sigma_points, dt=dt, **model_args)

    predicted_mean, predicted_covar = unscented_transform(
        sigmas=predicted_sigmas,
        Wm=Wm,
        Wc=Wc,
        noise_covar=self.transition_model.covar(dt=dt),
        residual_fn=self.state_residual_fn,
    )
    predicted_state = Gaussian(
        mean=predicted_mean, covar=predicted_covar, weight=state.weight)

    filter_state = dict(
        predicted_sigmas=predicted_sigmas
    )
    return predicted_state, filter_state

  def update(self,
             predicted_state: Gaussian,
             measurement: np.ndarray,
             predicted_sigmas: np.ndarray,
             Wm: np.ndarray,
             Wc: np.ndarray,
             model_args: Optional[Dict] = dict(),
             ) -> Tuple[Gaussian, Dict]:
    """
    UKF update step

    Parameters
    ----------
    predicted_state : GaussianState
        Predicted state
    measurement : np.ndarray
        Measurement vectors
    predicted_sigmas : np.ndarray 
        2n + 1 x n matrix of sigma points in prediction space
    Wm : np.ndarray
        Mean weights for the unscented transform
    Wc : np.ndarray
        Covariance weights for the unscented transform

    Returns
    -------
    Tuple[GaussianState, Dict]
        A tuple containing the:
        - Predicted state
        - Filter state dict with sigma points in measurement space
    """

    # Unscented transform in measurement space
    measured_sigmas = self.measurement_model(predicted_sigmas, **model_args)
    z_pred, S = unscented_transform(
        sigmas=measured_sigmas,
        Wm=Wm,
        Wc=Wc,
        noise_covar=self.measurement_model.covar(),
        residual_fn=self.measurement_residual_fn,
    )

    # Standard kalman update
    x_pred, P_pred = predicted_state.mean, predicted_state.covar
    z = measurement
    Pxz = np.einsum('...n, ...ni, ...nj -> ...ij',
                    Wc,
                    self.state_residual_fn(predicted_sigmas, x_pred),
                    self.measurement_residual_fn(measured_sigmas, z_pred))
    y = self.measurement_residual_fn(z, z_pred)
    K = Pxz @ np.linalg.inv(S)
    x_post = x_pred + np.einsum('...ij, ...j -> ...i', K, y)
    P_post = P_pred - K @ S @ K.swapaxes(-1, -2)
    P_post = 0.5*(P_post + P_post.swapaxes(-1, -2))
    post_state = Gaussian(
        mean=x_post, covar=P_post, weight=predicted_state.weight
    )

    filter_state = dict(
        measured_sigmas=measured_sigmas,
    )
    return post_state, filter_state


def unscented_transform(sigmas: np.ndarray,
                        Wm: np.ndarray,
                        Wc: np.ndarray,
                        noise_covar: np.ndarray,
                        residual_fn: callable = np.subtract,
                        ) -> Tuple[np.ndarray, np.ndarray]:
  # Mean
  x = np.einsum('...n, ...ni -> ...i', Wm, sigmas)

  # Covariance
  y = residual_fn(sigmas, x[..., None, :])
  P = np.einsum('...n, ...ni, ...nj -> ...ij', Wc, y, y) + noise_covar

  return x, P
