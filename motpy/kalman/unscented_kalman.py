from typing import Dict, Optional, Tuple

import numpy as np

from motpy.kalman.sigma_points import merwe_scaled_sigma_points, merwe_sigma_weights
from motpy.models.measurement import MeasurementModel
from motpy.models.transition import TransitionModel
from motpy.distributions.gaussian import GaussianState


class UnscentedKalmanFilter():
  def __init__(self,
               transition_model: TransitionModel,
               measurement_model: MeasurementModel,
               state_residual_fn: callable = np.subtract,
               measurement_residual_fn: callable = np.subtract,
               # Sigma point parameters
               alpha: float = 0.1,
               beta: float = 2,
               kappa: float = 0,
               ):
    self.transition_model = transition_model
    self.measurement_model = measurement_model
    self.state_residual_fn = state_residual_fn
    self.measurement_residual_fn = measurement_residual_fn

    self.alpha = alpha
    self.beta = beta
    self.kappa = kappa

  def predict(self,
              state: GaussianState,
              dt: float,
              metadata: Optional[dict] = dict(),
              ) -> Tuple[GaussianState, Dict]:
    # Compute sigma points and weights
    sigmas = merwe_scaled_sigma_points(
        x=state.mean,
        P=state.covar,
        alpha=self.alpha,
        beta=self.beta,
        kappa=self.kappa,
        subtract_fn=self.state_residual_fn)
    Wm, Wc = merwe_sigma_weights(
        ndim_state=state.mean.shape[-1],
        alpha=self.alpha,
        beta=self.beta,
        kappa=self.kappa)

    # Transform sigma points to the prediction space
    predicted_sigams = self.transition_model(sigmas, dt=dt, noise=False)

    predicted_mean, predicted_covar = self.unscented_transform(
        sigmas=predicted_sigams,
        Wm=Wm,
        Wc=Wc,
        noise_covar=self.transition_model.covar(dt=dt),
        residual_fn=self.state_residual_fn,
    )
    predicted_state = GaussianState(
        mean=predicted_mean, covar=predicted_covar, weight=state.weight)

    metadata.update({'predicted_sigmas': predicted_sigams})
    return predicted_state, metadata

  def update(self,
             predicted_state: GaussianState,
             measurement: np.ndarray,
             metadata: Optional[dict] = dict(),
             ) -> Tuple[GaussianState, Dict]:

    # Extract information from predict step
    predicted_sigmas = metadata.get('predicted_sigmas')
    if predicted_sigmas is None:
      raise ValueError('No sigma points found in the predicted state metadata')

    Wm, Wc = merwe_sigma_weights(
        ndim_state=predicted_sigmas.shape[-1],
        alpha=self.alpha,
        beta=self.beta,
        kappa=self.kappa)

    # Unscented transform in measurement space
    measured_sigmas = self.measurement_model(predicted_sigmas, noise=False)
    z_pred, S = UnscentedKalmanFilter.unscented_transform(
        sigmas=measured_sigmas,
        Wm=Wm,
        Wc=Wc,
        noise_covar=self.measurement_model.covar(),
        residual_fn=self.measurement_residual_fn,
    )

    # Standard kalman update
    x_pred, P_pred = predicted_state.mean, predicted_state.covar
    z = measurement
    Pxz = np.einsum('k, ...ki, ...kj -> ...ij',
                    Wc,
                    self.state_residual_fn(predicted_sigmas, x_pred),
                    self.measurement_residual_fn(measured_sigmas, z_pred))
    y = self.measurement_residual_fn(z, z_pred)
    K = Pxz @ np.linalg.inv(S)
    x_post = x_pred + np.einsum('...ij, ...j -> ...i', K, y)
    P_post = P_pred - K @ S @ K.swapaxes(-1, -2)
    P_post = (P_post + P_post.swapaxes(-1, -2)) / 2
    post_state = GaussianState(
        mean=x_post, covar=P_post, weight=predicted_state.weight)

    metadata.update({
        'measured_sigmas': measured_sigmas,
        'cache': {'innovation_covar': S,
                  'kalman_gain': K,
                  'predicted_measurement': z_pred}
    })
    return post_state, metadata

  @staticmethod
  def unscented_transform(sigmas: np.ndarray,
                          Wm: np.ndarray,
                          Wc: np.ndarray,
                          noise_covar: np.ndarray,
                          residual_fn: callable = np.subtract,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    # Mean computation
    x = np.dot(Wm, sigmas)

    # Covariance computation
    y = residual_fn(sigmas, x[..., None, :])
    P = np.einsum('k, ...ki, ...kj -> ...ij', Wc, y, y) + noise_covar

    return x, P
