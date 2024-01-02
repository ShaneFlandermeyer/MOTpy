from typing import Tuple

import numpy as np

from motpy.kalman.sigma_points import merwe_scaled_sigma_points
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
              ) -> GaussianState:
    # Compute sigma points and weights
    sigmas, Wm, Wc = merwe_scaled_sigma_points(
        x=state.mean,
        P=state.covar,
        alpha=self.alpha,
        beta=self.beta,
        kappa=self.kappa,
        subtract_fn=self.state_residual_fn,
    )
    # Transform points to the prediction space
    sigmas_f = np.zeros_like(sigmas)
    for i in range(len(sigmas)):
      sigmas_f[i] = self.transition_model(sigmas[i], dt=dt, noise=False)

    pred_mean, pred_covar = self.unscented_transform(
        sigmas=sigmas_f,
        Wm=Wm,
        Wc=Wc,
        noise_covar=self.transition_model.covar(dt=dt),
        residual_fn=self.state_residual_fn,
    )
    pred_state = GaussianState(mean=pred_mean, covar=pred_covar,
                               metadata=dict(sigmas_f=sigmas_f, Wm=Wm, Wc=Wc))
    return pred_state

  def update(self,
             predicted_state: GaussianState,
             measurement: np.ndarray,
             ) -> GaussianState:
    # Extract information from the predict step
    sigmas_f = predicted_state.metadata['sigmas_f']
    Wm = predicted_state.metadata['Wm']
    Wc = predicted_state.metadata['Wc']

    # Transform sigma points to measurement space
    n_sigma_points, ndim_state = sigmas_f.shape
    ndim_measurement = measurement.size
    sigmas_h = np.zeros((n_sigma_points, ndim_measurement))
    for i in range(n_sigma_points):
      sigmas_h[i] = self.measurement_model(sigmas_f[i], noise=False)

    # State update
    x_post, P_post, S, K, z_pred = self.ukf_update(
        x_pred=predicted_state.mean,
        P_pred=predicted_state.covar,
        z=measurement,
        R=self.measurement_model.covar(),
        sigmas_f=sigmas_f,
        sigmas_h=sigmas_h,
        Wm=Wm,
        Wc=Wc,
        state_residual_fn=self.state_residual_fn,
        measurement_residual_fn=self.measurement_residual_fn,
    )
    post_state = GaussianState(mean=x_post, covar=P_post,
                               metadata=dict(S=S, K=K, z_pred=z_pred))
    return post_state

  @staticmethod
  def unscented_transform(sigmas: np.ndarray,
                          Wm: np.ndarray,
                          Wc: np.ndarray,
                          noise_covar: np.ndarray,
                          residual_fn: callable = np.subtract,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the unscented transform to compute the mean and covariance from a set of sigma points

    Parameters
    ----------
    sigmas : np.ndarray
        Array of sigma points, where each row is a point in N-d space.
    Wm : np.ndarray
        Mean weight matrix
    Wc : np.ndarray
        Covariance weight matrix
    Q : np.ndarray
        Process noise matrix
    residual_fn : callable
        Function handle to compute the residual. This must be specified manually for nonlinear quantities such as angles, which cannot be subtracted directly. Default is np.subtract

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
          - Mean vector computed by applying the unscented transform to the input sigma points
          - The covariance computed by applying the unscented transform to the input sigma points

    """
    # Mean computation
    x = np.dot(Wm, sigmas)

    # Covariance computation
    y = residual_fn(sigmas, x[np.newaxis, :])
    P = np.einsum('k, ki, kj->ij', Wc, y, y) + noise_covar

    return x, P

  @staticmethod
  def ukf_update(
      x_pred: np.ndarray,
      P_pred: np.ndarray,
      z: np.ndarray,
      R: np.ndarray,
      # Sigma point parameters
      sigmas_f: np.ndarray,
      sigmas_h: np.ndarray,
      Wm: np.ndarray,
      Wc: np.ndarray,
      state_residual_fn: callable,
      measurement_residual_fn: callable,
  ) -> Tuple[np.ndarray]:
    """
    Unscented Kalman filter update step

    Parameters
    ----------
    measurement : np.ndarray
        New measurement to use for the update
    predicted_state : np.ndarray
        State vector after the prediction step
    predicted_covar : np.ndarray
        Covariance after the prediction step
    sigmas_h : np.ndarray
        Sigma points in measurement space
    Wm : np.ndarray
        Mean weights from the prediction step
    Wc : np.ndarray
        Covariance weights from the prediction step
    measurement_model : callable
        Measurement function
    R : np.ndarray
        Measurement noise. For now, assumed to be a matrix

    Returns
    -------
    Tuple[np.ndarray]
        A tuple containing the following:
          - Updated state vector
          - Updated covariance matrix
          - Innovation covariance
          - Kalman gain
          - Predicted measurement
    """

    # Compute the mean and covariance of the measurement prediction using the unscented transform
    z_pred, S = UnscentedKalmanFilter.unscented_transform(
        sigmas=sigmas_h,
        Wm=Wm,
        Wc=Wc,
        noise_covar=R,
        residual_fn=measurement_residual_fn,
    )

    # Compute the cross-covariance of the state and measurements
    Pxz = np.einsum('k, ki, kj->ij',
                    Wc,
                    state_residual_fn(sigmas_f, x_pred),
                    measurement_residual_fn(sigmas_h, z_pred))

    # Update the state vector and covariance
    y = measurement_residual_fn(z, z_pred)
    K = Pxz @ np.linalg.inv(S)
    x_post = x_pred + K @ y
    P_post = P_pred - K @ S @ K.T
    return x_post, P_post, S, K, z_pred
