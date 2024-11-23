import numpy as np
from typing import Tuple
import scipy


def merwe_scaled_sigma_points(x: np.ndarray,
                              P: np.ndarray,
                              # Merwe parameters
                              alpha: float,
                              beta: float,
                              kappa: float,
                              subtract_fn: callable = np.subtract,
                              ) -> Tuple[np.ndarray, ...]:
  """
  Compute sigma points (and the weights for each point) using Van der Merwe's algorithm

  Parameters
  ----------
  mean : np.ndarray
      Input mean
  covar : np.ndarray
      Input covariance
  alpha : float
      Van der Merwe alpha parameter
  beta : float
      Van der Merwe beta parameter
  kappa : float
      Van der Merwe kappa parameter

  Returns
  -------
  Tuple[np.ndarray, np.ndarray, np.ndarray]
      A tuple containing:
        - An array where each COLUMN contains the position of a sigma point. This is because the numpy cholesky function returns a lower triangular matrix by default.
        - An array containing the weights for each sigma point mean
        - An array containing the weights for each sigma point covariance
  """
  ndim_state = x.shape[-1]
  lambda_ = alpha**2 * (ndim_state + kappa) - ndim_state

  # Sigma points
  U = np.linalg.cholesky((ndim_state + lambda_) * P).swapaxes(-1, -2)
  x = x[..., None, :]
  sigmas = np.concatenate([x, subtract_fn(x, -U), subtract_fn(x, +U)], axis=-2)

  return sigmas


def merwe_sigma_weights(ndim_state: int,
                        # Merwe parameters
                        alpha: float,
                        beta: float,
                        kappa: float,
                        ) -> Tuple[np.ndarray, ...]:
  lambda_ = alpha**2 * (ndim_state + kappa) - ndim_state

  Wc = np.full(2 * ndim_state + 1, 1 / (2 * (ndim_state + lambda_)))
  Wm = np.full(2 * ndim_state + 1, 1 / (2 * (ndim_state + lambda_)))
  Wc[0] = lambda_ / (ndim_state + lambda_) + (1 - alpha**2 + beta)
  Wm[0] = lambda_ / (ndim_state + lambda_)

  return Wm, Wc
