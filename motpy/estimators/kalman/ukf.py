from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple

import numpy as np

import motpy.distributions.gaussian as gaussian
from motpy.distributions import Distribution, Gaussian
from motpy.estimators import StateEstimator
from motpy.gate import EllipsoidalGate
from motpy.models.measurement import MeasurementModel
from motpy.models.transition import TransitionModel


@dataclasses.dataclass
class SigmaPointDistribution:
  """
  Data class containing a distribution object and its sigma points

  NOTE: We assume Wm and Wc are the same for all elements
  """
  distribution: Optional[Distribution] = None
  sigma_points: Optional[np.ndarray] = None
  Wm: Optional[np.ndarray] = None
  Wc: Optional[np.ndarray] = None

  # Give the distribution attributes to this class
  def __getattr__(self, name):
    if self.distribution is not None and hasattr(self.distribution, name):
      return getattr(self.distribution, name)
    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'")

  def __setattr__(self, name, value):
    if self.distribution is not None and hasattr(self.distribution, name):
      setattr(self.distribution, name, value)
    else:
      super().__setattr__(name, value)

  def __getitem__(self, idx: int) -> SigmaPointDistribution:
    return SigmaPointDistribution(
        distribution=self.distribution[idx],
        sigma_points=self.sigma_points[idx],
        Wm=self.Wm,
        Wc=self.Wc
    )

  def __setitem__(self, idx: int, value: SigmaPointDistribution) -> None:
    self.distribution[idx] = value.distribution
    self.sigma_points[idx] = value.sigma_points
    self.Wm = value.Wm
    self.Wc = value.Wc

  def append(self,
             value: SigmaPointDistribution,
             axis: int = 0
             ) -> SigmaPointDistribution:
    return SigmaPointDistribution(
        distribution=self.distribution.append(value.distribution),
        sigma_points=np.append(
            self.sigma_points, value.sigma_points, axis=axis
        ) if self.sigma_points is not None else None,
        Wm=self.Wm,
        Wc=self.Wc
    )


class UnscentedKalmanFilter(StateEstimator):
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
              state: SigmaPointDistribution,
              dt: float,
              **kwargs,
              ) -> SigmaPointDistribution:
    # Transform sigma points to prediction space
    predicted_sigmas = self.transition_model(
        state.sigma_points, dt=dt, **kwargs
    )

    x_pred, P_pred = unscented_transform(
        sigmas=predicted_sigmas,
        Wm=state.Wm,
        Wc=state.Wc,
        noise_covar=self.transition_model.covar(dt=dt),
        residual_fn=self.state_residual_fn,
    )

    predicted_state = SigmaPointDistribution(
        distribution=Gaussian(
            mean=x_pred,
            covar=P_pred,
            weight=state.distribution.weight
        ),
        sigma_points=predicted_sigmas,
        Wm=state.Wm,
        Wc=state.Wc
    )

    return predicted_state

  def update(self,
             state: SigmaPointDistribution,
             measurement: np.ndarray,
             **kwargs
             ) -> Tuple[SigmaPointDistribution, Dict[str, Any]]:

    # Unscented transform in measurement space
    measured_sigmas = self.measurement_model(
        state.sigma_points, **kwargs
    )
    z_pred, S = unscented_transform(
        sigmas=measured_sigmas,
        Wm=state.Wm,
        Wc=state.Wc,
        noise_covar=self.measurement_model.covar(),
        residual_fn=self.measurement_residual_fn,
    )

    # Standard kalman update
    x_pred = state.distribution.mean
    P_pred = state.distribution.covar
    z = measurement
    Pxz = np.einsum(
        '...n, ...ni, ...nj -> ...ij',
        state.Wc,
        self.state_residual_fn(state.sigma_points, x_pred[..., None, :]),
        self.measurement_residual_fn(measured_sigmas, z_pred[..., None, :])
    )
    y = self.measurement_residual_fn(z, z_pred)
    K = Pxz @ np.linalg.inv(S)
    x_post = x_pred + np.einsum('...ij, ...j -> ...i', K, y)
    P_post = P_pred - K @ S @ K.swapaxes(-1, -2)
    P_post = 0.5 * (P_post + P_post.swapaxes(-1, -2))

    post_state = SigmaPointDistribution(
        distribution=Gaussian(
            mean=x_post,
            covar=P_post,
            weight=state.distribution.weight
        ),
        sigma_points=measured_sigmas,
        Wm=state.Wm,
        Wc=state.Wc
    )

    return post_state

  def likelihood(
      self,
      measurement: np.ndarray,
      state: SigmaPointDistribution,
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
    measured_sigmas = self.measurement_model(state.sigma_points, **kwargs)
    z_pred, S = unscented_transform(
        sigmas=measured_sigmas,
        Wm=state.Wm,
        Wc=state.Wc,
        noise_covar=self.measurement_model.covar(),
        residual_fn=self.measurement_residual_fn,
    )
    return gaussian.likelihood(
        x=measurement,
        mean=z_pred,
        covar=S
    )

  def gate(self,
           measurements: np.ndarray,
           state: SigmaPointDistribution,
           pg: float,
           **kwargs,
           ) -> np.ndarray:
    assert self.measurement_model is not None

    measured_sigmas = self.measurement_model(state.sigma_points, **kwargs)
    z_pred, S = unscented_transform(
        sigmas=measured_sigmas,
        Wm=state.Wm,
        Wc=state.Wc,
        noise_covar=self.measurement_model.covar(),
        residual_fn=self.measurement_residual_fn,
    )
    gate = EllipsoidalGate(pg=pg, ndim=measurements.shape[-1])
    gate_mask, _ = gate(
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
  # Mean
  x = np.einsum('...n, ...ni -> ...i', Wm, sigmas)

  # Covariance
  y = residual_fn(sigmas, x[..., None, :])
  P = np.einsum('...n, ...ni, ...nj -> ...ij', Wc, y, y) + noise_covar

  return x, P
