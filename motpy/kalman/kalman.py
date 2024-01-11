from typing import List, Optional, Tuple
import warnings
import numpy as np

from motpy.models.measurement.base import MeasurementModel
from motpy.models.transition.base import TransitionModel
from motpy.distributions.gaussian import GaussianState
import motpy.distributions.gaussian as gaussian
from motpy.gate import EllipsoidalGate
import torch


class KalmanFilter():
  def __init__(self,
               transition_model: Optional[TransitionModel] = None,
               measurement_model: Optional[MeasurementModel] = None,
               ):
    self.transition_model = transition_model
    self.measurement_model = measurement_model

  def predict(self,
              state: GaussianState,
              dt: float,
              ) -> GaussianState:
    assert self.transition_model is not None

    x, P = state.mean, state.covar
    F = torch.tensor(self.transition_model.matrix(dt=dt))
    Q = torch.tensor(self.transition_model.covar(dt=dt))

    x_pred = x @ F.T
    P_pred = F @ P @ F.T + Q

    # # Clear cache from previous update step
    # meta = state.metadata.copy()
    # if 'cache' in meta:
    #   for key in ['S', 'K', 'P_post']:
    #     meta['cache'].pop(key, None)

    return GaussianState(mean=x_pred, covar=P_pred)

  def update(self,
             measurement: torch.Tensor,
             predicted_state: GaussianState) -> GaussianState:
    assert self.measurement_model is not None

    x_pred, P_pred = predicted_state.mean, predicted_state.covar
    z = measurement
    H = torch.tensor(self.measurement_model.matrix())
    R = torch.tensor(self.measurement_model.covar())

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ torch.inverse(S)
    P_post = P_pred - K @ S @ K.mT
    P_post = (P_post + P_post.mT) / 2

    z_pred = x_pred @ H.T
    x_post = x_pred + torch.einsum('...ij, ...j -> ...i', K, z - z_pred)

    # Use cached values if available
    # cache = predicted_state.metadata.get('cache', {})

    # S = cache['S'] if 'S' in cache else H @ P_pred @ H.T + R
    # K = cache['K'] if 'K' in cache else P_pred @ H.T @ np.linalg.inv(S)
    # if 'P_post' in cache:
    #   P_post = cache['P_post']
    # else:
    #   P_post = P_pred - K @ S @ K.T
    #   P_post = (P_post + P_post.T) / 2

    # z_pred = H @ x_pred
    # x_post = x_pred + K @ (z - z_pred) if z is not None else None

    # Update cache
    # meta = predicted_state.metadata
    # meta['cache'] = meta.get('cache', {})
    # meta['cache'].update(dict(S=S, K=K, P_post=P_post))

    return GaussianState(mean=x_post, covar=P_post)

  def gate(self,
           measurements: List[np.ndarray],
           predicted_state: GaussianState,
           pg: float = 0.999,
           ) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Gate measurements using the predicted state

    Parameters
    ----------
    measurements : np.ndarray
        Measurements
    predicted_state : GaussianState
        Predicted state
    pg : float
        Gate probability

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Measurements in the gate and their indices
    """
    assert self.measurement_model is not None

    warnings.warn('This function is not implemented yet.')
    return measurements, np.arange(len(measurements))
    x, P = predicted_state.mean, predicted_state.covar
    H = self.measurement_model.matrix()
    R = self.measurement_model.covar()
    z_pred = self.measurement_model(x, noise=False)
    S = H @ P @ H.T + R

    gate = EllipsoidalGate(pg=pg, ndim=measurements[0].size)
    return gate(measurements=measurements,
                predicted_measurement=z_pred,
                innovation_covar=S)

  def likelihood(
      self,
      measurement: np.ndarray,
      predicted_state: GaussianState,
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

    x, P = predicted_state.mean, predicted_state.covar
    z = torch.tensor(np.array(measurement))
    H = torch.tensor(self.measurement_model.matrix())
    R = torch.tensor(self.measurement_model.covar())
    return gaussian.likelihood(
        z=z,
        z_pred=x @ H.T,
        P_pred=P,
        H=H,
        R=R,
    )
