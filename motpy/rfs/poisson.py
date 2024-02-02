from __future__ import annotations
import copy
import numpy as np
from typing import Callable, List, Optional, Tuple, Union

from motpy.kalman import KalmanFilter
from motpy.measures import mahalanobis
from motpy.rfs.bernoulli import Bernoulli
from motpy.distributions.gaussian import mix_gaussians, GaussianState
from scipy.stats import multivariate_normal


class Poisson:
  """
  Class to hold all poisson distributions. Methods include birth, prediction, merge, prune, recycle.
  """

  def __init__(
      self,
      birth_distribution: GaussianState,
      init_distribution: Optional[GaussianState] = None,
  ):
    self.birth_distribution = birth_distribution

    self.distribution = init_distribution

  def __repr__(self):
    return f"""Poisson(birth_distribution={self.birth_distribution},
  distribution={self.distribution})"""

  def __len__(self):
    return len(self.distribution)

  def __getitem__(self, idx):
    return self.distribution[idx]

  def append(self,
             weight: Union[float, np.ndarray],
             state: GaussianState) -> None:
    """
    Append a new Gaussian state and its corresponding weight to the Poisson process.

    The weight is converted to logarithmic scale before being appended.

    Parameters
    ----------
    weight : float
        The weight corresponding to the state. This is converted to a logarithmic scale for internal storage.
    state : GaussianState
        The Gaussian state to be appended.

    Returns
    -------
    None
    """
    state = state if isinstance(state, list) else [state]
    self.weights = np.append(self.weights, weight)
    self.distribution.append(state)

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float) -> Poisson:
    # Predict existing PPP density
    pred_ppp = copy.copy(self)

    pred_ppp.distribution.weight *= ps
    pred_ppp.distribution = state_estimator.predict(
        state=pred_ppp.distribution, dt=dt)
    pred_ppp.distribution.append(pred_ppp.birth_distribution)

    return pred_ppp

  # @profile
  def update(self,
             measurement: np.ndarray,
             pd: np.ndarray,
             likelihoods: np.ndarray,
             in_gate: np.ndarray,
             state_estimator: KalmanFilter,
             clutter_intensity: float,
             ) -> Tuple[Bernoulli, float]:
    n_in_gate = np.count_nonzero(in_gate)
    if n_in_gate == 0:
      # No measurements in gate
      return None, 0

    # If a measurement is associated to a PPP component, we create a new Bernoulli whose existence probability depends on likelihood of measurement
    mixture_up = state_estimator.update(
        measurement=measurement, predicted_state=self.distribution[in_gate])
    mixture_up.weight *= likelihoods[in_gate] * pd[in_gate]

    # Create a new Bernoulli component based on updated weights
    sum_w_up = np.sum(mixture_up.weight)
    sum_w_total = sum_w_up + clutter_intensity
    r = sum_w_up / sum_w_total

    # Compute the state using moment matching across all PPP components
    mean, covar = mix_gaussians(
        means=mixture_up.mean,
        covars=mixture_up.covar,
        weights=mixture_up.weight)
    bern = Bernoulli(r=r, state=GaussianState(mean=mean, covar=covar))
    return bern, sum_w_total

  def prune(self, threshold: float) -> Poisson:
    pruned = copy.copy(self)
    # Prune components with existence probability below threshold
    keep = self.distribution.weight > threshold
    pruned.distribution = self.distribution[keep]

    return pruned

  def merge(self, threshold: float) -> Poisson:
    """
    Merge components that are close to each other.

    TODO: Currently assumes there is no thresholding
    """
    assert len(self) == 2 * len(self.birth_distribution)

    nbirth = len(self.birth_distribution)
    dist = self.distribution[:nbirth]
    birth_dist = self.birth_distribution

    wmix = np.concatenate(
        (dist.weight[None, ...], birth_dist.weight[None, ...]), axis=0)
    wmix /= np.sum(wmix + 1e-15, axis=0)
    Pmix = np.concatenate(
        (dist.covar[None, ...], birth_dist.covar[None, ...]), axis=0)
    merged_distribution = GaussianState(
        mean=dist.mean,
        covar=np.einsum('i..., i...jk -> ...jk', wmix, Pmix),
        weight=dist.weight + birth_dist.weight,
    )
    merged = Poisson(birth_distribution=self.birth_distribution,
                     init_distribution=merged_distribution)
    return merged
  
  def intensity(self, grid: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Compute the intensity of the Poisson process at a grid of points.

    Parameters
    ----------
    grid : np.ndarray
        Query points
    H : np.ndarray
        Matrix for extracting relevant dims

    Returns
    -------
    np.ndarray
        Intensity grid
    """
    raise NotImplementedError(
        "Intensity currently not supported with GaussianMixture API")
    intensity = np.zeros(grid.shape[:-1])
    for i, state in enumerate(self.states):
      mean = H @ state.mean[0]
      cov = H @ state.covar[0] @ H.T
      rv = multivariate_normal(mean=mean, cov=cov)
      intensity += self.weights[i] * rv.pdf(grid)
    return intensity
