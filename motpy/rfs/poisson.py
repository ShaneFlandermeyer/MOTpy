from __future__ import annotations
import copy
import time
import jax
import numpy as np
from typing import Callable, List, Optional, Tuple, Union

from motpy.kalman import KalmanFilter
from motpy.measures import mahalanobis
from motpy.rfs.bernoulli import Bernoulli
from motpy.distributions.gaussian import match_moments, GaussianState
# from sklearn.cluster import DBSCAN
# from motpy.measures import pairwise_euclidean
from motpy.distributions.gaussian import merge_mixture
from motpy.common import nextpow2


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

  def append(self, state: GaussianState) -> None:
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
    r = sum_w_up / (sum_w_total + 1e-15)

    # Compute the state using moment matching across all PPP components
    if n_in_gate == 1:
      mean = mixture_up.mean
      covar = mixture_up.covar
    else:
      mean, covar = match_moments(
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
    """
    fast_merge = True
    if fast_merge:
      nbirth = len(self.birth_distribution)
      dist = self.distribution[:nbirth]
      birth_dist = self.birth_distribution

      wmix = np.stack((dist.weight, birth_dist.weight), axis=0)
      wmix /= np.sum(wmix + 1e-15, axis=0)
      Pmix = np.stack((dist.covar, birth_dist.covar), axis=0)
      merged_distribution = GaussianState(
          mean=dist.mean,
          covar=np.einsum('i..., i...jk -> ...jk', wmix, Pmix),
          weight=dist.weight + birth_dist.weight,
      )
    else:
      # Pad inputs to the next power of 2
      means = self.distribution.mean
      covars = self.distribution.covar
      weights = self.distribution.weight

      npad = nextpow2(means.shape[0]) - means.shape[0]
      means = np.append(means, np.zeros((npad, *means.shape[1:])), axis=0)
      covars = np.append(covars, np.zeros((npad, *covars.shape[1:])), axis=0)
      weights = np.append(weights, np.zeros(npad))

      means, covars, weights = merge_mixture(
          means=means,
          covars=covars,
          weights=weights,
          threshold=threshold)

      valid = weights > 0
      merged_distribution = GaussianState(
          mean=np.asarray(means)[valid],
          covar=np.asarray(covars)[valid],
          weight=np.asarray(weights)[valid]
      )

    return Poisson(birth_distribution=self.birth_distribution,
                   init_distribution=merged_distribution)

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


if __name__ == '__main__':
  grid_extents = np.array([-1, 1])
  nx = ny = 10
  n_init = 100
  ngrid = nx * ny

  dx = np.diff(grid_extents).item() / nx
  dy = np.diff(grid_extents).item() / ny
  ppp_x_pos, ppp_y_pos = np.meshgrid(
      np.linspace(*(grid_extents+[dx/2, -dx/2]), nx),
      np.linspace(*(grid_extents+[dy/2, -dy/2]), ny))
  ppp_x_vel = np.zeros_like(ppp_x_pos)
  ppp_y_vel = np.zeros_like(ppp_x_pos)

  init_mean = np.stack(
      (ppp_x_pos.ravel(), ppp_x_vel.ravel(),
       ppp_y_pos.ravel(), ppp_y_vel.ravel()), axis=-1)
  init_covar = np.repeat(np.diag([dx**2, 0, dy**2, 0])[np.newaxis, ...],
                         len(init_mean), axis=0)
  init_weights = np.ones((nx, ny))

  undetected_dist = GaussianState(mean=init_mean,
                                  covar=init_covar,
                                  weight=init_weights.ravel())

  birth_rate = 0.1
  birth_weights = np.random.uniform(
      low=0, high=2*birth_rate/ngrid, size=(nx, ny))
  birth_mean = init_mean.copy()
  birth_covar = init_covar.copy()
  birth_dist = GaussianState(mean=birth_mean,
                             covar=birth_covar,
                             weight=birth_weights.ravel())

  ppp = Poisson(birth_distribution=birth_dist,
                init_distribution=undetected_dist)
  ppp.append(birth_dist)
  ppp.merge(threshold=0.1)
