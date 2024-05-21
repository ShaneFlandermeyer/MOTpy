from __future__ import annotations
import copy
import time
import jax
import numpy as np
from typing import Callable, List, Optional, Tuple, Union

from motpy.kalman import KalmanFilter
from motpy.measures import mahalanobis
from motpy.rfs.bernoulli import MultiBernoulli
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
    pred_ppp.distribution, filter_state = state_estimator.predict(
        state=pred_ppp.distribution, dt=dt)
    pred_ppp.distribution.append(pred_ppp.birth_distribution)

    return pred_ppp

  def update(self,
             measurement: np.ndarray,
             pd: np.ndarray,
             likelihoods: np.ndarray,
             in_gate: np.ndarray,
             state_estimator: KalmanFilter,
             clutter_intensity: float,
             ) -> Tuple[MultiBernoulli, float]:
    n_in_gate = np.count_nonzero(in_gate)
    if n_in_gate == 0:
      # No measurements in gate
      return None, 0

    # If a measurement is associated to a PPP component, we create a new Bernoulli whose existence probability depends on likelihood of measurement
    mixture_up, filter_state = state_estimator.update(
        predicted_state=self.distribution[in_gate], measurement=measurement)
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

    bern = MultiBernoulli(r=r, state=GaussianState(mean=mean, covar=covar))
    return bern, sum_w_total

  def prune(self, threshold: float) -> Poisson:
    pruned = copy.copy(self)
    # Prune components with existence probability below threshold
    keep = self.distribution.weight > threshold
    pruned.distribution = self.distribution[keep]

    return pruned

  def merge(self) -> Poisson:
    """
    Merge components that are close to each other.
    """
    nbirth = len(self.birth_distribution)
    dist = self.distribution[:nbirth]
    birth_dist = self.birth_distribution
    merged_distribution = GaussianState(
        mean=dist.mean,
        covar=dist.covar,
        weight=dist.weight + birth_dist.weight,
    )

    return Poisson(birth_distribution=self.birth_distribution,
                   init_distribution=merged_distribution)