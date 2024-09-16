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

  @property
  def shape(self) -> Tuple[int]:
    return self.distribution.shape

  @property
  def size(self) -> int:
    return self.distribution.size

  def __getitem__(self, idx):
    return self.distribution[idx]

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float) -> Poisson:
    predicted = copy.deepcopy(self)

    predicted.distribution.weight *= ps
    predicted.distribution, filter_state = state_estimator.predict(
        state=predicted.distribution, dt=dt)
    predicted.distribution = predicted.distribution.append(
        self.birth_distribution)

    return predicted

  def prune(self, threshold: float) -> Poisson:
    """
    Prune components below weight threshold
    """
    pruned = copy.deepcopy(self)
    keep = self.distribution.weight > threshold
    pruned.distribution = self.distribution[keep]

    return pruned

  def merge(self) -> Poisson:
    """
    Merge the birth distribution back into the main distribution.

    NOTE: This assumes that only the birth distribution contributes, such that we only need to change the weights and covariance matrices.
    """
    assert isinstance(self.birth_distribution, GaussianState)
    nbirth = self.birth_distribution.size
    dist = self.distribution[:nbirth]
    birth_dist = self.birth_distribution

    merged_distribution = GaussianState(
        mean=dist.mean,
        covar=dist.covar,
        weight=dist.weight + birth_dist.weight,
    )

    return Poisson(birth_distribution=self.birth_distribution,
                   init_distribution=merged_distribution)
