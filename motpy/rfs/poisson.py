from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple

from motpy.kalman import KalmanFilter
from motpy.distributions.gaussian import GaussianState


class Poisson:
  """
  Class to hold all poisson distributions. Methods include birth, prediction, merge, prune, recycle.
  """

  def __init__(
      self,
      birth_distribution: GaussianState,
      distribution: Optional[GaussianState] = None,
  ):
    self.birth_distribution = birth_distribution
    self.distribution = distribution

  def __repr__(self):
    return f"""Poisson(birth_distribution={self.birth_distribution},
  distribution={self.distribution})"""

  @property
  def shape(self) -> Tuple[int]:
    return self.distribution.shape

  @property
  def size(self) -> int:
    return self.distribution.size

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

  def prune(
          self,
          threshold: float,
          meta: Optional[List[Dict[str, Any]]] = None
  ) -> Poisson:
    """
    Prune components below weight threshold
    """
    pruned = copy.deepcopy(self)
    valid = pruned.distribution.weight > threshold
    pruned.distribution = pruned.distribution[valid]

    if meta is None:
      new_meta = None
    else:
      new_meta = [meta[i] for i in range(len(meta)) if valid[i]]

    return pruned, new_meta

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
                   distribution=merged_distribution)
