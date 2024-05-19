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


class Poisson:
  """
  Class to hold all poisson distributions. Methods include birth, prediction, merge, prune, recycle.
  """

  def __init__(
      self,
      birth_distribution: GaussianState,
      init_distribution: Optional[GaussianState] = None,
      metadata: Optional[dict] = dict(),
  ):
    self.birth_distribution = birth_distribution

    self.distribution = init_distribution
    self.metadata = metadata

  def __repr__(self):
    return f"""Poisson(birth_distribution={self.birth_distribution},
  distribution={self.distribution})"""

  def __len__(self):
    return len(self.distribution)

  def __getitem__(self, idx):
    return self.distribution[idx]

  def append(self, state: GaussianState) -> None:
    self.distribution.append(state)

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float) -> Poisson:
    # Predict existing PPP density
    pred_ppp = copy.copy(self)

    pred_ppp.distribution.weight *= ps
    pred_ppp.distribution, pred_ppp.metadata = state_estimator.predict(
        state=pred_ppp.distribution, dt=dt, metadata=self.metadata)
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
    mixture_up, _ = state_estimator.update(
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
    return Poisson(birth_distribution=self.birth_distribution,
                   init_distribution=merged_distribution)
    


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
