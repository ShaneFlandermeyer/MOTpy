from __future__ import annotations
import copy
import numpy as np
from typing import Callable, List, Tuple, Union

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
      birth_weights: np.ndarray = None,
      birth_states: GaussianState = None,
  ):
    self.birth_weights = birth_weights
    self.birth_states = birth_states

    self.weights = np.array([])
    self.states = GaussianState(
        mean=np.empty([0, birth_states.state_dim]),
        covar=np.empty([0, birth_states.state_dim, birth_states.state_dim]))

  def __repr__(self):
    return f"Poisson(weights={self.weights.tolist()}, \nstates={self.states})"

  def __len__(self):
    assert len(self.weights) == len(self.states)
    return len(self.weights)

  def __iter__(self):
    return zip(self.weights, self.states)

  def append(self,
             weight: Union[float, np.ndarray],
             state: Union[GaussianState, List[GaussianState]]) -> None:
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
    self.states.extend(state)

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float) -> Poisson:
    # Predict existing PPP density
    pred_ppp = Poisson(birth_weights=self.birth_weights,
                       birth_states=self.birth_states)
    pred_ppp.weights = np.concatenate((self.weights*ps, self.birth_weights))
    
    pred_ppp.states = state_estimator.predict(state=self.states, dt=dt)
    pred_ppp.states.append(self.birth_states)

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
    # Get PPP components in gate
    n_in_gate = np.count_nonzero(in_gate)
    if n_in_gate == 0:
      # No measurements in gate
      return None, 0

    gate_states = self.states[in_gate]
    gate_weights = self.weights[in_gate]
    likelihoods = likelihoods[in_gate]
    pds = pd[in_gate]

    # If a measurement is associated to a PPP component, we create a new Bernoulli whose existence probability depends on likelihood of measurement
    state_up = state_estimator.update(measurement=measurement,
                                      predicted_state=gate_states)
    weight_up = gate_weights * likelihoods * pds
    # state_up = []
    # weight_up = np.empty(n_in_gate)
    # for i in range(n_in_gate):
    #   # Update state and likelihoods for PPP components with measurement in gate
    #   state_up.append(state_estimator.update(measurement=measurement,
    #                                          predicted_state=gate_states[i]))
    #   weight_up[i] = gate_weights[i] * likelihoods[i] * pds[i]

    # Create a new Bernoulli component based on updated weights
    sum_w_up = np.sum(weight_up)
    sum_w_total = sum_w_up + clutter_intensity
    r = sum_w_up / sum_w_total

    # Compute the state using moment matching across all PPP components
    mean, covar = mix_gaussians(
        means=state_up.mean,
        covars=state_up.covar,
        weights=weight_up)
    bern = Bernoulli(r=r, state=GaussianState(mean=mean, covar=covar))
    return bern, sum_w_total

  # @profile
  def prune(self, threshold: float) -> Poisson:
    pruned = Poisson(birth_weights=self.birth_weights,
                     birth_states=self.birth_states)
    # Prune components with existence probability below threshold
    keep = self.weights > threshold
    pruned.weights = self.weights[keep]
    pruned.states = self.states[keep]

    return pruned

  # @profile
  def merge(self, threshold: float) -> Poisson:
    """
    Merge components that are close to each other.

    TODO: Only supports mahalanobis distance for now.
    """
    merged = Poisson(birth_weights=self.birth_weights,
                     birth_states=self.birth_states)
    old_weights = self.weights.copy()

    old_means = np.array([s.mean for s in self.states])
    old_covars = np.array([s.covar for s in self.states])
    while len(old_weights) > 0:
      # Find components that are close to each other
      dists = mahalanobis(
          mean=old_means[0], covar=old_covars[0], points=old_means)
      similar = dists < threshold
      # Mix components that are close to each other
      if np.any(similar):
        new_weight = np.sum(old_weights[similar])
        mix_mean, mix_covar = mix_gaussians(
            means=old_means[similar],
            covars=old_covars[similar],
            weights=old_weights[similar])
        mix_state = GaussianState(mean=mix_mean, covar=mix_covar)
        merged.append(weight=new_weight, state=mix_state)
        # Remove components that have been merged
        old_weights = old_weights[~similar]
        old_means = old_means[~similar]
        old_covars = old_covars[~similar]
      else:
        # No components are close to each other
        merged.weights.append(old_weights[0])
        merged.states.append(GaussianState(
            mean=old_means[0], covar=old_covars[0]))
        old_weights = old_weights[1:]

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
    intensity = np.zeros(grid.shape[:-1])
    for i, state in enumerate(self.states):
      mean = H @ state.mean
      cov = H @ state.covar @ H.T
      rv = multivariate_normal(mean=mean, cov=cov)
      intensity += self.weights[i] * rv.pdf(grid)
    return intensity
