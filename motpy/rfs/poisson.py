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
      birth_weights: List[float] = None,
      birth_states: List[GaussianState] = None,
  ):
    self.birth_weights = np.array(
        birth_weights) if birth_weights is not None else np.array([])
    self.birth_states = list(birth_states) if birth_states is not None else []

    self.weights = np.array([])
    self.states = []

  def __repr__(self):
    return f"PoissonPointProcess(log_weights={np.array(self.weights).tolist()}, states={self.states})"

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
    pred_ppp.states = [state_estimator.predict(
        state=state, dt=dt) for state in self.states] + self.birth_states

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

    gate_states = [s for i, s in enumerate(self.states) if in_gate[i]]
    gate_weights = self.weights[in_gate]
    likelihoods = likelihoods[in_gate]
    pds = pd[in_gate]

    # If a measurement is associated to a PPP component, we create a new Bernoulli whose existence probability depends on likelihood of measurement
    state_up = []
    weight_up = np.empty(n_in_gate)
    for i in range(n_in_gate):
      # Update state and likelihoods for PPP components with measurement in gate
      state_up.append(state_estimator.update(measurement=measurement,
                                             predicted_state=gate_states[i]))
      weight_up[i] = gate_weights[i] * likelihoods[i] * pds[i]

    # Create a new Bernoulli component based on updated weights
    sum_w_up = np.sum(weight_up)
    sum_w_total = sum_w_up + clutter_intensity
    r = sum_w_up / sum_w_total

    # Compute the state using moment matching across all PPP components
    mean, covar = mix_gaussians(means=[state.mean for state in state_up],
                                covars=[state.covar for state in state_up],
                                weights=weight_up)
    bern = Bernoulli(r=r, state=GaussianState(mean=mean, covar=covar))
    return bern, sum_w_total

  def prune(self, threshold: float) -> Poisson:
    pruned = copy.deepcopy(self)
    # Prune components with existence probability below threshold
    keep = self.weights > threshold
    pruned.weights = self.weights[keep]
    pruned.states = [self.states[i]
                     for i in range(len(self.states)) if keep[i]]

    return pruned

  def merge(self, threshold: float) -> Poisson:
    """
    Merge components that are close to each other.

    TODO: Only supports mahalanobis distance for now.
    """
    merged = Poisson(birth_weights=self.birth_weights,
                     birth_states=self.birth_states)
    old_weights, old_states = self.weights.copy(), self.states.copy()
    while len(old_weights) > 0:
      # Find components that are close to each other
      dists = mahalanobis(ref_dist=old_states[0], states=old_states)
      similar = dists < threshold
      # Mix components that are close to each other
      if np.any(similar):
        new_weight = np.sum(old_weights[similar])
        mix_mean, mix_covar = mix_gaussians(
            means=[s.mean for i, s in enumerate(old_states) if similar[i]],
            covars=[s.covar for i, s in enumerate(old_states) if similar[i]],
            weights=old_weights[similar])
        mix_state = GaussianState(mean=mix_mean, covar=mix_covar)
        merged.append(weight=new_weight, state=mix_state)
        # Remove components that have been merged
        old_weights = old_weights[~similar]
        old_states = [s for i, s in enumerate(old_states) if not similar[i]]
      else:
        # No components are close to each other
        merged.weights.append(old_weights[0])
        merged.states.append(old_states[0])
        old_weights = old_weights[1:]
        old_states.pop(0)

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
