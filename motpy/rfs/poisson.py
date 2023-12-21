from __future__ import annotations
import copy
import numpy as np
from typing import List, Tuple

from motpy.kalman import KalmanFilter
from motpy.rfs.bernoulli import Bernoulli
from motpy.distributions.gaussian import mix_gaussians, GaussianState


def normalize_log_weights(log_w: np.ndarray) -> Tuple[np.ndarray, float]:
  """
  Normalize weights in log space to sum to 1 in linear space.

  Parameters
  ----------
  log_w : np.ndarray
      Array of log weights

  Returns
  -------
  Tuple[np.ndarray, float]
      - Array of normalized log weights
      - Sum of normalized log weights
  """
  max_w = np.max(log_w)
  log_sum_w = max_w + np.log(1 + np.sum(np.exp(log_w[log_w != max_w] - max_w)))
  log_w = log_w - log_sum_w
  return log_w, log_sum_w


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

  def append(self, weight: float, state: GaussianState):
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
    self.weights.append(weight)
    self.states.append(state)

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float) -> Poisson:
    # Predict existing PPP density
    pred_weights = self.weights * ps
    pred_states = []
    for state in self.states:
      pred_states.append(state_estimator.predict(state=state, dt=dt))

    # Incorporate PPP birth intensity into PPP intensity
    pred_ppp = copy.deepcopy(self)
    pred_ppp.weights = np.concatenate(
        (pred_weights, self.birth_weights))
    pred_ppp.states = pred_states + self.birth_states

    return pred_ppp

  def update(self,
             measurement: np.ndarray,
             pd: float,
             in_gate: np.ndarray,
             state_estimator: KalmanFilter,
             clutter_intensity: float,
             ) -> Tuple[Bernoulli, float]:

    # Get PPP components in gate
    n_in_gate = np.count_nonzero(in_gate)
    if n_in_gate == 0:
      # No measurements in gate
      return None, 0

    if pd == 0:
      # No measurements to update
      return None, 0

    gate_states = [s for i, s in enumerate(self.states) if in_gate[i]]
    gate_weights = [w for i, w in enumerate(self.weights) if in_gate[i]]

    # If a measurement is associated to a PPP component, we create a new Bernoulli whose existence probability depends on likelihood of measurement
    state_up = []
    weight_up = np.empty(n_in_gate)
    for i in range(n_in_gate):
      state = gate_states[i]
      w = gate_weights[i]

      # Update state and likelihoods for PPP components with measurement in gate
      state_up.append(state_estimator.update(measurement=measurement,
                                             predicted_state=state))
      l = state_estimator.likelihood(measurement=measurement,
                                     predicted_state=state)
      weight_up[i] = w * pd * l

    # Create a new Bernoulli component based on updated weights
    sum_w_up = np.sum(weight_up)
    norm_w_up = weight_up / sum_w_up
    sum_w_total = sum_w_up + clutter_intensity
    r = sum_w_up / sum_w_total

    # Compute the state using moment matching across all PPP components
    mean, covar = mix_gaussians(means=[state.mean for state in state_up],
                                covars=[state.covar for state in state_up],
                                weights=norm_w_up)
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
