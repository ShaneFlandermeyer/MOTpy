from __future__ import annotations
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
      birth_log_weights: List[float] = None,
      birth_states: List[GaussianState] = None,
  ):
    self.birth_log_weights = np.array(
        birth_log_weights) if birth_log_weights is not None else np.array([])
    self.birth_states = list(birth_states) if birth_states is not None else []

    self.log_weights = self.birth_log_weights
    self.states = self.birth_states

  def __repr__(self):
    return f"PoissonPointProcess(log_weights={np.array(self.log_weights).tolist()}, states={self.states})"

  def __len__(self):
    assert len(self.log_weights) == len(self.states)
    return len(self.log_weights)

  def __iter__(self):
    return zip(self.log_weights, self.states)

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float) -> Poisson:
    # Predict existing PPP density
    pred_weights = np.array(self.log_weights) + np.log(ps)
    pred_states = []
    for state in self.states:
      pred_states.append(state_estimator.predict(state=state, dt=dt))

    # Incorporate PPP birth intensity into PPP intensity
    pred_ppp = Poisson(
        birth_log_weights=np.concatenate(
            (pred_weights, self.birth_log_weights)),
        birth_states=pred_states + self.birth_states,
    )

    return pred_ppp

  def update(self,
             measurement: np.ndarray,
             pd: float,
             in_gate: np.ndarray,
             state_estimator: KalmanFilter,
             clutter_intensity: float,
             ) -> Tuple[Bernoulli, float]:
    # If a measurement is associated to a PPP component, we create a new Bernoulli whose existence probability depends on likelihood of measurement
    state_up = []
    weight_up = []
    for i in range(len(self.states)):
      if not in_gate[i]:
        continue
      # Update state and likelihoods for PPP components with measurement in gate
      state_up.append(state_estimator.update(measurement=measurement,
                                             predicted_state=self.states[i]))
      likelihood = state_estimator.likelihood(
          measurement=measurement.reshape((1, -1)),
          predicted_state=self.states[i])
      weight_up.append(self.log_weights[i] + np.log(pd) + np.log(likelihood))
    weight_up = np.array(weight_up)

    # Create a new Bernoulli component based on updated weights
    norm_log_w_up, sum_log_w_up = normalize_log_weights(weight_up)
    _, sum_log_w_total = normalize_log_weights(np.concatenate(
        (weight_up, np.log([clutter_intensity]))))
    r = np.exp(sum_log_w_up - sum_log_w_total)

    # Compute the state using moment matching across all PPP components
    mean, covar = mix_gaussians(means=[state.mean for state in state_up],
                                covars=[state.covar for state in state_up],
                                weights=np.exp(norm_log_w_up))
    bern = Bernoulli(r=r, state=GaussianState(mean=mean, covar=covar))
    return bern, sum_log_w_total
