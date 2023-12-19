import copy
from typing import List, Tuple

import numpy as np

from motpy.distributions.gaussian import GaussianState, mix_gaussians
from motpy.kalman import KalmanFilter
from motpy.rfs.bernoulli import Bernoulli, MultiBernoulli
from motpy.rfs.poisson import Poisson
from motpy.gate import EllipsoidalGate


class TOMBP:
  """
  Track-oriented Multi-Bernoulli/Poisson filter
  """

  def __init__(self,
               birth_weights: np.ndarray,
               birth_states: List[np.ndarray],
               pg: float = None,
               w_min: float = None,
               r_min: float = None,
               r_estimate_threshold: float = None,
               ):
    log_weights = np.log(birth_weights) if birth_weights is not None else None
    self.poisson = Poisson(birth_log_weights=log_weights,
                           birth_states=birth_states)
    self.mb = MultiBernoulli()

    self.pg = pg
    self.w_min = w_min
    self.r_min = r_min
    self.r_estimate_threshold = r_estimate_threshold

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float) -> Tuple[MultiBernoulli, Poisson]:
    # Predict existing tracks
    pred_mb = self.mb.predict(state_estimator=state_estimator, ps=ps, dt=dt)

    # Predict PPP intensity and incorporate birth into PPP
    pred_poisson = self.poisson.predict(
        state_estimator=state_estimator, ps=ps, dt=dt)

    return pred_mb, pred_poisson

  def update(self,
             measurements: List[np.ndarray],
             state_estimator: KalmanFilter,
             pd: float,
             clutter_intensity: float = 0
             ) -> Tuple[MultiBernoulli, Poisson]:
    ndim_meas = measurements[0].size
    gate = EllipsoidalGate(pg=self.pg, ndim=ndim_meas)

    # Update existing tracks
    state_hypos = np.empty((len(self.mb), len(measurements)+1), dtype=object)
    w_upd = np.zeros((len(self.mb), len(measurements)+1))
    in_gate_mb = np.zeros((len(self.mb), len(measurements)), dtype=bool)
    for i, bern in enumerate(self.mb):
      # Missed detection hypothesis
      state_hypos[i, 0] = bern.update(measurement=None, pd=pd)
      w_upd[i, 0] = 1 - bern.r + bern.r * (1 - pd)

      # Gate measurements
      S = state_estimator.update(
          measurement=None, predicted_state=bern.state).metadata['S']
      valid_meas, valid_inds = gate(measurements=measurements,
                                    predicted_measurement=bern.state.mean,
                                    innovation_covar=S)
      in_gate_mb[i, valid_inds] = True

      # Create hypotheses from measurement updates
      likelihoods = np.exp(bern.log_likelihood(
          measurements=valid_meas, pd=pd, state_estimator=state_estimator))
      w_upd[i, valid_inds+1] = bern.r * pd * likelihoods
      for j, z in zip(valid_inds+1, valid_meas):
        state_hypos[i, j] = bern.update(
            measurement=z, pd=pd, state_estimator=state_estimator)

    # Create a new track for each measurement by updating PPP with measurement
    r_new = np.empty(len(measurements))
    state_new = []
    w_new = np.empty(len(measurements))
    for i, z in enumerate(measurements):
      bern, log_w_new = self.poisson.update(
          measurement=z,
          pd=pd,
          # TODO: Add gating
          in_gate=np.ones(len(self.poisson), dtype=bool),
          state_estimator=state_estimator,
          clutter_intensity=clutter_intensity,
      )
      state_new.append(bern.state)
      r_new[i] = bern.r
      w_new[i] = np.exp(log_w_new)

    # Undetected PPP update
    self.poisson.log_weights += np.log(1 - pd)

    # Use SPA to compute marginal association probabilities
    p_upd, p_new = self.spa(w_upd=w_upd, w_new=w_new)

    # Update existing tracks using marginal probs
    valid_mb = np.concatenate(
        (np.ones((len(self.mb), 1), dtype=bool), in_gate_mb), axis=1)
    for i, bern in enumerate(self.mb):
      valid = valid_mb[i]
      ri = np.array([h.r for h in state_hypos[i, valid]])
      xi = np.array([h.state.mean for h in state_hypos[i, valid]])
      Pi = np.array([h.state.covar for h in state_hypos[i, valid]])

      mix_r = np.sum(ri * p_upd[i, valid])
      mix_weights = p_upd[i, valid] * ri / mix_r
      mean, covar = mix_gaussians(means=xi, covars=Pi, weights=mix_weights)
      mix_state = GaussianState(mean=mean, covar=covar)

      self.mb[i].r = mix_r
      self.mb[i].state = mix_state

    # Form new tracks
    for i in range(len(measurements)):
      bern = Bernoulli(r=r_new[i] * p_new[i], state=state_new[i])
      self.mb.append(bern)

    # Bernoulli recycling
    if self.r_min is not None:
      self.mb, self.poisson = self.recycle(r_min=self.r_min)
    # PPP pruning
    if self.w_min is not None:
      self.poisson = self.poisson.prune(threshold=self.r_min)

  @staticmethod
  def spa(w_upd: np.ndarray, w_new: np.ndarray, eps: float = 1e-4
          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute marginal association probabilities using the Sum-Product Algorithm, allowing for a time-varying number of objects.

    Parameters
    ----------
    w_upd : np.ndarray
        Single-object association weights for existing objects
    w_new : np.ndarray
        Single-object association weights for potential new objects
    eps : float, optional
        Termination error threshold, by default 1e-4

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - p_upd : np.ndarray
            Marginal association probabilities for existing objects
        - p_new : np.ndarray
            Marginal association probabilities for new objects
    """

    n, mp1 = w_upd.shape
    m = mp1 - 1

    mu_ba = np.ones((n, m))
    mu_ab = np.zeros((n, m))

    delta = np.inf
    while delta > eps:
      mu_ba_old = mu_ba

      w_muba = w_upd[:, 1:] * mu_ba
      mu_ab = w_upd[:, 1:] / (w_upd[:, 0][:, np.newaxis] +
                              np.sum(w_muba, axis=1, keepdims=True) - w_muba)
      mu_ba = 1 / (w_new + np.sum(mu_ab, axis=0, keepdims=True) - mu_ab)

      delta = np.max(np.abs(mu_ba - mu_ba_old))

    # Compute marginal association probabilities
    mu_ba = np.concatenate((np.ones((n, 1)), mu_ba), axis=1)
    p_upd = w_upd * mu_ba / (np.sum(w_upd * mu_ba, axis=1, keepdims=True))

    p_new = w_new / (w_new + np.sum(mu_ab, axis=0))

    return p_upd, p_new

  def recycle(self, r_min: float) -> Tuple[MultiBernoulli, Poisson]:
    poisson = copy.deepcopy(self.poisson)
    mb = copy.deepcopy(self.mb)

    for bern in self.mb:
      if bern.r < r_min:
        log_w = np.log(bern.r)
        state = bern.state

        poisson.log_weights = np.concatenate(
            (self.poisson.log_weights, log_w))
        poisson.states.append(state)

        mb.remove(bern)

    return mb, poisson
