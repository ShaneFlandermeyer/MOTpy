import copy
from typing import List, Tuple

import numpy as np

from motpy.distributions.gaussian import GaussianState, mix_gaussians
from motpy.kalman import KalmanFilter
from motpy.rfs.bernoulli import Bernoulli, MultiBernoulli
from motpy.rfs.poisson import Poisson
from motpy.gate import EllipsoidalGate


class TOMBP:
  def __init__(self,
               birth_weights: np.ndarray,
               birth_states: List[np.ndarray],
               pg: float = None,
               w_min: float = None,
               r_min: float = None,
               r_estimate_threshold: float = None,
               ):
    self.poisson = Poisson(birth_weights=birth_weights,
                           birth_states=birth_states)
    self.mb = []

    self.pg = pg
    self.w_min = w_min
    self.r_min = r_min
    self.r_estimate_threshold = r_estimate_threshold

  def predict(self, state_estimator, dt, Ps):
    """
    PREDICT MULTI-BERNOULLI AND POISSON COMPONENTS
    Input:
    r(i), x[:,i] and P[:,:,i] give the probability of existence, state 
    estimate and covariance for the i-th multi-Bernoulli component (track)
    lambdau(k), xu[:,k] and Pu[:,:,k] give the intensity, state estimate and 
    covariance for the k-th mixture component of the unknown target Poisson
    Point Process (PPP)
    model is structure describing target birth and transition models
    Output:
    Predicted components in same format as input
    """
    # Implement prediction algorithm

    # Predict existing tracks
    pred_mb = copy.deepcopy(self.mb)
    for i, bern in enumerate(self.mb):
      pred_mb[i] = bern.predict(state_estimator=state_estimator, ps=Ps, dt=dt)

    # Predict existing PPP intensity
    pred_poisson = self.poisson.predict(
        state_estimator=state_estimator, ps=Ps, dt=dt)

    # Not shown in paper--truncate low weight components
    pred_poisson = pred_poisson.prune(threshold=self.w_min)

    return pred_mb, pred_poisson

  def update(self, z: List[np.ndarray], Pd: float, state_estimator: KalmanFilter, lambda_fa: float):
    n = len(self.mb)
    nu = len(self.poisson)
    m = len(z)

    # Update existing tracks
    wupd = np.zeros((n, m + 1))
    mb_hypos = np.empty((n, m + 1), dtype=object)
    in_gate_mb = np.zeros((n, m), dtype=bool)
    for i, bern in enumerate(self.mb):
      # Create missed detection hypothesis
      wupd[i, 0] = 1 - bern.r + bern.r * (1 - Pd)
      mb_hypos[i, 0] = bern.update(measurement=None, pd=Pd)

      # Create hypotheses with measurement updates

      valid_meas, valid_inds = state_estimator.gate(
          measurements=z, predicted_state=bern.state, pg=self.pg)
      in_gate_mb[i, valid_inds] = True
      l = np.zeros(len(z))
      if len(valid_inds) > 0:
        l[valid_inds] = state_estimator.likelihood(
            measurement=valid_meas, predicted_state=bern.state)
      for j in valid_inds:
        wupd[i, j + 1] = bern.r * Pd * l[j]
        mb_hypos[i, j+1] = bern.update(
            pd=Pd, measurement=z[j], state_estimator=state_estimator)

    # TODO: Do gating
    wnew = np.zeros(m)
    new_berns = []
    for j in range(m):
      bern, wnew[j] = self.poisson.update(
          measurement=z[j],
          in_gate=np.ones(nu, dtype=bool),
          pd=Pd,
          state_estimator=state_estimator,
          clutter_intensity=lambda_fa)
      new_berns.append(bern)

    poisson_upd = copy.deepcopy(self.poisson)
    # Update (i.e., thin) intensity of unknown targets
    poisson_upd.weights = poisson_upd.weights * (1 - Pd)

    # Not shown in paper--truncate low weight components
    poisson_upd = poisson_upd.prune(threshold=self.w_min)

    if wupd.size == 0:
      pupd = np.empty_like(wupd)
      pnew = np.ones_like(wnew)
    else:
      pupd, pnew = self.spa(wupd=wupd, wnew=wnew)

    mb_upd = self.tomb(pupd=pupd, mb_hypos=mb_hypos, pnew=pnew,
                       new_berns=new_berns, in_gate_mb=in_gate_mb)

    return mb_upd, poisson_upd

  def tomb(self, pupd: np.ndarray, mb_hypos: np.ndarray, pnew: np.ndarray, new_berns: List[Bernoulli], in_gate_mb: np.ndarray):
    
    # Add false alarm hypotheses as valid
    valid_hypos = np.concatenate(
      (np.ones((len(self.mb), 1), dtype=bool), in_gate_mb), axis=1)

    # Form continuing tracks
    tomb_mb = []
    for i in range(len(self.mb)):
      valid = valid_hypos[i]
      rupd = np.array([bern.r for bern in mb_hypos[i, valid]])
      xupd = [bern.state.mean for bern in mb_hypos[i, valid]]
      Pupd = [bern.state.covar for bern in mb_hypos[i, valid]]

      pr = pupd[i, valid] * rupd
      r = np.sum(pr)
      pr = pr / r
      x, P = mix_gaussians(means=xupd, covars=Pupd, weights=pr)

      bern_upd = Bernoulli(r=r, state=GaussianState(mean=x, covar=P))
      tomb_mb.append(bern_upd)

    # Form new tracks (already single hypothesis)
    for j in range(len(new_berns)):
      r = pnew[j] * new_berns[j].r
      new_bern = Bernoulli(r=r, state=new_berns[j].state)
      tomb_mb.append(new_bern)

    # Truncate tracks with low probability of existence (not shown in algorithm)
    tomb_mb = [bern for bern in tomb_mb if bern.r >= self.r_min]

    return tomb_mb

  @staticmethod
  def spa(wupd: np.ndarray, wnew: np.ndarray, eps: float = 1e-4
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

    n, mp1 = wupd.shape
    m = mp1 - 1

    mu_ba = np.ones((n, m))
    mu_ba_old = np.zeros((n, m))
    mu_ab = np.zeros((n, m))

    while np.max(np.abs(mu_ba - mu_ba_old)) > eps:
      mu_ba_old = mu_ba

      w_muba = wupd[:, 1:] * mu_ba
      mu_ab = wupd[:, 1:] / (wupd[:, 0][:, np.newaxis] +
                             np.sum(w_muba, axis=1, keepdims=True) - w_muba)
      mu_ba = 1 / (wnew + np.sum(mu_ab, axis=0, keepdims=True) - mu_ab)

    # Compute marginal association probabilities
    mu_ba = np.concatenate((np.ones((n, 1)), mu_ba), axis=1)
    p_upd = wupd * mu_ba / (np.sum(wupd * mu_ba, axis=1, keepdims=True))

    p_new = wnew / (wnew + np.sum(mu_ab, axis=0))

    return p_upd, p_new
