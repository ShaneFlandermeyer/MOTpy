import copy
from typing import List, Tuple

import numpy as np

from motpy.distributions.gaussian import GaussianState, mix_gaussians
from motpy.kalman import KalmanFilter
from motpy.rfs.bernoulli import Bernoulli
from motpy.rfs.poisson import Poisson
from motpy.gate import EllipsoidalGate


class TOMBP:
  """
  Track-oriented Marginal MeMBer-Poisson Filter (TOMBP) for multi-object tracking.
  
  Attributes
  ----------
  poisson : Poisson
      The Poisson point process (PPP) representing the intensity of new objects.
  mb : list
      The list of Bernoulli components representing the multi-Bernoulli (MB).
  pg : float
      The gating size in terms of the probability of Gaussian error.
  w_min : float
      The minimum weight below which a component in the PPP is pruned.
  r_min : float
      The minimum probability of existence below which a Bernoulli component is pruned.
  r_estimate_threshold : float
      The threshold for the probability of existence above which an object is estimated to exist.
  """

  def __init__(self,
               birth_weights: np.ndarray,
               birth_states: List[np.ndarray],
               pg: float = None,
               w_min: float = None,
               r_min: float = None,
               r_estimate_threshold: float = None,
               ):
    """
    Initializes the Track-Oriented Multi-Bernoulli (TOMB) process.

    Parameters
    ----------
    birth_weights : np.ndarray
        The weights of the birth components in the Poisson point process (PPP) representing the intensity of new objects.
    birth_states : List[np.ndarray]
        The states of the birth components in the PPP representing the intensity of new objects.
    pg : float, optional
        The gating size in terms of the probability of Gaussian error, by default None.
    w_min : float, optional
        The minimum weight below which a component in the PPP is pruned, by default None.
    r_min : float, optional
        The minimum probability of existence below which a Bernoulli component is pruned, by default None.
    r_estimate_threshold : float, optional
        The threshold for the probability of existence above which an object is estimated to exist, by default None.
    """
    self.poisson = Poisson(birth_weights=birth_weights,
                           birth_states=birth_states)
    self.mb = []

    self.pg = pg
    self.w_min = w_min
    self.r_min = r_min
    self.r_estimate_threshold = r_estimate_threshold

  def predict(self,
              state_estimator: KalmanFilter,
              dt: float,
              ps: float):
    """
    Predicts the state of the multi-object system in the next time step.

    Parameters
    ----------
    state_estimator : KalmanFilter
        The Kalman filter used for state estimation.
    dt : float
        The time step size.
    ps : float
        The survival probability, i.e., the probability that each object will continue to exist.

    Returns
    -------
    Tuple[List[Bernoulli], Poisson]
        A tuple containing the list of predicted Bernoulli components representing the multi-Bernoulli mixture (MBM), 
        and the predicted Poisson point process (PPP) representing the intensity of new objects.
    """
    # Implement prediction algorithm

    # Predict existing tracks
    pred_mb = [bern.predict(state_estimator=state_estimator,
                            ps=ps, dt=dt) for bern in self.mb]

    # Predict existing PPP intensity
    pred_poisson = self.poisson.predict(
        state_estimator=state_estimator, ps=ps, dt=dt)

    # Not shown in paper--truncate low weight components
    pred_poisson = pred_poisson.prune(threshold=self.w_min)

    return pred_mb, pred_poisson

  def update(self, z, Pd, state_estimator, lambda_fa):
    """
    Updates the state of the multi-object system based on the given measurements.

    Parameters
    ----------
    z : np.ndarray
        The array of measurements.
    Pd : float
        The detection probability, i.e., the probability that each object will be detected.
    state_estimator : KalmanFilter
        The Kalman filter used for state estimation.
    lambda_fa : float
        The false alarm rate, representing the density of spurious measurements.

    Returns
    -------
    Tuple[List[Bernoulli], Poisson]
        A tuple containing the list of updated Bernoulli components representing the multi-Bernoulli mixture (MBM), 
        and the updated Poisson point process (PPP) representing the intensity of new objects.
    """
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
      mb_hypos[i, 0] = bern.update(measurement=None,
                                   pd=Pd,
                                   state_estimator=state_estimator)

      if self.pg == 1 or self.pg is None:
        valid_meas = z
        valid_inds = np.arange(m)
      else:
        valid_meas, valid_inds = state_estimator.gate(
            measurements=z, predicted_state=bern.state, pg=self.pg)
      in_gate_mb[i, valid_inds] = True

      # Create hypotheses with measurement updates
      if len(valid_inds) > 0:
        l_mb = np.zeros(m)
        l_mb[valid_inds] = state_estimator.likelihood(
            measurement=valid_meas, predicted_state=bern.state)
      for j in valid_inds:
        wupd[i, j + 1] = bern.r * Pd * l_mb[j]
        mb_hypos[i, j + 1] = bern.update(
            pd=Pd, measurement=z[j], state_estimator=state_estimator)
        if j == 0:
          # Add cached state estimation values to bernoulli state.
          new_meta = mb_hypos[i, j+1].state.metadata.copy()
          new_meta.update(bern.state.metadata)
          bern.state.metadata = new_meta

    # Create a new track for each measurement by updating PPP with measurement
    wnew = np.zeros(m)
    new_berns = []

    # Gate PPP components and pre-compute likelihoods
    in_gate_poisson = np.zeros((nu, m), dtype=bool)
    l_ppp = np.zeros((nu, m))
    for k, state in enumerate(self.poisson.states):
      if self.pg == 1 or self.pg is None:
        valid_meas = z
        valid_inds = np.arange(m)
      else:
        valid_meas, valid_inds = state_estimator.gate(
            measurements=z, predicted_state=state, pg=self.pg)
      in_gate_poisson[k, valid_inds] = True
      l_ppp[k, valid_inds] = state_estimator.likelihood(
          measurement=valid_meas, predicted_state=state)

    for j in range(m):
      bern, wnew[j] = self.poisson.update(
          measurement=z[j],
          pd=Pd,
          likelihoods=l_ppp[:, j],
          in_gate=np.ones(nu, dtype=bool),
          state_estimator=state_estimator,
          clutter_intensity=lambda_fa)
      if wnew[j] > 0:
        new_berns.append(bern)

    # Update (i.e., thin) intensity of unknown targets
    poisson_upd = copy.deepcopy(self.poisson)
    poisson_upd.weights *= 1 - Pd

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

  def tomb(self,
           pupd: np.ndarray,
           mb_hypos: np.ndarray,
           pnew: np.ndarray,
           new_berns: List[Bernoulli],
           in_gate_mb: np.ndarray) -> List[Bernoulli]:
    """
    Implements the Track-Oriented Multi-Bernoulli (TOMB) update step.

    Parameters
    ----------
    pupd : np.ndarray
        The marginal association probabilities for existing objects.
    mb_hypos : np.ndarray
        The Bernoulli components representing the hypotheses for existing objects.
    pnew : np.ndarray
        The marginal association probabilities for potential new objects.
    new_berns : List[Bernoulli]
        The Bernoulli components representing the hypotheses for potential new objects.
    in_gate_mb : np.ndarray
        A boolean array indicating which measurements are inside the gating region for each existing object.

    Returns
    -------
    List[Bernoulli]
        The list of updated Bernoulli components representing the multi-Bernoulli mixture (MBM) after the TOMB update step.
    """

    # Add false alarm hypothesis as valid
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

      new_bern = Bernoulli(r=r, state=GaussianState(mean=x, covar=P))
      tomb_mb.append(new_bern)

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