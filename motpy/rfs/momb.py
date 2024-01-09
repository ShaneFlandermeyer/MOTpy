import copy
from typing import Callable, List, Tuple

import numpy as np

from motpy.distributions.gaussian import GaussianState, mix_gaussians
from motpy.kalman import KalmanFilter
from motpy.rfs.bernoulli import Bernoulli
from motpy.rfs.poisson import Poisson


class MOMBP:
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
               poisson_merge_threshold: float = None,
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
    self.poisson_merge_threshold = poisson_merge_threshold
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

  @profile
  def update(self,
             measurements: List[np.ndarray],
             state_estimator: KalmanFilter,
             pd: Callable,
             lambda_fa: float):
    """
    Updates the state of the multi-object system based on the given measurements.

    Parameters
    ----------
    measurements : np.ndarray
        The array of measurements.
    pd : float
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
    m = len(measurements)

    # Update existing tracks
    wupd = np.zeros((n, m + 1))
    mb_hypos = np.zeros((n, m + 1), dtype=object)
    in_gate_mb = np.zeros((n, m), dtype=bool)
    for i, bern in enumerate(self.mb):
      # Create missed detection hypothesis
      wupd[i, 0] = 1 - bern.r + bern.r * (1 - pd(bern.state))
      mb_hypos[i, 0] = bern.update(measurement=None,
                                   pd=pd(bern.state),
                                   state_estimator=state_estimator)

      valid_meas, valid_inds = state_estimator.gate(
          measurements=measurements, predicted_state=bern.state, pg=self.pg)
      in_gate_mb[i, valid_inds] = True
      # Create hypotheses with measurement updates
      if len(valid_meas) > 0:
        l_mb = np.zeros(m)
        l_mb[valid_inds] = state_estimator.likelihood(
            measurement=valid_meas, predicted_state=bern.state)
      for j in valid_inds:
        mb_hypos[i, j + 1] = bern.update(
            pd=pd, measurement=measurements[j], state_estimator=state_estimator)
        wupd[i, j + 1] = bern.r * pd(mb_hypos[i, j+1].state) * l_mb[j]

    # Create a new track for each measurement by updating PPP with measurement
    wnew = np.zeros(m)
    new_berns = []

    # Gate PPP components and pre-compute likelihoods
    in_gate_poisson = np.zeros((nu, m), dtype=bool)
    l_ppp = np.zeros((nu, m))
    pd_ppp = np.zeros(nu)
    for k, state in enumerate(self.poisson.states):
      pd_ppp[k] = pd(state)
      if pd_ppp[k] == 0:
        continue
      valid_meas, valid_inds = state_estimator.gate(
          measurements=measurements, predicted_state=state, pg=self.pg)
      in_gate_poisson[k, valid_inds] = True
      if len(valid_meas) > 0:
        l_ppp[k, valid_inds] = state_estimator.likelihood(
            measurement=valid_meas, predicted_state=state)

    for j in range(m):
      bern, wnew[j] = self.poisson.update(
          measurement=measurements[j],
          pd=pd_ppp,
          likelihoods=l_ppp[:, j],
          in_gate=in_gate_poisson[:, j],
          state_estimator=state_estimator,
          clutter_intensity=lambda_fa)
      if wnew[j] > 0:
        new_berns.append(bern)

    # Update (i.e., thin) intensity of unknown targets
    poisson_upd = copy.copy(self.poisson)
    poisson_upd.weights *= 1 - pd_ppp
    # poisson_upd.states = self.poisson.states.copy()

    # Not shown in paper--truncate low weight components
    if self.w_min is not None:
      poisson_upd = poisson_upd.prune(threshold=self.w_min)
    if self.poisson_merge_threshold is not None:
      poisson_upd = poisson_upd.merge(threshold=self.poisson_merge_threshold)

    # TODO: This requires m > 0
    if wupd.size == 0:
      pupd = np.zeros_like(wupd)
      pnew = np.ones_like(wnew)
    else:
      pupd, pnew = self.spa(wupd=wupd, wnew=wnew)

    mb_upd = self.momb(pupd=pupd, mb_hypos=mb_hypos, pnew=pnew,
                       new_berns=new_berns, in_gate_mb=in_gate_mb)

    return mb_upd, poisson_upd

  def momb(self,
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

    n = len(self.mb)
    mp1 = pupd.shape[1]
    m = mp1 - 1

    momb_mb = []
    for i in range(n):
      r = mb_hypos[i, 0].r
      p = pupd[i, 0]
      new_bern = Bernoulli(r=r*p, state=mb_hypos[i, 0].state)
      momb_mb.append(new_bern)

    for j in range(len(new_berns)):
      valid = in_gate_mb[:, j]
      rupd = np.array([bern.r for bern in mb_hypos[valid, j+1]])
      xupd = [bern.state.mean for bern in mb_hypos[valid, j+1]]
      Pupd = [bern.state.covar for bern in mb_hypos[valid, j+1]]

      if n == 0:
        x, P = new_berns[j].state.mean, new_berns[j].state.covar
        r = pnew[j]*new_berns[j].r
      else:
        pr = np.append(pupd[valid, j+1]*rupd, pnew[j]*new_berns[j].r)
        r = np.sum(pr)
        xmix = xupd + [new_berns[j].state.mean]
        Pmix = Pupd + [new_berns[j].state.covar]

        x, P = mix_gaussians(means=np.array(xmix), 
                             covars=np.array(Pmix), weights=pr)

      new_bern = Bernoulli(r=r, state=GaussianState(mean=x, covar=P))
      momb_mb.append(new_bern)

    # Truncate tracks with low probability of existence (not shown in algorithm)
    momb_mb = [bern for bern in momb_mb if bern.r >= self.r_min]

    return momb_mb

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
                             np.sum(w_muba, axis=1, keepdims=True) - w_muba + 1e-15)
      mu_ba = 1 / (wnew + np.sum(mu_ab, axis=0, keepdims=True) - mu_ab + 1e-15)

    # Compute marginal association probabilities
    mu_ba = np.concatenate((np.ones((n, 1)), mu_ba), axis=1)
    p_upd = wupd * mu_ba / \
        (np.sum(wupd * mu_ba, axis=1, keepdims=True) + 1e-15)

    p_new = wnew / (wnew + np.sum(mu_ab, axis=0) + 1e-15)

    return p_upd, p_new
