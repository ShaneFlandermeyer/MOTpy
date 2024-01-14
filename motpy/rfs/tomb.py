import copy
from typing import Callable, List, Tuple

import numpy as np

from motpy.distributions.gaussian import GaussianMixture, GaussianState, mix_gaussians
from motpy.kalman import KalmanFilter
from motpy.rfs.bernoulli import Bernoulli, MultiBernoulli
from motpy.rfs.poisson import Poisson


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
               birth_distribution: GaussianMixture,
               pg: float = None,
               w_min: float = None,
               r_min: float = None,
               merge_poisson: bool = False,
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
    self.poisson = Poisson(birth_distribution=birth_distribution)
    self.mb = MultiBernoulli()

    self.pg = pg
    self.w_min = w_min
    self.r_min = r_min
    self.merge_poisson = merge_poisson
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
    pred_mb = self.mb.predict(state_estimator=state_estimator, ps=ps, dt=dt)

    # Predict existing PPP intensity
    pred_poisson = self.poisson.predict(
        state_estimator=state_estimator, ps=ps, dt=dt)

    # Not shown in paper--truncate low weight components
    if self.w_min is not None:
      pred_poisson = pred_poisson.prune(threshold=self.w_min)
    if self.merge_poisson:
      assert self.w_min is None, "Poisson merging currently assumes there is no pruning"
      pred_poisson = pred_poisson.merge(threshold=None)
    return pred_mb, pred_poisson

  def update(self,
             measurements: np.ndarray,
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
    l_mb = np.zeros((n, m))

    # We create one MB hypothesis for each measurement, plus one missed detection hypothesis
    mb_hypos = [MultiBernoulli() for _ in range(n)]

    in_gate_mb = np.zeros((n, m), dtype=bool)
    # Create missed detection hypothesis
    if n > 0:
      wupd[:, 0] = 1 - self.mb.r + self.mb.r * (1 - pd(self.mb.state))
      r_post = self.mb.r * (1 - pd(self.mb.state)) / wupd[:, 0]
      state_post = self.mb.state
      for i in range(n):
        mb_hypos[i].append(r=r_post[i], state=state_post[i])

      # Gate MB components and compute likelihoods for state-measurement pairs
      if m > 0:
        in_gate_mb = state_estimator.gate(
            measurements=measurements, predicted_state=self.mb.state, pg=self.pg)

        l_mb = np.zeros((n, m))
        used_meas = np.argwhere(np.any(in_gate_mb, axis=0)).flatten()
        used_mb = np.argwhere(np.any(in_gate_mb, axis=1)).flatten()
        l_mb[np.ix_(used_mb, used_meas)] = state_estimator.likelihood(
            measurement=measurements[used_meas], 
            predicted_state=self.mb.state[used_mb])

        # Create hypotheses for each state-measurement pair
        for j in range(m):
          valid = in_gate_mb[:, j]
          valid_inds = np.nonzero(valid)[0]
          if np.any(valid):
            r_post = np.ones(np.count_nonzero(valid))
            state_post = state_estimator.update(
                measurement=measurements[j],
                predicted_state=self.mb[valid].state)
            wupd[valid, j + 1] = self.mb[valid].r * \
                pd(state_post) * l_mb[valid, j]
            for i in range(np.count_nonzero(valid)):
              mb_hypos[valid_inds[i]].append(r=r_post[i], state=state_post[i])

    # Create a new track for each measurement by updating PPP with measurement
    wnew = np.zeros(m)
    new_berns = MultiBernoulli()
    
    # TODO: Vectorize pd func
    pd_ppp = pd(self.poisson.distribution)
    if isinstance(pd_ppp, float):
      pd_ppp = np.full(nu, pd_ppp)
      
    if m > 0:
      # Gate PPP components
      in_gate_poisson = state_estimator.gate(
          measurements=measurements,
          predicted_state=self.poisson.distribution,
          pg=self.pg)

      # Compute likelihoods for PPP components with at least one measurement in the gate and measurements in at least one gate
      l_ppp = np.zeros((nu, m))
      used_meas = np.argwhere(np.any(in_gate_poisson, axis=0)).flatten()
      used_ppp = np.argwhere(np.any(in_gate_poisson, axis=1)).flatten()
      l_ppp[np.ix_(used_ppp, used_meas)] = state_estimator.likelihood(
          measurement=measurements[used_meas], 
          predicted_state=self.poisson.distribution[used_ppp])

      for j in range(m):
        bern, wnew[j] = self.poisson.update(
            measurement=measurements[j],
            pd=pd_ppp,
            likelihoods=l_ppp[:, j],
            in_gate=in_gate_poisson[:, j],
            state_estimator=state_estimator,
            clutter_intensity=lambda_fa)
        if wnew[j] > 0:
          new_berns.append(r=bern.r, state=bern.state)

    # Update (i.e., thin) intensity of unknown targets
    poisson_upd = copy.deepcopy(self.poisson)
    poisson_upd.distribution.weights *= 1 - pd_ppp

    if wupd.size == 0:
      pupd = np.zeros_like(wupd)
      pnew = np.ones_like(wnew)
    elif m == 0:
      pupd = wupd
      pnew = np.zeros_like(wnew)
    else:
      pupd, pnew = self.spa(wupd=wupd, wnew=wnew)

    mb_upd = self.tomb(pupd=pupd, mb_hypos=mb_hypos, pnew=pnew[wnew > 0],
                       new_berns=new_berns, in_gate_mb=in_gate_mb)

    return mb_upd, poisson_upd

  def tomb(self,
           pupd: np.ndarray,
           mb_hypos: List[MultiBernoulli],
           pnew: np.ndarray,
           new_berns: MultiBernoulli,
           in_gate_mb: np.ndarray) -> MultiBernoulli:
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
    tomb_mb = MultiBernoulli()
    for i in range(len(self.mb)):
      valid = valid_hypos[i]
      if not np.any(valid):
        continue
      rupd = mb_hypos[i].r
      xupd = mb_hypos[i].state.means
      Pupd = mb_hypos[i].state.covars
      pr = pupd[i, valid] * rupd
      r = np.sum(pr)
      x, P = mix_gaussians(means=xupd, covars=Pupd, weights=pr)

      tomb_mb.append(r=r,
                     state=GaussianMixture(means=x, covars=P, weights=None))

    # Form new tracks (already single hypothesis)
    if len(new_berns) > 0:
      tomb_mb.append(r=pnew*new_berns.r, state=new_berns.state)

    # Truncate tracks with low probability of existence (not shown in algorithm)
    if len(tomb_mb) > 0 and self.r_min is not None:
      tomb_mb = tomb_mb[tomb_mb.r > self.r_min]

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
                             np.sum(w_muba, axis=1, keepdims=True) - w_muba + 1e-15)
      mu_ba = 1 / (wnew + np.sum(mu_ab, axis=0, keepdims=True) - mu_ab + 1e-15)

    # Compute marginal association probabilities
    mu_ba = np.concatenate((np.ones((n, 1)), mu_ba), axis=1)
    p_upd = wupd * mu_ba / \
        (np.sum(wupd * mu_ba, axis=1, keepdims=True) + 1e-15)

    p_new = wnew / (wnew + np.sum(mu_ab, axis=0) + 1e-15)

    return p_upd, p_new
