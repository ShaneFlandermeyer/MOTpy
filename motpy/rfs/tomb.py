import copy
from typing import Callable, Dict, List, Tuple

import jax
import numpy as np

from motpy.distributions.gaussian import GaussianState, match_moments
from motpy.kalman import KalmanFilter
from motpy.rfs.bernoulli import MultiBernoulli
from motpy.rfs.poisson import Poisson


class TOMBP:
  """
  Track-oriented multi-Bernoulli/Poisson (TOMB/P) filter
  """

  def __init__(self,
               birth_distribution: GaussianState,
               undetected_distribution: GaussianState = None,
               pg: float = None,
               w_min: float = None,
               r_min: float = None,
               merge_poisson: bool = False,
               ):
    """

    Parameters
    ----------
    birth_distribution : GaussianMixture
        The birth distribution for the Poisson point process (PPP).
    pg : float, optional
        Gate probability. If None, gating is not performed, by default None
    w_min : float, optional
        Weight threshold for PPP pruning. If none, pruning is not performed, by default None
    r_min : float, optional
        Existence probability threshold for MB pruning. If none, pruning is not performed, by default None
    merge_threshold : bool, optional
        If True, similar PPP components are merged.
    """
    self.poisson = Poisson(
        birth_distribution=birth_distribution, init_distribution=undetected_distribution)
    self.mb = MultiBernoulli()
    self.metadata = {
        'mb': [],
        'ppp': [],
    }
    self.id_counter = 0

    self.pg = pg
    self.r_min = r_min
    self.w_min = w_min
    self.merge_poisson = merge_poisson

  def predict(self,
              state_estimator: KalmanFilter,
              dt: float,
              ps_func: float) -> Tuple[MultiBernoulli, Poisson]:
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
    meta = self.metadata.copy()

    # Predict existing tracks
    if len(self.mb) > 0:
      ps_mb = ps_func(self.mb.state)
      pred_mb, filter_state = self.mb.predict(
          state_estimator=state_estimator, ps=ps_mb, dt=dt)
      meta['filter_state'] = filter_state
    else:
      pred_mb = MultiBernoulli()

    # Predict existing PPP intensity
    ps_poisson = ps_func(self.poisson.distribution)
    pred_poisson = self.poisson.predict(
        state_estimator=state_estimator, ps=ps_poisson, dt=dt)

    # Not shown in paper--truncate low weight components
    if self.w_min is not None:
      pred_poisson = pred_poisson.prune(threshold=self.w_min)
    if self.merge_poisson:
      pred_poisson = pred_poisson.merge()
    return pred_mb, pred_poisson

  def update(self,
             measurements: np.ndarray,
             state_estimator: KalmanFilter,
             pd_func: Callable,
             lambda_fa: float) -> Tuple[MultiBernoulli, Poisson]:
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
    m = len(measurements) if measurements is not None else 0

    # Update existing tracks
    wupd = np.zeros((n, m + 1))
    l_mb = np.zeros((n, m))

    # We create one MB hypothesis for each measurement, plus one missed detection hypothesis
    mb_hypos = [MultiBernoulli() for _ in range(n)]

    in_gate_mb = np.zeros((n, m), dtype=bool)

    if n > 0:

      # Create missed detection hypothesis
      pd = pd_func(self.mb.state)
      wupd[:, 0] = (1 - self.mb.r) + self.mb.r * (1 - pd)
      r_post = self.mb.r * (1 - pd) / (wupd[:, 0] + 1e-15)
      state_post = self.mb.state
      for i in range(n):
        mb_hypos[i].append(r=r_post[i], state=state_post[i])

      # Gate MB components and compute likelihoods for state-measurement pairs
      if m > 0:
        in_gate_mb = state_estimator.gate(
            measurements=measurements, predicted_state=self.mb.state, pg=self.pg)

        l_mb = np.zeros((n, m))
        used_meas_mb = np.argwhere(np.any(in_gate_mb, axis=0)).ravel()
        used_mb = np.argwhere(np.any(in_gate_mb, axis=1)).ravel()
        l_mb[np.ix_(used_mb, used_meas_mb)] = state_estimator.likelihood(
            measurement=measurements[used_meas_mb],
            predicted_state=self.mb.state[used_mb])

        # Create hypotheses for each state-measurement pair
        for j in range(m):
          valid = in_gate_mb[:, j]
          valid_inds = np.nonzero(valid)[0]
          n_valid = np.count_nonzero(valid)
          if np.any(valid):
            r_post = np.ones(n_valid)
            state_post, _ = state_estimator.update(
                predicted_state=self.mb.state[valid],
                measurement=measurements[j])
            wupd[valid, j + 1] = self.mb[valid].r * \
                pd_func(state_post) * l_mb[valid, j]
            for i in range(n_valid):
              mb_hypos[valid_inds[i]].append(r=r_post[i], state=state_post[i])

    # Create a new track for each measurement by updating PPP with measurement
    w_new = np.zeros(m)
    new_berns = MultiBernoulli()

    pd_ppp = pd_func(self.poisson.distribution)
    if isinstance(pd_ppp, float):
      pd_ppp = np.full(nu, pd_ppp)

    if m > 0:
      # We only care about PPP components that can be detected here. This allows us to save some matrix inverses
      detectable_ppp = copy.copy(self.poisson)
      detectable_ppp.distribution = detectable_ppp.distribution[pd_ppp > 0]

      # Gate PPP components
      in_gate_poisson = state_estimator.gate(
          measurements=measurements,
          predicted_state=detectable_ppp.distribution,
          pg=self.pg)

      # Compute likelihoods for PPP components with at least one measurement in the gate and measurements in at least one gate
      l_ppp = np.zeros((len(detectable_ppp), m))
      used_meas_ppp = np.argwhere(np.any(in_gate_poisson, axis=0)).ravel()
      used_ppp = np.argwhere(np.any(in_gate_poisson, axis=1)).ravel()
      l_ppp[np.ix_(used_ppp, used_meas_ppp)] = state_estimator.likelihood(
          measurement=measurements[used_meas_ppp],
          predicted_state=detectable_ppp.distribution[used_ppp])

      for j in range(m):
        bern, w_new[j] = detectable_ppp.update(
            measurement=measurements[j],
            pd=pd_ppp[pd_ppp > 0],
            likelihoods=l_ppp[:, j],
            in_gate=in_gate_poisson[:, j],
            state_estimator=state_estimator,
            clutter_intensity=lambda_fa)
        if w_new[j] > 0:
          new_berns.append(r=bern.r, state=bern.state)

    # Update (i.e., thin) intensity of unknown targets
    poisson_post = copy.copy(self.poisson)
    poisson_post.distribution.weight *= 1 - pd_ppp

    if n == 0:
      p_upd = np.zeros_like(wupd)
      p_new = np.ones_like(w_new)
    elif m == 0:
      p_upd = wupd
      p_new = np.zeros_like(w_new)
    else:
      p_upd, p_new = self.spa(w_upd=wupd, w_new=w_new)

    mb_post, meta = self.tomb(p_upd=p_upd,
                              mb_hypos=mb_hypos,
                              p_new=p_new[w_new > 0],
                              new_berns=new_berns,
                              in_gate_mb=in_gate_mb)

    return mb_post, poisson_post, meta

  def tomb(self,
           p_upd: np.ndarray,
           mb_hypos: List[MultiBernoulli],
           p_new: np.ndarray,
           new_berns: MultiBernoulli,
           in_gate_mb: np.ndarray) -> MultiBernoulli:
    meta = copy.deepcopy(self.metadata)
    # Add false alarm hypothesis as valid
    valid_hypos = np.concatenate(
        (np.ones((len(self.mb), 1), dtype=bool), in_gate_mb), axis=1)

    # Form continuing tracks
    mb = MultiBernoulli()
    for i in range(len(self.mb)):
      valid = valid_hypos[i]
      n_valid = np.count_nonzero(valid)
      if n_valid == 0:
        continue
      rs = mb_hypos[i].r
      xs = mb_hypos[i].state.mean
      Ps = mb_hypos[i].state.covar
      pr = p_upd[i, valid] * rs
      r = np.sum(pr)
      if n_valid == 1:
        x, P = xs, Ps
      else:
        x, P = match_moments(means=xs, covars=Ps, weights=pr)

      mb.append(r=r, state=GaussianState(mean=x, covar=P))

    # Form new tracks (already single hypothesis)
    if len(new_berns) > 0:
      mb.append(r=p_new*new_berns.r, state=new_berns.state)
      new_ids = self._make_id(n=len(new_berns))
      for i in range(len(new_berns)):
        meta['mb'].append({'id': new_ids[i]})

    # Truncate tracks with low probability of existence (not shown in algorithm)
    if len(mb) > 0 and self.r_min is not None:
      meta['mb'] = [meta['mb'][i]
                    for i in range(len(mb)) if mb.r[i] > self.r_min]
      mb = mb[mb.r > self.r_min]

    return mb, meta

  @staticmethod
  def spa(w_upd: np.ndarray, w_new: np.ndarray,
          eps: float = 1e-4, max_iter: int = 10
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
    mu_ba_old = np.zeros((n, m))
    mu_ab = np.zeros((n, m))

    i = 0
    while True:
      mu_ba_old = mu_ba

      w_muba = w_upd[:, 1:] * mu_ba
      mu_ab = w_upd[:, 1:] / (w_upd[:, 0][:, np.newaxis] +
                              np.sum(w_muba, axis=1, keepdims=True) - w_muba + 1e-15)
      mu_ba = 1 / (w_new + np.sum(mu_ab, axis=0,
                   keepdims=True) - mu_ab + 1e-15)
      i += 1

      if np.max(np.abs(mu_ba - mu_ba_old)) < eps or i == max_iter:
        break

    # Compute marginal association probabilities
    mu_ba = np.concatenate((np.ones((n, 1)), mu_ba), axis=1)
    p_upd = w_upd * mu_ba / \
        (np.sum(w_upd * mu_ba, axis=1, keepdims=True) + 1e-15)

    p_new = w_new / (w_new + np.sum(mu_ab, axis=0) + 1e-15)

    return p_upd, p_new

  def _make_id(self, n: int) -> np.ndarray:
    ids = np.arange(self.id_counter, self.id_counter + n)
    self.id_counter += n
    return ids
