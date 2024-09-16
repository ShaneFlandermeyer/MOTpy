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
    self.mb = None
    self.metadata = {
        'mb': [],
        'ppp': [],
    }

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
    if self.mb is not None and self.mb.size > 0:
      ps_mb = ps_func(self.mb.state)
      predicted_mb, filter_state = self.mb.predict(
          state_estimator=state_estimator, ps=ps_mb, dt=dt)
      meta['filter_state'] = filter_state
    else:
      predicted_mb = self.mb

    # Predict existing PPP intensity
    ps_poisson = ps_func(self.poisson.distribution)
    predicted_poisson = self.poisson.predict(
        state_estimator=state_estimator, ps=ps_poisson, dt=dt)

    # Not shown in paper--truncate low weight components
    if self.w_min is not None:
      predicted_poisson = predicted_poisson.prune(threshold=self.w_min)
    if self.merge_poisson:
      predicted_poisson = predicted_poisson.merge()
    return predicted_mb, predicted_poisson

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

    ########################################################
    # Poisson update
    ########################################################
    pd_poisson = pd_func(self.poisson.distribution)
    if isinstance(pd_poisson, float):
      pd_poisson = np.full(self.poisson.size, pd_poisson)

    new_berns, w_new = self.bernoulli_birth(
        state_estimator=state_estimator,
        measurements=measurements,
        pd_poisson=pd_poisson,
        lambda_fa=lambda_fa)

    poisson_post = copy.deepcopy(self.poisson)
    poisson_post.distribution.weight *= 1 - pd_poisson

    ########################################################
    # MB Update
    ########################################################
    mb_hypos, mb_hypo_mask, w_upd = self.make_mb_hypos(
        state_estimator=state_estimator,
        measurements=measurements,
        pd_func=pd_func)

    n = self.mb.size if self.mb is not None else 0
    m = len(measurements) if measurements is not None else 0
    if n == 0:
      p_upd = np.zeros_like(w_upd)
      p_new = np.ones_like(w_new)
    elif m == 0:
      p_upd = w_upd
      p_new = np.zeros_like(w_new)
    else:
      p_upd, p_new = self.spa(w_upd=w_upd, w_new=w_new)

    mb_post, meta = self.tomb(p_upd=p_upd,
                              p_new=p_new,
                              mb_hypos=mb_hypos,
                              mb_hypo_mask=mb_hypo_mask,
                              new_berns=new_berns)

    return mb_post, poisson_post, meta

  def tomb(self,
           p_upd: np.ndarray,
           p_new: np.ndarray,
           mb_hypos: List[MultiBernoulli],
           mb_hypo_mask: np.ndarray,
           new_berns: MultiBernoulli
           ) -> MultiBernoulli:
    meta = copy.deepcopy(self.metadata)
    n_mb, mp1 = p_upd.shape
    m = mp1 - 1

    # NOTE: Makes Gaussian assumption
    mb = MultiBernoulli()
    if len(mb_hypos) > 0:
      state_dim = mb_hypos[0].state.state_dim
      rs = np.zeros((n_mb, m+1))
      xs = np.zeros((n_mb, m+1, state_dim))
      Ps = np.zeros((n_mb, m+1, state_dim, state_dim))
      # Transpose hypotheses to track-oriented format
      for im in range(m+1):
        if mb_hypos[im] is None:
          continue
        valid = mb_hypo_mask[:, im]
        rs[valid, im] = mb_hypos[im].r
        xs[valid, im] = mb_hypos[im].state.mean
        Ps[valid, im] = mb_hypos[im].state.covar

      # Marginalize over hypotheses
      for imb in range(n_mb):
        valid = mb_hypo_mask[imb]
        pr = p_upd[imb, valid] * rs[imb, valid]
        r = np.sum(pr)
        if np.count_nonzero(valid) == 1:
          x, P = xs[imb, valid], Ps[imb, valid]
        else:
          x, P = match_moments(
              means=xs[imb, valid], covars=Ps[imb, valid], weights=pr)
          x, P = x[None, ...], P[None, ...]
        mb = mb.append(r=np.array([r]), state=GaussianState(mean=x, covar=P))
        meta['mb'][imb].update(
            {'p_upd': p_upd[imb], 'p_new': 0, 'in_gate': valid})

    # Form new tracks
    n_new = new_berns.size
    if n_new > 0:
      mb = mb.append(r=p_new*new_berns.r, state=new_berns.state)
      meta['mb'].extend([{'p_new': p_new[i]} for i in range(n_new)])

    # Truncate tracks with low probability of existence (not shown in algorithm)
    if self.r_min is not None and mb.size > 0:
      valid = mb.r > self.r_min
      meta['mb'] = [meta['mb'][i] for i in range(mb.size) if valid[i]]
      mb = mb[valid]

    return mb, meta

  def bernoulli_birth(self,
                      state_estimator: KalmanFilter,
                      measurements: np.ndarray,
                      pd_poisson: np.ndarray,
                      lambda_fa: float,
                      ) -> Tuple[MultiBernoulli, np.ndarray]:
    m = len(measurements)
    n_u = self.poisson.size
    w_new = np.zeros(m)

    state_dim = self.poisson.distribution.state_dim
    rs = np.zeros(m)
    means = np.zeros((m, state_dim))
    covars = np.zeros((m, state_dim, state_dim))
    if m > 0:
      # Valid poisson-measurement pairs
      # Valid = in gate and detectable (pd > 0)
      detectable = pd_poisson > 0
      valid = np.zeros((n_u, m), dtype=bool)
      valid[detectable] = state_estimator.gate(
          measurements=measurements,
          state=self.poisson.distribution[detectable],
          pg=self.pg)

      # Compute likelihoods for all valid Poisson-measurement pairs
      l_poisson = np.zeros((n_u, m))
      valid_meas = np.argwhere(np.any(valid, axis=0)).ravel()
      valid_poisson = np.argwhere(np.any(valid, axis=1)).ravel()
      l_poisson[np.ix_(valid_poisson, valid_meas)] = state_estimator.likelihood(
          measurement=measurements[valid_meas],
          state=self.poisson.distribution[valid_poisson])

      # Create a new track hypothesis for each measurement
      for im in range(m):
        n_valid = np.count_nonzero(valid[:, im])
        if n_valid == 0:
          continue

        mixture, _ = state_estimator.update(
            state=self.poisson.distribution[valid[:, im]],
            measurement=measurements[im])
        mixture.weight *= l_poisson[detectable, im] * pd_poisson[detectable]

        sum_w_mixture = np.sum(mixture.weight)
        w_new[im] = sum_w_mixture + lambda_fa
        rs[im] = sum_w_mixture / (w_new[im] + 1e-15)

        # Reduce the mixture to a single Gaussian
        # NOTE: Makes Gaussian assumption for Poisson components
        if n_valid == 1:
          means[im], covars[im] = mixture.mean, mixture.covar
        else:
          means[im], covars[im] = match_moments(
              means=mixture.mean,
              covars=mixture.covar,
              weights=mixture.weight)

      # Create Bernoulli components for each measurement
      new_berns = MultiBernoulli(
          r=rs,
          state=GaussianState(mean=means, covar=covars, weight=None)
      )

    return new_berns, w_new

  def make_mb_hypos(self,
                    state_estimator: KalmanFilter,
                    measurements: np.ndarray,
                    pd_func: Callable
                    ) -> Tuple[List[MultiBernoulli], np.ndarray, np.ndarray]:
    n = self.mb.size if self.mb is not None else 0
    m = len(measurements) if measurements is not None else 0

    # One MB hypothesis per measurement (including missed detection event)
    hypos = []
    mask = np.ones((n, m+1), dtype=bool)
    w_upd = np.zeros((n, m+1))
    if n > 0:
      # Missed detection hypothesis
      pd = pd_func(self.mb.state)
      w_upd[:, 0] = (1 - self.mb.r) + self.mb.r * (1 - pd)
      r_post = self.mb.r * (1 - pd) / (w_upd[:, 0] + 1e-15)
      state_post = self.mb.state
      hypos.append(MultiBernoulli(r=r_post, state=state_post))

      # Gate MB components and compute likelihoods for state-measurement pairs
      if m > 0:
        in_gate = state_estimator.gate(
            measurements=measurements,
            state=self.mb.state,
            pg=self.pg)
        mask[:, 1:] = in_gate

        l_mb = np.zeros((n, m))
        valid_meas = np.argwhere(np.any(in_gate, axis=0)).ravel()
        valid_mb = np.argwhere(np.any(in_gate, axis=1)).ravel()
        l_mb[np.ix_(valid_mb, valid_meas)] = state_estimator.likelihood(
            measurement=measurements[valid_meas],
            state=self.mb.state[valid_mb])

        # Create hypotheses for each state-measurement pair
        for im in range(m):
          valid = in_gate[:, im]
          n_valid = np.count_nonzero(valid)
          if n_valid > 0:
            state_post, _ = state_estimator.update(
                state=self.mb.state[valid],
                measurement=measurements[im])
            r_post = np.ones(n_valid)
            hypos.append(MultiBernoulli(r=r_post, state=state_post))
            w_upd[valid, im+1] = self.mb[valid].r * \
                pd_func(state_post) * l_mb[valid, im]
          else:
            hypos.append(None)

    return hypos, mask, w_upd

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
