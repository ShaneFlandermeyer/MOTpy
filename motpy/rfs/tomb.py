import copy
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

from motpy.distributions.gaussian import Gaussian, merge_gaussians
from motpy.distributions import Distribution
from motpy.estimators import StateEstimator
from motpy.rfs.bernoulli import MultiBernoulli
from motpy.rfs.poisson import Poisson


class TOMBP:
  """
  Track-oriented multi-Bernoulli/Poisson (TOMB/P) filter
  """

  def __init__(self,
               birth_state: Distribution,
               undetected_state: Optional[Distribution] = None,
               pg: Optional[float] = None,
               poisson_pd_threshold: Optional[float] = None,
               ):
    """
    Parameters
    ----------
    birth_distribution : Distribution
        The birth distribution for the Poisson component
    undetected_distribution: Optional[Distribution]
        The initial distribution for the Poisson component
    pg : float, optional
        Gate probability for measurement association
    poisson_pd_gate_threshold : float, optional
        Detection probability threshold which determines if a Poisson component is detectable. This gate is applied before standard gating to reduce the number of matrix inverses required. If none, all Poisson components are considered detectable, by default None
    """""
    self.poisson = Poisson(
        birth_distribution=birth_state,
        state=undetected_state,
    )
    self.mb = MultiBernoulli()
    self.metadata = dict(
        mb=[],
        poisson=[dict() for _ in range(self.poisson.size)],
    )

    self.pg = pg if pg is not None else 1.0
    if poisson_pd_threshold is None:
      self.poisson_pd_threshold = 0
    else:
      self.poisson_pd_threshold = poisson_pd_threshold

  def predict(self,
              state_estimator: StateEstimator,
              dt: float,
              ps_model: float,
              **kwargs
              ) -> Tuple[MultiBernoulli, Poisson]:
    """
    Propagate the multi-object state forward in time.

    Parameters
    ----------
    state_estimator : KalmanFilter
        The Kalman filter used for state estimation.
    dt : float
        The time step size.
    ps_model: float
        Handle to a modeltion which takes the state as input and returns the survival probability.

    Returns
    -------
    Tuple[MultiBernoulli, Poisson]
      - The MB distribution after prediction
      - The poisson distribution after prediction
    """
    meta = self.metadata.copy()

    # Predict MB
    if self.mb.size > 0:
      ps_mb = ps_model(self.mb.state)
      predicted_mb = self.mb.predict(
          state_estimator=state_estimator, ps=ps_mb, dt=dt, **kwargs
      )
    else:
      predicted_mb = self.mb

    # Predict Poisson
    ps_poisson = ps_model(self.poisson.state)
    predicted_poisson = self.poisson.predict(
        state_estimator=state_estimator,
        ps=ps_poisson,
        dt=dt,
        **kwargs
    )

    # Update metadata
    meta['poisson'] = meta['poisson'] + \
        [dict() for _ in range(predicted_poisson.size)]

    return predicted_mb, predicted_poisson, meta

  def update(self,
             measurements: np.ndarray,
             state_estimator: StateEstimator,
             pd_model: Callable,
             lambda_fa: float,
             **kwargs
             ) -> Tuple[MultiBernoulli, Poisson]:
    """
    Updates the state of the multi-object system based on measurements.

    Parameters
    ----------
    measurements : np.ndarray
        Array of measurement vectors
    state_estimator : KalmanFilter
        The Kalman filter used for state estimation.
    pd_model : float
        Detection probability modeltion handle. This modeltion takes the state as input and returns the detection probability.
    lambda_fa : float
        The false alarm density per unit volume.

    Returns
    -------
    Tuple[MultiBernoulli, Poisson]
      - Updated MB distribution
      - Updated Poisson distribution
    """
    meta = self.metadata.copy()

    ########################################################
    # Poisson update
    ########################################################
    pd_poisson = pd_model(self.poisson.state)
    if isinstance(pd_poisson, float):
      pd_poisson = np.full(self.poisson.size, pd_poisson)

    new_berns, w_new = self.bernoulli_birth(
        state_estimator=state_estimator,
        measurements=measurements,
        pd_poisson=pd_poisson,
        lambda_fa=lambda_fa,
        **kwargs
    )

    poisson_post = Poisson(
        birth_distribution=self.poisson.birth_distribution,
        state=copy.deepcopy(self.poisson.state)
    )
    poisson_post.state.weight *= 1 - pd_poisson

    # Update metadata
    meta['pd_poisson'] = pd_poisson
    ########################################################
    # MB Update
    ########################################################
    mb_hypotheses, hypothesis_mask, w_updated = self.make_mb_hypotheses(
        state_estimator=state_estimator,
        measurements=measurements,
        pd_model=pd_model,
        **kwargs
    )

    ########################################################
    # Data association and track updates
    ########################################################
    n = self.mb.size
    m = len(measurements) if measurements is not None else 0
    if n == 0:
      p_updated = np.zeros_like(w_updated)
      p_new = np.ones_like(w_new)
    elif m == 0:
      p_updated = w_updated
      p_new = np.zeros_like(w_new)
    else:
      p_updated, p_new = self.spa(
          w_updated=w_updated, w_new=w_new, max_iter=1000
      )

    mb_post, meta['mb'] = self.tomb(
        p_updated=p_updated,
        p_new=p_new,
        mb_hypotheses=mb_hypotheses,
        hypothesis_mask=hypothesis_mask,
        new_berns=new_berns,
        mb_meta=meta['mb']
    )

    return mb_post, poisson_post, meta

  def bernoulli_birth(self,
                      state_estimator: StateEstimator,
                      measurements: np.ndarray,
                      pd_poisson: np.ndarray,
                      lambda_fa: float,
                      **kwargs
                      ) -> Tuple[MultiBernoulli, np.ndarray]:
    """
    Create new Bernoulli components from the Poisson distribution based on measurements

    NOTE: Makes Gaussian state assumption for Poisson components

    Parameters
    ----------
    state_estimator : KalmanFilter
      The Kalman filter used for state estimation.
    measurements : np.ndarray
      Array of measurement vectors
    pd_poisson : np.ndarray
      Detection probabilities for the Poisson components
    lambda_fa : float
      False alarm density per unit volume

    Returns
    -------
    Tuple[MultiBernoulli, np.ndarray]
      - The new Bernoulli components
      - The weights of the new Bernoulli components
    """
    m = len(measurements)
    n_u = self.poisson.size
    w_new = np.zeros(m)

    state_dim = self.poisson.state.state_dim
    r = np.zeros(m)
    means = np.zeros((m, state_dim))
    covars = np.zeros((m, state_dim, state_dim))
    if m > 0:
      in_gate = np.zeros((n_u, m), dtype=bool)
      detectable = pd_poisson > self.poisson_pd_threshold
      if self.pg is None or self.pg == 1:
        in_gate[detectable] = True
      else:
        in_gate[detectable] = state_estimator.gate(
            measurements=measurements,
            state=self.poisson.state[detectable],
            pg=self.pg,
            **kwargs
        )

      gated_measurements = np.argwhere(np.any(in_gate, axis=0)).ravel()
      gated_poisson = np.argwhere(np.any(in_gate, axis=1)).ravel()
      gated_inds = np.ix_(gated_poisson, gated_measurements)

      # Compute likelihoods for all valid Poisson-measurement pairs
      l_poisson = np.zeros((n_u, m))
      l_poisson[gated_inds] = state_estimator.likelihood(
          measurement=measurements[gated_measurements],
          state=self.poisson.state[gated_poisson],
          **kwargs
      )

      # Create a new track hypothesis for each measurement
      for i_m in range(m):
        gated = in_gate[:, i_m]
        n_valid = np.count_nonzero(gated)
        if n_valid == 0:
          continue

        mixture = state_estimator.update(
            state=self.poisson.state[gated],
            measurement=measurements[i_m],
            **kwargs
        )
        mixture.weight *= l_poisson[gated, i_m] * pd_poisson[gated]

        sum_w_mixture = np.sum(mixture.weight)
        w_new[i_m] = sum_w_mixture + lambda_fa
        r[i_m] = sum_w_mixture / (w_new[i_m] + 1e-15)

        # Mixture reduction
        if n_valid > 1:
          _, means[i_m], covars[i_m] = merge_gaussians(
              means=mixture.mean,
              covars=mixture.covar,
              weights=mixture.weight,
          )
        else:
          means[i_m] = mixture.mean
          covars[i_m] = mixture.covar

      # Create Bernoulli components for each measurement
      new_berns = MultiBernoulli(
          r=r,
          state=Gaussian(mean=means, covar=covars, weight=None)
      )
    else:
      new_berns = MultiBernoulli()

    return new_berns, w_new

  def make_mb_hypotheses(self,
                         state_estimator: StateEstimator,
                         measurements: np.ndarray,
                         pd_model: Callable,
                         **kwargs,
                         ) -> Tuple[List[MultiBernoulli], np.ndarray, np.ndarray]:
    """
    Create MB data association hypotheses for existing measurement-track pairs


    Parameters
    ----------
    state_estimator : KalmanFilter
        Kalman filter used for state estimation
    measurements : np.ndarray
        Array of measurement vectors
    pd_model : Callable
        Function handle to the detection probability modeltion

    Returns
    -------
    Tuple[List[MultiBernoulli], np.ndarray, np.ndarray]
        - List of MB hypotheses
        - Mask indicating which hypotheses are valid
        - Data association weights for each hypothesis
    """
    n = self.mb.size if self.mb is not None else 0
    m = len(measurements) if measurements is not None else 0

    # Create track-oriented hypotheses (one per existing track object)
    mask = np.zeros((n, m+1), dtype=bool)
    w_updated = np.zeros((n, m+1))
    if n == 0:
      hypos = []
    else:
      state_post = self.mb.state.empty(
          shape=(n, m+1), state_dim=self.mb.state.state_dim
      )
      r_post = np.zeros((n, m+1))
      # Each object has a missed detection hypothesis
      mask[:, 0] = True
      pd = pd_model(self.mb.state)
      w_updated[:, 0] = (1 - self.mb.r) + self.mb.r * (1 - pd)
      state_post[:, 0] = self.mb.state
      r_post[:, 0] = self.mb.r * (1 - pd) / (w_updated[:, 0] + 1e-15)

      # ...and a hypothesis for each measurement
      if m > 0:
        if self.pg is None or self.pg == 1:
          in_gate = np.ones((n, m), dtype=bool)
        else:
          in_gate = state_estimator.gate(
              measurements=measurements,
              state=self.mb.state,
              pg=self.pg,
              **kwargs
          )
        gated_measurements = np.argwhere(np.any(in_gate, axis=0)).ravel()
        # gated_mb = np.argwhere(np.any(in_gate, axis=1)).ravel()
        # gated_inds = np.ix_(gated_mb, gated_measurements)
        mask[:, 1:] = in_gate

        l_mb = np.zeros((n, m))
        for im in gated_measurements:
          mb_mask = in_gate[:, im]
          l_mb[mb_mask, im] = state_estimator.likelihood(
              measurement=measurements[im],
              state=self.mb.state[mb_mask],
              **kwargs
          ).squeeze(-1)

          state_post[mb_mask, 1+im] = state_estimator.update(
              state=self.mb.state[mb_mask],
              measurement=measurements[im],
              **kwargs
          )
          w_updated[mb_mask, 1+im] = self.mb.r[mb_mask] * \
              pd_model(state_post[mb_mask, im+1]) * l_mb[mb_mask, im]
          r_post[mb_mask, 1+im] = 1

      hypos = [
          MultiBernoulli(state=state_post[i, mask[i]], r=r_post[i, mask[i]])
          for i in range(n)
      ]
    return hypos, mask, w_updated

  def tomb(self,
           p_updated: np.ndarray,
           p_new: np.ndarray,
           mb_hypotheses: List[MultiBernoulli],
           hypothesis_mask: np.ndarray,
           new_berns: MultiBernoulli,
           mb_meta: List[Dict[str, Any]]
           ) -> MultiBernoulli:
    """
    Add new Bernoulli components to the filter and marginalize existing components across measurement hypotheses

    NOTE: Makes Gaussian state assumption in marginalization step

    Parameters
    ----------
    p_updated : np.ndarray
        Association probabilities for existing Bernoulli components
    p_new : np.ndarray
        Association probabilities for new Bernoulli components
    mb_hypos : List[MultiBernoulli]
        Measurement-oriented MB hypotheses. This is a list of MultiBernoulli objects, where each MB is a set of possible track associations for a single measurement.
    mb_hypo_mask : np.ndarray
        Mask indicating which hypotheses in mb_hypos are valid.
    new_berns : MultiBernoulli
        New Bernoulli components created from the Poisson distribution

    Returns
    -------
    MultiBernoulli
        The updated MB distribution
    """
    mb = MultiBernoulli()
    meta = mb_meta.copy()

    # Marginalize over measurements for existing tracks
    for i in range(len(mb_hypotheses)):
      mask = hypothesis_mask[i]
      r = mb_hypotheses[i].r
      x = mb_hypotheses[i].state.mean
      P = mb_hypotheses[i].state.covar
      pr = p_updated[i, mask] * r
      if pr.size == 1:
        updated = False
        r = pr
      else:
        updated = True
        r, x, P = merge_gaussians(
            means=x,
            covars=P,
            weights=pr,
        )
        x, P = x[None, :], P[None, ...]
      mb = mb.append(r=r, state=Gaussian(mean=x, covar=P, weight=None))
      meta[i] = meta[i].copy()
      meta[i].update(new=False, updated=updated, in_gate=mask[1:])

    # Form new tracks
    n_new = new_berns.size
    if n_new > 0:
      mb = mb.append(r=p_new * new_berns.r, state=new_berns.state)
      meta.extend([dict(new=True) for i in range(n_new)])

    return mb, meta

  @staticmethod
  def spa(w_updated: np.ndarray,
          w_new: np.ndarray,
          eps: float = 1e-4,
          max_iter: Optional[int] = None
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
    if max_iter is None:
      max_iter = np.inf

    n, mp1 = w_updated.shape
    m = mp1 - 1

    mu_ba = np.ones((n, m))
    mu_ba_old = np.zeros((n, m))
    mu_ab = np.zeros((n, m))

    i = 0
    while True:
      mu_ba_old = mu_ba

      w_muba = w_updated[:, 1:] * mu_ba
      mu_ab = w_updated[:, 1:] / (w_updated[:, 0][:, np.newaxis] +
                                  np.sum(w_muba, axis=1, keepdims=True) - w_muba + 1e-15)
      mu_ba = 1 / (w_new + np.sum(mu_ab, axis=0,
                                  keepdims=True) - mu_ab + 1e-15)
      i += 1

      if np.max(np.abs(mu_ba - mu_ba_old)) < eps or i == max_iter:
        break
    # Compute marginal association probabilities
    mu_ba = np.concatenate((np.ones((n, 1)), mu_ba), axis=1)
    p_updated = w_updated * mu_ba / \
        (np.sum(w_updated * mu_ba, axis=1, keepdims=True) + 1e-15)

    p_new = w_new / (w_new + np.sum(mu_ab, axis=0) + 1e-15)

    return p_updated, p_new
