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

  # TODO: Replace model with internal stuff
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

  def update(self, z, Pd, H, R, lambda_fa):
    lambdau = self.poisson.weights
    xu = np.array([state.mean for state in self.poisson.states]
                  ).swapaxes(0, -1)
    Pu = np.array([state.covar for state in self.poisson.states]
                  ).swapaxes(0, -1)
    r = np.array([bern.r for bern in self.mb])
    x = np.array([bern.state.mean for bern in self.mb]).swapaxes(0, -1)
    P = np.array([bern.state.covar for bern in self.mb]).swapaxes(0, -1)

    # Extract parameters from model
    lambdab_threshold = 1e-4

    # Interpret sizes from inputs
    n = len(r)
    stateDimensions, nu = xu.shape
    measDimensions, m = z.shape

    # Allocate memory for existing tracks
    wupd = np.zeros((n, m + 1))
    rupd = np.zeros((n, m + 1))
    xupd = np.zeros((stateDimensions, n, m + 1))
    Pupd = np.zeros((stateDimensions, stateDimensions, n, m + 1))

    # Allocate memory for new tracks
    wnew = np.zeros(m)
    rnew = np.zeros(m)
    xnew = np.zeros((stateDimensions, m))
    Pnew = np.zeros((stateDimensions, stateDimensions, m))

    # Allocate temporary working for new tracks
    Sk = np.zeros((measDimensions, measDimensions, nu))
    Kk = np.zeros((stateDimensions, measDimensions, nu))
    Pk = np.zeros((stateDimensions, stateDimensions, nu))
    ck = np.zeros(nu)
    sqrt_det2piSk = np.zeros(nu)
    yk = np.zeros((stateDimensions, nu))

    # Update existing tracks
    for i in range(n):
      # Create missed detection hypothesis
      wupd[i, 0] = 1 - r[i] + r[i] * (1 - Pd)
      rupd[i, 0] = r[i] * (1 - Pd) / wupd[i, 0]
      xupd[:, i, 0] = x[:, i]
      Pupd[:, :, i, 0] = P[:, :, i]

      # Create hypotheses with measurement updates
      S = H @ P[:, :, i] @ H.T + R
      sqrt_det2piS = np.sqrt(np.linalg.det(2 * np.pi * S))
      K = P[:, :, i] @ H.T @ np.linalg.inv(S)
      Pplus = P[:, :, i] - K @ H @ P[:, :, i]
      for j in range(m):
        v = z[:, j] - H @ x[:, i]
        wupd[i, j + 1] = r[i] * Pd * \
            np.exp(-0.5 * v.T @ np.linalg.inv(S) @ v) / sqrt_det2piS
        rupd[i, j + 1] = 1
        xupd[:, i, j + 1] = x[:, i] + K @ v
        Pupd[:, :, i, j + 1] = Pplus

    # Create a new track for each measurement by updating PPP with measurement
    for k in range(nu):
      Sk[:, :, k] = H @ Pu[:, :, k] @ H.T + R
      sqrt_det2piSk[k] = np.sqrt(np.linalg.det(2 * np.pi * Sk[:, :, k]))
      Kk[:, :, k] = Pu[:, :, k] @ H.T @ np.linalg.inv(Sk[:, :, k])
      Pk[:, :, k] = Pu[:, :, k] - Kk[:, :, k] @ H @ Pu[:, :, k]
    for j in range(m):
      for k in range(nu):
        v = z[:, j] - H @ xu[:, k]
        ck[k] = lambdau[k] * Pd * \
            np.exp(-0.5 * v.T @
                   np.linalg.inv(Sk[:, :, k]) @ v) / sqrt_det2piSk[k]
        yk[:, k] = xu[:, k] + Kk[:, :, k] @ v
      C = np.sum(ck)
      wnew[j] = C + lambda_fa
      rnew[j] = C / wnew[j]
      ck = ck / C
      xnew[:, j] = yk @ ck
      for k in range(nu):
        v = xnew[:, j] - yk[:, k]
        Pnew[:, :, j] = Pnew[:, :, j] + ck[k] * (Pk[:, :, k] + np.outer(v, v))

    # Update (i.e., thin) intensity of unknown targets
    lambdau = (1 - Pd) * lambdau

    # Not shown in paper--truncate low weight components
    ss = lambdau > lambdab_threshold
    lambdau = lambdau[ss]
    xu = xu[:, ss]
    Pu = Pu[:, :, ss]

    if wupd.size == 0:
      pupd = np.empty_like(wupd)
      pnew = np.ones_like(wnew)
    else:
      pupd, pnew = self.spa(wupd=wupd, wnew=wnew)

    mb_upd = self.tomb(pupd=pupd, rupd=rupd, xupd=xupd, Pupd=Pupd, pnew=pnew,
                        rnew=rnew, xnew=xnew, Pnew=Pnew)
    

    poisson_upd = Poisson(birth_weights=self.poisson.birth_weights,
                          birth_states=self.poisson.birth_states)
    poisson_upd.weights = lambdau
    poisson_upd.states = [GaussianState(
        mean=xu[:, i], covar=Pu[:, :, i]) for i in range(len(lambdau))]

    return mb_upd, poisson_upd

  def tomb(self, pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew):
    
    # Form continuing tracks
    tomb_mb = []
    for i in range(len(self.mb)):
      pr = pupd[i, :] * rupd[i, :]
      r = np.sum(pr)
      pr = pr / r
      x = np.dot(xupd[:, i, :].squeeze(), pr)
      P = np.zeros_like(Pupd[:, :, i, 0])
      for j in range(len(pr)):
        v = x - xupd[:, i, j]
        P = P + pr[j] * (Pupd[:, :, i, j] + np.outer(v, v))
        
      new_bern = Bernoulli(r=r, state=GaussianState(mean=x, covar=P))
      tomb_mb.append(new_bern)

    # Form new tracks (already single hypothesis)
    for j in range(len(pnew)):
        r = pnew[j] * rnew[j]
        x = xnew[:, j]
        P = Pnew[:, :, j]
        new_bern = Bernoulli(r=r, state=GaussianState(mean=x, covar=P))
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
