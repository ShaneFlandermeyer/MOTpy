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
    log_weights = np.log(birth_weights) if birth_weights is not None else None
    self.poisson = Poisson(birth_log_weights=log_weights,
                           birth_states=birth_states)
    self.mb = MultiBernoulli()

    self.pg = pg
    self.w_min = w_min
    self.r_min = r_min
    self.r_estimate_threshold = r_estimate_threshold

  # TODO: Replace model with internal stuff
  def predict(self, r, x, P, lambdau, xu, Pu, F, Q, Ps, lambdab, xb, Pb):
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


      # Get birth parameters from model
      nb = len(lambdab)
      lambdab_threshold = 1e-4

      # Interpret length of inputs
      n = len(r)
      nu = len(lambdau)

      # Implement prediction algorithm

      # Predict existing tracks
      for i in range(n):
          r[i] = Ps * r[i]
          x[:, i] = F @ x[:, i]
          P[:, :, i] = F @ P[:, :, i] @ F.T + Q

      # Predict existing PPP intensity
      for k in range(nu):
          lambdau[k] = Ps * lambdau[k]
          xu[:, k] = F @ xu[:, k]
          Pu[:, :, k] = F @ Pu[:, :, k] @ F.T + Q

      # Incorporate birth intensity into PPP

      # Allocate memory
      lambdau = np.concatenate((lambdau, np.zeros(nb)))
      xu = np.hstack((xu, np.zeros((xu.shape[0], nb))))
      Pu = np.concatenate((Pu, np.zeros((Pu.shape[0], Pu.shape[1], nb))), axis=2)

      for k in range(nb):
          lambdau[nu + k] = lambdab[k]
          xu[:, nu + k] = xb[:, k]
          Pu[:, :, nu + k] = Pb[:, :, k]

      # Not shown in paper--truncate low weight components
      ss = lambdau > lambdab_threshold
      lambdau = lambdau[ss]
      xu = xu[:, ss]
      Pu = Pu[:, :, ss]

      return r, x, P, lambdau, xu, Pu
    
  def update(self, lambdau, xu, Pu, r, x, P, z, Pd, H, R, lambda_fa):
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
              wupd[i, j + 1] = r[i] * Pd * np.exp(-0.5 * v.T @ np.linalg.inv(S) @ v) / sqrt_det2piS
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
              ck[k] = lambdau[k] * Pd * np.exp(-0.5 * v.T @ np.linalg.inv(Sk[:, :, k]) @ v) / sqrt_det2piSk[k]
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

      return lambdau, xu, Pu, wupd, rupd, xupd, Pupd, wnew, rnew, xnew, Pnew

  def tomb(self, pupd, rupd, xupd, Pupd, pnew, rnew, xnew, Pnew):
    r_threshold = 1e-4

    # Infer sizes
    nold, mp1 = pupd.shape
    stateDimensions = xnew.shape[0]
    m = mp1 - 1
    n = nold + m

    # Allocate memory
    r = np.zeros(n)
    x = np.zeros((stateDimensions, n))
    P = np.zeros((stateDimensions, stateDimensions, n))

    # Form continuing tracks
    for i in range(nold):
        pr = pupd[i, :] * rupd[i, :]
        r[i] = np.sum(pr)
        pr = pr / r[i]
        x[:, i] = np.dot(xupd[:, i, :].squeeze(), pr)
        for j in range(mp1):
            v = x[:, i] - xupd[:, i, j]
            P[:, :, i] = P[:, :, i] + pr[j] * (Pupd[:, :, i, j] + np.outer(v, v))

    # Form new tracks (already single hypothesis)
    r[nold:] = pnew * rnew
    x[:, nold:] = xnew
    P[:, :, nold:] = Pnew

    # Truncate tracks with low probability of existence (not shown in algorithm)
    ss = r > r_threshold
    r = r[ss]
    x = x[:, ss]
    P = P[:, :, ss]

    return r, x, P

  def lbp(self, wupd, wnew):
    """
    LOOPY BELIEF PROPAGATION APPROXIMATION OF MARGINAL ASSOCIATION PROBABILITIES
    Input:
    wupd(i,j+1) is PDA likelihood for track i/measurment j 
    wupd(i,1) is miss likelihood for target i
    wnew(j) is false alarm/new target intensity for measurement j
    Output:
    Estimates of marginal association probabilities in similar format.
    """

    n, mp1 = wupd.shape
    m = mp1 - 1

    eps_conv_threshold = 1e-4

    mu = np.ones((n, m))  # mu_ba
    mu_old = np.zeros((n, m))
    nu = np.zeros((n, m))  # mu_ab

    pupd = np.zeros((n, mp1))
    pnew = np.zeros(m)

    # Run LBP iteration
    if n > 0 and m > 0:
      while np.max(np.abs(mu - mu_old)) > eps_conv_threshold:
        mu_old = copy.deepcopy(mu)

        for i in range(n):
          prd = wupd[i, 1:] * mu[i, :]
          s = wupd[i, 0] + np.sum(prd)
          nu[i, :] = wupd[i, 1:] / (s - prd)

        for j in range(m):
          s = wnew[j] + np.sum(nu[:, j])
          mu[:, j] = 1. / (s - nu[:, j])
          
    # Calculate outputs--for existing tracks then for new tracks
    for i in range(n):
      s = wupd[i, 0] + np.sum(wupd[i, 1:] * mu[i, :])
      pupd[i, 0] = wupd[i, 0] / s
      pupd[i, 1:] = wupd[i, 1:] * mu[i, :] / s

    for j in range(m):
      s = wnew[j] + np.sum(nu[:, j])
      pnew[j] = wnew[j] / s

    return pupd, pnew