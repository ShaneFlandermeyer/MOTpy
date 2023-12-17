from motpy.rfs.poisson import Poisson
from motpy.rfs.bernoulli import MultiBernoulli
from motpy.kalman import KalmanFilter
from typing import List, Tuple
import numpy as np
from motpy.gate import EllipsoidalGate
import copy


class PMBMFilter:
  def __init__(self,
               birth_weights: np.ndarray,
               birth_states: List[np.ndarray],
               pg: float,
               w_min: float,
               num_hypo_max: int,
               r_min: float,
               r_min_estimate: float,
               ):
    self.poisson = Poisson(birth_log_weights=np.log(birth_weights),
                                       birth_states=birth_states)
    self.mbm = []

    # PMBM parameters
    self.pg = pg
    self.w_min = w_min
    self.num_hypo_max = num_hypo_max
    self.r_min = r_min
    self.r_min_estimate = r_min_estimate

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float) -> Tuple[Poisson, List[MultiBernoulli]]:
    pred_ppp = self.poisson.predict(
        state_estimator=state_estimator, ps=ps, dt=dt)

    pred_mbm = []
    for mb in self.mbm:
      mb_pred = mb.predict(state_estimator=state_estimator, ps=ps, dt=dt)
      pred_mbm.append(mb_pred)

    return pred_ppp, pred_mbm

  def update(self,
             measurements: List[np.ndarray],
             state_estimator: KalmanFilter,
             pd: float) -> Tuple[Poisson, List[MultiBernoulli]]:

    meas_dim = state_estimator.measurement_model.ndim
    gate = EllipsoidalGate(pg=self.pg, ndim=meas_dim)

    # Gate measurements w.r.t. Gaussian components of Poisson prior
    valid_matrix_u = np.zeros(
        (len(measurements), len(self.poisson)), dtype=int)
    for i, (w, x) in enumerate(self.poisson):
      z_pred = state_estimator.measurement_model(x.mean, noise=False)
      H = state_estimator.measurement_model.matrix(x=x.mean, noise=False)
      R = state_estimator.measurement_model.covar()
      S = H @ x.covar @ H.T + R
      S = (S + S.T) / 2
      valid_meas, valid_inds = gate(measurements=measurements,
                                    predicted_measurement=z_pred,
                                    innovation_covar=S)
      valid_matrix_u[valid_inds, i] = 1

    # TODO: Create Bernoulli component for undetected associations
    pass
    # valid_inds_u = np.unique(valid_inds_u)
    # for i in valid_inds_u:
    #   pass

    # TODO: Gate each local hypothesis

    # TODO: Update detected objects

    # TODO: Update undetected objects
    # raise NotImplementedError

    # Update for undetected objects that remain undetected
    self.poisson.log_weights += np.log(1 - pd)

    # TODO: Reduce

    return self.poisson, self.mbm
