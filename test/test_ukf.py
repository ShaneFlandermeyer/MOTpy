import matplotlib.pyplot as plt
import numpy as np
import pytest
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints

from motpy.distributions.gaussian import Gaussian
from motpy.kalman import UnscentedKalmanFilter
from motpy.kalman.sigma_points import (merwe_scaled_sigma_points,
                                       merwe_sigma_weights)
from motpy.models.measurement.range_bearing import RangeBearingModel
from motpy.models.transition import ConstantVelocity


def test_predict():

  seed = 0
  dt = 1
  w = 0.01
  alpha = 0.1
  beta = 2
  kappa = 0
  state = Gaussian(
      mean=np.array([0, 1, 0, 1]),
      covar=np.diag([1., 0.5, 1., 0.5]))
  state_dim = state.state_dim

  # Motpy UKF
  cv = ConstantVelocity(ndim_state=state_dim, w=w, seed=seed)
  ukf = UnscentedKalmanFilter(transition_model=cv, measurement_model=None)
  points = merwe_scaled_sigma_points(
      x=state.mean, P=state.covar, alpha=alpha, beta=beta, kappa=kappa
  )
  Wm, Wc = merwe_sigma_weights(
      ndim_state=state_dim, alpha=alpha, beta=beta, kappa=kappa
  )
  pred_state, filter_state = ukf.predict(
      state=state, dt=dt, sigma_points=points, Wm=Wm, Wc=Wc
  )

  # Filterpy UKF
  points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2, kappa=0)
  ukf_filterpy = UKF(dim_x=4, dim_z=0, dt=dt, hx=None, fx=cv, points=points)
  ukf_filterpy.x = state.mean
  ukf_filterpy.P = state.covar
  ukf_filterpy.Q = cv.covar(dt=dt)
  ukf_filterpy.predict()

  assert np.allclose(pred_state.mean, ukf_filterpy.x)
  assert np.allclose(pred_state.covar, ukf_filterpy.P)


def test_update():
  alpha = 0.1
  beta = 2
  kappa = 0
  state = Gaussian(
      mean=np.array([0, 1, 0, 1]),
      covar=np.diag([1., 0.5, 1., 0.5]))
  state_dim = state.state_dim
  ground_truth = np.array([0, 1, 0, 1])
  R = np.diag([0.1, np.deg2rad(0.1)])
  range_bearing = RangeBearingModel(covar=R)
  cv = ConstantVelocity(ndim_state=state_dim, w=0)
  z = range_bearing(ground_truth, noise=False)
  
  pred_sigmas = merwe_scaled_sigma_points(
      x=state.mean, P=state.covar, alpha=alpha, beta=beta, kappa=kappa
  )
  Wm, Wc = merwe_sigma_weights(
      ndim_state=state_dim, alpha=alpha, beta=beta, kappa=kappa
  )
  
  # Motpy UKF
  ukf = UnscentedKalmanFilter(
      transition_model=None,
      measurement_model=range_bearing
  )
  motpy_update, update_state = ukf.update(
      predicted_state=state,
      measurement=z,
      predicted_sigmas=pred_sigmas,
      Wm=Wm,
      Wc=Wc
  )
  
  # Filterpy UKF
  points = MerweScaledSigmaPoints(
      n=state_dim, alpha=alpha, beta=beta, kappa=kappa)
  filterpy_update = UKF(dim_x=4, dim_z=2, dt=0, fx=None,
                        hx=range_bearing, points=points)
  filterpy_update.x = state.mean
  filterpy_update.P = state.covar
  filterpy_update.R = R
  filterpy_update.sigmas_f = pred_sigmas
  filterpy_update.update(z)


  assert np.allclose(motpy_update.mean, filterpy_update.x)
  assert np.allclose(motpy_update.covar, filterpy_update.P)


if __name__ == '__main__':
  test_update()
  pytest.main([__file__])
