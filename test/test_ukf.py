import numpy as np

import pytest
from motpy.models.transition import ConstantVelocity
from motpy.kalman import UnscentedKalmanFilter
from motpy.distributions.gaussian import Gaussian
import matplotlib.pyplot as plt
from motpy.models.measurement.range_bearing import RangeBearingModel


def test_predict():
  seed = 0

  state = Gaussian(
      mean=np.array([0, 1, 0, 1]),
      covar=np.diag([1., 0.5, 1., 0.5]))

  # Motpy UKF
  w = 0.01
  cv = ConstantVelocity(ndim_state=4, w=w, seed=seed)
  ukf = UnscentedKalmanFilter(transition_model=cv, measurement_model=None)
  filter_state = {
      'alpha': 0.1,
      'beta': 2,
      'kappa': 0
  }
  dt = 1
  pred_state, filter_state = ukf.predict(
      state=state, dt=dt, filter_state=filter_state)

  # Filterpy UKF
  from filterpy.kalman import UnscentedKalmanFilter as UKF
  from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
  points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2, kappa=0)
  ukf_filterpy = UKF(dim_x=4, dim_z=0, dt=dt, hx=None, fx=cv, points=points)
  ukf_filterpy.x = state.mean
  ukf_filterpy.P = state.covar
  ukf_filterpy.Q = cv.covar(dt=dt)
  ukf_filterpy.predict()

  assert np.allclose(pred_state.mean, ukf_filterpy.x)
  assert np.allclose(pred_state.covar, ukf_filterpy.P)


def test_update():
  seed = 0
  state = Gaussian(
      mean=np.array([0, 1, 0, 1]),
      covar=np.diag([1., 0.5, 1., 0.5]))
  ground_truth = np.array([0, 1, 0, 1])

  # Motpy UKF
  R = np.diag([0.1, np.deg2rad(0.1)])
  cv = ConstantVelocity(ndim_state=4, w=0, seed=seed)
  range_bearing = RangeBearingModel(covar=R, seed=seed)
  ukf = UnscentedKalmanFilter(
      transition_model=cv, measurement_model=range_bearing)
  filter_state = {
      'alpha': 0.1,
      'beta': 2,
      'kappa': 0
  }
  dt = 0

  # Get the appropriate state dict from prediction
  pred_state, filter_state = ukf.predict(
      state=state, dt=dt, filter_state=filter_state)
  z = range_bearing(ground_truth, noise=True)
  pred_state, filter_state = ukf.update(
      predicted_state=pred_state, measurement=z, filter_state=filter_state)
  
  # Filterpy UKF
  from filterpy.kalman import UnscentedKalmanFilter as UKF
  from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
  points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2, kappa=0)
  ukf_filterpy = UKF(dim_x=4, dim_z=2, dt=dt, hx=range_bearing, fx=cv, points=points)
  ukf_filterpy.x = state.mean
  ukf_filterpy.P = state.covar
  ukf_filterpy.Q = cv.covar(dt=dt)
  ukf_filterpy.R = R
  ukf_filterpy.predict()
  ukf_filterpy.update(z)

  assert np.allclose(pred_state.mean, ukf_filterpy.x)
  assert np.allclose(pred_state.covar, ukf_filterpy.P)
  


# def test_ukf():
#   # Generate ground truth
#   n_steps = 20
#   current_time = last_update = 0
#   dt = 1

#   # Generate ground truth
#   trajectory = []
#   init_mean, init_covar = np.array([0, 1, 0, 1]), np.diag([1., 0.5, 1., 0.5])
#   trajectory.append(init_mean)
#   cv = ConstantVelocity(ndim=2, w=0.05)
#   for i in range(n_steps):
#     state = cv(trajectory[-1], dt=dt, noise=False)
#     trajectory.append(state)

#   # Generate measurements
#   range_bearing = NonlinearMeasurementModel(R=np.diag([np.deg2rad(0.1), 0.1]))
#   measurements = []
#   for state in trajectory:
#     z = range_bearing(state, noise=False)
#     measurements.append(z)

#   # Test the UKF
#   ukf = UnscentedKalmanFilter(
#       transition_model=cv, measurement_model=range_bearing)
#   filter_state = {
#       'alpha': 0.1,
#       'beta': 2,
#       'kappa': 0
#   }
#   track_states = [GaussianState(mean=init_mean, covar=init_covar)]
#   pred_state, filter_state = ukf.predict(
#       state=track_states[-1],
#       dt=0,
#       filter_state=filter_state)
#   for i, m in enumerate(measurements):
#     post_state, filter_state = ukf.update(
#         predicted_state=pred_state,
#         measurement=m,
#         filter_state=filter_state)
#     pred_state, filter_state = ukf.predict(
#         state=post_state, dt=dt, filter_state=filter_state)
#     track_states.append(post_state)

#   true_states = np.stack([state for state in trajectory]).T
#   track_states = np.stack([state.mean for state in track_states])[1:].T
#   track_pos = track_states[[0, 2], 0]
#   track_vel = track_states[[1, 3], 0]
#   pos_mse = np.mean(np.linalg.norm(true_states[[0, 2]] - track_pos, axis=1))
#   vel_mse = np.mean(np.linalg.norm(true_states[[1, 3]] - track_vel, axis=1))
#   assert pos_mse < 0.2
#   assert vel_mse < 0.1


# def test_linear_predict():
#   seed = 0
#   np.random.seed(seed)
#   cv = ConstantVelocity(ndim=2, w=0.05, seed=seed)
#   ukf = UnscentedKalmanFilter(transition_model=cv, measurement_model=None)
#   kf = KalmanFilter(transition_model=cv, measurement_model=None)
#   state = GaussianState(mean=np.random.uniform(size=4),
#                         covar=np.diag(np.random.uniform(size=4)))
#   dt = 1
#   kf_pred, meta = kf.predict(state, dt)
#   ukf_pred, meta = ukf.predict(state, dt)
#   assert np.allclose(kf_pred.mean, ukf_pred.mean)
#   assert np.allclose(kf_pred.covar, ukf_pred.covar)
if __name__ == '__main__':
  # test_update()
  pytest.main([__file__])
