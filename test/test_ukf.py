import numpy as np

import pytest
from motpy.models.transition import ConstantVelocity
from motpy.kalman import KalmanFilter, UnscentedKalmanFilter
from motpy.distributions.gaussian import GaussianState
import matplotlib.pyplot as plt


class NonlinearMeasurementModel():
  def __init__(self, R):
    self.R = R

  def __call__(self, state, noise=False):
    if noise:
      noise = np.random.multivariate_normal(
          np.zeros(2), self.R, size=state.shape[0] if state.ndim > 1 else 1)
    else:
      noise = 0
    x, y = state[..., 0], state[..., 2]
    azimuth = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return np.stack([azimuth, r], axis=-1) + noise

  def covar(self):
    return self.R


def test_ukf():
  # Generate ground truth
  n_steps = 20
  current_time = last_update = 0
  dt = 1

  # Generate ground truth
  trajectory = []
  init_mean, init_covar = np.array([0, 1, 0, 1]), np.diag([1., 0.5, 1., 0.5])
  trajectory.append(init_mean)
  cv = ConstantVelocity(ndim=2, w=0.05)
  for i in range(n_steps):
    state = cv(trajectory[-1], dt=dt, noise=False)
    trajectory.append(state)

  # Generate measurements
  range_bearing = NonlinearMeasurementModel(R=np.diag([np.deg2rad(0.1), 0.1]))
  measurements = []
  for state in trajectory:
    z = range_bearing(state, noise=False)
    measurements.append(z)

  # Test the UKF
  ukf = UnscentedKalmanFilter(
      transition_model=cv, measurement_model=range_bearing)
  track_states = [GaussianState(
      mean=init_mean, covar=init_covar)]
  pred_state, meta = ukf.predict(state=track_states[-1], dt=0)
  for i, m in enumerate(measurements):
    post_state, meta = ukf.update(
        predicted_state=pred_state,
        measurement=m,
        metadata=meta)
    pred_state, meta = ukf.predict(
        state=post_state, dt=dt, metadata=meta)
    track_states.append(post_state)

  true_states = np.stack([state for state in trajectory]).T
  track_states = np.stack([state.mean for state in track_states])[1:].T
  track_pos = track_states[[0, 2], 0]
  track_vel = track_states[[1, 3], 0]
  pos_mse = np.mean(np.linalg.norm(true_states[[0, 2]] - track_pos, axis=1))
  vel_mse = np.mean(np.linalg.norm(true_states[[1, 3]] - track_vel, axis=1))
  assert pos_mse < 0.2
  assert vel_mse < 0.1


def test_linear_predict():
  seed = 0
  np.random.seed(seed)
  cv = ConstantVelocity(ndim=2, w=0.05, seed=seed)
  ukf = UnscentedKalmanFilter(transition_model=cv, measurement_model=None)
  kf = KalmanFilter(transition_model=cv, measurement_model=None)
  state = GaussianState(mean=np.random.uniform(size=4),
                        covar=np.diag(np.random.uniform(size=4)))
  dt = 1
  kf_pred, meta = kf.predict(state, dt)
  ukf_pred, meta = ukf.predict(state, dt)
  assert np.allclose(kf_pred.mean, ukf_pred.mean)
  assert np.allclose(kf_pred.covar, ukf_pred.covar)


if __name__ == '__main__':
  test_ukf()
  pytest.main([__file__])
