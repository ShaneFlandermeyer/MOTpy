import numpy as np
import pytest

from motpy.distributions.gaussian import Gaussian
from motpy.distributions.mixture import static_reduce
from motpy.estimators.kalman import KalmanFilter, UnscentedKalmanFilter
from motpy.estimators.kalman.sigma_points import (merwe_scaled_sigma_points,
                                                  merwe_sigma_weights)
from motpy.estimators.kalman.ukf import UnscentedKalmanFilter
from motpy.models.measurement import LinearMeasurementModel
from motpy.models.transition import ConstantVelocity
from motpy.rfs.tomb import TOMBP


def make_data(dt, lambda_c, pd, n_steps):
  seed = 0
  np.random.seed(seed)

  noisy = False
  # Object trajectories
  paths = [[np.array([-90, 1, -90, 1])], [np.array([-90, 1, 90, -1])]]
  cv = ConstantVelocity(state_dim=4,
                        w=0.01,
                        position_inds=[0, 2],
                        velocity_inds=[1, 3],
                        seed=seed)
  linear = LinearMeasurementModel(
      state_dim=4, covar=np.eye(2), measured_dims=[0, 2], seed=seed)

  for i in range(n_steps):
    for path in paths:
      path.append(cv(path[-1], dt=dt, noise=noisy))

  # Measurements
  Z = []
  for k in range(n_steps):
    zk = []

    # Object measurements
    for path in paths:
      if np.random.uniform() < pd(path[k]):
        zk.append(linear(path[k], noise=noisy))

    # Clutter measurements
    for _ in range(np.random.poisson(lambda_c)):
      x = np.random.uniform(-100, 100)
      y = np.random.uniform(-100, 100)
      zk.append(np.array([x, y]))

    zk = np.array(zk)
    # Shuffle zk
    np.random.shuffle(zk)

    Z.append(zk)

  return paths, Z, cv, linear


def test_scenario_prune():
  """
  Test the algorithm with a simple multi-object scenario
  """

  def pd(x): return 0.8
  def ps(x): return 0.999
  lambda_c = 20
  dt = 1
  n_steps = 10
  volume = 200*200
  paths, Z, cv, linear = make_data(
      dt=dt, lambda_c=lambda_c, pd=pd, n_steps=n_steps)

  # Initialize TOMB filter
  birth_dist = Gaussian(
      mean=np.array([0, 0, 0, 0])[None, :],
      covar=np.diag([100, 1, 100, 1])[None, :]**2,
      weight=np.array([0.05]),
  )
  init_dist = Gaussian(
      mean=birth_dist.mean,
      covar=birth_dist.covar,
      weight=np.array([10.0])
  )
  tracker = TOMBP(birth_state=birth_dist, undetected_state=init_dist)
  kf = KalmanFilter(transition_model=cv, measurement_model=linear)

  for k in range(n_steps):
    tracker.mb, tracker.poisson, tracker.meta = tracker.predict(
        state_estimator=kf, dt=dt, ps_model=ps
    )
    tracker.poisson, _ = tracker.poisson.prune(threshold=1e-4)

    tracker.mb, tracker.poisson, tracker.metadata = tracker.update(
        measurements=Z[k],
        pd_model=pd,
        state_estimator=kf,
        lambda_fa=lambda_c/volume
    )
    if tracker.mb.size > 0:
      tracker.mb, tracker.metadata['mb'] = tracker.mb.prune(
          meta=tracker.metadata['mb'],
          valid_fn=lambda mb: mb.r > 1e-4,
      )

  assert tracker.mb.size == 54
  assert tracker.poisson.size == 4
  assert np.allclose(tracker.mb[0].r, 0.9999935076418562, atol=1e-6)
  assert np.allclose(tracker.mb[3].r, 0.9999901057591115, atol=1e-6)


def test_scenario_gate():
  """
  Test the algorithm with a simple multi-object scenario
  """

  def pd(x): return 0.8
  def ps(x): return 0.999
  lambda_c = 20
  dt = 1
  n_steps = 10
  volume = 200*200
  paths, Z, cv, linear = make_data(
      dt=dt, lambda_c=lambda_c, pd=pd, n_steps=n_steps
  )

  # Initialize TOMB filter
  birth_dist = Gaussian(
      mean=np.array([0, 0, 0, 0])[None, :],
      covar=np.diag([100, 1, 100, 1])[None, :]**2,
      weight=np.array([0.05]),
  )
  init_dist = Gaussian(
      mean=birth_dist.mean,
      covar=birth_dist.covar,
      weight=np.array([10.0])
  )
  tracker = TOMBP(
      birth_state=birth_dist, undetected_state=init_dist, pg=0.999
  )

  kf = KalmanFilter(transition_model=cv, measurement_model=linear)

  for k in range(n_steps):
    tracker.mb, tracker.poisson, tracker.meta = tracker.predict(
        state_estimator=kf, dt=dt, ps_model=ps)
    tracker.poisson.state = static_reduce(tracker.poisson.state)

    tracker.mb, tracker.poisson, tracker.metadata = tracker.update(
        measurements=Z[k],
        pd_model=pd,
        state_estimator=kf,
        lambda_fa=lambda_c/volume
    )
    if tracker.mb.size > 0:
      tracker.mb, tracker.metadata['mb'] = tracker.mb.prune(
          meta=tracker.metadata['mb'],
          valid_fn=lambda mb: mb.r > 1e-4,
      )

  assert tracker.mb.size == 53
  assert tracker.poisson.size == 1
  assert np.allclose(tracker.mb[0].r, 0.9999935076418562, atol=1e-6)
  assert np.allclose(tracker.mb[3].r, 0.9999901057591115, atol=1e-6)


def test_ukf_tomb():
  """
  Test the algorithm using a UKF for state estimation
  """

  def pd(x): return 0.8
  def ps(x): return 0.999
  lambda_c = 20
  dt = 1
  n_steps = 10
  volume = 200*200
  paths, Z, cv, linear = make_data(
      dt=dt, lambda_c=lambda_c, pd=pd, n_steps=n_steps)

  # Initialize TOMB filter
  birth_state = Gaussian(
      mean=np.array([0, 0, 0, 0])[None, :],
      covar=np.diag([100, 1, 100, 1])[None, :]**2,
      weight=np.array([0.05])
  )

  undetected_state = Gaussian(
      mean=birth_state.mean,
      covar=birth_state.covar,
      weight=np.array([10.0])
  )

  ukf = UnscentedKalmanFilter(transition_model=cv, measurement_model=linear)
  tracker = TOMBP(birth_state=birth_state, undetected_state=undetected_state)

  for k in range(n_steps):
    tracker.mb, tracker.poisson, tracker.metadata = tracker.predict(
        state_estimator=ukf,
        dt=dt,
        ps_model=ps,
    )
    tracker.poisson, _ = tracker.poisson.prune(threshold=1e-4)

    tracker.mb, tracker.poisson, tracker.metadata = tracker.update(
        measurements=Z[k],
        pd_model=pd,
        state_estimator=ukf,
        lambda_fa=lambda_c / volume
    )
    if tracker.mb.size > 0:
      tracker.mb, tracker.metadata['mb'] = tracker.mb.prune(
          meta=tracker.metadata['mb'],
          valid_fn=lambda mb: mb.r > 1e-4,
      )

  assert tracker.mb.size == 54
  assert tracker.poisson.size == 4
  assert np.allclose(tracker.mb[0].r, 0.9999935076418562, atol=1e-6)
  assert np.allclose(tracker.mb[3].r, 0.9999901057591115, atol=1e-6)


if __name__ == '__main__':
  
  # test_scenario_prune()
  # test_scenario_prune()
  # test_scenario_gate()
  test_ukf_tomb()
  pytest.main([__file__])
