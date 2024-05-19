from motpy.rfs.tomb import TOMBP
import pytest
import numpy as np
from motpy.distributions.gaussian import GaussianState
from motpy.kalman import KalmanFilter
from motpy.models.measurement import LinearMeasurementModel
from motpy.models.transition import ConstantVelocity


def make_data(dt, lambda_c, pd, n_steps):
  seed = 0
  np.random.seed(seed)

  noisy = False
  # Object trajectories
  paths = [[np.array([-90, 1, -90, 1])], [np.array([-90, 1, 90, -1])]]
  cv = ConstantVelocity(ndim_pos=2,
                        q=0.01,
                        position_mapping=[0, 2],
                        velocity_mapping=[1, 3],
                        seed=seed)
  linear = LinearMeasurementModel(
      ndim_state=4, covar=np.eye(2), measured_dims=[0, 2], seed=seed)

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
  birth_dist = GaussianState(
      mean=np.array([0, 0, 0, 0]),
      covar=np.diag([100, 1, 100, 1])**2,
      weight=np.array([0.05]),
  )
  init_dist = GaussianState(
      mean=birth_dist.mean,
      covar=birth_dist.covar,
      weight=[10.0])
  tracker = TOMBP(birth_distribution=birth_dist,
                  undetected_distribution=init_dist,
                  pg=1.0,
                  w_min=1e-4,
                  r_min=1e-4,
                  )
  kf = KalmanFilter(transition_model=cv, measurement_model=linear)

  for k in range(n_steps):
    tracker.mb, tracker.poisson = tracker.predict(
        state_estimator=kf, dt=dt, ps_func=ps)

    tracker.mb, tracker.poisson = tracker.update(
        measurements=Z[k], pd_func=pd, state_estimator=kf, lambda_fa=lambda_c/volume)

  assert len(tracker.mb) == 54
  assert len(tracker.poisson) == 4
  assert np.allclose(tracker.mb[0].r, 0.9999935076418562, atol=1e-6)
  assert np.allclose(tracker.mb[3].r, 0.9999901057591115, atol=1e-6)


def test_scenario_merge():
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
  birth_dist = GaussianState(
      mean=np.array([0, 0, 0, 0]),
      covar=np.diag([100, 1, 100, 1])**2,
      weight=np.array([0.05]),
  )
  init_dist = GaussianState(
      mean=birth_dist.mean,
      covar=birth_dist.covar,
      weight=[10.0])
  tracker = TOMBP(birth_distribution=birth_dist,
                  undetected_distribution=init_dist,
                  pg=1.0,
                  w_min=None,
                  merge_poisson=True,
                  r_min=1e-4,
                  )

  kf = KalmanFilter(transition_model=cv, measurement_model=linear)

  for k in range(n_steps):
    tracker.mb, tracker.poisson = tracker.predict(
        state_estimator=kf, dt=dt, ps_func=ps)

    tracker.mb, tracker.poisson = tracker.update(
        measurements=Z[k], pd_func=pd, state_estimator=kf, lambda_fa=lambda_c/volume)

  assert len(tracker.mb) == 54
  assert len(tracker.poisson) == 1
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
      dt=dt, lambda_c=lambda_c, pd=pd, n_steps=n_steps)

  # Initialize TOMB filter
  birth_dist = GaussianState(
      mean=np.array([0, 0, 0, 0]),
      covar=np.diag([100, 1, 100, 1])**2,
      weight=np.array([0.05]),
  )
  init_dist = GaussianState(
      mean=birth_dist.mean,
      covar=birth_dist.covar,
      weight=[10.0])
  tracker = TOMBP(birth_distribution=birth_dist,
                  undetected_distribution=init_dist,
                  pg=0.999,
                  w_min=None,
                  merge_poisson=True,
                  r_min=1e-4,
                  )

  kf = KalmanFilter(transition_model=cv, measurement_model=linear)

  for k in range(n_steps):
    tracker.mb, tracker.poisson = tracker.predict(
        state_estimator=kf, dt=dt, ps_func=ps)

    tracker.mb, tracker.poisson = tracker.update(
        measurements=Z[k], pd_func=pd, state_estimator=kf, lambda_fa=lambda_c/volume)

  assert len(tracker.mb) == 53
  assert len(tracker.poisson) == 1
  assert np.allclose(tracker.mb[0].r, 0.9999935076418562, atol=1e-6)
  assert np.allclose(tracker.mb[3].r, 0.9999901057591115, atol=1e-6)

if __name__ == '__main__':
  test_scenario_gate()
  pytest.main([__file__])
