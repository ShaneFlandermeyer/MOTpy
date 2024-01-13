import time
from motpy.rfs.momb import MOMBP
import pytest
import numpy as np
from motpy.distributions.gaussian import GaussianMixture, GaussianState
from motpy.kalman import KalmanFilter
from motpy.models.measurement import LinearMeasurementModel
from motpy.models.transition import ConstantVelocity
import matplotlib.pyplot as plt


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
  lambda_c = 20
  dt = 1
  n_steps = 10
  volume = 200*200
  paths, Z, cv, linear = make_data(
      dt=dt, lambda_c=lambda_c, pd=pd, n_steps=n_steps)

  # Initialize filter
  birth_dist = GaussianMixture(
      means=np.array([[0, 0, 0, 0]]),
      covars=np.array([np.diag([100, 1, 100, 1])**2]),
      weights=np.array([0.05]),
  )
  init_dist = GaussianMixture(
      means=birth_dist.means,
      covars=birth_dist.covars,
      weights=[10.0])
  momb = MOMBP(birth_distribution=birth_dist,
               pg=1,
               w_min=1e-4,
               merge_poisson=False,
               r_min=1e-4,
               r_estimate_threshold=0.5,
               )
  momb.poisson.distribution = init_dist

  kf = KalmanFilter(transition_model=cv, measurement_model=linear)

  for k in range(n_steps):
    momb.mb, momb.poisson = momb.predict(state_estimator=kf, dt=dt, ps=0.999)

    momb.mb, momb.poisson = momb.update(
        measurements=Z[k], pd=pd, state_estimator=kf, lambda_fa=lambda_c/volume)

  assert len(momb.mb) == 52
  assert len(momb.poisson) == 4
  assert np.allclose(momb.mb[36].r, 0.9986985737236855, atol=1e-6)
  assert np.allclose(momb.mb[51].r, 0.9980263987614411, atol=1e-6)


def test_scenario_merge():
  """
  Test the algorithm with a simple multi-object scenario
  """

  def pd(x): return 0.8
  lambda_c = 20
  dt = 1
  n_steps = 10
  volume = 200*200
  paths, Z, cv, linear = make_data(
      dt=dt, lambda_c=lambda_c, pd=pd, n_steps=n_steps)

  # Initialize TOMB filter
  birth_dist = GaussianMixture(
      means=np.array([[0, 0, 0, 0]]),
      covars=np.array([np.diag([100, 1, 100, 1])**2]),
      weights=np.array([0.05]),
  )
  init_dist = GaussianMixture(
      means=birth_dist.means,
      covars=birth_dist.covars,
      weights=10.0)
  momb = MOMBP(birth_distribution=birth_dist,
               pg=1,
               w_min=None,
               merge_poisson=True,
               r_min=1e-4,
               r_estimate_threshold=0.5,
               )
  momb.poisson.distribution = init_dist

  kf = KalmanFilter(transition_model=cv, measurement_model=linear)

  for k in range(n_steps):
    momb.mb, momb.poisson = momb.predict(state_estimator=kf, dt=dt, ps=0.999)

    momb.mb, momb.poisson = momb.update(
        measurements=Z[k], pd=pd, state_estimator=kf, lambda_fa=lambda_c/volume)

  assert len(momb.mb) == 52
  assert len(momb.poisson) == 1
  assert np.allclose(momb.mb[36].r, 0.9986985737236855, atol=1e-6)
  assert np.allclose(momb.mb[51].r, 0.9980263987614411, atol=1e-6)


if __name__ == '__main__':
  # test_scenario_merge()
  pytest.main([__file__])
