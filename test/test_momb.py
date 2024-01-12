import time
from motpy.rfs.momb import MOMBP
import pytest
import numpy as np
from motpy.distributions.gaussian import GaussianState
from motpy.kalman import KalmanFilter
from motpy.models.measurement import LinearMeasurementModel
from motpy.models.transition import ConstantVelocity
import matplotlib.pyplot as plt


# def test_predict():
#   raise NotImplementedError


# def test_update():
#   measurements = [np.array([0, 0]), np.array([1, 1]), np.array([100, 100])]
#   states = [GaussianState(mean=np.array([0, 0]), covar=np.eye(2)),
#             GaussianState(mean=np.array([1, 1]), covar=np.eye(2))]

#   tomb = TOMBP(birth_weights=np.array([0.01, 0.01]),
#                birth_states=copy.deepcopy(states),
#                w_min=1e-4, r_min=0.1,
#                r_estimate_threshold=None, pg=0.999)
#   # Add MB components
#   tomb.mb.append(Bernoulli(r=0.5, state=copy.deepcopy(states[0])))
#   tomb.mb.append(Bernoulli(r=0.5, state=copy.deepcopy(states[1])))

#   kf = KalmanFilter(measurement_model=LinearMeasurementModel(
#       ndim_state=2, covar=np.eye(2), measured_dims=[0, 1]))
#   tomb.update(measurements=measurements, state_estimator=kf, pd=0.9)


def test_scenario():
  """
  Test the algorithm with a simple multi-object scenario
  """
  seed = 0
  np.random.seed(seed)

  n_steps = 150
  def pd(x): return 0.8
  lambda_c = 20
  volume = 200*200
  dt = 1
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

    # Shuffle zk
    np.random.shuffle(zk)

    Z.append(zk)

  # Initialize TOMB filter
  momb = MOMBP(birth_weights=np.array([0.05]),
               birth_states=GaussianState(
                   mean=np.array([0, 0, 0, 0]),
                   covar=np.diag([100, 1, 100, 1])**2),
               pg=1,
               w_min=1e-4,
               r_min=1e-4,
               r_estimate_threshold=0.5)
  momb.poisson.states.append(momb.poisson.birth_states[0])
  momb.poisson.weights = np.append(momb.poisson.weights, 10)

  kf = KalmanFilter(transition_model=cv, measurement_model=linear)

  for k in range(10):
    momb.mb, momb.poisson = momb.predict(state_estimator=kf, dt=dt, ps=0.999)

    momb.mb, momb.poisson = momb.update(
        measurements=Z[k], pd=pd, state_estimator=kf, lambda_fa=lambda_c/volume)

  assert len(momb.mb) == 52
  assert len(momb.poisson) == 4
  assert np.allclose(momb.mb[36].r, 0.9986985737236855, atol=1e-6)
  assert np.allclose(momb.mb[51].r, 0.9980263987614411, atol=1e-6)


if __name__ == '__main__':
  # test_scenario()
  pytest.main([__file__])
