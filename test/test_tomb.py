import time
from motpy.rfs.bernoulli import Bernoulli
from motpy.rfs.tomb import TOMBP
import pytest
import numpy as np
from motpy.distributions.gaussian import GaussianState
from motpy.kalman import KalmanFilter
from motpy.models.measurement import LinearMeasurementModel
import copy
from motpy.models.transition import ConstantVelocity
import matplotlib.pyplot as plt
import scipy.io as io


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
  TODO: Test the algorithm with a simple multi-object scenario
  """
  seed = 0
  np.random.seed(seed)

  n_steps = 150
  pd = 0.5
  lambda_c = 20
  volume = 200*200
  dt = 1
  noisy = True

  # Object trajectories
  paths = [[np.array([-90, 1, -90, 1])], [np.array([-90, 1, 90, -1])]]
  cv = ConstantVelocity(ndim_pos=2,
                        q=0.01,
                        position_mapping=[0, 2],
                        velocity_mapping=[1, 3],
                        seed=seed)
  for i in range(n_steps):
    for path in paths:
      path.append(cv(path[-1], dt=dt, noise=noisy))

  # Measurements
  linear = LinearMeasurementModel(
      ndim_state=4, covar=np.eye(2), measured_dims=[0, 2], seed=seed)
  Z = []
  for k in range(n_steps):
    zk = []

    # Object measurements
    for path in paths:
      if np.random.uniform() < pd:
        zk.append(linear(path[k], noise=noisy))

    # Clutter measurements
    for _ in range(np.random.poisson(lambda_c)):
      x = np.random.uniform(-100, 100)
      y = np.random.uniform(-100, 100)
      zk.append(np.array([x, y]))

    # Shuffle zk
    np.random.shuffle(zk)

    Z.append(zk)

  # Load from matlab
  xlog = io.loadmat('test/xlog.mat')['xlog']

  Z = []
  measlog = io.loadmat('test/measlog.mat')['measlog']
  for meas_k in measlog:
    Z.append(list(meas_k[0].T))

  # Initialize TOMB filter
  tomb = TOMBP(birth_weights=[0.05],
               birth_states=[GaussianState(
                   mean=np.array([0, 0, 0, 0]),
                   covar=np.diag([100, 1, 100, 1])**2)],
               pg=0.99,
               w_min=1e-4,
               r_min=1e-4,
               r_estimate_threshold=0.5)
  tomb.poisson.states.append(tomb.poisson.birth_states[0])
  tomb.poisson.weights = np.append(tomb.poisson.weights, 10)


  kf = KalmanFilter(transition_model=cv, measurement_model=linear)
  for k in range(n_steps):
    start = time.time()
    # Predict
    tomb.mb, tomb.poisson = tomb.predict(state_estimator=kf, dt=dt, Ps=0.999)

    tomb.mb, tomb.poisson = tomb.update(
        z=Z[k], Pd=pd, state_estimator=kf, lambda_fa=lambda_c/volume)

    # print(np.max([bern.r for bern in tomb.mb]))
    # print(len(tomb.mb))
    # print(len(tomb.poisson))
    # for i, bern in enumerate(tomb.mb):
    #   if bern.r > 0.5:
    #     print(f"MB index: {i}, r: {bern.r}")
    print(f"Step {k}, {time.time() - start} seconds")

if __name__ == '__main__':
  test_scenario()
  # pytest.main([__file__])
