from motpy.rfs.bernoulli import Bernoulli
from motpy.rfs.tomb_matlab import TOMBP
import pytest
import numpy as np
from motpy.distributions.gaussian import GaussianState
from motpy.kalman import KalmanFilter
from motpy.models.measurement import LinearMeasurementModel
import copy
from motpy.models.transition import ConstantVelocity
import matplotlib.pyplot as plt
import scipy.io as io


def test_predict():
  raise NotImplementedError


def test_update():
  measurements = [np.array([0, 0]), np.array([1, 1]), np.array([100, 100])]
  states = [GaussianState(mean=np.array([0, 0]), covar=np.eye(2)),
            GaussianState(mean=np.array([1, 1]), covar=np.eye(2))]

  tomb = TOMBP(birth_weights=np.array([0.01, 0.01]),
               birth_states=copy.deepcopy(states),
               w_min=1e-4, r_min=0.1,
               r_estimate_threshold=None, pg=0.999)
  # Add MB components
  tomb.mb.append(Bernoulli(r=0.5, state=copy.deepcopy(states[0])))
  tomb.mb.append(Bernoulli(r=0.5, state=copy.deepcopy(states[1])))

  kf = KalmanFilter(measurement_model=LinearMeasurementModel(
      ndim_state=2, covar=np.eye(2), measured_dims=[0, 1]))
  tomb.update(measurements=measurements, state_estimator=kf, pd=0.9)


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

  # min_x = np.min([np.min([x[0] for x in path]) for path in paths])
  # max_x = np.max([np.max([x[0] for x in path]) for path in paths])
  # min_y = np.min([np.min([x[2] for x in path]) for path in paths])
  # max_y = np.max([np.max([x[2] for x in path]) for path in paths])
  # x_range = max_x - min_x
  # y_range = max_y - min_y
  # volume = x_range * y_range

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
  # # Convert paths to n_steps x 2 array
  # paths_array = np.zeros((n_steps, 2))
  # for i in range(n_steps):
  #   paths_array[i] = [paths[0][i][[0, 2]], paths[1][i][[0, 2]]]
  # savemat('paths.mat', {'paths': paths_array})
  # savemat('measurements.mat', {'measurements': Z})

  # Initialize TOMB filter
  tomb = TOMBP(birth_weights=[0.05],
               birth_states=[GaussianState(
                   mean=np.array([0, 0, 0, 0]),
                   covar=np.diag([100, 1, 100, 1])**2)],
               pg=1,
               w_min=1e-4,
               r_min=1e-4,
               r_estimate_threshold=0.5)
  tomb.poisson.states.append(tomb.poisson.birth_states[0])
  tomb.poisson.weights = np.append(tomb.poisson.weights, 10)

  # TODO: Hard-coded variables
  r = np.array([])
  x = np.zeros((4, 0))
  P = np.zeros((4, 4, 0))
  lambdau = tomb.poisson.weights
  xu = np.array([state.mean for state in tomb.poisson.states]).swapaxes(0, -1)
  Pu = np.array([state.covar for state in tomb.poisson.states]).swapaxes(0, -1)
  xb = np.array(
      [state.mean for state in tomb.poisson.birth_states]).swapaxes(0, -1)
  Pb = np.array(
      [state.covar for state in tomb.poisson.birth_states]).swapaxes(0, -1)

  kf = KalmanFilter(transition_model=cv, measurement_model=linear)
  for k in range(n_steps):
    # Predict
    tomb.mb, tomb.poisson = tomb.predict(state_estimator=kf, dt=dt, Ps=0.999)

    tomb.mb, tomb.poisson = tomb.update(
        z=np.array(Z[k]).T, Pd=pd, state_estimator=kf, lambda_fa=lambda_c/volume)

    print(np.max([bern.r for bern in tomb.mb]))
    print(len(tomb.mb))
    print(len(tomb.poisson))

    # Update
    # TODO: Convert to matlab version
    # tomb.update(measurements=Z[k],
    #             state_estimator=kf,
    #             pd=pd,
    #             clutter_intensity=lambda_c/volume)

    # Print MB components with r > 0.5
    # print(len(tomb.mb))
    # max_r = 0
    # for i, bern in enumerate(tomb.mb):
    #   max_r = max(max_r, bern.r)
    #   if bern.r > tomb.r_estimate_threshold:
    #     print(f"MB component {i} has r = {bern.r}")
    # print(max_r)
    # print(len(tomb.poisson))
    # print(f"Number of MB components: {len(tomb.mb)}")
    # print(f"Number of PPP components: {len(tomb.poisson)}")

  # Plot paths
  plt.figure()
  for path in paths:
    plt.plot([x[0] for x in path], [x[2] for x in path])
  # Plot measurements
  for k in range(n_steps):
    for z in Z[k]:
      plt.plot(z[0], z[1], 'rx')
  plt.show()

  raise NotImplementedError


if __name__ == '__main__':
  test_scenario()
  pytest.main([__file__])
