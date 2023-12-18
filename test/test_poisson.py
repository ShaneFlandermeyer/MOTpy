import numpy as np
import pytest

from motpy.distributions.gaussian import GaussianState
from motpy.kalman import ExtendedKalmanFilter
from motpy.rfs.poisson import Poisson


class ConstantTurnModel:
  __test__ = False

  def __init__(self, sigma_v, sigma_w):
    # State is [px, py, v, heading, turn-rate]
    self.ndim_state = 5
    self.sigma_v = sigma_v
    self.sigma_w = sigma_w

  def __call__(self, state, dt, noise=False):
    next_state = state + np.array([
        dt*state[2]*np.cos(state[3]),
        dt*state[2]*np.sin(state[3]),
        0,
        dt*state[4],
        0
    ]).reshape(state.shape)
    return next_state

  def covar(self, **kwargs):
    G = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 0],
                  [0, 1]])
    return G @ np.diag([self.sigma_v**2, self.sigma_w**2]) @ G.T

  def matrix(self, x, dt, **kwargs):
    return np.array([
        [1, 0, dt*np.cos(x[3]), -dt*x[2]*np.sin(x[3]), 0],
        [0, 1, dt*np.sin(x[3]), dt*x[2]*np.cos(x[3]), 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, dt],
        [0, 0, 0, 0, 1],
    ])


class RangeBearingModel:
  def __init__(self, sigma_r, sigma_b, s):
    self.ndim_meas = 2
    self.sigma_r = sigma_r
    self.sigma_b = sigma_b
    self.s = s  # Sensor position

  def __call__(self, state, noise=False):
    rng = np.linalg.norm(state[:2] - self.s)
    ber = np.arctan2(state[1] - self.s[1], state[0] - self.s[0])

    return np.array([rng, ber])

  def matrix(self, x):
    rng = np.linalg.norm(x[:2] - self.s)
    return np.array([
        [(x[0] - self.s[0]) / rng, (x[1] - self.s[1]) / rng, 0, 0, 0],
        [-(x[1] - self.s[1]) / rng**2,
         (x[0] - self.s[0]) / rng**2, 0, 0, 0],
    ])

  def covar(self, **kwargs):
    return np.diag([self.sigma_r**2, self.sigma_b**2])


def test_predict():
  T = 1
  ps = 0.2785
  ekf = ExtendedKalmanFilter(
      transition_model=ConstantTurnModel(sigma_v=1, sigma_w=np.pi/180),
      measurement_model=None,
  )
  # Weights should be LOG weights
  birth_weights = np.array([-0.2049]*4)
  birth_states = [
      GaussianState(mean=np.array([0.9058, 0.1270, 0.9134, 0.6324, 0.0975]),
                    covar=np.diag([1, 1, 1, 1*np.pi/90, 1*np.pi/90])**2)
      for _ in range(4)
  ]

  ppp = Poisson(
      birth_log_weights=birth_weights, birth_states=birth_states)
  # Use custom states and weights for testing
  ppp.states = []
  ppp.states.append(GaussianState(mean=np.array([0, 0, 5, 0, np.pi/180]),
                                  covar=np.eye(5)))
  ppp.states.append(GaussianState(mean=np.array([20, 20, -20, 0, np.pi/90]),
                                  covar=np.eye(5)))
  ppp.states.append(GaussianState(mean=np.array([-20, 10, -10, 0, np.pi/360]),
                                  covar=np.eye(5)))
  ppp.log_weights = np.array([-0.6035, -0.0434, -0.0357])

  ppp = ppp.predict(state_estimator=ekf, ps=ps, dt=T)

  expected_weights = np.array(
      [-1.8819, -1.3218, -1.3141, -0.2049, -0.2049, -0.2049, -0.2049,])
  expected_mean = np.array([5, 0, 5, 0.0175, 0.0175])
  expected_covar = np.array([[2, 0, 1, 0, 0],
                             [0, 26, 0, 5, 0.],
                             [1,  0, 2, 0, 0.],
                             [0,  5, 0, 2, 1.],
                             [0,  0, 0, 1, 1.00030462]])
  assert np.allclose(ppp.log_weights, expected_weights, atol=1e-4)
  assert np.allclose(ppp.states[0].mean, expected_mean, atol=1e-4)
  assert np.allclose(ppp.states[0].covar, expected_covar, atol=1e-4)
  assert len(ppp.states) == 7


def test_detected_update():
  pd = 0.8147
  clutter_intensity = 0.0091
  ekf = ExtendedKalmanFilter(
      transition_model=None,
      measurement_model=RangeBearingModel(
          sigma_r=5, sigma_b=np.pi/180, s=np.array([300, 400]))
  )

  ppp = Poisson(birth_log_weights=None, birth_states=None)
  ppp.states.append(GaussianState(mean=np.array([0, 0, 5, 0, np.pi/180]),
                                  covar=np.eye(5)))
  ppp.states.append(GaussianState(mean=np.array([20, 20, -20, 0, np.pi/90]),
                                  covar=np.eye(5)))
  ppp.states.append(GaussianState(mean=np.array([-20, 10, -10, 0, np.pi/360]),
                                  covar=np.eye(5)))
  ppp.log_weights = np.array([-2.0637, -0.0906, -0.4583])

  in_gate = np.array([True, False, True])
  z = ekf.measurement_model(ppp.states[0].mean)
  bern, total_log_likelihood = ppp.update(
      measurement=z, in_gate=in_gate, state_estimator=ekf,
      pd=pd, clutter_intensity=clutter_intensity)

  # From matlab unit tests
  expected_r = 0.9588984840787167
  expected_mean = np.array([-2.640505492844699, 1.3361450073693577,
                            2.986810645184825, 0.0, 0.016282066429689126])
  expected_covar = np.array([[45.95489241096722, -22.771572348910606, 34.29174484314454, 0.0, 0.01995012845888863], [-22.771572348910606, 12.487496687253945, -17.352262205214927, 0.0, -0.010095136938345755], [
                            34.29174484314454, -17.352262205214927, 27.144908943886485, 0.0, 0.015210491456831086], [0.0, 0.0, 0.0, 1.0, 0.0], [0.01995012845888863, -0.010095136938345755, 0.015210491456831086, 0.0, 1.0000088491052257]])
  expected_log_l = -1.50777059102861

  assert np.allclose(total_log_likelihood, expected_log_l, atol=1e-6)
  assert np.allclose(bern.r, expected_r, atol=1e-6)
  assert np.allclose(bern.state.mean, expected_mean, atol=1e-6)
  assert np.allclose(bern.state.covar, expected_covar, atol=1e-6)


if __name__ == '__main__':
  pytest.main([__file__])
