import numpy as np
import pytest

from motpy.distributions.gaussian import GaussianMixture, GaussianState
from motpy.kalman import ExtendedKalmanFilter
from motpy.rfs.poisson import Poisson


class ConstantTurnModel:
  __test__ = False

  def __init__(self, sigma_v, sigma_w):
    # State is [px, py, v, heading, turn-rate]
    self.ndim_state = 5
    self.sigma_v = sigma_v
    self.sigma_w = sigma_w

  def __call__(self, x, dt, noise=False):
    next_state = x + np.array([
        list(dt*x[:, 2]*np.cos(x[:, 3])),
        list(dt*x[:, 2]*np.sin(x[:, 3])),
        list(np.zeros(len(x))),
        list(dt*x[:, 4]),
        list(np.zeros(len(x))),
    ]).T
    return next_state

  def covar(self, **kwargs):
    G = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 0],
                  [0, 1]])
    return G @ np.diag([self.sigma_v**2, self.sigma_w**2]) @ G.T

  def matrix(self, x, dt, **kwargs):
    nx = len(x)
    F = np.empty((nx, 5, 5))
    for i in range(nx):
      F[i] = np.array([
          [1, 0, dt*np.cos(x[i, 3]), -dt*x[i, 2]*np.sin(x[i, 3]), 0],
          [0, 1, dt*np.sin(x[i, 3]), dt*x[i, 2]*np.cos(x[i, 3]), 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 1, dt],
          [0, 0, 0, 0, 1],
      ])
    return F


class RangeBearingModel:
  def __init__(self, sigma_r, sigma_b, s):
    self.ndim_meas = 2
    self.sigma_r = sigma_r
    self.sigma_b = sigma_b
    self.s = s  # Sensor position

  def __call__(self, x, noise=False):
    rng = np.linalg.norm(x[:, :2] - self.s, axis=1)
    ber = np.arctan2(x[:, 1] - self.s[1], x[:, 0] - self.s[0])

    return np.array([rng, ber]).T

  def matrix(self, x):
    rng = np.linalg.norm(x[:, :2] - self.s, axis=1)
    F = np.empty((len(x), 2, 5))
    for i in range(len(x)):
      F[i] = np.array([
          [(x[i, 0] - self.s[0]) / rng[i], (x[i, 1] - self.s[1]) / rng[i], 0, 0, 0],
          [-(x[i, 1] - self.s[1]) / rng[i]**2,
           (x[i, 0] - self.s[0]) / rng[i]**2, 0, 0, 0],
      ])
    return F

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
  birth_distribution = GaussianMixture(
      mean=np.array([[0.9058, 0.1270, 0.9134, 0.6324, 0.0975]]
                     ).repeat(4, axis=0),
      covar=np.diag([1, 1, 1, 1*np.pi/90, 1*np.pi/90]
                     )[None, ...].repeat(4, axis=0)**2,
      weight=np.array([0.8147]*4)
  )
  init_distribution = GaussianMixture(
      mean=np.array([[0, 0, 5, 0, np.pi/180],
                      [20, 20, -20, 0, np.pi/90],
                      [-20, 10, -10, 0, np.pi/360]]),
      covar=np.eye(5).reshape(1, 5, 5).repeat(3, axis=0),
      weight=np.exp(np.array([-0.6035, -0.0434, -0.0357])))
  ppp = Poisson(birth_state=birth_distribution,
                state=init_distribution)
  ppp = ppp.predict(state_estimator=ekf, ps=ps, dt=T)

  expected_weights = np.array(
      [0.15231, 0.26667, 0.26873, 0.8147, 0.8147, 0.8147, 0.8147])
  expected_mean = np.array([5, 0, 5, 0.0175, 0.0175])
  expected_covar = np.array([[2, 0, 1, 0, 0],
                             [0, 26, 0, 5, 0.],
                             [1,  0, 2, 0, 0.],
                             [0,  5, 0, 2, 1.],
                             [0,  0, 0, 1, 1.00030462]])
  assert np.allclose(ppp.state.weight, expected_weights, atol=1e-4)
  assert np.allclose(
      ppp.state.mean[0], expected_mean, atol=1e-4)
  assert np.allclose(
      ppp.state.covar[0], expected_covar, atol=1e-4)
  assert len(ppp.state) == 7


def test_measurement_update():
  pd = 0.8147
  clutter_intensity = 0.0091
  ekf = ExtendedKalmanFilter(
      transition_model=None,
      measurement_model=RangeBearingModel(
          sigma_r=5, sigma_b=np.pi/180, s=np.array([300, 400]))
  )

  init_distribution = GaussianMixture(
      mean=np.array([[0, 0, 5, 0, np.pi/180],
                      [20, 20, -20, 0, np.pi/90],
                      [-20, 10, -10, 0, np.pi/360]]),
      covar=np.eye(5).reshape(1, 5, 5).repeat(3, axis=0),
      weight=np.exp(np.array([-2.0637, -0.0906, -0.4583])))
  ppp = Poisson(birth_state=None,
                state=init_distribution)

  in_gate = np.array([True, False, True])
  z = ekf.measurement_model(ppp.state.mean[0][None, ...])
  likelihoods = ekf.likelihood(measurement=z, predicted_state=ppp.state)
  bern, bern_weight = ppp.update(
      measurement=z, in_gate=in_gate, state_estimator=ekf,
      pd=np.full(3, pd), clutter_intensity=clutter_intensity, likelihoods=likelihoods)

  # From matlab unit tests
  expected_r = 0.9588984840787167
  expected_mean = np.array([-2.6405, 1.3361, 2.9868, 0.0, 0.0162])
  expected_covar = np.array([
    [45.9548, -22.7715, 34.2917, 0.0, 0.01995], 
    [-22.7715, 12.4874, -17.3522, 0.0, -0.0100], 
    [34.2917, -17.3522, 27.1449, 0.0, 0.0152], 
    [0.0, 0.0, 0.0, 1.0, 0.0], 
    [0.0199, -0.0100, 0.0152, 0.0, 1.0]])
  expected_weight = 0.22140

  assert np.allclose(bern_weight, expected_weight, atol=1e-4)
  assert np.allclose(bern.r, expected_r, atol=1e-6)
  assert np.allclose(bern.state.mean, expected_mean, atol=1e-4)
  assert np.allclose(bern.state.covar, expected_covar, atol=1e-4)


if __name__ == '__main__':
  pytest.main([__file__])
