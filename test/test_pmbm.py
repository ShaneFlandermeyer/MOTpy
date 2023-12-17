import pytest
import numpy as np
from motpy.rfs.bernoulli import MultiBernoulli
from motpy.rfs.poisson import PoissonPointProcess
from motpy.distributions.gaussian import GaussianState
from motpy.rfs.pmbm import PMBMFilter
from motpy.kalman import ExtendedKalmanFilter


class ConstantTurnModel:
  __test__ = False

  def __init__(self, sigma_v, sigma_w):
    # State is [px, py, v, heading, turn-rate]
    self.ndim_state = 5
    self.sigma_v = sigma_v
    self.sigma_w = sigma_w

  def __call__(self, x, dt, noise=False):
    next_state = x + np.array([
        dt*x[2]*np.cos(x[3]),
        dt*x[2]*np.sin(x[3]),
        0,
        dt*x[4],
        0
    ]).reshape(x.shape)
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


def test_predict():
  T = 1
  motion_model = ConstantTurnModel(sigma_v=1, sigma_w=np.pi/180)

  # Birth model
  birth_weights = np.full(4, -0.2049)
  birth_states = [GaussianState(
      mean=np.array([0.9058, 0.1270, 0.9134, 0.6324, 0.0975]),
      covar=np.diag([1, 1, 1, 1*np.pi/90, 1*np.pi/90])**2) for _ in range(4)]
  pmbm = PMBMFilter(birth_weights=birth_weights, birth_states=birth_states)

  # Poisson components
  pmbm.poisson.states = []
  x = [
      np.array([0, 0, 5, 0, np.pi/180]),
      np.array([20, 20, -20, 0, np.pi/90]),
      np.array([-20, 10, -10, 0, np.pi/360])]
  P = np.eye(5)
  for i in range(len(x)):
    pmbm.poisson.states.append(GaussianState(mean=x[i], covar=P))
  pmbm.poisson.log_weights = np.array([-0.6035, -0.0434, -0.0357])

  # MBM component
  pmbm.mbm.append(MultiBernoulli(
      rs=np.full(3, 0.1419),
      states=[GaussianState(
          mean=np.array([0.4218, 0.9157, 0.7922, 0.9595, 0.6557]),
          covar=np.eye(5)) for _ in range(3)]))
  pmbm.mbm.append(MultiBernoulli(
      rs=np.full(2, 0.0357),
      states=[GaussianState(
          mean=np.array([0.8491, 0.9340, 0.6787, 0.7577, 0.7431]),
          covar=np.eye(5)) for _ in range(2)]))

  ps = 0.2785
  dt = 1
  pmbm.predict(
      state_estimator=ExtendedKalmanFilter(transition_model=motion_model),
      ps=ps, dt=dt)

  # Ground truth solution
  expected_ppp_w = np.array([-1.8818372196147983, -1.3217372196147983, -
                            1.3140372196147982, -0.2049, -0.2049, -0.2049, -0.2049])
  expected_ppp_states = []
  expected_ppp_states.append(GaussianState(
      mean=np.array([5., 0., 5., 0.01745329, 0.01745329]),
      covar=np.array([[2.,  0.,  1.,  0.,  0.],
                      [0., 26.,  0.,  5.,  0.],
                      [1.,  0.,  2.,  0.,  0.],
                      [0.,  5.,  0.,  2.,  1.],
                      [0.,  0.,  0.,  1.,  1.00030462]])))
  expected_ppp_states.append(GaussianState(
      mean=np.array([0.,  20., -20.,   0.03490659, 0.03490659]),
      covar=np.array([[2.,   0.,   1.,   0., 0.],
                      [0., 401.,   0., -20., 0.],
                      [1.,   0.,   2.,   0., 0.],
                      [0., -20.,   0.,   2., 1.],
                      [0.,   0.,   0.,   1., 1.00030462]])))
  expected_ppp_states.append(GaussianState(
      mean=np.array([-30,  10, -10,  8.72664626e-03, 8.72664626e-03]),
      covar=np.array([[2.,   0.,   1.,   0., 0.],
                      [0., 101.,   0., -10., 0.],
                      [1.,   0.,   2.,   0., 0.],
                      [0., -10.,   0.,   2., 1.],
                      [0.,   0.,   0.,   1., 1.00030462]])))
  expected_ppp_states.extend([GaussianState(
      mean=np.array([0.9058, 0.127,  0.9134, 0.6324, 0.0975]),
      covar=np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0012184696791468343, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0012184696791468343]])) for _ in range(4)])

  expected_r = [np.full(3, 0.03951915), np.array([0.009942450000000002])]
  expected_states = [
      GaussianState(
          mean=np.array([0.87646696, 1.56443631, 0.7922, 1.6152, 0.6557]),
          covar=np.array([[1.75025388, 0.17503461, 0.57392951, -0.64873631, 0.],
                          [0.17503461, 1.87732696, 0.81890471, 0.45466696, 0.],
                          [0.57392951, 0.81890471, 2., 0., 0.],
                          [-0.64873631, 0.45466696, 0., 2., 1.],
                          [0., 0., 0., 1., 1.00030462]])),
      GaussianState(
          mean=np.array([1.34212031, 1.40043827, 0.6787, 1.5008, 0.7431]),
          covar=np.array([[1.74524866, 0.26926947, 0.72641861, -0.46643827, 0.],
                          [0.26926947, 1.71538503, 0.6872525, 0.49302031, 0.],
                          [0.72641861, 0.6872525, 2., 0., 0.],
                          [-0.46643827, 0.49302031, 0., 2., 1.],
                          [0., 0., 0., 1., 1.00030462]]))
  ]

  assert np.allclose(pmbm.poisson.log_weights, expected_ppp_w)
  for i in range(len(pmbm.poisson.states)):
    assert np.allclose(pmbm.poisson.states[i].mean,
                       expected_ppp_states[i].mean)
    assert np.allclose(pmbm.poisson.states[i].covar,
                       expected_ppp_states[i].covar)

  assert np.allclose(pmbm.mbm[0].rs, expected_r[0])
  assert np.allclose(pmbm.mbm[1].rs, expected_r[1])
  for i in range(len(pmbm.mbm[0].states)):
    assert np.allclose(pmbm.mbm[0].states[i].mean,
                       expected_states[0].mean)
    assert np.allclose(pmbm.mbm[0].states[i].covar,
                       expected_states[0].covar)
  for i in range(len(pmbm.mbm[1].states)):
    assert np.allclose(pmbm.mbm[1].states[i].mean,
                       expected_states[1].mean)
    assert np.allclose(pmbm.mbm[1].states[i].covar,
                       expected_states[1].covar)


def test_update():
  raise NotImplementedError


if __name__ == '__main__':
  test_predict()
  pytest.main([__file__])
