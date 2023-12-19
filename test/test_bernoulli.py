from typing import Union
import pytest
import numpy as np
from motpy.rfs.bernoulli import Bernoulli
from motpy.kalman import KalmanFilter, UnscentedKalmanFilter
from motpy.distributions.gaussian import GaussianState


class TestConstantVelocityModel():
  __test__ = False

  def __init__(self,
               sigma: Union[float, np.ndarray] = 1,
               ):
    self.sigma = sigma

    self.ndim_state = 4

  def __call__(
      self,
      state: np.array,
      dt: float = 0,
      **kwargs,
  ) -> np.ndarray:
    next_state = np.dot(self.matrix(dt), state)
    return next_state

  def matrix(self, dt: float):
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return F

  def covar(self, dt: float):
    Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                  [0, dt**4/4, 0, dt**3/2],
                  [dt**3/2, 0, dt**2, 0],
                  [0, dt**3/2, 0, dt**2]])*self.sigma**2
    return Q


class LinearMeasurementModel():
  __test__ = False

  def __init__(self, sigma):
    self.sigma = sigma
    self.ndim_state = 4
    self.ndim_meas = 2

  def __call__(self, state, **kwargs):
    return np.dot(self.matrix(), state)

  def matrix(self):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

  def covar(self):
    return np.eye(self.ndim_meas) * self.sigma**2


def test_predict():
  dt = 1
  motion_model = TestConstantVelocityModel(sigma=1)
  kf = UnscentedKalmanFilter(transition_model=motion_model, measurement_model=None)
  ps = 0.8147
  r = 0.9058
  bern = Bernoulli(r=r,
                   state=GaussianState(mean=np.ones(4), covar=np.eye(4)))
  bern = bern.predict(state_estimator=kf, ps=ps, dt=dt)

  expected_mean = np.array([2, 2, 1, 1])
  expected_covar = np.array([
      [2.25, 0.0, 1.5, 0.0],
      [0.0, 2.25, 0., 1.5],
      [1.5, 0.0, 2., 0.0],
      [0.0, 1.5, 0.0, 2.]])
  assert np.allclose(bern.state.mean, expected_mean)
  assert np.allclose(bern.state.covar, expected_covar)
  assert bern.r == r*ps


def test_undetected_update():
  pd = 0.8147
  r = 0.9058
  bern = Bernoulli(r=r,
                   state=GaussianState(mean=np.ones(4), covar=np.eye(4)))
  init_state = bern.state
  bern = bern.update(pd=pd, measurement=None)

  assert bern.state == init_state
  assert bern.r == r * (1 - pd) / (1 - r + r * (1 - pd))


def test_detected_update():
  pd = 0.8147
  bern = Bernoulli(r=0.9058,
                   state=GaussianState(mean=np.ones(4), covar=np.eye(4)))
  measurement_model = LinearMeasurementModel(sigma=1)
  kf = KalmanFilter(transition_model=None, measurement_model=measurement_model)

  measurement = np.ones(2)
  bern = bern.update(pd=pd, measurement=measurement, state_estimator=kf)

  expected_mean = np.array([1, 1, 1, 1])
  expected_covar = np.array([
      [0.5, 0.0, 0.0, 0.0],
      [0.0, 0.5, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]])
  assert np.allclose(bern.state.mean, expected_mean)
  assert np.allclose(bern.state.covar, expected_covar)
  assert bern.r == 1 

def test_undetected_likelihood():
  pd = 0.8147
  r = 0.9058
  bern = Bernoulli(r=r,
                   state=GaussianState(mean=np.ones(4), covar=np.eye(4)))
  l = bern.log_likelihood(pd=pd, measurements=None)

  assert np.allclose(l, -1.3392400264406643, atol=1e-6)

def test_detected_likelihood():
  pd = 0.8147
  bern = Bernoulli(r=0.9058,
                   state=GaussianState(mean=np.ones(4), covar=np.eye(4)))
  measurement_model = LinearMeasurementModel(sigma=1)
  kf = KalmanFilter(transition_model=None, measurement_model=measurement_model)

  measurements = np.ones(2)
  l = bern.log_likelihood(pd=pd, measurements=measurements, state_estimator=kf)

  assert np.allclose(l, -2.8348963264948552, atol=1e-6)


if __name__ == '__main__':
  # test_predict()
  pytest.main([__file__])
