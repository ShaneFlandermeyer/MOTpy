import numpy as np
import pytest
import scipy.stats
from filterpy.kalman import predict, update

from motpy.distributions.gaussian import Gaussian
from motpy.estimators.kalman import KalmanFilter
from motpy.models.measurement import LinearMeasurementModel
from motpy.models.transition import ConstantVelocity


def test_predict():
  """
  Test kalman predict step. Example data from kalman filter ebook.
  """
  state = Gaussian(
      mean=np.array([11.35, 4.5]),
      covar=np.array([[545, 150], [150, 500]])
  )
  dt = 0.3
  cv = ConstantVelocity(state_dim=2, w=0.01)
  F = cv.matrix(dt=dt)
  Q = cv.covar(dt=dt)

  kf = KalmanFilter(
      transition_model=cv,
      measurement_model=None
  )
  pred_state = kf.predict(state=state, dt=dt)
  x_expected, P_expected = predict(
      x=state.mean, P=state.covar, F=F, Q=Q
  )
  assert np.allclose(pred_state.mean, x_expected)
  assert np.allclose(pred_state.covar, P_expected)


def test_update():
  """
  Test kalman update step. Example data from kalman filter ebook.
  """
  cv = ConstantVelocity(state_dim=2, w=0.01)
  lin = LinearMeasurementModel(
      state_dim=2, covar=np.array([[5.]]), measured_dims=[0])
  H = lin.matrix()
  R = lin.covar()
  z = 1
  state = Gaussian(
      mean=np.array([12.7, 4.5]),
      covar=np.array([[545, 150], [150, 500]])
  )
  kf = KalmanFilter(transition_model=cv, measurement_model=lin)

  post_state = kf.update(measurement=z, state=state)
  x_expected, P_expected = update(
      x=state.mean, P=state.covar, z=z, R=R, H=H
  )
  assert np.allclose(post_state.mean, x_expected)
  assert np.allclose(post_state.covar, P_expected)


def test_likelihood():
  linear = LinearMeasurementModel(state_dim=2, covar=np.eye(2))
  kf = KalmanFilter(transition_model=None, measurement_model=linear)

  state = Gaussian(mean=np.array([0, 0]), covar=np.eye(2))
  z = np.array([1, 1])
  l = kf.likelihood(z, state)

  l_expected = scipy.stats.multivariate_normal.pdf(
      z, state.mean, state.covar+linear.covar()
  )

  assert np.allclose(l, l_expected)


def test_gate():
  linear = LinearMeasurementModel(state_dim=2, covar=np.eye(2))
  kf = KalmanFilter(transition_model=None, measurement_model=linear)

  state = Gaussian(mean=np.array([0, 0]), covar=np.eye(2))
  z = np.array([1, 1])
  gate_mask = kf.gate(measurements=z, state=state, pg=1)
  expected = np.ones((1, 1))
  assert np.all(gate_mask == expected)

  z = np.array([5, 5])
  gate_mask = kf.gate(measurements=z, state=state, pg=0.999)
  expected = np.zeros((1, 1))
  assert np.all(gate_mask == expected)


if __name__ == '__main__':
  pytest.main([__file__])
