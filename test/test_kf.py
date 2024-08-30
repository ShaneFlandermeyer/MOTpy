import numpy as np
import pytest
from filterpy.kalman import predict, update

from motpy.distributions.gaussian import GaussianState
from motpy.kalman import KalmanFilter
import numpy as np

from motpy.distributions import GaussianState
from motpy.kalman import KalmanFilter
from motpy.models.transition import ConstantVelocity
from motpy.models.measurement import LinearMeasurementModel


class TestLinearTransitionModel():
  def __call__(self, state, dt, noise=False, **_):
    if noise:
      process_noise = self.covar()
    else:
      process_noise = 0

    return self.matrix(dt) @ state + process_noise

  def matrix(self, dt, **_):
    return np.array([[1, dt], [0, 1]])

  def covar(self, **_):
    return np.array([[0.588, 1.175],
                     [1.175, 2.35]])


class TestLinearMeasurementModel():
  def __call__(self, state):
    return self.matrix() @ state

  def matrix(self):
    return np.atleast_2d(np.array([1, 0]))

  def covar(self, dt=None):
    return np.array([[5.]])


def test_kalman_predict():
  """
  Test kalman predict step. Example data from kalman filter ebook.
  """
  state = GaussianState(
      state_dim=2,
      mean=np.array([11.35, 4.5]),
      covar=np.array([[545, 150], [150, 500]]))
  dt = 0.3
  transition_model = TestLinearTransitionModel()
  F = transition_model.matrix(dt)
  Q = transition_model.covar()

  kf = KalmanFilter(
      transition_model=TestLinearTransitionModel(),
      measurement_model=None)
  state_pred, meta = kf.predict(state=state, dt=dt)
  x_expected, P_expected = predict(
      x=state.mean, P=state.covar, F=F, Q=Q)
  assert np.allclose(state_pred.mean, x_expected)
  assert np.allclose(state_pred.covar, P_expected)


def test_kalman_update():
  """
  Test kalman update step. Example data from kalman filter ebook.
  """
  measurement_model = TestLinearMeasurementModel()
  R = measurement_model.covar()
  H = measurement_model.matrix()
  z = 1
  state = GaussianState(
      state_dim=2,
      mean=np.array([12.7, 4.5]),
      covar=np.array([[545, 150], [150, 500]]),
  )
  kf = KalmanFilter(
      transition_model=TestLinearTransitionModel(),
      measurement_model=TestLinearMeasurementModel(),
  )

  state_post, meta = kf.update(measurement=z, predicted_state=state)
  x_expected, P_expected = update(
      x=state.mean, P=state.covar, z=z, R=R, H=H)
  assert np.allclose(state_post.mean, x_expected)
  assert np.allclose(state_post.covar, P_expected)


if __name__ == '__main__':
  pytest.main([__file__])
