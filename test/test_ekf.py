import numpy as np
import pytest
from filterpy.kalman import predict, update

from motpy.distributions.gaussian import GaussianState
import numpy as np

from motpy.distributions import GaussianState
from motpy.kalman import ExtendedKalmanFilter
from motpy.models.transition import ConstantVelocity
from motpy.models.measurement import LinearMeasurementModel


class TestLinearTransitionModel():
  def __call__(self, prior, dt, noise=False):
    if noise:
      process_noise = self.covar()
    else:
      process_noise = 0

    return self.matrix(dt) @ prior + process_noise

  def matrix(self, state, dt):
    return np.array([[1, dt], [0, 1]])

  def covar(self, dt=None):
    return np.array([[0.588, 1.175],
                     [1.175, 2.35]])


class TestLinearMeasurementModel():
  def function(self, state):
    return self.matrix() @ state

  def matrix(self):
    return np.atleast_2d(np.array([1, 0]))

  def covar(self, dt=None):
    return np.array([[5.]])

def test_predict():
  """
  Test kalman predict step. Example data from kalman filter ebook.
  """
  state = GaussianState(mean=np.array([11.35, 4.5]),
                        covar=np.array([[545, 150], [150, 500]]))
  dt = 0.3
  transition_model = TestLinearTransitionModel()
  F = transition_model.matrix(state, dt)
  Q = transition_model.covar()

  ekf = ExtendedKalmanFilter(
      transition_model=TestLinearTransitionModel(),
      measurement_model=None)
  state_pred = ekf.predict(state=state, dt=dt)
  x_expected, P_expected = predict(
      x=state.mean[0], P=state.covar[0], F=F, Q=Q)
  assert np.allclose(state_pred.mean, x_expected)
  assert np.allclose(state_pred.covar, P_expected)


# def test_update():
#   """
#   Test kalman update step. Example data from kalman filter ebook.
#   """
#   measurement_model = TestLinearMeasurementModel()
#   R = measurement_model.covar()
#   H = measurement_model.matrix()
#   z = 1
#   state = GaussianState(mean=np.array([12.7, 4.5]),
#                         covar=np.array([[545, 150], [150, 500]]))
#   kf = ExtendedKalmanFilter(
#       transition_model=TestLinearTransitionModel(),
#       measurement_model=TestLinearMeasurementModel(),
#   )

#   state_post = kf.update(measurement=z, predicted_state=state)
#   x_expected, P_expected = update(
#       x=state.mean[0], P=state.covar[0], z=z, R=R, H=H)
#   assert np.allclose(state_post.mean, x_expected)
#   assert np.allclose(state_post.covar, P_expected)


if __name__ == '__main__':
  pytest.main([__file__])
