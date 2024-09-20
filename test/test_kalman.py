import numpy as np
import pytest
from filterpy.kalman import predict, update

from motpy.distributions.gaussian import Gaussian
from motpy.kalman import KalmanFilter
import numpy as np

from motpy.distributions import Gaussian
from motpy.kalman import KalmanFilter
from motpy.models.transition import ConstantVelocity
from motpy.models.measurement import LinearMeasurementModel


def test_predict():
  """
  Test kalman predict step. Example data from kalman filter ebook.
  """
  state = Gaussian(
      mean=np.array([11.35, 4.5]),
      covar=np.array([[545, 150], [150, 500]]))
  dt = 0.3
  cv = ConstantVelocity(ndim_state=2, w=0.01, seed=0)
  F = cv.matrix(dt=dt)
  Q = cv.covar(dt=dt)

  kf = KalmanFilter(
      transition_model=cv,
      measurement_model=None)
  pred_state, _ = kf.predict(state=state, dt=dt)
  x_expected, P_expected = predict(
      x=state.mean, P=state.covar, F=F, Q=Q)
  assert np.allclose(pred_state.mean, x_expected)
  assert np.allclose(pred_state.covar, P_expected)


def test_update():
  """
  Test kalman update step. Example data from kalman filter ebook.
  """
  cv = ConstantVelocity(ndim_state=2, w=0.01)
  lin = LinearMeasurementModel(
      ndim_state=2, covar=np.array([[5.]]), measured_dims=[0])
  H = lin.matrix()
  R = lin.covar()
  z = 1
  state = Gaussian(
      mean=np.array([12.7, 4.5]),
      covar=np.array([[545, 150], [150, 500]]),
  )
  kf = KalmanFilter(
      transition_model=cv,
      measurement_model=lin,
  )

  state_post, _ = kf.update(measurement=z, state=state)
  x_expected, P_expected = update(
      x=state.mean, P=state.covar, z=z, R=R, H=H)
  assert np.allclose(state_post.mean, x_expected)
  assert np.allclose(state_post.covar, P_expected)


if __name__ == '__main__':
  pytest.main([__file__])
