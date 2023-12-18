from motpy.rfs.tomb import TOMBP
import pytest
import numpy as np
from motpy.distributions.gaussian import GaussianState
from motpy.kalman import KalmanFilter
from motpy.models.measurement import LinearMeasurementModel
import copy

def test_predict():
  raise NotImplementedError


def test_update():
  measurements = [np.array([0, 0]), np.array([1, 1]), np.array([100, 100])]
  states = [GaussianState(mean=np.array([0, 0]), covar=np.eye(2)),
            GaussianState(mean=np.array([1, 1]), covar=np.eye(2))]

  tomb = TOMBP(birth_weights=np.array([0.01, 0.01]), birth_states=copy.deepcopy(states),
               w_min=None, r_min=None, r_estimate_threshold=None)
  # Add MB components
  tomb.mb.r = np.array([0.5, 0.5])
  tomb.mb.states = states

  kf = KalmanFilter(measurement_model=LinearMeasurementModel(
      ndim_state=2, covar=np.eye(2), measured_dims=[0, 1]))
  tomb.update(measurements=measurements, state_estimator=kf, pd=0.9)


if __name__ == '__main__':
  test_update()
  pytest.main([__file__])
