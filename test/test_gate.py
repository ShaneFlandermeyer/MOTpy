import pytest
import numpy as np
from motpy.gate import EllipsoidalGate


def test_single_dist_multi_query():
  measurements = np.array([[1, 1], [2, 2], [3, 3]])
  z_pred = np.array([1, 1])
  S = np.array(np.eye(2))
  pg = 0.99

  gate = EllipsoidalGate(pg=pg, ndim=2)
  in_gate, squared_dist = gate(measurements=measurements,
                               predicted_measurement=z_pred, innovation_covar=S)

  assert in_gate.shape == (3,)
  assert squared_dist.shape == (3,)
  assert np.all(in_gate == np.ones(3, dtype=bool))
  assert np.allclose(squared_dist, np.array([0, 2, 8]))


def test_multi_dist_multi_query():
  measurements = np.array([[1, 1], [2, 2], [3, 3]])
  z_pred = np.array([[1, 1], [2, 2]])
  S = np.array([np.eye(2)]*2)
  pg = 0.99

  gate = EllipsoidalGate(pg=pg, ndim=2)
  in_gate, squared_dist = gate(measurements=measurements,
                               predicted_measurement=z_pred, innovation_covar=S)

  assert in_gate.shape == (2, 3)
  assert squared_dist.shape == (2, 3)
  assert np.all(in_gate == np.ones((2, 3), dtype=bool))
  assert np.allclose(squared_dist, np.array([[0, 2, 8], [2, 0, 2]]))


if __name__ == "__main__":
  pytest.main([__file__])
