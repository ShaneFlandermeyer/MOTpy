import matplotlib.pyplot as plt
import numpy as np
import pytest
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints

from motpy.distributions.gaussian import Gaussian
from motpy.estimators.kalman import UnscentedKalmanFilter, SigmaPointDistribution
from motpy.estimators.kalman.sigma_points import (merwe_scaled_sigma_points,
                                                  merwe_sigma_weights)
from motpy.models.measurement.linear import LinearMeasurementModel
from motpy.models.measurement.range_bearing import RangeBearingModel
from motpy.models.transition import ConstantVelocity
import scipy.stats


def test_predict():

  seed = 0
  dt = 1
  w = 0.01
  alpha = 0.1
  beta = 2
  kappa = 0
  distribution = Gaussian(
      mean=np.array([0, 1, 0, 1]),
      covar=np.diag([1., 0.5, 1., 0.5])
  )
  state_dim = distribution.ndim

  # Motpy UKF
  cv = ConstantVelocity(ndim_state=state_dim, w=w, seed=seed)
  ukf = UnscentedKalmanFilter(transition_model=cv, measurement_model=None)
  sigmas = merwe_scaled_sigma_points(
      x=distribution.mean,
      P=distribution.covar,
      alpha=alpha,
      beta=beta,
      kappa=kappa
  )
  Wm, Wc = merwe_sigma_weights(
      ndim_state=state_dim, alpha=alpha, beta=beta, kappa=kappa
  )
  state = SigmaPointDistribution(
      distribution=distribution,
      sigma_points=sigmas,
      Wm=Wm,
      Wc=Wc
  )
  pred_state = ukf.predict(state=state, dt=dt)

  # Filterpy UKF
  sigmas = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2, kappa=0)
  expected = UKF(
      dim_x=4,
      dim_z=0,
      dt=dt,
      hx=None,
      fx=cv,
      points=sigmas
  )
  expected.x = distribution.mean
  expected.P = distribution.covar
  expected.Q = cv.covar(dt=dt)
  expected.predict()

  assert np.allclose(pred_state.distribution.mean, expected.x)
  assert np.allclose(pred_state.distribution.covar, expected.P)


def test_update():
  alpha = 0.1
  beta = 2
  kappa = 0
  distribution = Gaussian(
      mean=np.array([0, 1, 0, 1]),
      covar=np.diag([1., 0.5, 1., 0.5])
  )

  R = np.diag([0.1, np.deg2rad(0.1)])
  range_bearing = RangeBearingModel(covar=R)

  state_dim = distribution.ndim
  ground_truth = np.array([0, 1, 0, 1])
  z = range_bearing(ground_truth, noise=False)

  sigmas = merwe_scaled_sigma_points(
      x=distribution.mean, P=distribution.covar, alpha=alpha, beta=beta, kappa=kappa
  )
  Wm, Wc = merwe_sigma_weights(
      ndim_state=state_dim, alpha=alpha, beta=beta, kappa=kappa
  )

  # Motpy UKF
  state = SigmaPointDistribution(
      distribution=distribution,
      sigma_points=sigmas,
      Wm=Wm,
      Wc=Wc
  )
  ukf = UnscentedKalmanFilter(
      transition_model=None,
      measurement_model=range_bearing
  )

  post_state = ukf.update(state=state, measurement=z)

  # Filterpy UKF
  points = MerweScaledSigmaPoints(
      n=state_dim, alpha=alpha, beta=beta, kappa=kappa
  )
  expected = UKF(
      dim_x=4,
      dim_z=2,
      dt=0,
      fx=None,
      hx=range_bearing,
      points=points
  )
  expected.x = distribution.mean
  expected.P = distribution.covar
  expected.R = R
  expected.sigmas_f = sigmas
  expected.update(z)

  assert np.allclose(post_state.distribution.mean, expected.x)
  assert np.allclose(post_state.distribution.covar, expected.P)


def test_likelihood():
  state_dim = 2
  linear = LinearMeasurementModel(
      ndim_state=state_dim, covar=np.eye(state_dim)
  )
  ukf = UnscentedKalmanFilter(transition_model=None, measurement_model=linear)

  Wm, Wc = merwe_sigma_weights(
      ndim_state=state_dim, alpha=0.1, beta=2, kappa=0
  )
  distribution = Gaussian(mean=np.zeros(state_dim), covar=np.eye(state_dim))
  sigma_points = merwe_scaled_sigma_points(
      x=distribution.mean, P=distribution.covar, alpha=0.1, beta=2, kappa=0
  )
  state = SigmaPointDistribution(
      distribution=distribution,
      sigma_points=sigma_points,
      Wm=Wm,
      Wc=Wc
  )

  z = np.array([1, 1])
  l = ukf.likelihood(z, state)

  l_expected = scipy.stats.multivariate_normal.pdf(
      z, state.distribution.mean, state.distribution.covar + linear.covar()
  )

  assert np.allclose(l, l_expected)


def test_gate():
  state_dim = 2
  linear = LinearMeasurementModel(
      ndim_state=state_dim, covar=np.eye(state_dim)
  )
  ukf = UnscentedKalmanFilter(transition_model=None, measurement_model=linear)

  Wm, Wc = merwe_sigma_weights(
      ndim_state=state_dim, alpha=0.1, beta=2, kappa=0
  )
  distribution = Gaussian(mean=np.zeros(state_dim), covar=np.eye(state_dim))
  sigma_points = merwe_scaled_sigma_points(
      x=distribution.mean, P=distribution.covar, alpha=0.1, beta=2, kappa=0
  )
  state = SigmaPointDistribution(
      distribution=distribution,
      sigma_points=sigma_points,
      Wm=Wm,
      Wc=Wc
  )

  z = np.array([1, 1])
  gate_mask = ukf.gate(measurements=z, state=state, pg=1)
  expected = np.ones(1)
  assert np.all(gate_mask == expected)

  z = np.array([5, 5])
  gate_mask = ukf.gate(measurements=z, state=state, pg=0.999)
  expected = np.zeros(1)
  assert np.all(gate_mask == expected)


if __name__ == '__main__':
  pytest.main([__file__])
