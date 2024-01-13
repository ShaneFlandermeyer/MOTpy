from __future__ import annotations
import copy
import numpy as np
from typing import Callable, List, Optional, Tuple, Union

from motpy.kalman import KalmanFilter
from motpy.measures import mahalanobis
from motpy.rfs.bernoulli import Bernoulli
from motpy.distributions.gaussian import GaussianMixture, mix_gaussians, GaussianState
from scipy.stats import multivariate_normal


class Poisson:
  """
  Class to hold all poisson distributions. Methods include birth, prediction, merge, prune, recycle.
  """

  def __init__(
      self,
      birth_distribution: GaussianMixture,
      init_distribution: Optional[GaussianMixture] = None,
  ):
    self.birth_distribution = birth_distribution

    self.distribution = init_distribution
    if init_distribution is None:
      state_dim = birth_distribution.state_dim
      self.distribution = GaussianMixture(
          means=np.empty([0, state_dim]),
          covars=np.empty([0, state_dim, state_dim]),
          weights=np.array([]),
      )

  def __repr__(self):
    return f"Poisson(weights={self.weights.tolist()}, \nstates={self.states})"

  def __len__(self):
    assert len(self.weights) == len(self.states)
    return len(self.weights)

  def __iter__(self):
    return zip(self.weights, self.states)

  def append(self,
             weight: Union[float, np.ndarray],
             state: Union[GaussianState, List[GaussianState]]) -> None:
    """
    Append a new Gaussian state and its corresponding weight to the Poisson process.

    The weight is converted to logarithmic scale before being appended.

    Parameters
    ----------
    weight : float
        The weight corresponding to the state. This is converted to a logarithmic scale for internal storage.
    state : GaussianState
        The Gaussian state to be appended.

    Returns
    -------
    None
    """
    state = state if isinstance(state, list) else [state]
    self.weights = np.append(self.weights, weight)
    self.states.extend(state)

  def predict(self,
              state_estimator: KalmanFilter,
              ps: float,
              dt: float) -> Poisson:
    # Predict existing PPP density
    pred_ppp = copy.deepcopy(self)

    pred_ppp.distribution.weights *= ps
    pred_ppp.distribution = state_estimator.predict(
        state=pred_ppp.distribution, dt=dt)
    pred_ppp.distribution.append(pred_ppp.birth_distribution)

    return pred_ppp

  # @profile
  def update(self,
             measurement: np.ndarray,
             pd: np.ndarray,
             likelihoods: np.ndarray,
             in_gate: np.ndarray,
             state_estimator: KalmanFilter,
             clutter_intensity: float,
             ) -> Tuple[Bernoulli, float]:
    n_in_gate = np.count_nonzero(in_gate)
    if n_in_gate == 0:
      # No measurements in gate
      return None, 0

    # If a measurement is associated to a PPP component, we create a new Bernoulli whose existence probability depends on likelihood of measurement
    mixture_up = state_estimator.update(
        measurement=measurement, predicted_state=self.distribution[in_gate])
    mixture_up.weights *= likelihoods[in_gate] * pd[in_gate]

    # Create a new Bernoulli component based on updated weights
    sum_w_up = np.sum(mixture_up.weights)
    sum_w_total = sum_w_up + clutter_intensity
    r = sum_w_up / sum_w_total

    # Compute the state using moment matching across all PPP components
    mean, covar = mix_gaussians(
        means=mixture_up.means,
        covars=mixture_up.covars,
        weights=mixture_up.weights)
    bern = Bernoulli(r=r, state=GaussianState(mean=mean, covar=covar))
    return bern, sum_w_total

  def prune(self, threshold: float) -> Poisson:
    pruned = copy.deepcopy(self)
    # Prune components with existence probability below threshold
    keep = self.weights > threshold
    pruned.distribution = self.distribution[keep]

    return pruned

  def merge(self, threshold: float) -> Poisson:
    """
    Merge components that are close to each other.

    TODO: Currently assumes there is no thresholding
    """
    raise NotImplementedError(
        "Merging currently not supported with GaussianMixture API")
    nbirth = len(self.birth_states)
    assert len(self.states) == 2 * \
        nbirth, "Merging currently only supported when PPP states come directly from birth states"

    birth_states = self.states[-nbirth:]
    birth_weights = self.weights[-nbirth:]
    persistent_states = self.states[:-nbirth]
    persistent_weights = self.weights[:-nbirth]

    merged = Poisson(birth_weights=birth_weights, birth_states=birth_states)
    # Sum birth and consistent components, mix their distributions
    wmix = np.concatenate(
        (persistent_weights[:, None], birth_weights[:, None]), axis=1)
    wmix = wmix / np.sum(wmix + 1e-15, axis=1, keepdims=True)
    Pmix = np.concatenate(
        (persistent_states.covar[None, ...], birth_states.covar[None, ...]), axis=0)
    merged.states = GaussianState(
        mean=persistent_states.mean,
        covar=np.einsum('...i, i...jk -> ...jk', wmix, Pmix))
    merged.weights = birth_weights + persistent_weights
    return merged

  def intensity(self, grid: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Compute the intensity of the Poisson process at a grid of points.

    Parameters
    ----------
    grid : np.ndarray
        Query points
    H : np.ndarray
        Matrix for extracting relevant dims

    Returns
    -------
    np.ndarray
        Intensity grid
    """
    raise NotImplementedError(
        "Intensity currently not supported with GaussianMixture API")
    intensity = np.zeros(grid.shape[:-1])
    for i, state in enumerate(self.states):
      mean = H @ state.mean[0]
      cov = H @ state.covar[0] @ H.T
      rv = multivariate_normal(mean=mean, cov=cov)
      intensity += self.weights[i] * rv.pdf(grid)
    return intensity
