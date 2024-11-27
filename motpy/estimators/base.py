from motpy.distributions import Distribution
import numpy as np


class StateEstimator():
  def predict(self, state: Distribution, dt: float, **kwargs) -> Distribution:
    """
    Predict step
    """
    raise NotImplementedError

  def update(self,
             state: Distribution,
             measurement: np.ndarray,
             **kwargs
             ) -> Distribution:
    """
    Update state with a single measurement
    """
    raise NotImplementedError

  def update_vectorized(self,
                        state: Distribution,
                        measurements: np.ndarray,
                        **kwargs
                        ) -> Distribution:
    """
    Update multiple state-measurement pairs
    """
    raise NotImplementedError
