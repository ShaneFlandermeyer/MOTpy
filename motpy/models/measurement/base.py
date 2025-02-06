import abc
from typing import List
import numpy as np

class MeasurementModel(abc.ABC):
  """Base class for measurement models"""

  @abc.abstractmethod
  def __call__(self, x, **kwargs):
    """Measurement function"""
    pass
  
  @abc.abstractmethod
  def covar(self, **kwargs) -> np.ndarray:
    """Measurement covariance"""
    pass
