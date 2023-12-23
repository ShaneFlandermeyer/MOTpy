import abc
from typing import List


class MeasurementModel(abc.ABC):
  """Base class for measurement models"""

  @abc.abstractmethod
  def __call__(self, x, **kwargs):
    """Measurement function"""
    pass

  @abc.abstractmethod
  def matrix(self, **kwargs):
    """Measurement matrix"""
    pass
  
  @abc.abstractmethod
  def covar(self, **kwargs):
    """Measurement covariance"""
    pass
