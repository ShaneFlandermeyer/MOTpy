from abc import abstractmethod


class TransitionModel():
  """Base class for motion model objects"""

  @abstractmethod
  def __call__(self, state, dt, **kwargs):
    """Predict the next state given the current state and time step

    Parameters
    ----------
    state : np.ndarray
        Current state
    dt : float
        Time step

    Returns
    -------
    np.ndarray
        Predicted state
    """
    pass

  @abstractmethod
  def matrix(self, **kwargs):
    """Return the transition matrix

    Returns
    -------
    np.ndarray
        Transition matrix
    """
    pass

  @abstractmethod
  def covar(self, **kwargs):
    """Return the transition covariance matrix

    Returns
    -------
    np.ndarray
        Transition covariance matrix
    """
    pass
