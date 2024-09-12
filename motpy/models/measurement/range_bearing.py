from typing import List, Tuple
import numpy as np


class RangebearingModel():
  def __init__(self,
               covar,
               pos_dims: List[int] = [0, 2],
               seed: int = np.random.randint(0, 2**32-1)
               ):
    self.covar = covar
    self.pos_dims = pos_dims
    self.ndim = 2

    self.np_random = np.random.RandomState(seed)

  def __call__(self, state, noise: bool = False):
    state = np.array(state)
    x, y = state[..., self.pos_dims]
    
    range_ = np.sqrt(x**2 + y**2)
    bearing = np.arctan2(y, x)
    
    z = np.stack([range_, bearing], axis=-1)
    if noise:
      z += self.sample_noise(size=z.shape[:-1])
      
    return z

  def sample_noise(self, size: Tuple[int, ...]) -> np.ndarray:
    return self.np_random.multivariate_normal(
        mean=np.zeros(self.ndim), cov=self.covar, size=size)
