from typing import Tuple
import numpy as np


def systematic_resample(weights: np.ndarray, n: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
  offset = np.random.uniform(0, 1)
  positions = (np.arange(n) + offset) / n
  new_weights = np.full(n, 1 / n)
  inds = np.digitize(positions, bins=np.cumsum(weights))
  return inds, new_weights
