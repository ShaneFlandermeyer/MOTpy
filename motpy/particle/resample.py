import numpy as np


def systematic_resample(weights: np.ndarray) -> np.ndarray:
  N = len(weights)
  offset = np.random.uniform(0, 1)
  positions = (np.arange(N) + offset) / N
  return np.digitize(positions, bins=np.cumsum(weights))
